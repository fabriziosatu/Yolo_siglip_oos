"""
src/models/joint_pipeline.py
=============================
Complete joint pipeline: YOLO26 (with DetectNoDetach, differentiable
on box coordinates) + SigLIPv2 with RoI Align.

CHANGES compared to your version (the rest is UNCHANGED — same
double-threshold IoU matching, same _assign_roi_labels logic, same
differentiable flow through rois):

  1. SigLIP is now variant-aware: it's built with build_siglip_module(cfg)
     instead of always SigLIPModule(...), and forward() handles both
     output shapes ({"logit"} or {"logits_pos","logits_neg"}) under
     the uniform key "siglip_out".
  2. Synthetic negative mining now uses neg_ratio (proportional to the
     number of GTs in the image, as in Phase 3) instead of the fixed integer
     n_neg_synthetic. See _sample_negative_rois: the signature is the same,
     only how n_target is calculated in the caller changes
     (_add_synthetic_negatives).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, box_iou

from src.models.detector      import YOLO26Detector, FEATURE_STRIDE, FEATURE_CHANNELS
from src.models.siglip_module import build_siglip_module


ROI_OUTPUT_SIZE = 7


class JointPipeline(nn.Module):
    """
    Joint end-to-end model combining the YOLO26 detector and the SigLIP 
    verification module using RoI Align to extract features from predicted boxes.
    """

    def __init__(
        self,
        cfg,
        yolo_weights      = None,
        conf_threshold    = None,
        roi_size          = None,
        iou_pos_thr       = 0.30,
        iou_neg_thr       = 0.10,
        neg_ratio         = None,
        neg_iou_max       = None,
    ):
        """
        Note: now also accepts `cfg` (used to build SigLIP with the
        right variant). The loose parameters remain for compatibility
        with how trainer.py passed them — if not explicitly specified
        they are read from cfg.
        """
        super().__init__()

        self.cfg = cfg
        yolo_weights   = yolo_weights   or cfg.detector.model_name
        conf_threshold = conf_threshold if conf_threshold is not None else cfg.detector.conf_threshold
        roi_size       = roi_size       or cfg.data.roi_size
        neg_ratio      = neg_ratio      if neg_ratio is not None else cfg.negative_mining.neg_ratio
        neg_iou_max    = neg_iou_max    if neg_iou_max is not None else cfg.negative_mining.neg_iou_max

        self.roi_size    = roi_size
        self.roi_padding = cfg.data.roi_padding
        self.max_rois_per_image = cfg.detector.max_rois_per_image
        self.iou_pos_thr = iou_pos_thr
        self.iou_neg_thr = iou_neg_thr
        self.neg_ratio   = neg_ratio
        self.neg_iou_max = neg_iou_max
        self.siglip_variant = cfg.siglip.variant

        print("\n[1/2] Loading YOLO26...")
        self.detector = YOLO26Detector(
            weights              = yolo_weights,
            conf_threshold       = conf_threshold,
            train_conf_threshold = cfg.detector.train_conf_threshold,
        )

        print(f"\n[2/2] Loading SigLIPv2 (variant '{self.siglip_variant}')...")
        self.siglip = build_siglip_module(cfg)   # loads pretrained weights internally if configured

        self.feature_proj = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def encode_prompts(self, positive_prompts: list, negative_prompts: list):
        """Delegates to SigLIPModule — no-op for the vision_mlp variant."""
        self.siglip.encode_prompts(positive_prompts, negative_prompts)

    def forward(self, images, gt_boxes=None, gt_labels=None):
        """
        Main forward pass combining object detection and verification.

        Args:
            images:    (B, 3, 640, 640)
            gt_boxes:  list of B tensors (Ni, 4) YOLO norm format (xc,yc,w,h)
            gt_labels: unused (single class), kept for compatibility

        Returns: dict with (UNCHANGED except 'siglip_out' instead of 'logits_pos'):
            predictions, feature_map, rois, siglip_out, roi_labels_gt, n_rois
        """
        B      = images.shape[0]
        device = images.device

        predictions, feature_map, *_ = self.detector(images)
        filtered_preds = self.detector.filter_predictions(predictions)
        filtered_preds = self._cap_rois_per_image(filtered_preds)

        if gt_boxes is not None and self.training:
            gt_xyxy_list = [
                self._yolo_to_xyxy_pixel(gt.to(device), images.shape[-1])
                for gt in gt_boxes
            ]
            roi_labels_list, filtered_preds = self._assign_roi_labels(
                filtered_preds, gt_xyxy_list, device
            )
        else:
            gt_xyxy_list = [torch.zeros(0, 4, device=device)] * B
            roi_labels_list = [
                torch.ones(len(fp), dtype=torch.long, device=device)
                for fp in filtered_preds
            ]

        if self.training and self.neg_ratio > 0 and gt_boxes is not None:
            filtered_preds, roi_labels_list = self._add_synthetic_negatives(
                filtered_preds, roi_labels_list,
                gt_xyxy_list, img_size=images.shape[-1], device=device
            )

        rois = self.detector.predictions_to_roi_format(filtered_preds, B)
        roi_labels_gt = torch.cat(roi_labels_list) if roi_labels_list else \
                        torch.zeros(0, dtype=torch.long, device=device)

        if rois.shape[0] == 0:
            empty_out = ({"logit": torch.zeros(0, 1, device=device)}
                         if self.siglip_variant == "vision_mlp"
                         else {"logits_pos": torch.zeros(0, 1, device=device),
                               "logits_neg": torch.zeros(0, 1, device=device)})
            return {
                "predictions":   predictions,
                "feature_map":   feature_map,
                "rois":          rois,
                "siglip_out":    empty_out,
                "roi_labels_gt": roi_labels_gt,
                "n_rois":        0,
            }

        roi_features = roi_align(
            input          = feature_map,
            boxes          = self._pad_rois(rois, images.shape[-1]),
            output_size    = ROI_OUTPUT_SIZE,
            spatial_scale  = 1.0 / FEATURE_STRIDE,
            sampling_ratio = 2,
            aligned        = True,
        )

        roi_rgb   = self.feature_proj(roi_features)
        roi_crops = F.interpolate(
            roi_rgb, size=(self.roi_size, self.roi_size),
            mode='bilinear', align_corners=False,
        )

        siglip_out = self.siglip(roi_crops)   # dict — variant-aware

        return {
            "predictions":   predictions,
            "feature_map":   feature_map,
            "rois":          rois,
            "siglip_out":    siglip_out,
            "roi_labels_gt": roi_labels_gt.to(device),
            "n_rois":        rois.shape[0],
        }

    def remove_hooks(self):
        """Delegates hook cleanup to the detector."""
        self.detector.remove_hooks()

    def _cap_rois_per_image(self, filtered_preds: list) -> list:
        """
        Limits the number of predictions per image to self.max_rois_per_image,
        keeping those with highest confidence (column 4: [x1,y1,x2,y2,conf,cls]).
        If an image has fewer than the limit, it remains unchanged. <=0 -> no-op.
        """
        if self.max_rois_per_image <= 0:
            return filtered_preds
        capped = []
        for preds in filtered_preds:
            if preds.shape[0] > self.max_rois_per_image:
                conf = preds[:, 4]
                top_idx = torch.topk(conf, self.max_rois_per_image).indices
                # Reorder by original index — not necessary for
                # correctness (downstream order doesn't matter), but more readable
                # in logs/debug.
                top_idx, _ = torch.sort(top_idx)
                preds = preds[top_idx]
            capped.append(preds)
        return capped

    # ── UNCHANGED — same double threshold matching logic ──────────────

    def _assign_roi_labels(self, filtered_preds, gt_xyxy_list, device):
        """Assigns GT labels (1 for positive, 0 for negative) to RoIs based on IoU thresholds."""
        roi_labels_list  = []
        clean_preds_list = []

        for i, preds in enumerate(filtered_preds):
            preds = preds.to(device)
            gt    = gt_xyxy_list[i]

            if len(preds) == 0:
                roi_labels_list.append(torch.zeros(0, dtype=torch.long, device=device))
                clean_preds_list.append(preds)
                continue

            if len(gt) == 0:
                roi_labels_list.append(torch.zeros(len(preds), dtype=torch.long, device=device))
                clean_preds_list.append(preds)
                continue

            iou_mat   = box_iou(preds[:, :4], gt)
            max_iou,_ = iou_mat.max(dim=1)

            pos_mask  = max_iou >= self.iou_pos_thr
            neg_mask  = max_iou <  self.iou_neg_thr
            keep_mask = pos_mask | neg_mask

            labels = torch.where(
                pos_mask,
                torch.ones_like(max_iou,  dtype=torch.long),
                torch.zeros_like(max_iou, dtype=torch.long),
            )

            roi_labels_list.append(labels[keep_mask])
            clean_preds_list.append(preds[keep_mask])

        return roi_labels_list, clean_preds_list

    # ── CHANGED — n_target now proportional to n_GT (neg_ratio) ─────────

    def _add_synthetic_negatives(self, filtered_preds, roi_labels_list,
                                  gt_xyxy_list, img_size, device):
        """Generates hard negative examples dynamically and adds them to the ROI lists."""
        new_preds  = []
        new_labels = []

        for i, preds in enumerate(filtered_preds):
            preds  = preds.to(device)
            labels = roi_labels_list[i].to(device)
            gt     = gt_xyxy_list[i]

            # BEFORE: n_target = max(n_pos, self.n_neg_synthetic)  <- fixed integer
            # NOW: proportional to the number of GTs in the image, like Phase 3
            n_gt = len(gt)
            n_target = round(n_gt * self.neg_ratio) if n_gt > 0 else 0

            synthetic = self._sample_negative_rois(gt, img_size, n_target, device)

            if synthetic.shape[0] > 0:
                syn_labels = torch.zeros(synthetic.shape[0], dtype=torch.long, device=device)
                preds  = torch.cat([preds,  synthetic],  dim=0)
                labels = torch.cat([labels, syn_labels], dim=0)

            new_preds.append(preds)
            new_labels.append(labels)

        return new_preds, new_labels

    # ── UNCHANGED ────────────────────────────────────────────────────────

    def _sample_negative_rois(self, gt_xyxy, img_size, n_target, device,
                               max_attempts=200, min_size_frac=0.05, max_size_frac=0.50):
        """Randomly samples bounding boxes that do not overlap with ground truths."""
        if n_target <= 0:
            return torch.zeros(0, 6, device=device)

        min_sz = max(1, int(min_size_frac * img_size))
        max_sz = max(min_sz + 1, int(max_size_frac * img_size))

        negatives = []
        attempts  = 0

        while len(negatives) < n_target and attempts < max_attempts:
            attempts += 1

            w  = torch.randint(min_sz, max_sz, (1,)).item()
            h  = torch.randint(min_sz, max_sz, (1,)).item()
            x1 = torch.randint(0, max(1, img_size - w), (1,)).item()
            y1 = torch.randint(0, max(1, img_size - h), (1,)).item()

            candidate = torch.tensor([[x1, y1, x1+w, y1+h]], dtype=torch.float32, device=device)

            if len(gt_xyxy) > 0:
                if box_iou(candidate, gt_xyxy).max().item() > self.neg_iou_max:
                    continue

            negatives.append(
                torch.tensor([x1, y1, x1+w, y1+h, 0.0, 0.0], dtype=torch.float32, device=device)
            )

        return torch.stack(negatives) if negatives else torch.zeros(0, 6, device=device)

    def _pad_rois(self, rois: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Adds context around each box BEFORE RoI Align — same
        logic as crop_with_padding() in Phase 3 (pad = dimension * fraction,
        clamped to image edges), applied here to coordinates instead of
        pixels. self.roi_padding=0 -> no-op (no overhead).

        Maintains differentiability (no .detach()): SigLIP's gradient
        continues to flow back to the original box coordinates,
        consistent with the rest of the pipeline (DetectNoDetach).
        """
        if self.roi_padding <= 0 or rois.shape[0] == 0:
            return rois

        s = float(img_size)
        batch_idx = rois[:, 0:1]
        x1, y1, x2, y2 = rois[:, 1], rois[:, 2], rois[:, 3], rois[:, 4]
        w = x2 - x1
        h = y2 - y1
        pad_w = w * self.roi_padding
        pad_h = h * self.roi_padding

        x1p = (x1 - pad_w).clamp(0, s)
        y1p = (y1 - pad_h).clamp(0, s)
        x2p = (x2 + pad_w).clamp(0, s)
        y2p = (y2 + pad_h).clamp(0, s)

        return torch.cat([
            batch_idx, x1p.unsqueeze(1), y1p.unsqueeze(1),
            x2p.unsqueeze(1), y2p.unsqueeze(1),
        ], dim=1)

    @staticmethod
    def _yolo_to_xyxy_pixel(boxes, img_size):
        """Utility to convert YOLO normalized format to absolute xyxy format."""
        if len(boxes) == 0:
            return boxes
        s  = float(img_size)
        xc, yc, w, h = boxes[:,0]*s, boxes[:,1]*s, boxes[:,2]*s, boxes[:,3]*s
        x1 = (xc - w/2).clamp(0, s)
        y1 = (yc - h/2).clamp(0, s)
        x2 = (xc + w/2).clamp(0, s)
        y2 = (yc + h/2).clamp(0, s)
        return torch.stack([x1, y1, x2, y2], dim=1)