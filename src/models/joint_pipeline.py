"""
src/models/joint_pipeline.py
=============================
Pipeline joint completa: YOLO26 + SigLIPv2 con RoI Align.

Flusso forward:
  1. YOLO26  → predizioni (B, 300, 6) + feature map (B, 128, 40, 40)
  2. Filtra predizioni per confidence threshold
  3. IoU matching con GT → assegna label alle ROI di YOLO
       IoU >= iou_pos_thr → positivo (label=1, empty_shelf confermato)
       IoU <  iou_neg_thr → negativo da YOLO (label=0, falso positivo)
       zona intermedia    → scartato (ambiguo)
  4. Negative mining sintetico → aggiunge ROI casuali che non si
     sovrappongono con nessuna GT (label=0, scaffale pieno per esclusione)
  5. RoI Align → crop (N_total, 128, 7, 7)
  6. Proiezione + resize → (N_total, 3, 224, 224)
  7. SigLIPv2 + MLP → logit binario (N_total, 1)

Negative mining sintetico:
  Campiona ROI casuali sull'immagine e le accetta solo se hanno
  IoU < neg_iou_max con tutte le GT box (empty_shelf).
  L'assunzione e' valida per dataset di scaffali: la maggior parte
  dell'area e' occupata da prodotti, quindi una box casuale che non
  si sovrappone con nessuno spazio vuoto cade quasi certamente su
  un prodotto o uno scaffale pieno.
  Il bilanciamento e' 1:1 rispetto ai positivi trovati da YOLO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, box_iou

from src.models.detector      import YOLO26Detector, FEATURE_STRIDE, FEATURE_CHANNELS
from src.models.siglip_module import SigLIPModule


ROI_OUTPUT_SIZE = 7


class JointPipeline(nn.Module):

    def __init__(
        self,
        yolo_weights      = "yolo26n.pt",
        siglip_model_name = "google/siglip2-base-patch16-224",
        conf_threshold    = 0.25,
        lora_r_visual     = 4,
        lora_alpha_visual = 8,
        lora_dropout      = 0.10,
        roi_size          = 224,
        iou_pos_thr       = 0.30,
        iou_neg_thr       = 0.10,
        n_neg_synthetic   = 4,
        neg_iou_max       = 0.10,
    ):
        super().__init__()

        self.roi_size        = roi_size
        self.iou_pos_thr     = iou_pos_thr
        self.iou_neg_thr     = iou_neg_thr
        self.n_neg_synthetic = n_neg_synthetic
        self.neg_iou_max     = neg_iou_max

        print("\n[1/2] Carico YOLO26...")
        self.detector = YOLO26Detector(
            weights        = yolo_weights,
            conf_threshold = conf_threshold,
        )

        print("\n[2/2] Carico SigLIPv2 + LoRA...")
        self.siglip = SigLIPModule(
            model_name        = siglip_model_name,
            lora_r_visual     = lora_r_visual,
            lora_alpha_visual = lora_alpha_visual,
            lora_dropout      = lora_dropout,
        )

        self.feature_proj = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def forward(self, images, gt_boxes=None, gt_labels=None):
        """
        Args:
            images:    (B, 3, 640, 640)
            gt_boxes:  lista di B tensor (Ni, 4) formato YOLO norm (xc,yc,w,h)
            gt_labels: non usato (una sola classe), mantenuto per compatibilita'
        """
        B      = images.shape[0]
        device = images.device

        predictions, feature_map, *_ = self.detector(images)
        filtered_preds = self.detector.filter_predictions(predictions)

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

        if self.training and self.n_neg_synthetic > 0 and gt_boxes is not None:
            filtered_preds, roi_labels_list = self._add_synthetic_negatives(
                filtered_preds, roi_labels_list,
                gt_xyxy_list, img_size=images.shape[-1], device=device
            )

        rois = self.detector.predictions_to_roi_format(filtered_preds, B)
        roi_labels_gt = torch.cat(roi_labels_list) if roi_labels_list else \
                        torch.zeros(0, dtype=torch.long, device=device)

        if rois.shape[0] == 0:
            return {
                "predictions":   predictions,
                "feature_map":   feature_map,
                "rois":          rois,
                "logits_pos":    torch.zeros(0, 1, device=device),
                "roi_labels_gt": roi_labels_gt,
                "n_rois":        0,
            }

        roi_features = roi_align(
            input          = feature_map,
            boxes          = rois,
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

        logits_pos = self.siglip(roi_crops)   # (N, 1) — MLP binario

        return {
            "predictions":   predictions,
            "feature_map":   feature_map,
            "rois":          rois,
            "logits_pos":    logits_pos,
            "roi_labels_gt": roi_labels_gt.to(device),
            "n_rois":        rois.shape[0],
        }

    def remove_hooks(self):
        self.detector.remove_hooks()

    def _assign_roi_labels(self, filtered_preds, gt_xyxy_list, device):
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

    def _add_synthetic_negatives(self, filtered_preds, roi_labels_list,
                                  gt_xyxy_list, img_size, device):
        new_preds  = []
        new_labels = []

        for i, preds in enumerate(filtered_preds):
            preds  = preds.to(device)
            labels = roi_labels_list[i].to(device)
            gt     = gt_xyxy_list[i]

            n_pos    = int(labels.sum().item())
            n_target = max(n_pos, self.n_neg_synthetic)

            synthetic = self._sample_negative_rois(gt, img_size, n_target, device)

            if synthetic.shape[0] > 0:
                syn_labels = torch.zeros(synthetic.shape[0], dtype=torch.long, device=device)
                preds  = torch.cat([preds,  synthetic],  dim=0)
                labels = torch.cat([labels, syn_labels], dim=0)

            new_preds.append(preds)
            new_labels.append(labels)

        return new_preds, new_labels

    def _sample_negative_rois(self, gt_xyxy, img_size, n_target, device,
                               max_attempts=200, min_size_frac=0.05, max_size_frac=0.50):
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

    @staticmethod
    def _yolo_to_xyxy_pixel(boxes, img_size):
        if len(boxes) == 0:
            return boxes
        s  = float(img_size)
        xc, yc, w, h = boxes[:,0]*s, boxes[:,1]*s, boxes[:,2]*s, boxes[:,3]*s
        x1 = (xc - w/2).clamp(0, s)
        y1 = (yc - h/2).clamp(0, s)
        x2 = (xc + w/2).clamp(0, s)
        y2 = (yc + h/2).clamp(0, s)
        return torch.stack([x1, y1, x2, y2], dim=1)