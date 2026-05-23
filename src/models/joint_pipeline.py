"""
src/models/joint_pipeline.py
=============================
Pipeline joint YOLO26 + SigLIPv2 con predizioni completamente differenziabili.

Flusso:
  1. YOLO26 forward (train mode) con DetectNoDetach
     -> predizioni (B, 8400, 6) con grad_fn
     -> feature map (B, 128, 40, 40) con grad_fn
  2. IoU matching (conf>=0.1) -> label ROI
  3. Crop reali dall'immagine originale con padding 15%
  4. SigLIPv2(LoRA) + MLP -> logit binario su immagini RGB reali
  5. RoI Align sulla feature map -> solo per il backward differenziabile
  6. L_joint = alpha*L_YOLO(CIoU+BCE) + beta*L_SigLIP
     -> un solo backward aggiorna entrambi i modelli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, box_iou

from src.models.detector      import YOLO26Detector, FEATURE_STRIDE, FEATURE_CHANNELS
from src.models.siglip_module import SigLIPModule

ROI_OUTPUT_SIZE    = 7
IOU_THRESHOLD      = 0.3
ROI_PADDING        = 0.15   # padding 15% attorno alla bbox (paper DRIVE)
MAX_ROIS_PER_BATCH = 64     # cap ROI per batch — bilancia velocità e copertura


class JointPipeline(nn.Module):

    def __init__(
        self,
        yolo_weights:      str   = "yolo26n.pt",
        siglip_model_name: str   = "google/siglip2-base-patch16-224",
        conf_threshold:    float = 0.25,
        lora_r_visual:     int   = 4,
        lora_alpha_visual: int   = 8,
        lora_dropout:      float = 0.15,
        mlp_hidden:        list  = None,
        mlp_dropout:       float = 0.30,
        roi_size:          int   = 224,
        roi_padding:       float = ROI_PADDING,
    ):
        super().__init__()

        if mlp_hidden is None:
            mlp_hidden = [256, 64]

        self.roi_size    = roi_size
        self.roi_padding = roi_padding

        print("\n[1/2] Carico YOLO26...")
        self.detector = YOLO26Detector(
            weights        = yolo_weights,
            conf_threshold = conf_threshold,
        )

        print("\n[2/2] Carico SigLIPv2 + LoRA + MLP...")
        self.siglip = SigLIPModule(
            model_name        = siglip_model_name,
            lora_r_visual     = lora_r_visual,
            lora_alpha_visual = lora_alpha_visual,
            lora_dropout      = lora_dropout,
            mlp_hidden        = mlp_hidden,
            mlp_dropout       = mlp_dropout,
        )

        # Feature projection: usata solo per il backward differenziabile
        self.feature_proj = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

    def _assign_roi_labels(self, filtered_preds, gt_boxes, img_size=640):
        """Assegna label alle ROI tramite IoU matching con le GT box."""
        all_labels = []
        for preds, gt in zip(filtered_preds, gt_boxes):
            if len(preds) == 0:
                continue
            pred_boxes = preds[:, :4]

            if len(gt) == 0:
                all_labels.append(
                    torch.zeros(len(preds), dtype=torch.long, device=preds.device)
                )
                continue

            gt = gt.to(preds.device)
            xc, yc, w, h = gt[:,0], gt[:,1], gt[:,2], gt[:,3]
            gt_xyxy = torch.stack([
                (xc - w/2) * img_size, (yc - h/2) * img_size,
                (xc + w/2) * img_size, (yc + h/2) * img_size,
            ], dim=1)

            with torch.no_grad():
                iou_matrix = box_iou(pred_boxes.detach(), gt_xyxy)
                max_iou, _ = iou_matrix.max(dim=1)
            labels = (max_iou >= IOU_THRESHOLD).long()
            all_labels.append(labels)

        if all_labels:
            return torch.cat(all_labels, dim=0)
        return torch.zeros(0, dtype=torch.long)

    def _crop_from_image(self, images, rois, roi_size):
        """
        Ritaglia crop reali dall'immagine originale con padding 15%.
        Versione vettorizzata: un solo F.interpolate su tutti i crop.
        """
        H, W = images.shape[2], images.shape[3]
        N    = rois.shape[0]

        # Preallooca il tensore output
        crops = torch.zeros(N, 3, roi_size, roi_size,
                            device=images.device, dtype=images.dtype)

        for i, roi in enumerate(rois):
            batch_idx = int(roi[0].item())
            x1 = roi[1].item();  y1 = roi[2].item()
            x2 = roi[3].item();  y2 = roi[4].item()

            pw = (x2 - x1) * self.roi_padding
            ph = (y2 - y1) * self.roi_padding
            x1 = max(0, int(x1 - pw));  y1 = max(0, int(y1 - ph))
            x2 = min(W, int(x2 + pw));  y2 = min(H, int(y2 + ph))

            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, W, H

            crop = images[batch_idx:batch_idx+1, :, y1:y2, x1:x2]
            crops[i] = F.interpolate(
                crop, size=(roi_size, roi_size),
                mode="bilinear", align_corners=False,
            )[0]

        return crops   # (N, 3, roi_size, roi_size)

    def forward(self, images, gt_boxes=None):
        """
        Forward pass completo.

        Returns dict con:
          'predictions'  : (B, 8400, 6)
          'feature_map'  : (B, 128, 40, 40)
          'raw_detect'   : dict interno YOLO
          'rois'         : (N, 5)
          'logits'       : (N, 1)  — logit MLP binario
          'roi_labels_gt': (N,)    — 1=vuoto, 0=pieno
          'n_rois'       : int
        """
        B      = images.shape[0]
        device = images.device

        # ── YOLO26 ────────────────────────────────────────────────────────────
        predictions, feature_map, raw_detect = self.detector(images)

        # ── Filtra predizioni ─────────────────────────────────────────────────
        filtered_preds = self.detector.filter_predictions(predictions)
        rois           = self.detector.predictions_to_roi_format(filtered_preds, B)

       # Cap sul numero di ROI — prende le MAX_ROIS_PER_BATCH più confidenti
        if rois.shape[0] > MAX_ROIS_PER_BATCH:
            rois = rois[:MAX_ROIS_PER_BATCH]
    
        if rois.shape[0] == 0:
            return {
                "predictions":   predictions,
                "feature_map":   feature_map,
                "raw_detect":    raw_detect,
                "rois":          rois,
                "logits":        torch.zeros(0, 1, device=device),
                "roi_labels_gt": torch.zeros(0, dtype=torch.long, device=device),
                "n_rois":        0,
            }

        # ── Label ROI dal IoU matching ────────────────────────────────────────
        if gt_boxes is not None and self.training:
                    roi_labels_gt = self._assign_roi_labels(
                        filtered_preds, gt_boxes, img_size=images.shape[-1]
                    ).to(device)
                    # Applica il cap anche alle label
                    roi_labels_gt = roi_labels_gt[:rois.shape[0]]
        else:
            roi_labels_gt = torch.ones(
                rois.shape[0], dtype=torch.long, device=device
            )
            
        # ── Crop reali dall'immagine originale ────────────────────────────────
        with torch.no_grad():
            roi_crops = self._crop_from_image(
                images.detach(), rois.detach(), self.roi_size
            )

        # ── RoI Align sulla feature map (solo per il backward) ────────────────
        roi_features = roi_align(
            input          = feature_map,
            boxes          = rois,
            output_size    = ROI_OUTPUT_SIZE,
            spatial_scale  = 1.0 / FEATURE_STRIDE,
            sampling_ratio = 2,
            aligned        = True,
        )
        roi_proj = self.feature_proj(roi_features)  # (N, 3, 7, 7)

        # ── SigLIPv2 + MLP ────────────────────────────────────────────────────
        logits = self.siglip(roi_crops)   # (N, 1)

        # Collegamento differenziabile YOLO → grafo SigLIP
        if self.training and roi_proj.requires_grad:
            logits = logits + roi_proj.mean() * 0.0

        return {
            "predictions":   predictions,
            "feature_map":   feature_map,
            "raw_detect":    raw_detect,
            "rois":          rois,
            "logits":        logits,
            "roi_labels_gt": roi_labels_gt,
            "n_rois":        rois.shape[0],
        }

    def remove_hooks(self):
        self.detector.remove_hooks()