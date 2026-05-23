"""
src/training/losses.py
=======================
Loss functions per il joint training:

  1. SigLIPBCELoss   — BCE binaria sul logit MLP (sostituisce SigLIPSigmoidLoss)
  2. YOLOLossWrapper — CIoU + BCE custom differenziabile (NON usa E2ELoss)
  3. JointLoss       — alpha*L_YOLO + beta*L_SigLIP con pesi FISSI

IMPORTANTE: YOLOLossWrapper usa la nostra CIoU+BCE custom con Max IoU Assigner
differenziabile — NON usa E2ELoss/TAL di Ultralytics.

alpha e beta sono fissi (calcolati con analyze_losses_custom.py):
  alpha=0.89, beta=0.11  con vincolo alpha+beta=1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


# ── 1. SigLIP BCE Loss ────────────────────────────────────────────────────────

class SigLIPBCELoss(nn.Module):
    """
    BCE binaria sul logit singolo dell'MLP classificatore.

    Il text encoder è stato rimosso su indicazione della professoressa.
    Al suo posto un MLP classifica direttamente l'embedding visivo di SigLIP.

    Input:  logits (N, 1), roi_labels (N,)  [1=vuoto, 0=pieno]
    Output: scalare BCE
    """

    def __init__(self, pos_weight: float = None):
        super().__init__()
        self._pos_weight = pos_weight

    def forward(self, logits, roi_labels):
        if len(roi_labels) == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        targets = roi_labels.float().unsqueeze(1)   # (N, 1)
        pw = None
        if self._pos_weight is not None:
            pw = torch.tensor([self._pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="mean"
        )

    def get_predictions(self, logits, threshold=0.5):
        return (torch.sigmoid(logits).squeeze(1) >= threshold).long()


# ── 2. YOLO Loss Wrapper — CIoU+BCE custom differenziabile ───────────────────

class YOLOLossWrapper:
    """
    CIoU + BCE con predizioni differenziabili (grad_fn=True).

    Non usa E2ELoss/TAL di Ultralytics — usa un Max IoU Assigner
    differenziabile che mantiene il grafo computazionale intatto.
    """

    def __init__(self, total_epochs: int = 50):
        self.img_size = 640.0

    def compute(
        self,
        gt_boxes:    list,
        gt_labels:   list,
        images:      torch.Tensor,
        device:      torch.device,
        predictions: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:

        if predictions is None:
            return torch.tensor(0.0, requires_grad=True, device=device)

        B          = predictions.shape[0]
        total_ciou = torch.tensor(0.0, device=device)
        total_bce  = torch.tensor(0.0, device=device)
        n_matched  = 0

        for i in range(B):
            pred       = predictions[i]
            gt         = gt_boxes[i].to(device)
            pred_boxes = pred[:, :4]
            pred_conf  = pred[:, 4]

            if gt.shape[0] == 0:
                target_conf = torch.zeros_like(pred_conf)
                total_bce   = total_bce + F.binary_cross_entropy_with_logits(
                    pred_conf, target_conf, reduction="mean"
                )
                continue

            xc = gt[:, 0] * self.img_size
            yc = gt[:, 1] * self.img_size
            w  = gt[:, 2] * self.img_size
            h  = gt[:, 3] * self.img_size
            gt_xyxy = torch.stack([
                xc - w/2, yc - h/2, xc + w/2, yc + h/2
            ], dim=1)

            with torch.no_grad():
                iou_matrix    = box_iou(pred_boxes.detach(), gt_xyxy)
                best_pred_idx = iou_matrix.max(dim=0).indices

            matched_preds = pred_boxes[best_pred_idx]
            ciou          = self._ciou_loss(matched_preds, gt_xyxy)
            total_ciou    = total_ciou + ciou
            n_matched     += gt.shape[0]

            target_conf                = torch.zeros_like(pred_conf)
            target_conf[best_pred_idx] = 1.0
            total_bce = total_bce + F.binary_cross_entropy_with_logits(
                pred_conf, target_conf, reduction="mean"
            )

        if n_matched > 0:
            total_ciou = total_ciou / n_matched
        total_bce = total_bce / B

        return total_ciou + total_bce

    def _ciou_loss(self, pred, gt):
        ix1 = torch.max(pred[:, 0], gt[:, 0])
        iy1 = torch.max(pred[:, 1], gt[:, 1])
        ix2 = torch.min(pred[:, 2], gt[:, 2])
        iy2 = torch.min(pred[:, 3], gt[:, 3])
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

        pw = (pred[:, 2] - pred[:, 0]).clamp(min=1e-7)
        ph = (pred[:, 3] - pred[:, 1]).clamp(min=1e-7)
        gw = (gt[:, 2]   - gt[:, 0]).clamp(min=1e-7)
        gh = (gt[:, 3]   - gt[:, 1]).clamp(min=1e-7)
        union = pw * ph + gw * gh - inter + 1e-7
        iou   = inter / union

        pcx = (pred[:, 0] + pred[:, 2]) / 2
        pcy = (pred[:, 1] + pred[:, 3]) / 2
        gcx = (gt[:, 0]   + gt[:, 2])   / 2
        gcy = (gt[:, 1]   + gt[:, 3])   / 2
        enc_x1   = torch.min(pred[:, 0], gt[:, 0])
        enc_y1   = torch.min(pred[:, 1], gt[:, 1])
        enc_x2   = torch.max(pred[:, 2], gt[:, 2])
        enc_y2   = torch.max(pred[:, 3], gt[:, 3])
        enc_diag = (enc_x2-enc_x1)**2 + (enc_y2-enc_y1)**2 + 1e-7
        dist_pen = ((pcx-gcx)**2 + (pcy-gcy)**2) / enc_diag

        v = (4 / (torch.pi**2)) * (torch.atan(gw/gh) - torch.atan(pw/ph))**2
        with torch.no_grad():
            alpha_c = v / (1 - iou + v + 1e-7)

        return (1 - iou + dist_pen + alpha_c * v).mean()


# ── 3. Joint Loss — pesi FISSI ────────────────────────────────────────────────

class JointLoss(nn.Module):
    """
    L_joint = alpha * L_YOLO(CIoU+BCE) + beta * L_SigLIP(BCE)

    Pesi fissi con vincolo alpha+beta=1.
    Calcolati con analyze_losses_custom.py su 20 batch reali.
    """

    def __init__(self, alpha: float = 0.89, beta: float = 0.11):
        super().__init__()
        assert abs(alpha + beta - 1.0) < 1e-6, \
            f"alpha+beta deve essere 1, ma {alpha}+{beta}={alpha+beta}"
        self.alpha = alpha
        self.beta  = beta
        print(f"  JointLoss: alpha={alpha:.2f}, beta={beta:.2f} "
              f"(alpha+beta={alpha+beta:.2f})")

    def forward(self, loss_yolo, loss_siglip):
        loss_joint = self.alpha * loss_yolo + self.beta * loss_siglip
        return loss_joint, loss_yolo, loss_siglip