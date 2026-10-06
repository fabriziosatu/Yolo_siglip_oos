"""
src/training/losses.py
=======================
Loss functions per il joint training.

MODIFICHE rispetto alla vostra versione (YOLOLossWrapper e JointLoss
restano ESATTAMENTE come le avevate — CIoU+BCE custom differenziabile,
pesi alpha/beta fissi):

  - SigLIPBCELoss: INVARIATA (variante vision_mlp).
  - SigLIPSigmoidLoss: NUOVA, per la variante completa (2 logit pos/neg,
    stessa loss usata in train_fase3_full_lora.py).
  - compute_siglip_loss(): NUOVO dispatcher — sceglie la loss giusta in
    base a cfg.siglip.variant, cosi' trainer.py non deve avere if/else
    sparsi sulla variante.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


# ── 1a. SigLIP BCE Loss (variante vision_mlp) — INVARIATA ────────────────────

class SigLIPBCELoss(nn.Module):
    """BCE binaria sul logit singolo dell'MLP classificatore."""

    def __init__(self, pos_weight: float = None):
        super().__init__()
        self._pos_weight = pos_weight

    def forward(self, logits, roi_labels):
        if len(roi_labels) == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits.device)
        targets = roi_labels.float().unsqueeze(1)
        pw = None
        if self._pos_weight is not None:
            pw = torch.tensor([self._pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="mean")

    def get_predictions(self, logits, threshold=0.5):
        return (torch.sigmoid(logits).squeeze(1) >= threshold).long()


# ── 1b. SigLIP Sigmoid Loss (variante completa) — NUOVA ───────────────────────

class SigLIPSigmoidLoss(nn.Module):
    """
    BCE indipendente su logit_pos e logit_neg — stessa loss di
    train_fase3_full_lora.py. Usata SOLO da variant="completa".
    """

    def forward(self, logits_pos, logits_neg, roi_labels):
        if len(roi_labels) == 0:
            return torch.tensor(0.0, requires_grad=True, device=logits_pos.device)
        labels = roi_labels.float().unsqueeze(1)
        loss_pos = F.binary_cross_entropy_with_logits(logits_pos, labels, reduction="mean")
        loss_neg = F.binary_cross_entropy_with_logits(logits_neg, 1.0 - labels, reduction="mean")
        return (loss_pos + loss_neg) / 2.0


def compute_siglip_loss(variant: str, siglip_out: dict, roi_labels: torch.Tensor,
                         bce_loss_fn: SigLIPBCELoss, sigmoid_loss_fn: SigLIPSigmoidLoss) -> torch.Tensor:
    """Dispatcher unico chiamato da trainer.py."""
    if variant == "vision_mlp":
        return bce_loss_fn(siglip_out["logit"], roi_labels)
    elif variant == "completa":
        return sigmoid_loss_fn(siglip_out["logits_pos"], siglip_out["logits_neg"], roi_labels)
    else:
        raise ValueError(f"Variante SigLIP sconosciuta: '{variant}'")


# ── 2. YOLO Loss Wrapper — CIoU+BCE custom differenziabile — INVARIATA ───────

class YOLOLossWrapper:

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
            gt_xyxy = torch.stack([xc - w/2, yc - h/2, xc + w/2, yc + h/2], dim=1)

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


def compute_balanced_weights(mean_yolo: float, mean_siglip: float,
                              mode: str = "pure", floor: float = 0.3) -> tuple:
    """
    Calcola (alpha, beta) per la JointLoss a partire dalla magnitudo media
    osservata delle due loss, con tre strategie possibili (vedi discussione
    sul divario di scala strutturale tra L_YOLO — regressione geometrica,
    mai vicina a 0 anche per un detector ottimo — e L_SigLIP —
    classificazione binaria, puo' arrivare vicino a 0 se ben separata):

      "pure"  — alpha = L_SigLIP/(L_YOLO+L_SigLIP), beta = 1-alpha.
                Bilanciamento magnitudo-per-magnitudo "onesto", ma con
                gap di scala ampi (es. 100x) spinge alpha vicino a 0 —
                rischio che L_SigLIP domini quasi del tutto il gradiente
                che arriva alle coordinate delle box (via DetectNoDetach).

      "log"   — usa log(1+L) invece di L prima di normalizzare. Attenua
                lo squilibrio mantenendo comunque "chi parte piu' piccolo
                pesa di piu'", senza la schiacciata totale della versione
                lineare quando il gap e' di ordini di grandezza.

      "floor" — alpha = max(alpha_pure, floor), beta = 1-alpha. Pragmatico:
                garantisce che L_YOLO mantenga sempre un peso minimo
                (default 0.3) nel gradiente sulle coordinate delle box,
                sacrificando un po' il criterio di bilanciamento puro.

    Returns:
        (alpha, beta) — non arrotondati, arrotonda tu a valle se serve
    """
    import math

    if mode == "pure":
        alpha = mean_siglip / (mean_yolo + mean_siglip)
    elif mode == "log":
        log_yolo = math.log(1.0 + mean_yolo)
        log_sig  = math.log(1.0 + mean_siglip)
        alpha = log_sig / (log_yolo + log_sig)
    elif mode == "floor":
        alpha_pure = mean_siglip / (mean_yolo + mean_siglip)
        alpha = max(alpha_pure, floor)
    else:
        raise ValueError(f"mode sconosciuta: '{mode}' (usa 'pure'/'log'/'floor')")

    beta = 1.0 - alpha
    return alpha, beta


# ── 3. Joint Loss — pesi FISSI — INVARIATA ────────────────────────────────────

class JointLoss(nn.Module):

    def __init__(self, alpha: float = 0.49, beta: float = 0.51):
        super().__init__()
        assert abs(alpha + beta - 1.0) < 1e-6, \
            f"alpha+beta deve essere 1, ma {alpha}+{beta}={alpha+beta}"
        self.alpha = alpha
        self.beta  = beta
        print(f"  JointLoss: alpha={alpha:.3f}, beta={beta:.3f} (alpha+beta={alpha+beta:.3f})")

    def forward(self, loss_yolo, loss_siglip):
        loss_joint = self.alpha * loss_yolo + self.beta * loss_siglip
        return loss_joint, loss_yolo, loss_siglip
