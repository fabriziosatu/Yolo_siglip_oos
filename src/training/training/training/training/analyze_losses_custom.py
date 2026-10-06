"""
scripts/analyze_losses.py
==========================
Analizza la magnitudo di L_YOLO(CIoU+BCE) e L_SigLIP(BCE) su batch reali
per determinare i valori corretti di alpha e beta nella joint loss.

IMPORTANTE: usa CIoU+BCE, NON E2ELoss. I pesi alpha/beta devono essere
calcolati sulla stessa loss usata nel training della Fase 2.

Differenza tra --phase 1 e --phase 2:
  phase 1 → batch con soli positivi (come il training YOLO standalone)
             L_SigLIP sara' artificialmente bassa (nessun negativo)
  phase 2 → batch con positivi + negativi sintetici generati dalla pipeline
             L_SigLIP riflette il vero regime di training della Fase 2
             USA QUESTO per calcolare alpha e beta corretti

Esegui con:
  python scripts/analyze_losses.py --phase 2
  python scripts/analyze_losses.py --phase 2 --yolo_weights weights/cluster_training/phase1_yolo_clean/weights/best.pt
"""

import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, ".")

from src.data.dataset          import build_dataloaders
from src.models.joint_pipeline import JointPipeline
from src.training.losses       import SigLIPBCELoss, YOLOLossWrapper
from src.utils.config          import CFG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_weights", type=str,
        default="weights/cluster_training/phase1_yolo_clean/weights/best.pt",
        help="Pesi YOLO da caricare",
    )
    parser.add_argument(
        "--phase", type=int, default=2, choices=[1, 2],
        help="Modalita' dataset: 1=solo positivi, 2=pos+neg "
             "(default: 2 — usa questo per calcolare alpha/beta corretti)",
    )
    parser.add_argument(
        "--n_batches", type=int, default=20,
        help="Numero di batch da analizzare (default: 20)",
    )
    parser.add_argument(
        "--batch", type=int, default=4,
        help="Batch size (default: 4)",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode   = f"phase{args.phase}"

    print("=" * 60)
    print("  Analisi magnitudo — CIoU+BCE + SigLIP(BCE)")
    print("=" * 60)
    print(f"  YOLO weights : {args.yolo_weights}")
    print(f"  Dataset mode : {mode}  "
          f"({'pos+neg con mining sintetico' if args.phase == 2 else 'solo positivi'})")
    print(f"  Batch size   : {args.batch}")
    print(f"  N batches    : {args.n_batches}")
    if args.phase == 1:
        print(f"\n  ⚠ Stai usando phase1: L_SigLIP sara' bassa perche' non ci")
        print(f"    sono negativi nel batch. Per calcolare alpha/beta corretti")
        print(f"    usa --phase 2.")
    print()

    CFG.data.batch_size     = args.batch
    CFG.detector.model_name = args.yolo_weights

    # ── Inizializza pipeline ──────────────────────────────────────────────────
    # Nota: nessun encode_prompts — il text encoder e' stato rimosso,
    # al suo posto un MLP classificatore binario sull'embedding visivo
    pipeline = JointPipeline(
        yolo_weights      = args.yolo_weights,
        siglip_model_name = CFG.siglip.model_name,
        conf_threshold    = CFG.detector.conf_threshold,
        lora_r_visual     = CFG.siglip.lora_r_visual,
        lora_alpha_visual = CFG.siglip.lora_alpha_visual,
        lora_dropout      = CFG.siglip.lora_dropout,
    ).to(device)

    siglip_loss_fn = SigLIPBCELoss()
    yolo_loss_fn   = YOLOLossWrapper(total_epochs=50)

    # ── DataLoader con la modalita' corretta ──────────────────────────────────
    train_loader, _, _ = build_dataloaders(
        img_size   = CFG.data.img_size,
        batch_size = args.batch,
        mode       = mode,
    )

    yolo_values = []
    sig_values  = []
    n_rois_list = []

    pipeline.train()
    print(f"  Analizzo {args.n_batches} batch in modalita' {mode}...\n")

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= args.n_batches:
                break

            images = batch["images"].to(device)
            boxes  = batch["boxes"]
            labels = batch["labels"]

            try:
                # La pipeline genera i negativi sintetici internamente
                # tramite _sample_negative_rois() durante il forward
                output = pipeline(images, gt_boxes=boxes, gt_labels=labels)

                # L_YOLO: CIoU + BCE sulle predizioni del detector
                loss_y = yolo_loss_fn.compute(
                    gt_boxes    = boxes,
                    gt_labels   = labels,
                    images      = images,
                    device      = device,
                    predictions = output["predictions"],
                )

                # L_SigLIP: BCE binaria su logits_pos (MLP output)
                # roi_labels_gt: 1=empty_shelf, 0=negativo sintetico
                if output["n_rois"] > 0:
                    loss_s = siglip_loss_fn(
                        output["logits_pos"],
                        output["roi_labels_gt"],
                    )
                else:
                    loss_s = torch.tensor(0.0)

                n_pos = int(output["roi_labels_gt"].sum().item())
                n_neg = output["n_rois"] - n_pos

                yolo_values.append(loss_y.item())
                sig_values.append(loss_s.item())
                n_rois_list.append(output["n_rois"])

                print(f"  Batch {i+1:2d}: "
                      f"L_YOLO={loss_y.item():.4f}  "
                      f"L_SigLIP={loss_s.item():.4f}  "
                      f"ROIs={output['n_rois']}  "
                      f"(pos={n_pos} neg_sintetici={n_neg})")

            except Exception as e:
                print(f"  Batch {i+1}: errore — {e}")
                import traceback
                traceback.print_exc()

    if not yolo_values:
        print("Nessun batch analizzato.")
        return

    yolo_arr = np.array(yolo_values)
    sig_arr  = np.array([v for v in sig_values if v > 0])

    # ── Statistiche ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STATISTICHE (modalita' {mode})")
    print(f"{'='*60}")
    print(f"  L_YOLO   : media={yolo_arr.mean():.4f}  "
          f"std={yolo_arr.std():.4f}  "
          f"min={yolo_arr.min():.4f}  max={yolo_arr.max():.4f}")

    if len(sig_arr) == 0:
        print("  L_SigLIP : nessun batch con ROI — abbassa conf_threshold")
        return

    mu_y = yolo_arr.mean()
    mu_s = sig_arr.mean()

    print(f"  L_SigLIP : media={mu_s:.4f}  "
          f"std={sig_arr.std():.4f}  "
          f"min={sig_arr.min():.4f}  max={sig_arr.max():.4f}")
    print(f"  Rapporto L_YOLO / L_SigLIP: {mu_y/mu_s:.4f}")
    print(f"  ROIs medi per batch         : {np.mean(n_rois_list):.1f}")

    # ── Calcolo alpha e beta ottimali ─────────────────────────────────────────
    # Criterio: alpha * L_YOLO = beta * L_SigLIP  con alpha + beta = 1
    # => alpha = mu_s / (mu_y + mu_s)
    alpha_opt = mu_s / (mu_y + mu_s)
    beta_opt  = 1.0 - alpha_opt
    alpha_r   = round(alpha_opt, 2)
    beta_r    = round(1.0 - alpha_r, 2)

    print(f"\n{'='*60}")
    print(f"  VALORI OTTIMALI alpha/beta (vincolo alpha+beta=1)")
    print(f"{'='*60}")
    print(f"""
  Calcolo:
    alpha = L_SigLIP / (L_YOLO + L_SigLIP)
          = {mu_s:.4f} / ({mu_y:.4f} + {mu_s:.4f})
          = {alpha_opt:.4f}
    beta  = 1 - alpha = {beta_opt:.4f}

  Arrotondati:
    alpha = {alpha_r}   (peso di L_YOLO nella joint loss)
    beta  = {beta_r}   (peso di L_SigLIP nella joint loss)

  Verifica bilanciamento:
    alpha * L_YOLO   = {alpha_r} * {mu_y:.4f} = {alpha_r * mu_y:.4f}
    beta  * L_SigLIP = {beta_r}  * {mu_s:.4f} = {beta_r  * mu_s:.4f}
    Contributi: {"BILANCIATI ✓" if abs(alpha_r*mu_y - beta_r*mu_s) < 0.05 else "leggermente sbilanciati"}
    """)

    print(f"  CONFRONTO CON VALORI IN CONFIG (alpha=0.89, beta=0.11):")
    print(f"    Attuali: 0.89 * {mu_y:.4f} = {0.89*mu_y:.4f}  "
          f"vs  0.11 * {mu_s:.4f} = {0.11*mu_s:.4f}")
    print(f"    Nuovi:   {alpha_r} * {mu_y:.4f} = {alpha_r*mu_y:.4f}  "
          f"vs  {beta_r} * {mu_s:.4f} = {beta_r*mu_s:.4f}")

    print(f"\n  => Aggiorna src/utils/config.py:")
    print(f"     alpha: float = {alpha_r}")
    print(f"     beta:  float = {beta_r}")

    if args.phase == 1:
        print(f"\n  ⚠ ATTENZIONE: questi valori sono calcolati su batch senza negativi.")
        print(f"    L_SigLIP e' sottostimata. Riesegui con --phase 2 per valori corretti.")


if __name__ == "__main__":
    main()