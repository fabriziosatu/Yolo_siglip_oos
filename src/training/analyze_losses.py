"""
scripts/analyze_losses.py
==========================
Analizza la magnitudo di L_YOLO(E2ELoss) e L_SigLIP su batch reali
per determinare i valori corretti di alpha e beta nella joint loss.

Vincolo imposto dalla professoressa: alpha + beta = 1

Calcolo:
  alpha = L_SigLIP / (L_YOLO + L_SigLIP)
  beta  = 1 - alpha

Esegui con:
  python scripts/analyze_losses.py
  python scripts/analyze_losses.py --yolo_weights weights/phase1_yolo/best.pt
"""

import sys, argparse, torch, numpy as np
sys.path.insert(0, ".")

from src.data.dataset          import build_dataloaders
from src.models.joint_pipeline import JointPipeline
from src.training.losses       import SigLIPSigmoidLoss, YOLOLossWrapper
from src.utils.config          import CFG


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_weights", type=str,
                        default="weights/phase1_yolo/best.pt")
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument("--batch",     type=int, default=4)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Analisi magnitudo delle loss (E2ELoss + SigLIP)")
    print("=" * 60)
    print(f"  YOLO weights : {args.yolo_weights}")
    print(f"  Batch size   : {args.batch}")
    print(f"  N batches    : {args.n_batches}")
    print()

    CFG.data.batch_size     = args.batch
    CFG.detector.model_name = args.yolo_weights

    pipeline = JointPipeline(
        yolo_weights      = args.yolo_weights,
        siglip_model_name = CFG.siglip.model_name,
        conf_threshold    = CFG.detector.conf_threshold,
        lora_r_visual     = CFG.siglip.lora_r_visual,
        lora_alpha_visual = CFG.siglip.lora_alpha_visual,
        lora_r_text       = CFG.siglip.lora_r_text,
        lora_alpha_text   = CFG.siglip.lora_alpha_text,
    ).to(device)

    pipeline.encode_prompts(
        positive_prompts = CFG.siglip.positive_prompts,
        negative_prompts = CFG.siglip.negative_prompts,
    )

    siglip_loss_fn = SigLIPSigmoidLoss()
    yolo_loss_fn   = YOLOLossWrapper(
        yolo_model   = pipeline.detector.model,
        total_epochs = 50,
    )

    train_loader, _, _ = build_dataloaders(
        img_size=CFG.data.img_size, batch_size=args.batch
    )

    yolo_values, sig_values, n_rois_list = [], [], []

    pipeline.train()
    print(f"  Analizzo {args.n_batches} batch...\n")

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= args.n_batches:
                break
            images = batch["images"].to(device)
            boxes  = batch["boxes"]
            labels = batch["labels"]
            try:
                output = pipeline(images, gt_boxes=boxes)
                loss_y = yolo_loss_fn.compute(
                    gt_boxes=boxes, gt_labels=labels,
                    images=images, device=device,
                )
                if output["n_rois"] > 0:
                    loss_s = siglip_loss_fn(
                        output["logits_pos"],
                        output["logits_neg"],
                        output["roi_labels_gt"],
                    )
                else:
                    loss_s = torch.tensor(0.0)

                yolo_values.append(loss_y.item())
                sig_values.append(loss_s.item())
                n_rois_list.append(output["n_rois"])
                print(f"  Batch {i+1:2d}: L_YOLO={loss_y.item():.4f}  "
                      f"L_SigLIP={loss_s.item():.4f}  "
                      f"ROIs={output['n_rois']}")
            except Exception as e:
                print(f"  Batch {i+1}: errore — {e}")

    if not yolo_values:
        print("Nessun batch analizzato.")
        return

    yolo_arr = np.array(yolo_values)
    sig_arr  = np.array([v for v in sig_values if v > 0])

    print(f"\n{'='*60}")
    print(f"  RISULTATI SU {len(yolo_values)} BATCH")
    print(f"{'='*60}")
    print(f"  L_YOLO:   media={yolo_arr.mean():.4f}  std={yolo_arr.std():.4f}  "
          f"min={yolo_arr.min():.4f}  max={yolo_arr.max():.4f}")

    if len(sig_arr) > 0:
        mu_y = yolo_arr.mean()
        mu_s = sig_arr.mean()
        print(f"  L_SigLIP: media={mu_s:.4f}  std={sig_arr.std():.4f}  "
              f"min={sig_arr.min():.4f}  max={sig_arr.max():.4f}")
        print(f"\n  Rapporto L_YOLO / L_SigLIP: {mu_y/mu_s:.2f}")

        # ── Calcolo alpha e beta con vincolo alpha+beta=1 ─────────────────────
        alpha_opt = mu_s / (mu_y + mu_s)
        beta_opt  = 1.0 - alpha_opt
        alpha_r   = round(alpha_opt, 2)
        beta_r    = round(beta_opt,  2)

        print(f"\n{'='*60}")
        print(f"  VALORI OTTIMALI (vincolo alpha+beta=1)")
        print(f"{'='*60}")
        print(f"""
  Calcolo:
    alpha = L_SigLIP / (L_YOLO + L_SigLIP)
    alpha = {mu_s:.4f} / ({mu_y:.4f} + {mu_s:.4f}) = {alpha_opt:.4f}
    beta  = 1 - alpha = {beta_opt:.4f}

  Arrotondati:  alpha = {alpha_r},  beta = {beta_r}
  Verifica alpha+beta = {alpha_r}+{beta_r} = {alpha_r+beta_r:.2f}  {'✓' if abs(alpha_r+beta_r-1.0)<0.01 else '✗'}

  Contributi bilanciati:
    alpha * L_YOLO   = {alpha_r} * {mu_y:.4f} = {alpha_r*mu_y:.4f}
    beta  * L_SigLIP = {beta_r}  * {mu_s:.4f} = {beta_r*mu_s:.4f}
        """)

        print(f"  => Aggiorna src/utils/config.py:")
        print(f"     alpha: float = {alpha_r}")
        print(f"     beta:  float = {beta_r}")

    print(f"\n  ROIs medi per batch: {np.mean(n_rois_list):.1f}")


if __name__ == "__main__":
    main()