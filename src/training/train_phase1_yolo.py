"""
src/training/train_phase1_yolo.py
==================================
FASE 1: Addestra YOLO26 standalone sul dataset degli scaffali vuoti.

Output:
  weights/cluster_training/phase1_yolo/weights/best.pt  <- pesi migliori
  weights/cluster_training/phase1_yolo/weights/last.pt  <- pesi finali

Esegui con:
  python src/training/train_phase1_yolo.py
  python src/training/train_phase1_yolo.py --epochs 30
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, ".")

from src.utils.config import CFG


def parse_args():
    parser = argparse.ArgumentParser(description="Fase 1 — Training YOLO26 standalone")
    parser.add_argument("--epochs",  type=int, default=30,               help="Numero di epoche")
    parser.add_argument("--batch",   type=int, default=8,                help="Batch size")
    parser.add_argument("--imgsz",   type=int, default=640,              help="Dimensione immagine")
    parser.add_argument("--device",  type=str, default="0",              help="Device (0=GPU, cpu)")
    parser.add_argument("--name",    type=str, default="phase1_yolo",    help="Nome cartella output")
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    # Percorsi dal config
    yaml_path   = CFG.data.dataset_yaml
    save_dir     = Path("weights/cluster_training") / args.name
    best_weights = save_dir / "weights/best.pt"

    print("=" * 60)
    print("  FASE 1 — Training YOLO26 standalone")
    print("=" * 60)
    print(f"  Dataset  : {yaml_path}")
    print(f"  Epoche   : {args.epochs}")
    print(f"  Batch    : {args.batch}")
    print(f"  Output   : {save_dir}")
    print()

    if not yaml_path.exists():
        print(f"✗ dataset.yaml non trovato in {yaml_path}")
        sys.exit(1)

    model = YOLO("yolo26n.pt")

    model.train(
        data          = str(yaml_path),
        epochs        = args.epochs,
        batch         = args.batch,
        imgsz         = args.imgsz,
        device        = args.device,
        project       = "weights/cluster_training",
        name          = args.name,
        exist_ok      = True,
        lr0           = 1e-3,
        lrf           = 0.01,
        momentum      = 0.937,
        weight_decay  = 5e-4,
        warmup_epochs = 3,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.1,
        flipud=0.0, fliplr=0.5, mosaic=0.8, mixup=0.05,
        save=True, save_period=-1, patience=10,
        optimizer="AdamW", cos_lr=True, amp=True,
        verbose=True, plots=True,
    )

    if best_weights.exists():
        print(f"\n{'='*60}")
        print(f"  ✓ Fase 1 completata!")
        print(f"  Pesi migliori: {best_weights}")
        print(f"\n  Per avviare la Fase 2:")
        print(f"  python src/training/train_phase2_joint.py")
        print(f"{'='*60}")
    else:
        print(f"\n⚠ Pesi non trovati in {best_weights}")


if __name__ == "__main__":
    main()