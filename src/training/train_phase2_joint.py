"""
scripts/train_phase2_joint.py
==============================
FASE 2: Joint training YOLO26 + SigLIPv2 con ProgLoss.

Carica i pesi YOLO pre-addestrati dalla Fase 1 e avvia
il joint training con la pipeline completa.

Differenze rispetto al training standalone:
  - YOLO è inizializzato con i pesi della Fase 1 (già specializzato)
  - SigLIP viene fine-tunato con LoRA per adattarsi al dominio
  - ProgLoss bilancia le due loss durante il training
  - I gradienti di SigLIP risalgono via RoI Align fino al backbone YOLO
"""

import argparse
import sys
sys.path.insert(0, ".")


def parse_args():
    parser = argparse.ArgumentParser(description="Fase 2 — Joint training YOLO26 + SigLIPv2")
    parser.add_argument(
        "--yolo_weights", type=str,
        default="weights/phase1_yolo/weights/best.pt",
        help="Pesi YOLO dalla Fase 1"
    )
    parser.add_argument("--epochs",  type=int,   default=10,  help="Numero di epoche")
    parser.add_argument("--batch",   type=int,   default=4,   help="Batch size")
    parser.add_argument("--data_dir", type=str,  default=None, help="Percorso dataset (default: config)")
    parser.add_argument("--resume",  type=str,   default=None, help="Checkpoint da cui riprendere")
    return parser.parse_args()


def main():
    args = parse_args()

    from pathlib import Path
    from src.utils.config import CFG

    print("=" * 60)
    print("  FASE 2 — Joint training YOLO26 + SigLIPv2")
    print("=" * 60)

    # Verifica che i pesi YOLO esistano
    yolo_weights = Path(args.yolo_weights)
    if not yolo_weights.exists():
        # Fallback: usa i pesi pretrainati base se la Fase 1 non è stata eseguita
        print(f"⚠ Pesi Fase 1 non trovati: {yolo_weights}")
        print("  Uso yolo26n.pt come fallback (consigliato eseguire prima la Fase 1)")
        yolo_weights = "yolo26n.pt"
    else:
        print(f"  Pesi YOLO  : {yolo_weights}")

    print(f"  Epoche     : {args.epochs}")
    print(f"  Batch      : {args.batch}")
    print()

    # Aggiorna la config con i parametri da CLI
    CFG.training.num_epochs = args.epochs
    CFG.data.batch_size     = args.batch
    CFG.detector.model_name = str(yolo_weights)
    if args.data_dir:
        CFG.data.data_dir = args.data_dir

    # Importa il trainer
    from src.training.trainer import Trainer

    trainer = Trainer(cfg=CFG)

    # Eventualmente riprendi da checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Avvia il joint training
    history = trainer.train()

    print(f"\n{'='*60}")
    print(f"  ✓ Fase 2 completata!")
    print(f"  Best model: weights/phase2_joint/best_model.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()