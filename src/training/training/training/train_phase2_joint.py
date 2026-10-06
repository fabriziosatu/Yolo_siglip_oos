"""
src/training/train_phase2_joint.py
==============================
FASE 2: Joint training YOLO26 + SigLIPv2 + MLP con JointLoss.

Carica i pesi YOLO pre-addestrati dalla Fase 1 e avvia
il joint training con la pipeline completa.

Architettura:
  - YOLO26     : detector, inizializzato con i pesi della Fase 1
  - SigLIPv2   : encoder visivo fine-tunato con LoRA sul dominio scaffali
  - MLP        : classificatore binario (256 -> 64 -> 1) applicato sulle
                 ROI estratte da SigLIP — sostituisce il text encoder
  - RoI Align  : estrae feature dalla feature map YOLO per ogni ROI
                 proposta, le proietta in RGB e le passa a SigLIP

Loss:
  L_joint = alpha * L_YOLO(CIoU+BCE) + beta * L_SigLIP(BCE)
  con alpha=0.49, beta=0.51 — pesi fissi calcolati con analyze_losses_custom.py
  Un solo backward aggiorna YOLO e SigLIP+MLP insieme.

Negativi:
  Generati sinteticamente on-the-fly dalla JointPipeline tramite
  _sample_negative_rois() — nessun dataset esterno necessario.
  Bilanciamento 1:1 con i positivi trovati da YOLO.
"""

import argparse
import sys
sys.path.insert(0, ".")


def parse_args():
    parser = argparse.ArgumentParser(description="Fase 2 - Joint training YOLO26 + SigLIPv2")
    parser.add_argument(
        "--yolo_weights", type=str,
        default="weights/phase1_yolo/weights/best.pt",
        help="Pesi YOLO dalla Fase 1"
    )
    parser.add_argument("--epochs",   type=int, default=10,   help="Numero di epoche")
    parser.add_argument("--batch",    type=int, default=4,    help="Batch size")
    parser.add_argument("--data_dir", type=str, default=None, help="Percorso dataset (default: config)")
    parser.add_argument("--resume",   type=str, default=None, help="Checkpoint da cui riprendere")
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Cartella di salvataggio pesi (default: weights/cluster_training/phase2_joint)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from pathlib import Path
    from src.utils.config import CFG

    print("=" * 60)
    print("  FASE 2 - Joint training YOLO26 + SigLIPv2")
    print("=" * 60)

    # Verifica che i pesi YOLO esistano
    yolo_weights = Path(args.yolo_weights)
    if not yolo_weights.exists():
        print(f"Pesi Fase 1 non trovati: {yolo_weights}")
        print("  Uso yolo26n.pt come fallback (consigliato eseguire prima la Fase 1)")
        yolo_weights = "yolo26n.pt"
    else:
        print(f"  Pesi YOLO  : {yolo_weights}")

    print(f"  Epoche     : {args.epochs}")
    print(f"  Batch      : {args.batch}")

    # Aggiorna la config con i parametri da CLI
    CFG.training.num_epochs = args.epochs
    CFG.data.batch_size     = args.batch
    CFG.detector.model_name = str(yolo_weights)
    if args.data_dir:
        CFG.data.data_dir = args.data_dir
    if args.save_dir:
        CFG.training.save_dir = Path(args.save_dir)

    print(f"  Save dir   : {CFG.training.save_dir}")
    print()

    # Importa il trainer
    from src.training.trainer import Trainer

    trainer = Trainer(cfg=CFG)

    # Eventualmente riprendi da checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Avvia il joint training
    history = trainer.train()

    print(f"\n{'='*60}")
    print(f"  Fase 2 completata!")
    print(f"  Best model: {CFG.training.save_dir}/best_model.pt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()