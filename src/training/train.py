"""
scripts/train.py
=================
Entry point per il training della pipeline joint.

Uso:
    python scripts/train.py
    python scripts/train.py --resume weights/checkpoint_epoch010.pt
"""

import sys
import argparse
sys.path.insert(0, ".")

from src.training.trainer import Trainer
from src.utils.config     import CFG


def parse_args():
    parser = argparse.ArgumentParser(description="Training joint pipeline YOLO26 + SigLIPv2")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path a un checkpoint da cui riprendere il training"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Numero di epoche (sovrascrive config)"
    )
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Batch size (sovrascrive config)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Sovrascrivi config se passati da CLI
    if args.epochs:
        CFG.training.num_epochs = args.epochs
    if args.batch:
        CFG.data.batch_size = args.batch

    # Inizializza il trainer
    trainer = Trainer(cfg=CFG)

    # Eventualmente riprendi da checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Avvia il training
    history = trainer.train()