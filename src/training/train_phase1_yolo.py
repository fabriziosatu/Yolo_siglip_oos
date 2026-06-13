"""
src/training/train_phase1_yolo.py
==================================
FASE 1: Addestra YOLO26 standalone sul dataset degli scaffali vuoti.

Parametri di training allineati ai valori ufficiali YOLO26n
(da YOLO-JUICE paper, EAAI 2025, Tab. 2 — modalita' detector_only).

Data augmentation:
  Default (--augment NON specificato): tutta l'augmentazione e' disabilitata.
  Con --augment: vengono applicati i valori ufficiali YOLO26n.

Output:
  weights/cluster_training/{name}/weights/best.pt
  weights/cluster_training/{name}/weights/last.pt

Esegui con:
  # Senza augmentazione (default)
  python src/training/train_phase1_yolo.py

  # Con augmentazione ufficiale YOLO26n
  python src/training/train_phase1_yolo.py --augment

  # Con tutti i parametri espliciti
  python src/training/train_phase1_yolo.py --epochs 80 --batch 16 --augment
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, ".")

from src.utils.config import CFG


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fase 1 — Training YOLO26 standalone"
    )
    parser.add_argument("--epochs",   type=int,   default=80,
        help="Numero di epoche (default: 80)")
    parser.add_argument("--batch",    type=int,   default=16,
        help="Batch size (default: 16)")
    parser.add_argument("--imgsz",    type=int,   default=640,
        help="Dimensione immagine (default: 640)")
    parser.add_argument("--device",   type=str,   default="0",
        help="Device: 0=GPU, cpu (default: 0)")
    parser.add_argument("--name",     type=str,   default="phase1_yolo_clean",
        help="Nome cartella output (default: phase1_yolo_clean)")
    parser.add_argument("--lr0",      type=float, default=None,
        help="Learning rate iniziale (default: 0.0054 ufficiale YOLO26n)")
    parser.add_argument("--patience", type=int,   default=50,
        help="Early stopping patience (default: 50)")
    parser.add_argument("--augment",  action="store_true", default=False,
        help="Abilita data augmentation con valori ufficiali YOLO26n. "
             "Se non specificato, tutta l'augmentazione e' disabilitata.")
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    yaml_path    = CFG.data.dataset_yaml
    save_dir     = Path("weights/cluster_training") / args.name
    best_weights = save_dir / "weights/best.pt"

    lr0 = args.lr0 if args.lr0 is not None else 0.0054

    print("=" * 60)
    print("  FASE 1 — Training YOLO26 standalone")
    print("=" * 60)
    print(f"  Dataset      : {yaml_path}")
    print(f"  Epoche       : {args.epochs}")
    print(f"  Batch        : {args.batch}")
    print(f"  lr0          : {lr0}")
    print(f"  Augmentation : {'ABILITATA (valori ufficiali YOLO26n)' if args.augment else 'DISABILITATA'}")
    print(f"  Output       : {save_dir}")
    print()

    if not yaml_path.exists():
        print(f"  X dataset.yaml non trovato in {yaml_path}")
        sys.exit(1)

    # ── Parametri augmentazione ───────────────────────────────────────────────
    # Con --augment: valori ufficiali YOLO26n (EAAI 2025, Tab. 2)
    # Senza --augment: tutto a 0.0 — nessuna trasformazione applicata
    if args.augment:
        aug_params = dict(
            hsv_h        = 0.015,  # jitter tonalita' colore
            hsv_s        = 0.7,    # jitter saturazione
            hsv_v        = 0.4,    # jitter luminosita'
            bgr          = 0.0,    # swap canali BGR (non usato)
            degrees      = 1.11,   # rotazione casuale
            translate    = 0.071,  # traslazione casuale
            scale        = 0.56,   # scala casuale
            flipud       = 0.0,    # flip verticale (non usato)
            fliplr       = 0.5,    # flip orizzontale
            mosaic       = 0.7,    # composizione 4 immagini
            close_mosaic = 10,     # disabilita mosaic ultimi 10 ep
            mixup        = 0.012,  # mixup tra immagini
        )
    else:
        aug_params = dict(
            hsv_h        = 0.0,
            hsv_s        = 0.0,
            hsv_v        = 0.0,
            bgr          = 0.0,
            degrees      = 0.0,
            translate    = 0.0,
            scale        = 0.0,
            flipud       = 0.0,
            fliplr       = 0.0,
            mosaic       = 0.0,
            close_mosaic = 0,
            mixup        = 0.0,
        )

    model = YOLO("yolo26n.pt")

    model.train(
        data     = str(yaml_path),
        epochs   = args.epochs,
        batch    = args.batch,
        imgsz    = args.imgsz,
        device   = args.device,
        project  = "weights/cluster_training",
        name     = args.name,
        exist_ok = True,

        # ── Ottimizzatore ─────────────────────────────────────────────────────
        optimizer    = "SGD",      # ufficiale YOLO26n per detector_only
        lr0          = lr0,
        lrf          = 0.0495,     # ufficiale YOLO26n
        momentum     = 0.947,      # ufficiale YOLO26n
        weight_decay = 0.00064,    # ufficiale YOLO26n
        cos_lr       = True,

        # ── Warmup ────────────────────────────────────────────────────────────
        warmup_epochs   = 0.9,     # ufficiale YOLO26n
        warmup_momentum = 0.8,
        warmup_bias_lr  = 0.1,

        # ── Loss weights ──────────────────────────────────────────────────────
        box = 5.63,                # ufficiale YOLO26n
        cls = 0.56,                # ufficiale YOLO26n
        dfl = 9.04,                # ufficiale YOLO26n

        # ── Architettura ──────────────────────────────────────────────────────
        freeze      = 10,          # congela primi 10 layer backbone
        multi_scale = False,

        # ── Augmentazione (controllata dal flag --augment) ────────────────────
        **aug_params,

        # ── Saving e logging ──────────────────────────────────────────────────
        save        = True,
        save_period = -1,
        patience    = args.patience,
        amp         = True,
        workers     = 4,
        verbose     = True,
        plots       = True,
    )

    if best_weights.exists():
        print(f"\n{'='*60}")
        print(f"  Fase 1 completata!")
        print(f"  Pesi migliori: {best_weights}")
        print(f"\n  Per avviare la Fase 2:")
        print(f"  python src/training/train_phase2_joint.py")
        print(f"{'='*60}")
    else:
        print(f"\n  Pesi non trovati in {best_weights}")


if __name__ == "__main__":
    main()