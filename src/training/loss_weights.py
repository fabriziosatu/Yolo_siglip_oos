"""
src/training/loss_weights.py
==============================
Calcola alpha/beta della JointLoss (bilanciamento per magnitudo, vedi
discussione) a partire da magnitudo di L_YOLO/L_SigLIP GIA' MISURATE —
nessuna GPU necessaria, solo aritmetica.

Modifica i due dizionari YOLO_MAGNITUDES / SIGLIP_MAGNITUDES sotto con i
tuoi valori, poi lancia lo script: calcola le 3 strategie (pura/log/floor)
per ogni combinazione YOLO x SigLIP.

Uso:
    python src/training/loss_weights.py
    python src/training/loss_weights.py --floor 0.3
"""

import sys
import argparse
sys.path.insert(0, ".")

from src.training.losses import compute_balanced_weights


# ── Modifica questi due dizionari con i tuoi valori misurati ──────────────────

YOLO_MAGNITUDES = {
    "yolo_aug":    1.227,
    "yolo_no_aug": 0.634,
}

SIGLIP_MAGNITUDES = {
    "completa_r16_neg0.25":    0.0110,
    "completa_r16_neg0.5":     0.0073,
    "vision_mlp_r16_neg0.25":  0.0065,
    "vision_mlp_r16_neg0.5":   0.0059,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Calcola alpha/beta della joint loss da magnitudo note")
    parser.add_argument("--floor", type=float, default=0.3,
                         help="Peso minimo per alpha nella strategia 'floor' (default 0.3)")
    return parser.parse_args()


def print_strategies(mean_yolo: float, mean_siglip: float, floor: float, label: str = ""):
    print(f"\n  {label}")
    print(f"    L_YOLO={mean_yolo:.4f}  L_SigLIP={mean_siglip:.4f}  "
          f"rapporto={mean_yolo/mean_siglip:.2f}x")
    for mode in ["pure", "log", "floor"]:
        a, b = compute_balanced_weights(mean_yolo, mean_siglip, mode=mode, floor=floor)
        a_r, b_r = round(a, 4), round(1.0 - round(a, 4), 4)
        print(f"    [{mode:5s}] alpha={a_r:.4f}  beta={b_r:.4f}   "
              f"(alpha*L_YOLO={a_r*mean_yolo:.4f}  beta*L_SigLIP={b_r*mean_siglip:.4f})")


def main():
    args = parse_args()

    print(f"\n{'='*100}")
    print(f"  Calcolo alpha/beta da magnitudo note (nessuna GPU necessaria)")
    print(f"{'='*100}")

    for yolo_name, l_yolo in YOLO_MAGNITUDES.items():
        for sig_name, l_sig in SIGLIP_MAGNITUDES.items():
            print_strategies(l_yolo, l_sig, args.floor, label=f"{yolo_name} / {sig_name}")

    print(f"\n{'='*100}")
    print(f"  Legenda: pure = bilanciamento magnitudo puro | log = attenuazione log(1+L) "
          f"(spesso non aiuta se il gap e' enorme) | floor = alpha minimo forzato a {args.floor}")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
