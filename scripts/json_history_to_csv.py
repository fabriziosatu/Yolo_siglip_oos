"""
json_history_to_csv.py
========================
Ricostruisce un results.csv (stesso schema di train_fase3_lora.py) a partire
da training_history.json — utile quando results.csv risulta troncato/vuoto
(es. una seconda esecuzione dello stesso job ha riaperto il file in scrittura
e l'ha svuotato senza completare nessuna epoca, mentre training_history.json
viene sovrascritto solo DOPO che un'epoca finisce, quindi resta quello buono
dell'esecuzione precedente).

Uso:
    python json_history_to_csv.py \
        --json weights/fase3_siglip_lora/fase3_siglip_lora_r16_neg1/training_history.json \
        --out_csv weights/fase3_siglip_lora/fase3_siglip_lora_r16_neg1/results.csv
"""

import argparse
import csv
import json
from pathlib import Path


def main(args):
    json_path = Path(args.json)
    out_csv = Path(args.out_csv)

    with open(json_path) as f:
        history = json.load(f)

    n = len(history["epoch"])
    if n == 0:
        print(f"[Attenzione] {json_path} non contiene nessuna epoca. Niente da ricostruire.")
        return

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
                          "val/box_loss", "val/cls_loss", "val/dfl_loss",
                          "train_acc", "val_acc"])
        for i in range(n):
            writer.writerow([
                history["epoch"][i], 0.0, history["train_loss"][i], 0.0,
                0.0, history["val_loss"][i], 0.0,
                history["train_acc"][i], history["val_acc"][i],
            ])

    print(f"[Fatto] {n} epoche ricostruite da {json_path}")
    print(f"[Output] {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Ricostruisce results.csv da training_history.json")
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())