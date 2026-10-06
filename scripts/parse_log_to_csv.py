"""
parse_log_to_csv.py
=====================
Ricostruisce un results.csv (stesso schema usato da train_fase3_lora.py /
analyze_yolo_loss_magnitude.py) leggendo le righe "Epoch N/50 | ..." gia'
stampate nel log .out di un job SLURM — utile quando il job e' stato
interrotto (scancel, timeout) PRIMA che lo script arrivasse a scrivere
results.csv su disco (che con la versione vecchia dello script avveniva
solo a fine training).

Uso:
    python parse_log_to_csv.py --log jobs/logs/fase3_r8_524148.out \
        --out_csv weights/fase3_siglip_lora/fase3_siglip_lora_r8_neg0/results.csv
"""

import argparse
import csv
import re
from pathlib import Path

# Esempio di riga da matchare:
# Epoch   1/50 | train_loss=0.0096 train_acc=1.000 | val_loss=0.0000 val_acc=1.000
LINE_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s*\|\s*"
    r"train_loss=([\d.]+)\s+train_acc=([\d.]+)\s*\|\s*"
    r"val_loss=([\d.]+)\s+val_acc=([\d.]+)"
)


def parse_log(log_path: Path):
    rows = []
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                epoch, train_loss, train_acc, val_loss, val_acc = m.groups()
                rows.append({
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": float(val_loss),
                    "val_acc": float(val_acc),
                })
    return rows


def main(args):
    log_path = Path(args.log)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_log(log_path)
    if not rows:
        print(f"[Attenzione] Nessuna riga 'Epoch N/50 | ...' trovata in {log_path}. "
              f"Controlla che il path sia giusto e che il job abbia stampato almeno un'epoca.")
        return

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
                          "val/box_loss", "val/cls_loss", "val/dfl_loss",
                          "train_acc", "val_acc"])
        for r in rows:
            writer.writerow([r["epoch"], 0.0, r["train_loss"], 0.0,
                              0.0, r["val_loss"], 0.0,
                              r["train_acc"], r["val_acc"]])

    print(f"[Fatto] {len(rows)} epoche estratte da {log_path}")
    print(f"[Output] {out_csv}")
    print(f"[Info] Ora puoi lanciare analyze_yolo_loss_magnitude.py puntando a questo csv.")


def parse_args():
    parser = argparse.ArgumentParser(description="Ricostruisce results.csv da un log .out di SLURM")
    parser.add_argument("--log", type=str, required=True, help="Path al file .out del job")
    parser.add_argument("--out_csv", type=str, required=True, help="Dove scrivere il results.csv ricostruito")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)