"""
best_epoch_summary.py
=======================
Per ciascun results.csv passato, trova l'epoca con val/cls_loss minimo
(quella corrispondente al checkpoint best.pt salvato) e stampa la sua
val_acc — per confrontare le PRESTAZIONI reali tra run diversi, non solo
la magnitudo della loss (che da sola puo' essere fuorviante quando si
confrontano neg_ratio diversi, per via del diverso bilanciamento classi).

Uso:
    python best_epoch_summary.py \
        weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r8_neg05/results.csv \
        weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r8_neg025/results.csv \
        weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r16_neg05/results.csv \
        weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r16_neg025/results.csv
"""
import csv
import sys
from pathlib import Path


import statistics as stats


def summarize(csv_path: str):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    rows_sorted = sorted(rows, key=lambda r: int(r["epoch"]))
    best_idx = min(range(len(rows_sorted)), key=lambda i: float(rows_sorted[i]["val/cls_loss"]))
    best_row = rows_sorted[best_idx]

    # Finestra "regime genuino": dalla prima epoca fino a quella del best
    # checkpoint INCLUSA — evita sia il rumore di un singolo punto sia la
    # contaminazione da overfitting delle epoche successive al best.
    window = rows_sorted[:best_idx + 1]
    train_losses_window = [float(r["train/cls_loss"]) for r in window]

    return {
        "epoch": best_row["epoch"],
        "n_window": len(window),
        "train_loss_best_epoch": float(best_row["train/cls_loss"]),
        "train_loss_window_mean": stats.mean(train_losses_window),
        "train_loss_window_median": stats.median(train_losses_window),
        "val_loss": float(best_row["val/cls_loss"]),
        "val_acc": float(best_row["val_acc"]),
        "train_acc": float(best_row["train_acc"]),
        "n_epochs": len(rows),
    }


def main():
    print(f"{'Run':30s} {'Best_ep':>8s} {'train_ep_best':>14s} {'val_loss':>9s} "
          f"{'val_acc':>9s} {'train_acc':>10s} {'#epoche':>8s}")
    print("-" * 92)
    for path in sys.argv[1:]:
        info = summarize(path)
        label = Path(path).parent.name
        if info is None:
            print(f"{label:30s}  (results.csv vuoto o non trovato)")
            continue
        print(f"{label:30s} {info['epoch']:>8s} {info['train_loss_best_epoch']:>14.4f} "
              f"{info['val_loss']:>9.4f} {info['val_acc']:>9.4f} {info['train_acc']:>10.4f} "
              f"{info['n_epochs']:>8d}")


if __name__ == "__main__":
    main()