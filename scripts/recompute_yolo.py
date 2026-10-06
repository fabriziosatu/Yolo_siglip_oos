"""
src/evaluation/recompute_yolo_baseline_metrics.py
=====================================================
Ricalcola le metriche standard di YOLO26 baseline (Precision, Recall, F1,
mAP@50, mAP@50-95, e le loss) per entrambi i detector (Augmentation /
No Augmentation), su Validation Set e Test Set — stessa struttura della
tabella Excel di riferimento.

Perche' due fonti diverse per Val e Test:
  - VALIDATION SET: le metriche (incluse le loss) sono gia' state calcolate
    durante il training stesso, un'epoca alla volta, e salvate nel
    results.csv originale — stessa fonte gia' usata per la magnitudo di
    Fase 1. Non serve rilanciare nulla: si legge la riga dell'epoca
    migliore (o l'ultima, a seconda di --epoch_mode).
  - TEST SET: non fa parte del training loop, quindi non esistono loss
    di test nei log — l'unico modo per ottenere Precision/Recall/mAP e'
    lanciare una validazione esplicita di Ultralytics su quello split
    (model.val(split='test')). Coerente con la tua tabella, che infatti
    non riporta colonne di loss per il Test Set.

F1 = 2*Precision*Recall / (Precision+Recall) — stessa formula usata
ovunque nel progetto (equivalente a 2*TP/(2*TP+FP+FN)).

Uso:
    python src/evaluation/recompute_yolo_baseline_metrics.py \
        --aug_weights weights/fase1_baseline/aug/best.pt \
        --aug_results_csv weights/fase1_baseline/aug/results.csv \
        --noaug_weights weights/fase1_baseline/no_aug/best.pt \
        --noaug_results_csv weights/fase1_baseline/no_aug/results.csv \
        --data data.yaml \
        --out_csv results/yolo_baseline_metrics.csv \
        --epoch_mode best
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def safe_f1(p, r):
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def read_val_from_results_csv(csv_path: Path, epoch_mode: str):
    """
    Legge Precision/Recall/mAP/loss dal results.csv del training originale.
    epoch_mode='best' -> riga con val/box_loss+val/cls_loss+val/dfl_loss minima
    epoch_mode='last' -> ultima riga del file
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} e' vuoto")

    def tot_val_loss(row):
        return (float(row.get("val/box_loss", 0) or 0)
                + float(row.get("val/cls_loss", 0) or 0)
                + float(row.get("val/dfl_loss", 0) or 0))

    if epoch_mode == "best":
        row = min(rows, key=tot_val_loss)
    else:
        row = rows[-1]

    precision = float(row.get("metrics/precision(B)", 0) or 0)
    recall = float(row.get("metrics/recall(B)", 0) or 0)
    map50 = float(row.get("metrics/mAP50(B)", 0) or 0)
    map5095 = float(row.get("metrics/mAP50-95(B)", 0) or 0)
    box_loss = float(row.get("val/box_loss", 0) or 0)
    cls_loss = float(row.get("val/cls_loss", 0) or 0)
    dfl_loss = float(row.get("val/dfl_loss", 0) or 0)

    return {
        "Precision": precision, "Recall": recall, "F1": safe_f1(precision, recall),
        "mAP@50": map50, "mAP@50-95": map5095,
        "box_loss": box_loss, "cls_loss": cls_loss, "dfl_loss": dfl_loss,
        "tot_loss": box_loss + cls_loss + dfl_loss,
        "epoch": row.get("epoch", "?"),
    }


def run_test_val(weights_path: str, data_yaml: str):
    """Lancia una validazione esplicita di Ultralytics sullo split 'test'."""
    model = YOLO(weights_path)
    results = model.val(data=data_yaml, split="test", verbose=False)
    rd = results.results_dict
    precision = float(rd.get("metrics/precision(B)", 0))
    recall = float(rd.get("metrics/recall(B)", 0))
    map50 = float(rd.get("metrics/mAP50(B)", 0))
    map5095 = float(rd.get("metrics/mAP50-95(B)", 0))
    return {
        "Precision": precision, "Recall": recall, "F1": safe_f1(precision, recall),
        "mAP@50": map50, "mAP@50-95": map5095,
    }


def main(args):
    detectors = {
        "No Augmentation": {"weights": args.noaug_weights, "results_csv": args.noaug_results_csv},
        "Augmentation": {"weights": args.aug_weights, "results_csv": args.aug_results_csv},
    }

    out_rows = []

    print("=" * 60)
    print("VALIDATION SET (da results.csv del training originale)")
    print("=" * 60)
    val_metrics = {}
    for det_name, paths in detectors.items():
        m = read_val_from_results_csv(Path(paths["results_csv"]), args.epoch_mode)
        val_metrics[det_name] = m
        print(f"\n[{det_name}] (epoca {m['epoch']}, criterio={args.epoch_mode})")
        for k in ["Precision", "Recall", "F1", "mAP@50", "mAP@50-95",
                  "box_loss", "cls_loss", "dfl_loss", "tot_loss"]:
            print(f"    {k:12s} = {m[k]:.3f}")
        out_rows.append({"Split": "Validation", "Detector": det_name, **{
            k: round(m[k], 4) for k in ["Precision", "Recall", "F1", "mAP@50", "mAP@50-95",
                                          "box_loss", "cls_loss", "dfl_loss", "tot_loss"]}})

    print("\n" + "=" * 60)
    print("TEST SET (validazione esplicita Ultralytics, split='test')")
    print("=" * 60)
    for det_name, paths in detectors.items():
        print(f"\n[{det_name}] Valido su test set...")
        m = run_test_val(paths["weights"], args.data)
        print(f"[{det_name}]")
        for k in ["Precision", "Recall", "F1", "mAP@50", "mAP@50-95"]:
            print(f"    {k:12s} = {m[k]:.3f}")
        out_rows.append({"Split": "Test", "Detector": det_name, **{
            k: round(m[k], 4) for k in ["Precision", "Recall", "F1", "mAP@50", "mAP@50-95"]},
            "box_loss": "n/d", "cls_loss": "n/d", "dfl_loss": "n/d", "tot_loss": "n/d"})

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Split", "Detector", "Precision", "Recall", "F1", "mAP@50", "mAP@50-95",
                  "box_loss", "cls_loss", "dfl_loss", "tot_loss"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\n[Output] {out_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aug_weights", required=True)
    p.add_argument("--aug_results_csv", required=True)
    p.add_argument("--noaug_weights", required=True)
    p.add_argument("--noaug_results_csv", required=True)
    p.add_argument("--data", required=True, help="data.yaml usato per il training, con path a val/test")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--epoch_mode", choices=["best", "last"], default="best",
                    help="Quale epoca del results.csv usare per il Validation Set "
                         "(default: 'best' = val_loss totale minima)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())