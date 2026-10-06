"""
src/evaluation/compute_yolo_baseline_multimatch.py
======================================================
Calcola Precision/Recall(TPR)/F1/FPR/FNR di YOLO26 baseline (senza alcun
filtro SigLIP) usando la STESSA metodologia gia' usata per la colonna
"F1 Baseline" nelle tabelle di confronto (multi-match: una predizione
puo' "coprire" piu' GT, nessun vincolo di esclusivita' 1-a-1; IoU=0.0
significa "qualunque sovrapposizione positiva conta gia' come match").

Non serve nessun modello SigLIP: solo YOLO — molto piu' leggero e veloce
di evaluate_fase3_siglip.py, utile quando serve SOLO la baseline con
questa metodologia (es. per popolare/verificare la colonna F1 Baseline
di compute_fscore.py senza rilanciare un'intera valutazione Fase 3).

Formule (identiche a compute_fscore.py):
    Precision = TP / (TP + FP)
    Recall    = TP / GT          (= TPR)
    F1        = 2*TP / (2*TP + FP + FN)
    FPR       = FP / GT
    FNR       = FN / GT

Uso:
    python src/evaluation/compute_yolo_baseline_multimatch.py \
        --aug_weights weights/fase1_baseline/aug/best.pt \
        --noaug_weights weights/fase1_baseline/no_aug/best.pt \
        --images dataset_finale/test/images \
        --labels dataset_finale/test/labels \
        --split_name test \
        --out_csv results/yolo_baseline_multimatch_test.csv \
        --conf 0.25 0.5 \
        --iou_thr 0.0 0.25 0.5 0.75

Per includere anche il validation set nello stesso file, rilancia con
--images/--labels puntati a dataset_finale/val/... e --split_name val,
usando lo stesso --out_csv con --append (aggiunge righe invece di
sovrascrivere).
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO
from PIL import Image


def load_test_pairs(img_dir: Path, lbl_dir: Path):
    pairs = []
    paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    for img_path in paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = Image.open(img_path)
        W, H = img.size
        gt_boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * W; y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W; y2 = (cy + bh / 2) * H
            gt_boxes.append([x1, y1, x2, y2])
        pairs.append({"path": img_path, "gt": gt_boxes})
    return pairs


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0]); y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2]); y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0])*(box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def multi_match(pred_boxes, gt_boxes, iou_thr):
    """Stessa convenzione usata ovunque nel progetto (evaluate_fase3_siglip.py):
    una pred e' TP se copre almeno una GT; una GT e' FN se nessuna pred la copre."""
    mat_pred = [False] * len(pred_boxes)
    gt_covered = [False] * len(gt_boxes)
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi] = True
                gt_covered[gi] = True
    tp = sum(mat_pred)
    fp = len(pred_boxes) - tp
    fn = sum(1 for c in gt_covered if not c)
    return tp, fp, fn


def safe_precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def safe_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_weights", required=True)
    parser.add_argument("--noaug_weights", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--split_name", default="test",
                         help="Etichetta da scrivere in colonna 'Split' (default: test)")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--conf", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--iou_thr", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75])
    parser.add_argument("--append", action="store_true",
                         help="Aggiunge righe a --out_csv invece di sovrascriverlo "
                              "(utile per unire piu' split nello stesso file)")
    parser.add_argument("--max_images", type=int, default=None)
    args = parser.parse_args()

    print("[1/2] Carico YOLO aug/noaug...")
    detectors = {"aug": YOLO(args.aug_weights), "noaug": YOLO(args.noaug_weights)}

    pairs = load_test_pairs(Path(args.images), Path(args.labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"[2/2] Immagini ({args.split_name}): {len(pairs)}")

    rows = []
    for det_name, yolo in detectors.items():
        for conf in args.conf:
            # Un solo giro di inferenza per conf, riusato per tutte le soglie IoU
            per_image_boxes = []
            for sample in pairs:
                res = yolo(sample["path"], conf=conf, verbose=False)
                boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if res and res[0].boxes is not None else []
                per_image_boxes.append((boxes, sample["gt"]))

            for iou_thr in args.iou_thr:
                total_tp = total_fp = total_fn = total_gt = 0
                for boxes, gt_boxes in per_image_boxes:
                    tp, fp, fn = multi_match(boxes, gt_boxes, iou_thr)
                    total_tp += tp; total_fp += fp; total_fn += fn
                    total_gt += len(gt_boxes)

                precision = safe_precision(total_tp, total_fp)
                recall = total_tp / total_gt if total_gt > 0 else 0.0
                f1 = safe_f1(total_tp, total_fp, total_fn)
                fpr = total_fp / total_gt if total_gt > 0 else 0.0
                fnr = total_fn / total_gt if total_gt > 0 else 0.0

                rows.append({
                    "Split": args.split_name, "Detector": det_name,
                    "conf": conf, "iou_thr": iou_thr,
                    "TP": total_tp, "FP": total_fp, "FN": total_fn, "GT": total_gt,
                    "Precision": round(precision, 4), "Recall_TPR": round(recall, 4),
                    "F1": round(f1, 4), "FPR": round(fpr, 4), "FNR": round(fnr, 4),
                })
                print(f"  [{det_name}] conf={conf} iou={iou_thr} -> "
                      f"P={precision:.4f} R={recall:.4f} F1={f1:.4f} FPR={fpr:.4f} FNR={fnr:.4f}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Split", "Detector", "conf", "iou_thr", "TP", "FP", "FN", "GT",
                  "Precision", "Recall_TPR", "F1", "FPR", "FNR"]

    mode = "a" if (args.append and out_csv.exists()) else "w"
    write_header = not (args.append and out_csv.exists())
    with open(out_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\n[Output] {out_csv} ({'aggiunto' if mode == 'a' else 'creato'}, {len(rows)} righe)")


if __name__ == "__main__":
    main()