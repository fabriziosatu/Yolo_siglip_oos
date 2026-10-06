"""
src/evaluation/fase2_siglip_frozen.py
======================================
FASE 2 — Pipeline di verifica con SigLIP2 FROZEN (zero-shot, nessun training).

Architettura (vedi foglio 1, schema 2):

    immagine originale
        -> YOLO (Fase 1, pesi gia' addestrati, frozen)
        -> BB predetta
        -> crop della BB con padding 15% (vedi paper DRIVE, sez. 4.4:
           il crop con padding batte sia l'immagine intera che l'overlay
           del box sull'immagine intera)
        -> SigLIP2 giant (google/siglip2-giant-opt-patch16-384), FROZEN
        -> classificazione zero-shot testo-immagine:
               "an empty supermarket shelf"  vs  "a supermarket shelf full of products"
        -> True/False (scaffale vuoto confermato / rigettato)

Uso:
    python src/evaluation/fase2_siglip_frozen.py \
        --yolo_weights weights/fase1_baseline/aug/best.pt \
        --data_dir dataset_finale \
        --split test \
        --out_dir output/fase2_siglip_frozen \
        --padding 0.15 \
        --conf 0.25

Output prodotto in --out_dir:
    predictions.csv       - una riga per BB predetta da YOLO, con verdetto SigLIP2 e bce_loss
    metrics_comparison.txt- metriche YOLO-only vs pipeline Fase 1+2
    summary.png           - tabella comparativa YOLO-only vs pipeline
    loss_magnitude.txt    - statistiche di L_VLM (BCE) su tutte le box valutate,
                             utile per calibrare alpha/beta della Fase 4
    loss_magnitude.png    - istogramma della distribuzione di L_VLM

Nota sulla loss di magnitudo (L_VLM):
    Qui SigLIP e' frozen, quindi questa BCE NON viene usata per aggiornare
    pesi: serve solo come lettura diagnostica del punto di partenza, per
    confrontarla con la magnitudo di L_YOLO (vedi
    scripts/analyze_yolo_loss_magnitude.py sui log della Fase 1) e calibrare
    alpha/beta della loss congiunta di Fase 4.
    Etichetta di verita' usata per la BCE: 1 se la box predetta da YOLO
    matcha una GT reale (yolo_is_tp, IoU>=0.5), 0 altrimenti — cioe' "questa
    box e' davvero uno scaffale vuoto secondo l'annotazione".
"""

import math

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────
# Config di default (calibrabili da CLI)
# ──────────────────────────────────────────────────────────────────────────

SIGLIP_CKPT = "google/siglip2-giant-opt-patch16-384"

# Prompt zero-shot: coppia positiva/negativa usata per la similarità
# testo-immagine (segue lo spirito della Tabella 2 del paper DRIVE).
PROMPT_EMPTY = "an empty supermarket shelf, missing products, out of stock"
PROMPT_FULL = "a supermarket shelf full of products, well stocked"

IOU_MATCH_THRESHOLD = 0.5  # per il matching TP/FP/FN contro la GT


# ──────────────────────────────────────────────────────────────────────────
# Utility geometriche
# ──────────────────────────────────────────────────────────────────────────

def load_yolo_labels(label_path: Path, img_w: int, img_h: int):
    """
    Legge un file di label YOLO (formato: class cx cy w h, normalizzato [0,1])
    e restituisce una lista di box in pixel assoluti [x1, y1, x2, y2].
    Solo la classe 0 (empty_shelf) viene considerata.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if cls != 0:
                continue
            cx, cy, w, h = map(float, parts[1:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def iou(box_a, box_b):
    """IoU fra due box [x1, y1, x2, y2]."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def crop_with_padding(image: Image.Image, box, padding_frac: float):
    """
    Ritaglia `image` sulla box [x1, y1, x2, y2], allargando ciascun lato
    di `padding_frac` * (dimensione del lato), poi clampando ai bordi
    dell'immagine originale. Segue la configurazione "Cropped Image with
    Padding" del paper DRIVE (padding ottimale = 15%).
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1

    pad_w = w * padding_frac
    pad_h = h * padding_frac

    x1p = max(0, x1 - pad_w)
    y1p = max(0, y1 - pad_h)
    x2p = min(image.width, x2 + pad_w)
    y2p = min(image.height, y2 + pad_h)

    return image.crop((x1p, y1p, x2p, y2p))


# ──────────────────────────────────────────────────────────────────────────
# Wrapper SigLIP2 — classificazione zero-shot frozen
# ──────────────────────────────────────────────────────────────────────────

class SigLIP2ZeroShotClassifier:
    """
    Wrapper minimale attorno a SigLIP2 (frozen) per classificazione
    zero-shot binaria: scaffale vuoto vs scaffale pieno.

    Usa il modello completo (vision + text tower) perche' la Fase 2
    richiede similarita' testo-immagine senza alcun training; la sola
    vision tower (usata invece in Fase 3 con MLP) non basta qui.
    """

    def __init__(self, ckpt: str = SIGLIP_CKPT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SigLIP2] Carico {ckpt} su {self.device} (puo' richiedere qualche minuto)...")
        self.model = AutoModel.from_pretrained(ckpt).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)

        self.texts = [PROMPT_EMPTY, PROMPT_FULL]

        # Precalcola gli embedding testuali una volta sola (frozen, non cambiano)
        with torch.no_grad():
            text_inputs = self.processor(
                text=self.texts, padding="max_length", max_length=64,
                return_tensors="pt"
            ).to(self.device)
            self.text_embeds = self.model.get_text_features(**text_inputs)
            self.text_embeds = self.text_embeds / self.text_embeds.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def classify(self, crop: Image.Image):
        """
        Restituisce (is_empty: bool, score_empty: float, score_full: float).
        Le due probabilita' derivano dalla logistic (sigmoid) di SigLIP,
        coerente con la formulazione contrastiva del modello.
        """
        image_inputs = self.processor(images=[crop], return_tensors="pt").to(self.device)
        image_embeds = self.model.get_image_features(**image_inputs)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        logit_bias = getattr(self.model, "logit_bias", 0.0)

        logits = image_embeds @ self.text_embeds.T * logit_scale + logit_bias
        probs = torch.sigmoid(logits).squeeze(0)  # [prob_empty, prob_full]

        score_empty = probs[0].item()
        score_full = probs[1].item()
        is_empty = score_empty > score_full

        return is_empty, score_empty, score_full


# ──────────────────────────────────────────────────────────────────────────
# Pipeline principale
# ──────────────────────────────────────────────────────────────────────────

def run_pipeline(args):
    data_dir = Path(args.data_dir)
    images_dir = data_dir / args.split / "images"
    labels_dir = data_dir / args.split / "labels"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[YOLO] Carico pesi Fase 1: {args.yolo_weights}")
    yolo = YOLO(args.yolo_weights)

    siglip = SigLIP2ZeroShotClassifier(device=args.device)

    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"[Dati] {len(image_paths)} immagini in {images_dir}")

    csv_path = out_dir / "predictions.csv"
    rows = []

    # Contatori per le metriche
    yolo_tp = yolo_fp = yolo_fn = 0
    pipe_tp = pipe_fp = pipe_fn = 0
    bce_losses = []  # magnitudo di L_VLM, una voce per ogni box valutata

    for img_idx, img_path in enumerate(image_paths):
        image = Image.open(img_path).convert("RGB")
        gt_boxes = load_yolo_labels(
            labels_dir / (img_path.stem + ".txt"), image.width, image.height
        )
        gt_matched = [False] * len(gt_boxes)
        gt_matched_pipe = [False] * len(gt_boxes)

        results = yolo.predict(source=str(img_path), conf=args.conf, verbose=False)[0]

        pred_boxes = []
        if results.boxes is not None:
            for b in results.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                conf = float(b.conf[0])
                pred_boxes.append([x1, y1, x2, y2, conf])

        # Ordina per confidence decrescente per il matching greedy
        pred_boxes.sort(key=lambda b: -b[4])

        for x1, y1, x2, y2, conf in pred_boxes:
            box = [x1, y1, x2, y2]

            # --- Matching YOLO-only (baseline Fase 1) ---
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                if gt_matched[j]:
                    continue
                cur_iou = iou(box, gt)
                if cur_iou > best_iou:
                    best_iou, best_j = cur_iou, j
            is_tp_yolo = best_iou >= IOU_MATCH_THRESHOLD
            if is_tp_yolo:
                gt_matched[best_j] = True
                yolo_tp += 1
            else:
                yolo_fp += 1

            # --- Fase 2: verifica SigLIP2 sul crop con padding ---
            crop = crop_with_padding(image, box, args.padding)
            is_empty, score_empty, score_full = siglip.classify(crop)

            # --- Magnitudo L_VLM: BCE fra score_empty e l'etichetta vera ---
            # (etichetta vera = is_tp_yolo, cioe' se la box e' davvero uno
            # scaffale vuoto secondo la GT). SigLIP e' frozen, questa BCE
            # NON aggiorna pesi: e' solo una lettura diagnostica.
            eps = 1e-7
            p = min(max(score_empty, eps), 1 - eps)
            y = 1.0 if is_tp_yolo else 0.0
            bce_loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
            bce_losses.append(bce_loss)

            # --- Matching pipeline (solo se SigLIP conferma "vuoto") ---
            # Se SigLIP dice "pieno", la predizione viene scartata:
            # non conta ne' come TP ne' come FP per la pipeline.
            is_tp_pipe = False
            if is_empty:
                best_iou_p, best_j_p = 0.0, -1
                for j, gt in enumerate(gt_boxes):
                    if gt_matched_pipe[j]:
                        continue
                    cur_iou = iou(box, gt)
                    if cur_iou > best_iou_p:
                        best_iou_p, best_j_p = cur_iou, j
                is_tp_pipe = best_iou_p >= IOU_MATCH_THRESHOLD
                if is_tp_pipe:
                    gt_matched_pipe[best_j_p] = True
                    pipe_tp += 1
                else:
                    pipe_fp += 1

            rows.append({
                "image": img_path.name,
                "x1": round(x1, 1), "y1": round(y1, 1),
                "x2": round(x2, 1), "y2": round(y2, 1),
                "yolo_conf": round(conf, 3),
                "yolo_is_tp": is_tp_yolo,
                "siglip_is_empty": is_empty,
                "siglip_score_empty": round(score_empty, 4),
                "siglip_score_full": round(score_full, 4),
                "pipeline_is_tp": is_tp_pipe,
                "bce_loss": round(bce_loss, 5),
            })

        # FN: GT non matchate
        yolo_fn += sum(1 for m in gt_matched if not m)
        pipe_fn += sum(1 for m in gt_matched_pipe if not m)

        if (img_idx + 1) % 50 == 0:
            print(f"  ... {img_idx + 1}/{len(image_paths)} immagini processate")

    # ── Scrivi CSV ──────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Output] CSV salvato in {csv_path}")

    # ── Calcola metriche ────────────────────────────────────────────────
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    yolo_p, yolo_r, yolo_f1 = prf(yolo_tp, yolo_fp, yolo_fn)
    pipe_p, pipe_r, pipe_f1 = prf(pipe_tp, pipe_fp, pipe_fn)

    metrics_txt = out_dir / "metrics_comparison.txt"
    with open(metrics_txt, "w") as f:
        f.write("=== Fase 1 (solo YOLO) ===\n")
        f.write(f"TP={yolo_tp}  FP={yolo_fp}  FN={yolo_fn}\n")
        f.write(f"Precision={yolo_p:.4f}  Recall={yolo_r:.4f}  F1={yolo_f1:.4f}\n\n")
        f.write("=== Fase 2 (YOLO + SigLIP2 frozen zero-shot) ===\n")
        f.write(f"TP={pipe_tp}  FP={pipe_fp}  FN={pipe_fn}\n")
        f.write(f"Precision={pipe_p:.4f}  Recall={pipe_r:.4f}  F1={pipe_f1:.4f}\n")
    print(f"[Output] Metriche salvate in {metrics_txt}")

    # ── Tabella comparativa PNG ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 2.2))
    ax.axis("off")
    table_data = [
        ["", "Precision", "Recall", "F1", "TP", "FP", "FN"],
        ["Fase 1 (YOLO)", f"{yolo_p:.3f}", f"{yolo_r:.3f}", f"{yolo_f1:.3f}", yolo_tp, yolo_fp, yolo_fn],
        ["Fase 2 (+SigLIP2)", f"{pipe_p:.3f}", f"{pipe_r:.3f}", f"{pipe_f1:.3f}", pipe_tp, pipe_fp, pipe_fn],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=150)
    print(f"[Output] Tabella salvata in {out_dir / 'summary.png'}")

    print("\n=== RIEPILOGO ===")
    print(f"Fase 1 (YOLO)     -> P={yolo_p:.3f} R={yolo_r:.3f} F1={yolo_f1:.3f}")
    print(f"Fase 2 (+SigLIP2) -> P={pipe_p:.3f} R={pipe_r:.3f} F1={pipe_f1:.3f}")

    # ── Magnitudo L_VLM (per calibrazione alpha/beta di Fase 4) ─────────
    if bce_losses:
        import statistics as stats
        loss_mean = stats.mean(bce_losses)
        loss_median = stats.median(bce_losses)
        loss_std = stats.stdev(bce_losses) if len(bce_losses) > 1 else 0.0
        loss_min = min(bce_losses)
        loss_max = max(bce_losses)

        loss_txt = out_dir / "loss_magnitude.txt"
        with open(loss_txt, "w") as f:
            f.write("=== Magnitudo L_VLM (BCE, SigLIP2 frozen zero-shot) ===\n")
            f.write(f"N box valutate = {len(bce_losses)}\n")
            f.write(f"mean   = {loss_mean:.4f}\n")
            f.write(f"median = {loss_median:.4f}\n")
            f.write(f"std    = {loss_std:.4f}\n")
            f.write(f"min    = {loss_min:.4f}\n")
            f.write(f"max    = {loss_max:.4f}\n")
            f.write(
                "\nNota: SigLIP e' frozen (zero-shot), questo e' un valore "
                "diagnostico di partenza. Confrontalo con la magnitudo di "
                "L_YOLO (scripts/analyze_yolo_loss_magnitude.py) e, quando "
                "disponibile, con la loss reale di training della Fase 3 "
                "(LoRA) per calibrare alpha/beta della Fase 4.\n"
            )
        print(f"[Output] Magnitudo L_VLM salvata in {loss_txt}")
        print(f"  mean={loss_mean:.4f}  median={loss_median:.4f}  std={loss_std:.4f}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(bce_losses, bins=30, color="steelblue", edgecolor="black")
        ax.axvline(loss_mean, color="red", linestyle="--", label=f"mean={loss_mean:.3f}")
        ax.axvline(loss_median, color="orange", linestyle="--", label=f"median={loss_median:.3f}")
        ax.set_xlabel("BCE loss (L_VLM, SigLIP2 frozen)")
        ax.set_ylabel("Numero di box")
        ax.set_title("Distribuzione magnitudo L_VLM — Fase 2")
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_magnitude.png", dpi=150)
        print(f"[Output] Istogramma salvato in {out_dir / 'loss_magnitude.png'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Fase 2: pipeline YOLO + SigLIP2 frozen zero-shot")
    parser.add_argument("--yolo_weights", type=str, required=True,
                         help="Pesi YOLO Fase 1, es. weights/fase1_baseline/aug/best.pt")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Cartella dataset (es. dataset_finale), con sottocartelle images/ labels/")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir", type=str, default="output/fase2_siglip_frozen")
    parser.add_argument("--conf", type=float, default=0.25, help="Soglia di confidenza YOLO")
    parser.add_argument("--padding", type=float, default=0.15,
                         help="Padding relativo applicato al crop prima di SigLIP2 (0.15 = 15%%)")
    parser.add_argument("--device", type=str, default=None,
                         help="cuda / cpu — default: auto-detect")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
