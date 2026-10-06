"""
src/evaluation/threshold_sweep.py
====================================
Calcola il trade-off TPR/FPR/FNR al variare della SOGLIA di decisione di
SigLIP fine-tunato — a costo zero (nessun retraining, un solo passaggio
di inferenza sul test set).

Motivazione (vedi discussione in chat): la decisione attuale ("vuoto" se
il punteggio grezzo > 0) e' una soglia di comodo, non necessariamente
quella ottimale. Cambiare soglia sposta il compromesso FP/FN su TUTTE le
box (non solo su un sottoinsieme specifico gia' osservato) — e' l'unico
modo legittimo di "correggere" il comportamento del modello senza
memorizzare risposte specifiche (data snooping), che invaliderebbe la
valutazione.

Punteggio grezzo:
  - "mlp" : il logit dell'MLP head (prima della sigmoid) — decisione
            attuale: > 0.
  - "full": logit_pos - logit_neg (differenza tra i due logit contrastivi)
            — decisione attuale: > 0.

Uso:
    python src/evaluation/threshold_sweep.py \
        --aug_weights weights/fase1_baseline/aug/best.pt \
        --noaug_weights weights/fase1_baseline/no_aug/best.pt \
        --arch full \
        --checkpoint_dir weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r16_neg025/best \
        --test_images dataset_finale/test/images \
        --test_labels dataset_finale/test/labels \
        --out_dir output/fase3_threshold_sweep \
        --conf 0.25 \
        --iou_thr 0.0

Output in --out_dir:
    threshold_sweep.json         - dati grezzi per ogni soglia testata
    threshold_sweep_table.txt    - tabella leggibile
    threshold_sweep_curves.png   - TPR/FPR/FNR vs soglia, per detector
    roc_curve.png                - TPR vs FPR (stile ROC), soglie annotate
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from peft import PeftModel


SIGLIP_CKPT = "google/siglip2-giant-opt-patch16-384"

FT_PROMPT_POSITIVE = ["an empty shelf", "a retail shelf with no products",
                       "an out-of-stock shelf section"]
FT_PROMPT_NEGATIVE = ["a shelf full of products", "a well-stocked retail shelf",
                       "a shelf with items on it"]


# ══════════════════════════════════════════════════════════════════════
# Utility geometriche
# ══════════════════════════════════════════════════════════════════════

def load_test_pairs(img_dir: Path, lbl_dir: Path):
    pairs = []
    paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    for img_path in paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
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
        pairs.append({"path": img_path, "image": img, "gt": gt_boxes})
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


def matched_gt_indices(pred_box, gt_boxes, iou_thr):
    """Ritorna l'insieme (frozenset) di indici GT che questa box matcha."""
    matched = set()
    for gi, gb in enumerate(gt_boxes):
        v = iou(pred_box, gb)
        if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
            matched.add(gi)
    return frozenset(matched)


def crop_with_padding(image: Image.Image, box, padding_frac: float):
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    pad_w = w * padding_frac; pad_h = h * padding_frac
    x1p = max(0, x1 - pad_w); y1p = max(0, y1 - pad_h)
    x2p = min(image.width, x2 + pad_w); y2p = min(image.height, y2 + pad_h)
    return image.crop((x1p, y1p, x2p, y2p))


# ══════════════════════════════════════════════════════════════════════
# Classificatori — stessa architettura di evaluate_fase3_siglip.py, con
# l'aggiunta di un metodo score() che restituisce il punteggio GREZZO
# (prima della soglia), invece del solo verdetto booleano.
# ══════════════════════════════════════════════════════════════════════

class SigLIPMLPFineTunedClassifier:
    def __init__(self, checkpoint_dir: str, ckpt: str = SIGLIP_CKPT,
                 use_bf16: bool = True, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        full_model = AutoModel.from_pretrained(ckpt, torch_dtype=dtype)
        hidden_size = full_model.config.vision_config.hidden_size
        self.vision_model = PeftModel.from_pretrained(
            full_model.vision_model, str(checkpoint_dir / "lora_adapters")
        ).to(self.device).eval()

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(hidden_size // 2, 1),
        )
        self.mlp_head.load_state_dict(torch.load(checkpoint_dir / "mlp_head.pt", map_location=self.device))
        self.mlp_head = self.mlp_head.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)

    @torch.no_grad()
    def score(self, crop: Image.Image) -> float:
        """Punteggio grezzo: il logit prima della sigmoid. Decisione attuale: >0."""
        backbone_dtype = next(self.vision_model.parameters()).dtype
        pixel_values = self.processor(images=[crop], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=backbone_dtype)
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output.float()
        logit = self.mlp_head(pooled).squeeze(-1)
        return logit.item()


class SigLIPFullFineTunedClassifier:
    def __init__(self, checkpoint_dir: str, ckpt: str = SIGLIP_CKPT,
                 use_bf16: bool = True, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        full_model = AutoModel.from_pretrained(ckpt, torch_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)
        self.visual_encoder = PeftModel.from_pretrained(
            full_model.vision_model, str(checkpoint_dir / "lora_adapters_visual")
        ).to(self.device).eval()
        self.text_encoder = PeftModel.from_pretrained(
            full_model.text_model, str(checkpoint_dir / "lora_adapters_text")
        ).to(self.device).eval()

        scale_bias = torch.load(checkpoint_dir / "scale_bias.pt", map_location=self.device)
        self.logit_scale = scale_bias["logit_scale"].to(self.device)
        self.logit_bias = scale_bias["logit_bias"].to(self.device)
        self.processor = AutoProcessor.from_pretrained(ckpt)

        with torch.no_grad():
            self.text_embeds_pos = self._encode_text(FT_PROMPT_POSITIVE)
            self.text_embeds_neg = self._encode_text(FT_PROMPT_NEGATIVE)

    @torch.no_grad()
    def _encode_text(self, prompts):
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=64).to(self.device)
        out = self.text_encoder(**tokens)
        embeds = F.normalize(out.pooler_output.float(), dim=-1)
        return F.normalize(embeds.mean(dim=0, keepdim=True), dim=-1)

    @torch.no_grad()
    def score(self, crop: Image.Image) -> float:
        """Punteggio grezzo: logit_pos - logit_neg. Decisione attuale: >0."""
        backbone_dtype = next(self.visual_encoder.parameters()).dtype
        pixel_values = self.processor(images=[crop], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=backbone_dtype)
        vis_out = self.visual_encoder(pixel_values=pixel_values)
        v = F.normalize(vis_out.pooler_output.float(), dim=-1)

        scale = self.logit_scale.exp().item(); bias = self.logit_bias.item()
        sim_pos = (v @ self.text_embeds_pos.T).item()
        sim_neg = (v @ self.text_embeds_neg.T).item()
        logit_pos = sim_pos*scale + bias; logit_neg = sim_neg*scale + bias
        return logit_pos - logit_neg


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_weights", required=True)
    parser.add_argument("--noaug_weights", required=True)
    parser.add_argument("--arch", choices=["mlp", "full"], required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--test_labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou_thr", type=float, default=0.0)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--n_thresholds", type=int, default=21,
                         help="Numero di soglie da testare, distribuite uniformemente "
                              "tra il punteggio minimo e massimo osservato")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Carico YOLO aug/noaug...")
    detectors = {"aug": YOLO(args.aug_weights), "noaug": YOLO(args.noaug_weights)}

    print(f"[2/3] Carico classificatore '{args.arch}' fine-tunato...")
    if args.arch == "mlp":
        classifier = SigLIPMLPFineTunedClassifier(args.checkpoint_dir, device=args.device)
    else:
        classifier = SigLIPFullFineTunedClassifier(args.checkpoint_dir, device=args.device)

    pairs = load_test_pairs(Path(args.test_images), Path(args.test_labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"[3/3] Immagini test: {len(pairs)}  conf={args.conf}  iou_thr={args.iou_thr}")

    # Per ciascun detector: lista per immagine di (n_gt, [(score, matched_gt_frozenset), ...])
    data_by_det = {det: [] for det in detectors}

    for img_idx, sample in enumerate(pairs, start=1):
        image = sample["image"]; gt_boxes = sample["gt"]
        for det_name, yolo in detectors.items():
            res = yolo(sample["path"], conf=args.conf, verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if res and res[0].boxes is not None else []
            entries = []
            for box in boxes:
                crop = crop_with_padding(image, box, args.padding)
                s = classifier.score(crop)
                matched = matched_gt_indices(box, gt_boxes, args.iou_thr)
                entries.append((s, matched))
            data_by_det[det_name].append((len(gt_boxes), entries))

        if img_idx % 100 == 0 or img_idx == len(pairs):
            print(f"  [{img_idx}/{len(pairs)}] elaborate", flush=True)

    # ── Range di soglie da testare ──
    all_scores = [s for det in data_by_det.values() for _, entries in det for s, _ in entries]
    if not all_scores:
        print("[Attenzione] Nessuna box trovata, impossibile calcolare soglie.")
        return
    smin, smax = min(all_scores), max(all_scores)
    thresholds = np.linspace(smin, smax, args.n_thresholds).tolist()
    # Assicura che la soglia attuale (0.0) sia sempre inclusa, per confronto diretto
    if smin < 0 < smax and 0.0 not in thresholds:
        thresholds.append(0.0)
        thresholds.sort()

    print(f"\n[Info] Punteggi osservati: min={smin:.3f}  max={smax:.3f}")
    print(f"[Info] Soglie testate: {len(thresholds)} (da {smin:.2f} a {smax:.2f}, 0.0 inclusa)")

    # ── Calcolo metriche per ciascuna soglia ──
    results = {}
    for det_name, images_data in data_by_det.items():
        for t in thresholds:
            total_gt = total_tp = total_fp = total_fn = 0
            for n_gt, entries in images_data:
                total_gt += n_gt
                kept = [(s, m) for s, m in entries if s > t]
                covered = set()
                for s, m in kept:
                    if m:
                        total_tp += 1
                        covered |= m
                    else:
                        total_fp += 1
                total_fn += n_gt - len(covered)
            tpr = total_tp / total_gt if total_gt else 0.0
            fpr = total_fp / total_gt if total_gt else 0.0
            fnr = total_fn / total_gt if total_gt else 0.0
            results[(det_name, round(t, 4))] = {
                "TP": total_tp, "FP": total_fp, "FN": total_fn, "GT": total_gt,
                "TPR": tpr, "FPR": fpr, "FNR": fnr,
            }

    # ── Output: JSON + tabella testuale ──
    out_json = {f"{det}__{t}": v for (det, t), v in results.items()}
    with open(out_dir / "threshold_sweep.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[Output] {out_dir / 'threshold_sweep.json'}")

    with open(out_dir / "threshold_sweep_table.txt", "w") as f:
        f.write(f"{'Detector':8s} | {'Soglia':8s} | {'TPR':6s} | {'FPR':6s} | {'FNR':6s} | "
                f"{'TP':6s} | {'FP':6s} | {'FN':6s}\n")
        f.write("-" * 70 + "\n")
        for det_name in detectors:
            for t in thresholds:
                r = results[(det_name, round(t, 4))]
                marker = "  <-- soglia attuale" if abs(t) < 1e-9 else ""
                f.write(f"{det_name:8s} | {t:8.3f} | {r['TPR']:.4f} | {r['FPR']:.4f} | "
                        f"{r['FNR']:.4f} | {r['TP']:6d} | {r['FP']:6d} | {r['FN']:6d}{marker}\n")
    print(f"[Output] {out_dir / 'threshold_sweep_table.txt'}")

    # ── Grafico 1: TPR/FPR/FNR vs soglia ──
    fig, axes = plt.subplots(1, len(detectors), figsize=(6*len(detectors), 5), squeeze=False)
    for i, det_name in enumerate(detectors):
        ax = axes[0][i]
        ts = thresholds
        tpr_vals = [results[(det_name, round(t, 4))]["TPR"] for t in ts]
        fpr_vals = [results[(det_name, round(t, 4))]["FPR"] for t in ts]
        fnr_vals = [results[(det_name, round(t, 4))]["FNR"] for t in ts]
        ax.plot(ts, tpr_vals, label="TPR", color="#2E8B57", marker="o", markersize=3)
        ax.plot(ts, fpr_vals, label="FPR", color="#C0392B", marker="o", markersize=3)
        ax.plot(ts, fnr_vals, label="FNR", color="#E8871E", marker="o", markersize=3)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1, label="soglia attuale (0.0)")
        ax.set_xlabel("Soglia sul punteggio grezzo")
        ax.set_ylabel("Valore metrica")
        ax.set_title(f"{det_name}")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.suptitle(f"Trade-off TPR/FPR/FNR al variare della soglia — arch={args.arch}, conf={args.conf}")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep_curves.png", dpi=150)
    print(f"[Output] {out_dir / 'threshold_sweep_curves.png'}")

    # ── Grafico 2: ROC-style TPR vs FPR ──
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"aug": "#1E2761", "noaug": "#E8871E"}
    for det_name in detectors:
        fpr_vals = [results[(det_name, round(t, 4))]["FPR"] for t in thresholds]
        tpr_vals = [results[(det_name, round(t, 4))]["TPR"] for t in thresholds]
        ax.plot(fpr_vals, tpr_vals, marker="o", markersize=4, color=colors.get(det_name, "black"), label=det_name)
        # evidenzia la soglia attuale
        cur = results[(det_name, round(0.0, 4))]
        ax.scatter([cur["FPR"]], [cur["TPR"]], color=colors.get(det_name, "black"), s=90, zorder=5,
                   edgecolor="white", linewidth=1.5)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"TPR vs FPR al variare della soglia — arch={args.arch}\n(pallino pieno = soglia attuale, 0.0)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve.png", dpi=150)
    print(f"[Output] {out_dir / 'roc_curve.png'}")

    print("\nCompletato.")


if __name__ == "__main__":
    main()