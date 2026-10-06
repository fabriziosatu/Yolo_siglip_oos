"""
src/evaluation/compare_new_fp_by_distractor.py
==================================================
Isola i "nuovi FP" introdotti dal fine-tuning — box che lo zero-shot
("Versione 3", Context Diagnostico) rigettava correttamente ma che il
modello SigLIP2 fine-tunato (full, con prompt context) conferma per
errore — e ne mostra la distribuzione per categoria di distrattore.

Motivazione (vedi discussione in chat): il fine-tuning riduce i FN ma
aumenta i FP rispetto allo zero-shot. L'ipotesi da verificare e' che i
nuovi FP siano concentrati nelle stesse categorie diagnostiche gia'
isolate in Fase 2 (foto sfocata, scaffale scuro, ecc.) — segno che il
negative mining sintetico usato in training non copre abbastanza bene
quei casi specifici.

A differenza di un confronto tra due run separate (che richiederebbe
salvare e poi unire risultati per-box da due script diversi, con rischio
di disallineamento), qui i due modelli girano NELLO STESSO passaggio,
sugli stessi identici box prodotti da YOLO — il confronto e' quindi
diretto e senza ambiguita' di matching.

Uso:
    python src/evaluation/compare_new_fp_by_distractor.py \
        --aug_weights weights/fase1_baseline/aug/best.pt \
        --noaug_weights weights/fase1_baseline/no_aug/best.pt \
        --full_dir weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r16_neg025/best \
        --test_images dataset_finale/test/images \
        --test_labels dataset_finale/test/labels \
        --out_dir output/fase3_new_fp_analysis \
        --conf 0.25 \
        --iou_thr 0.0 \
        --dump_dir output/fase3_new_fp_analysis/crops

Output in --out_dir:
    new_fp_by_distractor.json   - conteggi grezzi
    new_fp_by_distractor.png    - tabella riassuntiva (stile Fase 2)
Se --dump_dir e' specificato, salva anche i crop dei nuovi FP per ispezione manuale.
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from peft import PeftModel


SIGLIP_CKPT = "google/siglip2-giant-opt-patch16-384"

# Prompt "context" — identico a evaluate_fase2_siglip_v2.py, "Versione 3"
# (Context Diagnostico): decisione SOLO empty-vs-full, distrattori diagnostici.
ZS_CONTEXT = {
    "empty": "a bare empty supermarket shelf, visible back wall and shadow, no items",
    "full": "a supermarket shelf stocked with visible products",
    "distractors": [
        "a metallic shelf edge or divider, no products visible",
        "a closed freezer or refrigerator glass door",
        "a blurry out of focus photo of a shelf",
        "a dark, poorly lit shelf far from the camera",
    ],
}

# Prompt fissi del modello fine-tunato — identici a train_fase3_full_lora.py
FT_PROMPT_POSITIVE = ["an empty shelf", "a retail shelf with no products",
                       "an out-of-stock shelf section"]
FT_PROMPT_NEGATIVE = ["a shelf full of products", "a well-stocked retail shelf",
                       "a shelf with items on it"]
FT_CONTEXT_DISTRACTORS = ZS_CONTEXT["distractors"]  # stessi 4 usati per la diagnostica in Fase 3


# ══════════════════════════════════════════════════════════════════════
# Utility geometriche (identiche a evaluate_fase2/3_siglip.py)
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
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
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


def multi_match(pred_boxes, gt_boxes, iou_thr):
    mat_pred = [[] for _ in range(len(pred_boxes))]
    for pi, pb in enumerate(pred_boxes):
        for gb in gt_boxes:
            v = iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi].append(1)
    return [len(lst) > 0 for lst in mat_pred]  # is_tp_pred


def crop_with_padding(image: Image.Image, box, padding_frac: float):
    x1, y1, x2, y2 = box
    w = x2 - x1; h = y2 - y1
    pad_w = w * padding_frac; pad_h = h * padding_frac
    x1p = max(0, x1 - pad_w); y1p = max(0, y1 - pad_h)
    x2p = min(image.width, x2 + pad_w); y2p = min(image.height, y2 + pad_h)
    return image.crop((x1p, y1p, x2p, y2p))


# ══════════════════════════════════════════════════════════════════════
# Zero-shot "Versione 3" (Context Diagnostico) — decisione empty-vs-full,
# distrattori diagnostici, IDENTICO a evaluate_fase2_siglip_v2.py.
# ══════════════════════════════════════════════════════════════════════

class ZeroShotContextClassifier:
    def __init__(self, ckpt: str = SIGLIP_CKPT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ZeroShot] Carico {ckpt} su {self.device}...")
        self.model = AutoModel.from_pretrained(ckpt).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)
        texts = [ZS_CONTEXT["empty"], ZS_CONTEXT["full"]] + ZS_CONTEXT["distractors"]
        self.texts = texts
        with torch.no_grad():
            self.text_inputs = self.processor(
                text=texts, padding="max_length", max_length=64, return_tensors="pt"
            ).to(self.device)

    @torch.no_grad()
    def classify(self, crop: Image.Image):
        image_inputs = self.processor(images=[crop], return_tensors="pt").to(self.device)
        outputs = self.model(
            input_ids=self.text_inputs["input_ids"],
            attention_mask=self.text_inputs.get("attention_mask"),
            pixel_values=image_inputs["pixel_values"],
        )
        probs = torch.sigmoid(outputs.logits_per_image).squeeze(0)
        is_empty = probs[0].item() > probs[1].item()  # SOLO empty vs full
        return is_empty


# ══════════════════════════════════════════════════════════════════════
# SigLIP2 fine-tunato — completo, con prompt context (identico a
# evaluate_fase3_siglip.py, incluso il calcolo diagnostico dei distrattori)
# ══════════════════════════════════════════════════════════════════════

class SigLIPFullFineTunedClassifier:
    def __init__(self, checkpoint_dir: str, ckpt: str = SIGLIP_CKPT,
                 use_bf16: bool = True, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        print(f"[FineTuned] Carico backbone {ckpt}...")
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
            self.distractor_embeds = [(t, self._encode_text([t])) for t in FT_CONTEXT_DISTRACTORS]

    @torch.no_grad()
    def _encode_text(self, prompts):
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=64).to(self.device)
        out = self.text_encoder(**tokens)
        embeds = F.normalize(out.pooler_output.float(), dim=-1)
        return F.normalize(embeds.mean(dim=0, keepdim=True), dim=-1)

    @torch.no_grad()
    def classify(self, crop: Image.Image):
        """Restituisce (is_empty, winning_distractor)."""
        backbone_dtype = next(self.visual_encoder.parameters()).dtype
        pixel_values = self.processor(images=[crop], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=backbone_dtype)

        vis_out = self.visual_encoder(pixel_values=pixel_values)
        v = F.normalize(vis_out.pooler_output.float(), dim=-1)

        scale = self.logit_scale.exp().item(); bias = self.logit_bias.item()
        sim_pos = (v @ self.text_embeds_pos.T).item()
        sim_neg = (v @ self.text_embeds_neg.T).item()
        logit_pos = sim_pos*scale + bias; logit_neg = sim_neg*scale + bias
        is_empty = logit_pos > logit_neg

        best_label, best_logit = None, -1e9
        for label, emb in self.distractor_embeds:
            sim = (v @ emb.T).item()
            logit = sim*scale + bias
            if logit > best_logit:
                best_logit = logit; best_label = label
        winning_distractor = best_label if best_logit > max(logit_pos, logit_neg) else None

        return is_empty, winning_distractor


# ══════════════════════════════════════════════════════════════════════
# Tabella riassuntiva (stile Fase 2)
# ══════════════════════════════════════════════════════════════════════

def render_table(stats, out_path):
    W, row_h, head_h = 900, 40, 60
    labels = ["Metallico/divisorio", "Sportello frigo", "Foto sfocata", "Scaffale scuro"]
    keys = FT_CONTEXT_DISTRACTORS
    dets = list(stats.keys())
    H = head_h + 40 + row_h * (len(dets) * 2 + 1)
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        f_head = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        f_cell = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        f_title = f_head = f_cell = ImageFont.load_default()

    d.text((15, 12), "Nuovi FP introdotti dal fine-tuning — distribuzione per distrattore",
           font=f_title, fill=(26, 26, 46))
    y = head_h
    col_w = (W - 200) // len(labels)
    d.rectangle([0, y, W, y+30], fill=(30, 39, 97))
    d.text((10, y+7), "Detector", font=f_head, fill=(255,255,255))
    d.text((110, y+7), "# nuovi FP", font=f_head, fill=(255,255,255))
    for i, lbl in enumerate(labels):
        d.text((200+i*col_w+5, y+7), lbl, font=f_head, fill=(255,255,255))
    y += 30

    for det, info in stats.items():
        d.rectangle([0, y, W, y+row_h], fill=(247,247,247))
        d.text((10, y+12), det, font=f_cell, fill=(26,26,46))
        d.text((110, y+12), str(info["n_new_fp"]), font=f_cell, fill=(192,57,43))
        for i, key in enumerate(keys):
            pct = 100*info["wins"].get(key, 0)/info["n_new_fp"] if info["n_new_fp"] else 0
            d.text((200+i*col_w+5, y+12), f"{pct:.1f}%", font=f_cell, fill=(26,26,46))
        y += row_h

    img.save(out_path)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_weights", required=True)
    parser.add_argument("--noaug_weights", required=True)
    parser.add_argument("--full_dir", required=True,
                         help="Cartella best/ del checkpoint completo fine-tunato")
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--test_labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou_thr", type=float, default=0.0)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dump_dir", type=str, default=None,
                         help="Se specificato, salva su disco il crop (con padding) di ogni "
                              "'nuovo FP' — utile per ispezione manuale. Nome file: "
                              "<detector>_<indice>_<distrattore_o_nessuno>.jpg")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = None
    if args.dump_dir:
        dump_dir = Path(args.dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Dump] I crop dei nuovi FP verranno salvati in {dump_dir}")

    print("[1/4] Carico YOLO aug/noaug...")
    detectors = {"aug": YOLO(args.aug_weights), "noaug": YOLO(args.noaug_weights)}

    print("[2/4] Carico zero-shot 'Versione 3' (Context Diagnostico)...")
    zs_classifier = ZeroShotContextClassifier(device=args.device)

    print("[3/4] Carico modello fine-tunato completo (context)...")
    ft_classifier = SigLIPFullFineTunedClassifier(args.full_dir, device=args.device)

    pairs = load_test_pairs(Path(args.test_images), Path(args.test_labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"[4/4] Immagini test: {len(pairs)}  conf={args.conf}  iou_thr={args.iou_thr}")

    stats = {det: {"n_new_fp": 0, "n_total_fp_ft": 0, "wins": Counter()} for det in detectors}

    for img_idx, sample in enumerate(pairs, start=1):
        image = sample["image"]; gt_boxes = sample["gt"]
        for det_name, yolo in detectors.items():
            res = yolo(sample["path"], conf=args.conf, verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if res and res[0].boxes is not None else []
            if not boxes:
                continue
            is_tp_real = multi_match(boxes, gt_boxes, args.iou_thr)

            for box, real_tp in zip(boxes, is_tp_real):
                if real_tp:
                    continue  # ci interessano solo le box che sono VERI FP
                crop = crop_with_padding(image, box, args.padding)
                zs_verdict = zs_classifier.classify(crop)
                ft_verdict, winning_distractor = ft_classifier.classify(crop)

                if ft_verdict:
                    stats[det_name]["n_total_fp_ft"] += 1
                # "Nuovo FP": zero-shot rigettava correttamente, fine-tuned conferma per errore
                if (not zs_verdict) and ft_verdict:
                    stats[det_name]["n_new_fp"] += 1
                    if winning_distractor is not None:
                        stats[det_name]["wins"][winning_distractor] += 1
                    if dump_dir is not None:
                        idx = stats[det_name]["n_new_fp"]
                        distr_tag = "nessuno"
                        if winning_distractor is not None:
                            # etichetta breve leggibile nel nome file (prime 2 parole)
                            distr_tag = "_".join(winning_distractor.split()[:2])
                        fname = f"{det_name}_{idx:04d}_{distr_tag}_{sample['path'].stem}.jpg"
                        crop.convert("RGB").save(dump_dir / fname, quality=90)

        if img_idx % 100 == 0 or img_idx == len(pairs):
            print(f"  [{img_idx}/{len(pairs)}] " +
                  "  ".join(f"{d}:nuovi_fp={s['n_new_fp']}" for d, s in stats.items()), flush=True)

    # ── Output ──
    out_json = {det: {"n_new_fp": s["n_new_fp"], "n_total_fp_ft": s["n_total_fp_ft"],
                       "wins": dict(s["wins"].most_common())}
                for det, s in stats.items()}
    with open(out_dir / "new_fp_by_distractor.json", "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\n[Output] {out_dir / 'new_fp_by_distractor.json'}")

    render_table(stats, out_dir / "new_fp_by_distractor.png")
    print(f"[Output] {out_dir / 'new_fp_by_distractor.png'}")


if __name__ == "__main__":
    main()