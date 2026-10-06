"""
src/evaluation/evaluate_fase3_siglip.py
==========================================
End-to-end evaluation of the Phase 3 pipeline (YOLO frozen + FINE-TUNED 
SigLIP2 with LoRA), same schema as evaluate_fase2_siglip.py but with
the SigLIP verdict produced by trained weights instead of zero-shot.

Supports MULTIPLE CHECKPOINTS per architecture, each with a label:
    --mlp_dirs  <dir1> <dir2> ...   --mlp_labels  <lbl1> <lbl2> ...
    --full_dirs <dir1> <dir2> ...   --full_labels <lbl1> <lbl2> ...

For each MLP checkpoint 1 configuration is generated: "mlp_<label>".
For each full checkpoint 2 configurations are generated:
"full_<label>_simple" and "full_<label>_context" (same simple/context 
distinction as before, now repeated for each passed full checkpoint).

Example, to compare rank=16 at neg_ratio=0.25 AND neg_ratio=0.5/1.0
in the same run:
    --mlp_dirs  weights/.../fase3_siglip_lora_r16_neg1/best \
                weights/.../fase3_siglip_lora_r16_neg025_v2/best \
    --mlp_labels neg1 neg025 \
    --full_dirs  weights/.../fase3_siglip_lora_full_r16_neg05/best \
                 weights/.../fase3_siglip_lora_full_r16_neg025/best \
    --full_labels neg05 neg025

Generates configurations: mlp_neg1, mlp_neg025, full_neg05_simple,
full_neg05_context, full_neg025_simple, full_neg025_context.

Counting methodology: identical to Phase 2 (multi-match, TPR/FPR/FNR
normalized on #GT, YES/NO cases crossed with real TP/FPs).

Outputs in --out_dir:
    metrics_full.json           - all combinations conf x iou x det x config
    metrics_full_summary.txt    - same content in readable tabular format
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from peft import PeftModel


# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

SIGLIP_CKPT = "google/siglip2-giant-opt-patch16-384"
CROP_PADDING = 0.15  # consistent with fase2_siglip_frozen.py and train_fase3_lora.py

CONF_THRESHOLDS_DEFAULT = [0.25, 0.5]
IOU_THRESHOLDS_DEFAULT = [0.0, 0.25, 0.5, 0.75]

# Configurations are no longer fixed: they are built dynamically in
# main() starting from the labels passed with --mlp_labels/--full_labels
# (each MLP checkpoint -> 1 config, each full checkpoint -> 2 configs).

# Same fixed prompts used in training by train_fase3_full_lora.py — for
# the "full" model they are also needed in inference (the verdict comes from
# comparing image embeddings vs these texts).
PROMPT_POSITIVE = ["an empty shelf", "a retail shelf with no products",
                    "an out-of-stock shelf section"]
PROMPT_NEGATIVE = ["a shelf full of products", "a well-stocked retail shelf",
                    "a shelf with items on it"]

# Distractors ONLY diagnostic (identical to those of Phase 2/context) — the
# fine-tuned text encoder has never been optimized on these specific
# phrases (the training loss only saw PROMPT_POSITIVE/NEGATIVE),
# so here we test generalization to text never seen in training, not
# a guaranteed behavior. The decision (is_empty) NEVER depends on
# these — it remains always and only empty vs full, same correction already
# applied in Phase 2 after the TPR collapse with the 6-class argmax.
DISTRACTOR_PROMPTS = [
    "a metallic shelf edge or divider, no products visible",
    "a closed freezer or refrigerator glass door",
    "a blurry out of focus photo of a shelf",
    "a dark, poorly lit shelf far from the camera",
]


# ══════════════════════════════════════════════════════════════════════
# Geometric utilities (identical to evaluate_yolo_cosmos_v2.py)
# ══════════════════════════════════════════════════════════════════════

def load_test_pairs(img_dir: Path, lbl_dir: Path):
    """Loads image/label pairs. Each GT box is [x1,y1,x2,y2] in pixels."""
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
        pairs.append({"path": img_path, "image": img, "gt": gt_boxes, "W": W, "H": H})
    return pairs


def iou(box_a, box_b):
    """Calculates intersection over union for two boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def multi_match(pred_boxes, gt_boxes, iou_thr):
    """
    Multi-matching (same convention as the Cosmos project):
      is_tp_pred[i] = True if pred i matches at least one GT
      is_fn_gt[j]   = True if GT j is not matched by any pred
    """
    mat_pred = [[] for _ in range(len(pred_boxes))]
    mat_gt = [[] for _ in range(len(gt_boxes))]
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi].append(gi)
                mat_gt[gi].append(pi)
    is_tp_pred = [len(lst) > 0 for lst in mat_pred]
    is_fn_gt = [len(lst) == 0 for lst in mat_gt]
    return is_tp_pred, is_fn_gt


def crop_with_padding(image: Image.Image, box, padding_frac: float):
    """Crops the image according to the bounding box, adding the specified padding fraction."""
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


# ══════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════
# Fine-tuned SigLIP2 — "mlp" architecture (vision encoder + LoRA + MLP head)
# Same definition as SigLIPLoRAClassifier in train_fase3_lora.py, but
# here we LOAD the already trained adapters and head (inference only).
# ══════════════════════════════════════════════════════════════════════

class SigLIPMLPFineTunedClassifier:
    """Wrapper class managing the MLP branch inference of fine-tuned SigLIP2."""
    def __init__(self, checkpoint_dir: str, ckpt: str = SIGLIP_CKPT,
                 use_bf16: bool = True, device: str = None):
        """Initializes the base model and loads the PEFT adapters and the trained MLP head."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        print(f"[SigLIP2-MLP] Loading backbone {ckpt}...")
        full_model = AutoModel.from_pretrained(ckpt, torch_dtype=dtype)
        hidden_size = full_model.config.vision_config.hidden_size

        print(f"[SigLIP2-MLP] Loading LoRA adapter from {checkpoint_dir / 'lora_adapters'}")
        self.vision_model = PeftModel.from_pretrained(
            full_model.vision_model, str(checkpoint_dir / "lora_adapters")
        ).to(self.device).eval()

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )
        mlp_state = torch.load(checkpoint_dir / "mlp_head.pt", map_location=self.device)
        self.mlp_head.load_state_dict(mlp_state)
        self.mlp_head = self.mlp_head.to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(ckpt)

    @torch.no_grad()
    def classify(self, crop: Image.Image):
        """Returns is_empty: bool. No distractors (not applicable here)."""
        backbone_dtype = next(self.vision_model.parameters()).dtype
        pixel_values = self.processor(images=[crop], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=backbone_dtype)

        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output.float()
        logit = self.mlp_head(pooled).squeeze(-1)
        is_empty = torch.sigmoid(logit).item() > 0.5
        return is_empty, None  # None = no distractor, for compatibility with common interface


# ══════════════════════════════════════════════════════════════════════
# Fine-tuned SigLIP2 — "full" architecture (vision+text + LoRA, contrastive)
# Same definition as SigLIPFullLoRAModel in train_fase3_full_lora.py.
# ══════════════════════════════════════════════════════════════════════

class SigLIPFullFineTunedClassifier:
    """Wrapper class managing the full contrastive inference of fine-tuned SigLIP2."""
    def __init__(self, checkpoint_dir: str, ckpt: str = SIGLIP_CKPT,
                 use_bf16: bool = True, device: str = None):
        """Initializes model, tokenizer, and loads PEFT adapters for both vision and text models."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)
        dtype = torch.bfloat16 if use_bf16 else torch.float32

        print(f"[SigLIP2-Full] Loading backbone {ckpt}...")
        full_model = AutoModel.from_pretrained(ckpt, torch_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)

        print(f"[SigLIP2-Full] Loading visual adapter from {checkpoint_dir / 'lora_adapters_visual'}")
        self.visual_encoder = PeftModel.from_pretrained(
            full_model.vision_model, str(checkpoint_dir / "lora_adapters_visual")
        ).to(self.device).eval()

        print(f"[SigLIP2-Full] Loading text adapter from {checkpoint_dir / 'lora_adapters_text'}")
        self.text_encoder = PeftModel.from_pretrained(
            full_model.text_model, str(checkpoint_dir / "lora_adapters_text")
        ).to(self.device).eval()

        scale_bias = torch.load(checkpoint_dir / "scale_bias.pt", map_location=self.device)
        self.logit_scale = scale_bias["logit_scale"].to(self.device)
        self.logit_bias = scale_bias["logit_bias"].to(self.device)

        self.processor = AutoProcessor.from_pretrained(ckpt)

        # FIXED prompts (identical to those used in training) — precomputed
        # only once here, not recomputed at each classify() as in
        # training (there gradients on the text encoder were needed, here it's just
        # inference, the result is the same but much faster).
        with torch.no_grad():
            self.text_embeds_pos = self._encode_text(PROMPT_POSITIVE)
            self.text_embeds_neg = self._encode_text(PROMPT_NEGATIVE)
            # Distractors: single embedding per phrase (not averaged together,
            # unlike pos/neg — here they are needed individually to understand
            # WHICH distractor would "win", not an aggregated average).
            self.distractor_embeds = [self._encode_text([d]) for d in DISTRACTOR_PROMPTS]

    @torch.no_grad()
    def _encode_text(self, prompts):
        """Tokenizes and encodes text prompts using the fine-tuned text model."""
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=64).to(self.device)
        out = self.text_encoder(**tokens)
        embeds = F.normalize(out.pooler_output.float(), dim=-1)
        return F.normalize(embeds.mean(dim=0, keepdim=True), dim=-1)

    @torch.no_grad()
    def classify(self, crop: Image.Image, use_context: bool = False):
        """
        Returns (is_empty: bool, winning_distractor: str | None).

        The decision (is_empty) is ALWAYS AND ONLY empty vs full, whether you pass
        use_context=True or False — distractors, when computed, are
        PURELY diagnostic (same correct rule already in Phase 2, to
        avoid the TPR collapse caused by an argmax over multiple classes).

        use_context=False: identical behavior as before (no computation
            on distractors, faster).
        use_context=True: also computes which distractor would have the highest
            score among empty/full/distractors — only for diagnostics.
        """
        backbone_dtype = next(self.visual_encoder.parameters()).dtype
        pixel_values = self.processor(images=[crop], return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device, dtype=backbone_dtype)

        vis_out = self.visual_encoder(pixel_values=pixel_values)
        v = F.normalize(vis_out.pooler_output.float(), dim=-1)

        sim_pos = (v @ self.text_embeds_pos.T).item()
        sim_neg = (v @ self.text_embeds_neg.T).item()
        logit_pos = sim_pos * self.logit_scale.exp().item() + self.logit_bias.item()
        logit_neg = sim_neg * self.logit_scale.exp().item() + self.logit_bias.item()

        is_empty = logit_pos > logit_neg  # ALWAYS only empty vs full

        winning_distractor = None
        if use_context:
            distractor_scores = [(v @ emb.T).item() for emb in self.distractor_embeds]
            best_idx = max(range(len(distractor_scores)), key=lambda i: distractor_scores[i])
            best_sim = distractor_scores[best_idx]
            if best_sim > max(sim_pos, sim_neg):
                winning_distractor = DISTRACTOR_PROMPTS[best_idx]

        return is_empty, winning_distractor


# ══════════════════════════════════════════════════════════════════════
# Metrics (identical in spirit to evaluate_yolo_cosmos_v2.py,
# "cosmos_*" renamed to "siglip_*")
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(all_results, iou_thr, detector, arch):
    """Computes evaluating metrics tracking the performance of the YOLO+SigLIP pipeline."""
    total_gt = total_tp = total_fp = total_fn = 0
    yolo_total = siglip_yes_total = siglip_no_total = 0
    siglip_yes_were_tp = siglip_yes_were_fp = 0
    siglip_no_were_tp = siglip_no_were_fp = 0
    fn_missed_by_yolo = 0  # "Irreducible" FNs: GTs that NO YOLO box (even discarded ones) covers

    from collections import Counter
    distractor_wins_by_tp = Counter()  # only for arch with winning_distractor != None (full_context)
    distractor_wins_by_fp = Counter()
    n_real_tp_total = n_real_fp_total = 0

    for r in all_results:
        gt_boxes = r["gt"]
        pred_boxes = r["preds"][detector]
        responses = r["responses"][detector][arch]  # list of bools (True=yes/empty)
        winning_distractors = r.get("winning_distractors", {}).get(detector, {}).get(arch, [None] * len(pred_boxes))

        n_gt = len(gt_boxes)
        total_gt += n_gt
        yolo_total += len(pred_boxes)

        # Baseline: GT not covered by ANY YOLO box (regardless of SigLIP) —
        # this is the portion of FNs for which SigLIP cannot be held responsible.
        _, is_fn_gt_baseline = multi_match(pred_boxes, gt_boxes, iou_thr)
        fn_missed_by_yolo += sum(is_fn_gt_baseline)

        is_tp_real, _ = multi_match(pred_boxes, gt_boxes, iou_thr)

        for resp, is_real_tp, wd in zip(responses, is_tp_real, winning_distractors):
            if is_real_tp:
                n_real_tp_total += 1
            else:
                n_real_fp_total += 1
            if wd is not None:
                if is_real_tp:
                    distractor_wins_by_tp[wd] += 1
                else:
                    distractor_wins_by_fp[wd] += 1

            if resp:  # "yes" / empty confirmed
                siglip_yes_total += 1
                if is_real_tp:
                    siglip_yes_were_tp += 1
                else:
                    siglip_yes_were_fp += 1
            else:  # "no" / rejected
                siglip_no_total += 1
                if is_real_tp:
                    siglip_no_were_tp += 1
                else:
                    siglip_no_were_fp += 1

        kept_boxes = [b for b, resp in zip(pred_boxes, responses) if resp]

        if kept_boxes:
            is_tp_kept, is_fn_gt = multi_match(kept_boxes, gt_boxes, iou_thr)
            total_tp += sum(is_tp_kept)
            total_fp += sum(not tp for tp in is_tp_kept)
            total_fn += sum(is_fn_gt)
        else:
            total_fn += n_gt

    fpr = total_fp / total_gt if total_gt > 0 else 0.0
    fnr = total_fn / total_gt if total_gt > 0 else 0.0
    tpr = total_tp / total_gt if total_gt > 0 else 0.0

    # Delta FN specifically attributable to the SigLIP filter (not YOLO):
    fn_rejected_by_siglip = total_fn - fn_missed_by_yolo

    return {
        "FPR": fpr, "FNR": fnr, "TPR": tpr,
        "TP": total_tp, "FP": total_fp, "FN": total_fn, "GT": total_gt,
        "yolo_total": yolo_total,
        "siglip_yes_total": siglip_yes_total,
        "siglip_no_total": siglip_no_total,
        "siglip_yes_were_tp": siglip_yes_were_tp,
        "siglip_yes_were_fp": siglip_yes_were_fp,
        "siglip_no_were_tp": siglip_no_were_tp,
        "siglip_no_were_fp": siglip_no_were_fp,
        "fn_missed_by_yolo": fn_missed_by_yolo,
        "fn_rejected_by_siglip": fn_rejected_by_siglip,
        "n_real_tp_total": n_real_tp_total,
        "n_real_fp_total": n_real_fp_total,
        "distractor_wins_by_tp": dict(distractor_wins_by_tp.most_common()),
        "distractor_wins_by_fp": dict(distractor_wins_by_fp.most_common()),
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    """Main execution function parsing args and orchestrating the evaluation steps."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_weights", required=True)
    parser.add_argument("--noaug_weights", required=True)
    parser.add_argument("--mlp_dirs", nargs="+", required=True,
                         help="One or more best/ folders of fine-tuned MLP checkpoints")
    parser.add_argument("--mlp_labels", nargs="+", required=True,
                         help="Label for each --mlp_dirs, same order/length "
                              "(e.g. neg1 neg025) — generates config 'mlp_<label>'")
    parser.add_argument("--full_dirs", nargs="+", required=True,
                         help="One or more best/ folders of full fine-tuned checkpoints")
    parser.add_argument("--full_labels", nargs="+", required=True,
                         help="Label for each --full_dirs, same order/length "
                              "(e.g. neg05 neg025) — generates config 'full_<label>_simple' "
                              "and 'full_<label>_context'")
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--test_labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--conf", nargs="+", type=float, default=CONF_THRESHOLDS_DEFAULT)
    parser.add_argument("--iou_thr", nargs="+", type=float, default=IOU_THRESHOLDS_DEFAULT)
    parser.add_argument("--padding", type=float, default=CROP_PADDING)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if len(args.mlp_dirs) != len(args.mlp_labels):
        raise ValueError(f"--mlp_dirs ({len(args.mlp_dirs)}) and --mlp_labels "
                          f"({len(args.mlp_labels)}) must have the same length")
    if len(args.full_dirs) != len(args.full_labels):
        raise ValueError(f"--full_dirs ({len(args.full_dirs)}) and --full_labels "
                          f"({len(args.full_labels)}) must have the same length")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Evaluation pipeline YOLO + FINE-TUNED SigLIP2 (Phase 3) — complete grid")
    print("=" * 70)

    print(f"[1/3] Loading aug detector   <- {args.aug_weights}")
    yolo_aug = YOLO(args.aug_weights)
    print(f"[2/3] Loading noaug detector <- {args.noaug_weights}")
    yolo_noaug = YOLO(args.noaug_weights)
    detectors = {"aug": yolo_aug, "noaug": yolo_noaug}

    print(f"[3/3] Loading fine-tuned SigLIP2: "
          f"{len(args.mlp_dirs)} MLP checkpoints, {len(args.full_dirs)} full checkpoints")
    mlp_classifiers = {}
    for label, ckpt_dir in zip(args.mlp_labels, args.mlp_dirs):
        print(f"  [mlp:{label}] <- {ckpt_dir}")
        mlp_classifiers[label] = SigLIPMLPFineTunedClassifier(ckpt_dir, device=args.device)
    full_classifiers = {}
    for label, ckpt_dir in zip(args.full_labels, args.full_dirs):
        print(f"  [full:{label}] <- {ckpt_dir}")
        full_classifiers[label] = SigLIPFullFineTunedClassifier(ckpt_dir, device=args.device)

    # Configuration grid built dynamically from the provided labels:
    # each MLP checkpoint -> 1 config, each full checkpoint -> 2 configs (simple/context)
    config_names = [f"mlp_{lbl}" for lbl in args.mlp_labels]
    for lbl in args.full_labels:
        config_names.append(f"full_{lbl}_simple")
        config_names.append(f"full_{lbl}_context")

    pairs = load_test_pairs(Path(args.test_images), Path(args.test_labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"\n  Test images : {len(pairs)}")
    print(f"  CONF        : {args.conf}")
    print(f"  IOU         : {args.iou_thr}")
    print(f"  CONFIGS     : {config_names}")

    # ── YOLO inference (once per conf x detector) ───────────────────
    print("\nPhase 1/2: YOLO inference...")
    all_results = {}
    for conf in args.conf:
        results_conf = []
        for sample in pairs:
            entry = {"path": sample["path"], "gt": sample["gt"], "image": sample["image"],
                      "preds": {}, "responses": {}}
            for det_name, yolo in detectors.items():
                res = yolo(sample["path"], conf=conf, verbose=False)
                boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if res and res[0].boxes is not None else []
                entry["preds"][det_name] = boxes
                entry["responses"][det_name] = {}
            results_conf.append(entry)
        all_results[conf] = results_conf
        print(f"  conf={conf}: YOLO completed on {len(results_conf)} images")

    # ── SigLIP2 inference (for each conf x detector x configuration) ────
    print("\nPhase 2/2: SigLIP2 inference (crop with padding)...")
    PROGRESS_EVERY = 50
    from collections import Counter

    # Map configuration -> function that returns (is_empty, winning_distractor).
    # Name parsing: "mlp_<label>" or "full_<label>_simple"/"full_<label>_context".
    def classify_fn(config_name, crop):
        if config_name.startswith("mlp_"):
            label = config_name[len("mlp_"):]
            return mlp_classifiers[label].classify(crop)
        elif config_name.startswith("full_") and config_name.endswith("_simple"):
            label = config_name[len("full_"):-len("_simple")]
            return full_classifiers[label].classify(crop, use_context=False)
        elif config_name.startswith("full_") and config_name.endswith("_context"):
            label = config_name[len("full_"):-len("_context")]
            return full_classifiers[label].classify(crop, use_context=True)
        raise ValueError(f"Unknown configuration: {config_name}")

    distractor_stats = {}  # (conf, det, config) -> (Counter, n_boxes) — only for progress print
    for conf in args.conf:
        for det_name in detectors:
            for config_name in config_names:
                print(f"  det={det_name}  conf={conf}  config='{config_name}'")
                yes_count = no_count = boxes_done = 0
                distractor_win_counter = Counter()
                n_images = len(all_results[conf])
                for img_idx, entry in enumerate(all_results[conf], start=1):
                    boxes = entry["preds"][det_name]
                    image = entry["image"]
                    resps = []
                    wdist = []
                    for box in boxes:
                        crop = crop_with_padding(image, box, args.padding)
                        is_empty, winning_distractor = classify_fn(config_name, crop)
                        resps.append(is_empty)
                        wdist.append(winning_distractor)
                        if is_empty:
                            yes_count += 1
                        else:
                            no_count += 1
                        if winning_distractor is not None:
                            distractor_win_counter[winning_distractor] += 1
                    entry["responses"][det_name][config_name] = resps
                    entry.setdefault("winning_distractors", {}).setdefault(det_name, {})[config_name] = wdist
                    boxes_done += len(boxes)
                    if img_idx % PROGRESS_EVERY == 0 or img_idx == n_images:
                        print(f"    [{img_idx}/{n_images}] total_box={boxes_done} "
                              f"YES={yes_count} NO={no_count}", flush=True)
                if config_name.endswith("_context"):
                    distractor_stats[(conf, det_name, config_name)] = (distractor_win_counter, boxes_done)
                    total_wins = sum(distractor_win_counter.values())
                    if boxes_done > 0:
                        pct = 100 * total_wins / boxes_done
                        print(f"    [diagnostic, not used for decision] "
                              f"a distractor would have won on {total_wins}/{boxes_done} boxes ({pct:.1f}%)")
                        for label, count in distractor_win_counter.most_common():
                            label_pct = 100 * count / total_wins if total_wins else 0
                            print(f"        {count:5d} ({label_pct:5.1f}%) -> \"{label}\"")

    # ── Metrics on the full grid ──────────────────────────────────────
    print("\nCalculating metrics...")
    output = {}
    for conf in args.conf:
        for iou_t in args.iou_thr:
            for det_name in detectors:
                for config_name in config_names:
                    key = f"conf={conf}_iou={iou_t}_{det_name}_{config_name}"
                    m = compute_metrics(all_results[conf], iou_t, det_name, config_name)
                    m.update({"conf": conf, "iou": iou_t, "detector": det_name, "arch": config_name})
                    output[key] = m
                    print(f"  {det_name:6s} {config_name:18s} conf={conf} iou={iou_t} | "
                          f"TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} FNR={m['FNR']:.3f}")

    with open(out_dir / "metrics_full.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Output] {out_dir / 'metrics_full.json'}")

    # NOTE: distractor data (crossed with real TP/FP) is now
    # inside metrics_full.json for each conf/iou/det/full_context combination
    # (keys "distractor_wins_by_tp"/"distractor_wins_by_fp") — same schema
    # as evaluate_fase2_siglip.py, so visualize_fase3_siglip.py can generate
    # the distractors table without needing a separate file.

    with open(out_dir / "metrics_full_summary.txt", "w") as f:
        f.write(f"{'detector':8s} | {'config':12s} | {'conf':4s} | {'iou':4s} | "
                f"{'TPR':5s} | {'FPR':5s} | {'FNR':5s} | {'YOLO_tot':8s} | "
                f"{'YES':5s} | {'NO':5s}\n")
        f.write("-" * 95 + "\n")
        for key, m in output.items():
            f.write(f"{m['detector']:8s} | {m['arch']:12s} | {m['conf']:.2f} | {m['iou']:.2f} | "
                    f"{m['TPR']:.3f} | {m['FPR']:.3f} | {m['FNR']:.3f} | {m['yolo_total']:8d} | "
                    f"{m['siglip_yes_total']:5d} | {m['siglip_no_total']:5d}\n")
    print(f"[Output] {out_dir / 'metrics_full_summary.txt'}")

    print("\nCompleted.")


if __name__ == "__main__":
    main()