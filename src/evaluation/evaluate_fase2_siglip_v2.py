"""
src/evaluation/evaluate_fase2_siglip.py
==========================================
Complete evaluation of the Phase 2 pipeline (YOLO frozen + SigLIP2 frozen
zero-shot), replicating the evaluation schema already used for Cosmos
(evaluate_yolo_cosmos_v2.py), adapted to SigLIP2.

Evaluates 4 pipelines at once:
    2 detectors (aug / noaug) x 2 prompt sets (no_context / context)
on a grid of:
    conf     = [0.25, 0.5]
    iou_thr  = [0.0, 0.25, 0.5, 0.75]

Differences compared to the Cosmos version (explicitly declared, see
chat discussion):
  - Input to SigLIP2: ONLY the crop with 15% padding (not the full image
    with a drawn box + text coordinates) — for SigLIP2 the crop is the
    configuration that works (see DRIVE paper), the overlay drastically
    worsens the performance of a contrastive model.
  - "context" for SigLIP2 is NOT a step-by-step instruction (SigLIP doesn't
    "reason"): it's a multi-class comparison between the "empty" caption and
    multiple candidate captions (full + structural distractors), the
    decision is "empty wins over all others", no textual negation
    (contrastive models handle it poorly).
  - SigLIP responds with a similarity score (sigmoid), not with
    free text "yes"/"no" — here "yes"/"no" are derived labels
    (yes = SigLIP confirms "empty", no = SigLIP rejects it).

Counting methodology (identical to the Cosmos version):
  - multi-match: a pred "matches" a GT if IoU >= threshold (or IoU>0 if
    threshold=0), no 1-to-1 constraint.
  - TPR/FPR/FNR normalized on #GT (not on #pred).
  - YES/NO cases crossed with YOLO's real TP/FPs.

Usage:
    python src/evaluation/evaluate_fase2_siglip.py \
        --aug_weights weights/fase1_baseline/aug/best.pt \
        --noaug_weights weights/fase1_baseline/no_aug/best.pt \
        --test_images dataset_finale/test/images \
        --test_labels dataset_finale/test/labels \
        --out_dir output/fase2_evaluation_full \
        --conf 0.25 0.5 \
        --iou_thr 0.0 0.25 0.5 0.75

Outputs in --out_dir:
    metrics_full.json           - all combinations conf x iou x det x prompt
    metrics_full_summary.txt    - same content in readable tabular format

NOTE: this script produces ONLY the raw metrics (JSON + txt). To generate
the tables/plots (identical in style to those of the Cosmos project), run
subsequently:
    python src/evaluation/visualize_fase2_siglip.py \
        --json_path <out_dir>/metrics_full.json \
        --out_dir <out_dir>/vis \
        --prompt context
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor


# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

SIGLIP_CKPT = "google/siglip2-giant-opt-patch16-384"
CROP_PADDING = 0.15  # consistent with fase2_siglip_frozen.py

CONF_THRESHOLDS_DEFAULT = [0.25, 0.5]
IOU_THRESHOLDS_DEFAULT = [0.0, 0.25, 0.5, 0.75]

# Same PROMPT_SETS of fase2_siglip_frozen.py — if you modify them there,
# update them here as well (or import from fase2_siglip_frozen instead of duplicating,
# if you prefer a single source of truth).
PROMPT_SETS = {
    "no_context": {
        "empty": "an empty shelf with no products",
        "full": "a shelf filled with products",
        "distractors": [],
    },
    "context": {
        "empty": "a bare empty supermarket shelf, visible back wall and shadow, no items",
        "full": "a supermarket shelf stocked with visible products",
        "distractors": [
            "a metallic shelf edge or divider, no products visible",
            "a closed freezer or refrigerator glass door",
            "a blurry out of focus photo of a shelf",
            "a dark, poorly lit shelf far from the camera",
        ],
    },
}
PROMPT_NAMES = list(PROMPT_SETS.keys())

IOU_MATCH_THRESHOLD = 0.5  # not used directly (multi_match handles thresholds), kept for compatibility


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
# SigLIP2 — zero-shot multi-prompt-set classifier
# (loads the model ONLY once, precomputes texts for BOTH prompt
#  sets — avoids loading ~1B parameters to GPU twice)
# ══════════════════════════════════════════════════════════════════════

class SigLIP2MultiPromptClassifier:
    """Wrapper class for SigLIP2 handling inference logic."""
    
    def __init__(self, ckpt: str = SIGLIP_CKPT, device: str = None):
        """Initializes the SigLIP2 model and precomputes the tokenized textual inputs."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SigLIP2] Loading {ckpt} on {self.device}...")
        self.model = AutoModel.from_pretrained(ckpt).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)

        # Precompute the tokenized textual inputs for each prompt set
        self.text_inputs_by_set = {}
        for name, cfg in PROMPT_SETS.items():
            texts = [cfg["empty"], cfg["full"]] + list(cfg.get("distractors", []))
            with torch.no_grad():
                self.text_inputs_by_set[name] = {
                    "texts": texts,
                    "inputs": self.processor(
                        text=texts, padding="max_length", max_length=64,
                        return_tensors="pt",
                    ).to(self.device),
                }
            print(f"[SigLIP2] prompt_set '{name}': {len(texts)} captions -> {texts}")

    @torch.no_grad()
    def classify(self, crop: Image.Image, prompt_name: str):
        """
        Returns (is_empty: bool, winning_distractor: str | None).

        Decision (is_empty): ONLY empty vs full (indices 0 and 1), exactly
        as in "no_context" — distractors (if present) NO longer influence
        the decision. Reason: with the argmax on all captions,
        some distractors (e.g. "shelf edge, no products visible")
        share lexicon with "empty" ("no products") and often win
        even on genuinely empty crops, causing a TPR collapse (see chat analysis:
        FNs rose to 84% with the old rule).

        winning_distractor: ONLY diagnostic (not used for the decision) —
        the distractor text with a higher score than both "empty" and
        "full", or None if no distractor beats both. Serves to
        verify empirically if it's always the same distractor
        "winning" (lexical bias/artifact) or if the victory is distributed
        based on the real visual content (genuine selectivity).
        """
        text_data = self.text_inputs_by_set[prompt_name]
        texts = text_data["texts"]
        image_inputs = self.processor(images=[crop], return_tensors="pt").to(self.device)

        outputs = self.model(
            input_ids=text_data["inputs"]["input_ids"],
            attention_mask=text_data["inputs"].get("attention_mask"),
            pixel_values=image_inputs["pixel_values"],
        )
        logits = outputs.logits_per_image  # [1, n_texts]
        probs = torch.sigmoid(logits).squeeze(0)

        score_empty = probs[0].item()
        score_full = probs[1].item()
        is_empty = score_empty > score_full

        winning_distractor = None
        if probs.numel() > 2:
            distractor_scores = probs[2:]
            best_idx = int(torch.argmax(distractor_scores).item())
            best_score = distractor_scores[best_idx].item()
            if best_score > max(score_empty, score_full):
                winning_distractor = texts[2 + best_idx]

        return is_empty, winning_distractor


# ══════════════════════════════════════════════════════════════════════
# Metrics (identical in spirit to evaluate_yolo_cosmos_v2.py,
# "cosmos_*" renamed to "siglip_*")
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(all_results, iou_thr, detector, prompt):
    """Computes evaluating metrics tracking the performance of the YOLO+SigLIP pipeline."""
    total_gt = total_tp = total_fp = total_fn = 0
    yolo_total = siglip_yes_total = siglip_no_total = 0
    siglip_yes_were_tp = siglip_yes_were_fp = 0
    siglip_no_were_tp = siglip_no_were_fp = 0
    fn_missed_by_yolo = 0  # "Irreducible" FNs: GTs that NO YOLO box (even discarded ones) covers

    from collections import Counter
    distractor_wins_by_tp = Counter()  # label -> count, ONLY on boxes that were true TPs
    distractor_wins_by_fp = Counter()  # label -> count, ONLY on boxes that were true FPs
    n_real_tp_total = n_real_fp_total = 0

    for r in all_results:
        gt_boxes = r["gt"]
        pred_boxes = r["preds"][detector]
        responses = r["responses"][detector][prompt]  # list of bools (True=yes/empty)
        winning_distractors = r.get("winning_distractors", {}).get(detector, {}).get(prompt, [None] * len(pred_boxes))

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
    # both terms are "per GT" counts, so the subtraction is valid
    # (unlike per-box counts like siglip_yes/no_were_*, where
    # multi-match can generate double counts if multiple boxes match the same GT).
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
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--test_labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--conf", nargs="+", type=float, default=CONF_THRESHOLDS_DEFAULT)
    parser.add_argument("--iou_thr", nargs="+", type=float, default=IOU_THRESHOLDS_DEFAULT)
    parser.add_argument("--padding", type=float, default=CROP_PADDING)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Evaluation pipeline YOLO + SigLIP2 (Phase 2) — complete grid")
    print("=" * 70)

    print(f"[1/3] Loading aug detector   <- {args.aug_weights}")
    yolo_aug = YOLO(args.aug_weights)
    print(f"[2/3] Loading noaug detector <- {args.noaug_weights}")
    yolo_noaug = YOLO(args.noaug_weights)
    detectors = {"aug": yolo_aug, "noaug": yolo_noaug}

    print(f"[3/3] Loading SigLIP2 (both prompt sets precomputed)")
    siglip = SigLIP2MultiPromptClassifier(device=args.device)

    pairs = load_test_pairs(Path(args.test_images), Path(args.test_labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"\n  Test images : {len(pairs)}")
    print(f"  CONF        : {args.conf}")
    print(f"  IOU         : {args.iou_thr}")
    print(f"  PROMPT SETS : {PROMPT_NAMES}")

    # ── YOLO inference (once per conf x detector) ───────────────────
    print("\nPhase 1/2: YOLO inference...")
    all_results = {}
    for conf in args.conf:
        results_conf = []
        for sample in pairs:
            entry = {"path": sample["path"], "gt": sample["gt"], "image": sample["image"],
                      "preds": {}, "responses": {}, "winning_distractors": {}}
            for det_name, yolo in detectors.items():
                res = yolo(sample["path"], conf=conf, verbose=False)
                boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if res and res[0].boxes is not None else []
                entry["preds"][det_name] = boxes
                entry["responses"][det_name] = {}
                entry["winning_distractors"][det_name] = {}
            results_conf.append(entry)
        all_results[conf] = results_conf
        print(f"  conf={conf}: YOLO completed on {len(results_conf)} images")

    # ── SigLIP2 inference (for each conf x detector x prompt) ────────────
    print("\nPhase 2/2: SigLIP2 inference (crop with padding)...")
    PROGRESS_EVERY = 50
    from collections import Counter
    distractor_stats = {}  # (conf, det, prompt) -> (Counter per label, total_boxes)
    for conf in args.conf:
        for det_name in detectors:
            for prompt_name in PROMPT_NAMES:
                print(f"  det={det_name}  conf={conf}  prompt='{prompt_name}'")
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
                        is_empty, winning_distractor = siglip.classify(crop, prompt_name)
                        resps.append(is_empty)
                        wdist.append(winning_distractor)
                        if is_empty:
                            yes_count += 1
                        else:
                            no_count += 1
                        if winning_distractor is not None:
                            distractor_win_counter[winning_distractor] += 1
                    entry["responses"][det_name][prompt_name] = resps
                    entry["winning_distractors"][det_name][prompt_name] = wdist
                    boxes_done += len(boxes)
                    if img_idx % PROGRESS_EVERY == 0 or img_idx == n_images:
                        print(f"    [{img_idx}/{n_images}] total_box={boxes_done} "
                              f"YES={yes_count} NO={no_count}", flush=True)
                distractor_stats[(conf, det_name, prompt_name)] = (distractor_win_counter, boxes_done)
                total_distractor_wins = sum(distractor_win_counter.values())
                if boxes_done > 0 and prompt_name != "no_context":
                    pct = 100 * total_distractor_wins / boxes_done
                    print(f"    [diagnostic, not used for decision] "
                          f"a distractor would have won on {total_distractor_wins}/{boxes_done} "
                          f"boxes ({pct:.1f}%). Distribution by label:")
                    for label, count in distractor_win_counter.most_common():
                        label_pct = 100 * count / total_distractor_wins if total_distractor_wins else 0
                        print(f"        {count:5d} ({label_pct:5.1f}% of distractor wins) -> \"{label}\"")

    # ── Metrics on the full grid ──────────────────────────────────────
    print("\nCalculating metrics...")
    output = {}
    for conf in args.conf:
        for iou_t in args.iou_thr:
            for det_name in detectors:
                for prompt_name in PROMPT_NAMES:
                    key = f"conf={conf}_iou={iou_t}_{det_name}_{prompt_name}"
                    m = compute_metrics(all_results[conf], iou_t, det_name, prompt_name)
                    m.update({"conf": conf, "iou": iou_t, "detector": det_name, "prompt": prompt_name})
                    output[key] = m
                    print(f"  {det_name:6s} {prompt_name:12s} conf={conf} iou={iou_t} | "
                          f"TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} FNR={m['FNR']:.3f}")

    with open(out_dir / "metrics_full.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Output] {out_dir / 'metrics_full.json'}")

    # Distractors diagnostics (not used by the pipeline, only to understand how much
    # and WHICH distractor would have interfered if considered in the
    # decision — useful to distinguish systematic lexical bias (always the
    # same label winning) from genuine selectivity distributed on the
    # visual content).
    distractor_diag = {
        f"conf={c}_{det}_{p}": {
            "n_boxes": n,
            "total_distractor_wins": sum(counter.values()),
            "wins_by_label": dict(counter.most_common()),
        }
        for (c, det, p), (counter, n) in distractor_stats.items()
    }
    with open(out_dir / "distractor_diagnostics.json", "w") as f:
        json.dump(distractor_diag, f, indent=2)
    print(f"[Output] {out_dir / 'distractor_diagnostics.json'} (diagnostic only)")

    with open(out_dir / "metrics_full_summary.txt", "w") as f:
        f.write(f"{'detector':8s} | {'prompt':12s} | {'conf':4s} | {'iou':4s} | "
                f"{'TPR':5s} | {'FPR':5s} | {'FNR':5s} | {'YOLO_tot':8s} | "
                f"{'YES':5s} | {'NO':5s}\n")
        f.write("-" * 100 + "\n")
        for key, m in output.items():
            f.write(f"{m['detector']:8s} | {m['prompt']:12s} | {m['conf']:.2f} | {m['iou']:.2f} | "
                    f"{m['TPR']:.3f} | {m['FPR']:.3f} | {m['FNR']:.3f} | {m['yolo_total']:8d} | "
                    f"{m['siglip_yes_total']:5d} | {m['siglip_no_total']:5d}\n")
    print(f"[Output] {out_dir / 'metrics_full_summary.txt'}")

    print("\nCompleted. Raw metrics saved — to generate tables and plots, run:")
    print(f"  python src/evaluation/visualize_fase2_siglip.py --json_path {out_dir / 'metrics_full.json'} --out_dir {out_dir}/vis")


if __name__ == "__main__":
    main()