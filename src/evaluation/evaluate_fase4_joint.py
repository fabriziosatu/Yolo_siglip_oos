"""
src/evaluation/evaluate_fase4_joint.py
=========================================
End-to-end evaluation of the JOINT-TRAINED Phase 4 pipeline — same
scheme (conf x IoU grid, TPR/FPR/FNR, JSON+summary output) as
evaluate_fase3_siglip.py, but adapted to the inference of the ENTIRE
pipeline instead of an independent SigLIP classifier on pixel crops.

KEY DIFFERENCES compared to evaluate_fase3_siglip.py (because it is not a
simple 1:1 porting):

  1. In Phase 3, YOLO (frozen) and SigLIP (one among many checkpoints) were
     INDEPENDENT and interchangeable components — the grid crossed
     "2 detectors" x "N SigLIP classifiers". In Phase 4 this makes no sense:
     each joint-trained checkpoint contains ONE pair (YOLO, SigLIP) that
     were trained TOGETHER and are not separable/recombinable with
     another checkpoint. Here the grid is therefore: N joint checkpoints x
     conf x IoU — a single configuration "axis", not two crossed axes.

  2. In Phase 3, crops were extracted with crop_with_padding() on the original
     PIXELS of the image. In Phase 4 this is NOT correct: the joint-trained
     SigLIP weights adapted specifically to the crops that come out of the
     RoI Align on the YOLO feature map + the learnable feature_proj projection
     — not to pure pixel crops. Evaluating on pixel crops would evaluate weights
     outside their training domain. Thus, here we use the real
     JointPipeline.forward() in eval mode, exactly the same path used in training.

  3. Images are loaded with the same 640x640 letterbox used in training
     (via build_dataloaders/ShelfDataset), not with the original resolution +
     internal Ultralytics resize like in Phase 3 — this way predictions
     and GTs are in the same [0,640] pixel coordinate system without
     additional conversions.

  4. compute_metrics()/multi_match()/iou() are UNCHANGED (pure
     geometry/counting, independent of architecture) — same
     TP/FP/FN criteria, same normalization over #GT.

Usage:
    python src/evaluation/evaluate_fase4_joint.py \
        --checkpoints weights/fase4_joint/aug__completa__neg025/best_model.pt \
                      weights/fase4_joint/aug__vision_mlp__neg05/best_model.pt \
        --labels aug_completa_neg025 aug_vision_mlp_neg05 \
        --data_dir dataset_finale \
        --out_dir results/fase4_eval

Note on LoRA rank: the initial construction of the pipeline (before
loading the state_dict of the checkpoint) uses rank=16/alpha=32 by
default (consistent with the only rank you kept, see
--lora_rank/--lora_alpha in your Phase 3 .sh files). If a checkpoint was
trained with a different rank, overwrite with --lora_rank/--lora_alpha.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

import sys
import os

# Project root calculated from the FILE position, not from the launch
# folder — sys.path.insert(0, ".") only worked if the script was launched
# exactly after a "cd $HPC_ROOT"; if launched from another folder
# it failed with "ModuleNotFoundError: No module named 'src'".
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.utils.config import Config
from src.models.joint_pipeline import JointPipeline
from src.data.dataset import build_dataloaders


CONF_THRESHOLDS_DEFAULT = [0.25, 0.5]
IOU_THRESHOLDS_DEFAULT = [0.0, 0.25, 0.5, 0.75]


# ══════════════════════════════════════════════════════════════════════
# Geometry — UNCHANGED from evaluate_fase3_siglip.py
# ══════════════════════════════════════════════════════════════════════

def iou(box_a, box_b):
    """Computes the Intersection over Union (IoU) between two bounding boxes."""
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
    """Multi-matching, same convention as Phase 3:
    is_tp_pred[i] = True if pred i matches at least one GT
    is_fn_gt[j]   = True if GT j is not matched by any pred"""
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


# ══════════════════════════════════════════════════════════════════════
# Loading a joint-trained checkpoint
# ══════════════════════════════════════════════════════════════════════

def _robust_torch_load(ckpt_path: str, device: torch.device) -> dict:
    """
    Loads a checkpoint with automatic fallback if the file is corrupted
    (e.g., SLURM job killed mid-write, before the fix of atomic saves
    in trainer.py). If ckpt_path (typically best_model.pt) does not open,
    searches among the checkpoint_epochNNN.pt in the SAME folder and uses
    the readable one with the lowest best_val_loss — a reasonable
    approximation of the true best, not the true best (which is lost).
    """
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"  ⚠ '{ckpt_path}' is corrupted ({e}) — searching for a substitute among "
              f"numbered checkpoints in the same folder...")

    ckpt_dir = Path(ckpt_path).parent
    candidates = sorted(ckpt_dir.glob("checkpoint_epoch*.pt"))
    if not candidates:
        raise RuntimeError(
            f"'{ckpt_path}' is corrupted and there are no reserve "
            f"checkpoint_epochNNN.pt files in '{ckpt_dir}'. This checkpoint must be regenerated "
            f"(relaunch the training of this run)."
        )

    best_ck, best_path, best_loss = None, None, float("inf")
    for c in candidates:
        try:
            ck = torch.load(c, map_location=device, weights_only=False)
        except Exception:
            print(f"    ⚠ '{c.name}' is also corrupted, skipping")
            continue
        loss = ck.get("best_val_loss", float("inf"))
        if loss < best_loss:
            best_ck, best_path, best_loss = ck, c, loss

    if best_ck is None:
        raise RuntimeError(
            f"No readable checkpoint found in '{ckpt_dir}' "
            f"(neither '{Path(ckpt_path).name}' nor the checkpoint_epochNNN.pt). "
            f"This run must be relaunched from scratch."
        )

    print(f"  ✓ Using '{best_path.name}' as substitute "
          f"(recorded best_val_loss: {best_loss:.4f} — it's not guaranteed to be the "
          f"true best, it's the best among surviving checkpoints)")
    return best_ck


def load_joint_pipeline_from_checkpoint(
    ckpt_path: str, device: torch.device,
    yolo_arch_ref: str,
    lora_rank: int = 16, lora_alpha: int = 32,
    lora_rank_text: int = None, lora_alpha_text: int = None,
) -> JointPipeline:
    """
    Builds an 'empty' JointPipeline and loads the complete state_dict
    saved by Trainer._save_checkpoint(). The SigLIP variant
    (completa/vision_mlp) is read automatically from the checkpoint.

    Args:
        yolo_arch_ref: path to ANY real 1-class YOLO checkpoint
            (e.g., weights/fase1_baseline/aug/best.pt) — serves ONLY to
            build the correct nc=1 detector architecture
            (a generic .yaml like "yolo26n.yaml" builds an 80-class COCO
            architecture by default — mismatch guaranteed). The
            weights of this file are nonetheless overwritten immediately
            afterwards by the state_dict of the joint checkpoint, so any
            valid single-class .pt is fine, aug or no_aug.
        lora_rank_text/lora_alpha_text: if None, default to HALVED visual
            rank/alpha (16->8, 32->16) — convention observed in the real
            checkpoint of train_fase3_full_lora.py (the text encoder
            uses half the rank of the visual). Verify against the real
            script anyway if you doubt the exact value of alpha: wrong
            scaling doesn't give a shape error (alpha is not in the
            state_dict) but produces numerically incorrect results silently.
    """
    print(f"\n[Checkpoint] Loading {ckpt_path}...")
    ck = _robust_torch_load(ckpt_path, device)

    variant = ck.get("siglip_variant")
    if variant is None:
        raise ValueError(
            f"The checkpoint '{ckpt_path}' does not contain 'siglip_variant' — "
            f"probably saved by a previous version of trainer.py. "
            f"Specify the variant manually or regenerate the checkpoint."
        )
    print(f"  siglip_variant read from checkpoint: '{variant}'")

    if lora_rank_text is None:
        lora_rank_text = lora_rank // 2   # confirmed from real checkpoints: 16->8, 8->4
    if lora_alpha_text is None:
        # Confirmed from real checkpoints (adapter_config.json): alpha_text = 2 x rank_text
        # (r16->alpha32 visual, r8->alpha16 text; r8->alpha16 visual, r4->alpha8 text —
        # same "alpha = 2*rank" pattern for BOTH encoders, not a fixed value
        # equal to visual as assumed in a previous version of this default).
        lora_alpha_text = lora_rank_text * 2

    cfg = Config()
    cfg.siglip.variant = variant
    cfg.siglip.lora_r_visual = lora_rank
    cfg.siglip.lora_alpha_visual = lora_alpha
    cfg.siglip.lora_r_text = lora_rank_text
    cfg.siglip.lora_alpha_text = lora_alpha_text
    cfg.siglip.skip_pretrained_load = True   # useless: will be overwritten immediately
    cfg.detector.model_name = yolo_arch_ref  # correct nc=1 architecture, weights overwritten later

    pipeline = JointPipeline(cfg=cfg).to(device)

    missing, unexpected = pipeline.load_state_dict(ck["pipeline_state_dict"], strict=False)
    if missing or unexpected:
        print(f"  ⚠ WARNING while loading the state_dict:")
        print(f"    missing_keys:    {missing}")
        print(f"    unexpected_keys: {unexpected}")
        print(f"    Probable cause: --lora_rank/--lora_alpha (or the _text versions) "
              f"do not match the real checkpoint, or incorrect siglip_variant.")

    pipeline.eval()

    # Text embeddings (complete variant) are NOT in the state_dict (they are
    # not nn.Parameters, they are simple python attributes) — they must be explicitly
    # recomputed with the newly loaded text encoder. No-op for vision_mlp.
    pipeline.encode_prompts(
        positive_prompts=cfg.siglip.positive_prompts,
        negative_prompts=cfg.siglip.negative_prompts,
    )

    print(f"  ✓ Pipeline loaded (epoch {ck.get('epoch', '?')}, "
          f"best_val_loss={ck.get('best_val_loss', '?')})")
    return pipeline, variant


def siglip_decisions(variant: str, siglip_out: dict) -> list:
    """True = SigLIP confirms 'empty' for that ROI, False = rejected.
    Same semantics used during training (see losses.py)."""
    if variant == "vision_mlp":
        probs = torch.sigmoid(siglip_out["logit"]).squeeze(-1)
        return (probs > 0.5).tolist()
    elif variant == "completa":
        decisions = siglip_out["logits_pos"] > siglip_out["logits_neg"]
        return decisions.squeeze(-1).tolist()
    else:
        raise ValueError(f"Unknown variant: '{variant}'")


# ══════════════════════════════════════════════════════════════════════
# Metrics — UNCHANGED in spirit from evaluate_fase3_siglip.py
# (only the "detector" crossed with "arch" dimension removed: here we have
# A SINGLE detector+classifier per checkpoint, already chosen upstream)
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(all_results, iou_thr):
    """Computes basic performance metrics and identifies correct/incorrect SigLIP decisions."""
    total_gt = total_tp = total_fp = total_fn = 0
    yolo_total = siglip_yes_total = siglip_no_total = 0
    siglip_yes_were_tp = siglip_yes_were_fp = 0
    siglip_no_were_tp = siglip_no_were_fp = 0
    fn_missed_by_yolo = 0

    for r in all_results:
        gt_boxes = r["gt"]
        pred_boxes = r["preds"]
        responses = r["responses"]

        n_gt = len(gt_boxes)
        total_gt += n_gt
        yolo_total += len(pred_boxes)

        _, is_fn_gt_baseline = multi_match(pred_boxes, gt_boxes, iou_thr)
        fn_missed_by_yolo += sum(is_fn_gt_baseline)

        is_tp_real, _ = multi_match(pred_boxes, gt_boxes, iou_thr)

        for resp, is_real_tp in zip(responses, is_tp_real):
            if resp:
                siglip_yes_total += 1
                if is_real_tp:
                    siglip_yes_were_tp += 1
                else:
                    siglip_yes_were_fp += 1
            else:
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
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    """Main execution function to load checkpoints, run test batches, and log results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                         help="Path to one or more best_model.pt of Phase 4")
    parser.add_argument("--labels", nargs="+", required=True,
                         help="Label for each checkpoint, same order/length")
    parser.add_argument("--data_dir", required=True,
                         help="Root dataset folder (e.g. dataset_finale) — must contain "
                              "test/images and test/labels, same structure used in training "
                              "by build_dataloaders/ShelfDataset")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--conf", nargs="+", type=float, default=CONF_THRESHOLDS_DEFAULT)
    parser.add_argument("--iou_thr", nargs="+", type=float, default=IOU_THRESHOLDS_DEFAULT)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--yolo_arch_ref", required=True,
                         help="Path to ANY real 1-class YOLO checkpoint (e.g. "
                              "weights/fase1_baseline/aug/best.pt) — serves only to build "
                              "the correct architecture (nc=1) before overwriting weights. "
                              "A generic .yaml would build an 80-class COCO architecture "
                              "and the state_dict loading would fail.")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_rank_text", type=int, default=None,
                         help="Default: --lora_rank // 2 (convention observed in real "
                              "checkpoints of train_fase3_full_lora.py — the text encoder uses half "
                              "the rank of the visual)")
    parser.add_argument("--lora_alpha_text", type=int, default=None,
                         help="Default: 2 * lora_rank_text — confirmed reading "
                              "adapter_config.json on real checkpoints (r16/alpha32 and "
                              "r8/alpha16 for the text encoder). If wrong: no shape "
                              "error, but weights are numerically scaled incorrectly in a "
                              "silent way — always verify with the command "
                              "'cat lora_adapters_text/adapter_config.json' before trusting "
                              "a default.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if len(args.checkpoints) != len(args.labels):
        raise ValueError(f"--checkpoints ({len(args.checkpoints)}) and --labels "
                          f"({len(args.labels)}) must have the same length")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Evaluation JOINT pipeline (Phase 4) — full grid")
    print("=" * 70)

    # ── Test dataset — same 640 letterbox used in training ─────────────
    print(f"\n[Dataset] Loading test set from {args.data_dir}...")
    _, _, test_loader = build_dataloaders(
        img_size=args.img_size, batch_size=1,
        mode="phase2", data_dir=args.data_dir,
    )
    print(f"  {len(test_loader)} test images")

    # ── Preload all GTs only once (reused for each checkpoint) ───
    from src.models.joint_pipeline import JointPipeline as JP
    test_samples = []
    for batch in test_loader:
        images = batch["images"]
        gt_norm = batch["boxes"][0]
        gt_xyxy = JP._yolo_to_xyxy_pixel(gt_norm, args.img_size).tolist()
        test_samples.append({"images": images, "gt": gt_xyxy})

    output = {}

    for label, ckpt_path in zip(args.labels, args.checkpoints):
        pipeline, variant = load_joint_pipeline_from_checkpoint(
            ckpt_path, device, yolo_arch_ref=args.yolo_arch_ref,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_rank_text=args.lora_rank_text, lora_alpha_text=args.lora_alpha_text,
        )

        # ── Inference for each confidence threshold (cache reused for IoU grid) ──
        all_results_by_conf = {}
        for conf in args.conf:
            pipeline.detector.conf_threshold = conf
            results_conf = []
            print(f"\n  [{label}] conf={conf}: inference on {len(test_samples)} images...")
            with torch.no_grad():
                for i, sample in enumerate(test_samples, start=1):
                    images = sample["images"].to(device)
                    out = pipeline(images, gt_boxes=None)

                    rois = out["rois"]
                    if rois.shape[0] == 0:
                        pred_boxes, responses = [], []
                    else:
                        pred_boxes = rois[:, 1:5].tolist()
                        responses = siglip_decisions(variant, out["siglip_out"])

                    results_conf.append({
                        "gt": sample["gt"], "preds": pred_boxes, "responses": responses,
                    })
                    if i % 50 == 0 or i == len(test_samples):
                        n_yes = sum(responses)
                        print(f"    [{i}/{len(test_samples)}] total ROIs so far: "
                              f"{sum(len(r['preds']) for r in results_conf)}")
            all_results_by_conf[conf] = results_conf

        # ── Metrics on conf x IoU grid (no new inference) ──────────
        for conf in args.conf:
            for iou_t in args.iou_thr:
                key = f"conf={conf}_iou={iou_t}_{label}"
                m = compute_metrics(all_results_by_conf[conf], iou_t)

                # Fields for compatibility with compute_fscore.py (Phase 2/3 schema):
                #   n_real_tp_total/n_real_fp_total = TP/FP that YOLO produces BEFORE
                #   the SigLIP filter — the sum of correct/incorrect predictions
                #   regardless of what SigLIP decides (yes or no). fn_missed_by_yolo
                #   is already calculated by compute_metrics() with the same meaning
                #   as Phase 2/3 (FN that would exist even without SigLIP).
                #   detector/arch: derived from the checkpoint label, to make
                #   "Phase 4" appear in the same table with Detector/Config columns
                #   consistent with Phase 2/3 rows.
                n_real_tp_total = m.get("siglip_yes_were_tp", 0) + m.get("siglip_no_were_tp", 0)
                n_real_fp_total = m.get("siglip_yes_were_fp", 0) + m.get("siglip_no_were_fp", 0)
                detector_label = "No Augmentation" if label.startswith("noaug") or label.startswith("no_aug") \
                                 else "Augmentation"
                variant_label = "Full" if variant == "completa" else "mlp"
                if "neg025" in label:
                    neg_ratio_str = "0.25"
                elif "neg05" in label:
                    neg_ratio_str = "0.5"
                else:
                    neg_ratio_str = "?"
                arch_label = f"{variant_label}, rank 16, neg_ratio {neg_ratio_str}, " \
                             f"padding 50, JOINT (Phase 4)"

                m.update({
                    "conf": conf, "iou": iou_t, "config": label, "variant": variant,
                    "n_real_tp_total": n_real_tp_total,
                    "n_real_fp_total": n_real_fp_total,
                    "detector": detector_label,
                    "arch": arch_label,
                })
                output[key] = m
                print(f"  {label:30s} conf={conf} iou={iou_t} | "
                      f"TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} FNR={m['FNR']:.3f}")

        pipeline.remove_hooks()
        del pipeline
        torch.cuda.empty_cache()

    with open(out_dir / "metrics_full.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Output] {out_dir / 'metrics_full.json'}")

    with open(out_dir / "metrics_full_summary.txt", "w") as f:
        f.write(f"{'config':30s} | {'conf':4s} | {'iou':4s} | "
                f"{'TPR':5s} | {'FPR':5s} | {'FNR':5s} | {'YOLO_tot':8s} | "
                f"{'YES':5s} | {'NO':5s}\n")
        f.write("-" * 100 + "\n")
        for key, m in output.items():
            f.write(f"{m['config']:30s} | {m['conf']:.2f} | {m['iou']:.2f} | "
                    f"{m['TPR']:.3f} | {m['FPR']:.3f} | {m['FNR']:.3f} | {m['yolo_total']:8d} | "
                    f"{m['siglip_yes_total']:5d} | {m['siglip_no_total']:5d}\n")
    print(f"[Output] {out_dir / 'metrics_full_summary.txt'}")

    print("\nCompleted.")


if __name__ == "__main__":
    main()