# Improving Out-Of-Stock Detection in Retail Environments via Vision Language Models (YOLO26 + SigLIP2)

This repository contains the final solution of the thesis project: a **domain-adapted, fine-tuned Vision-Language pipeline** for Out-of-Stock (OOS) retail shelf detection, combining a fast geometric detector (YOLO26) with a fine-tuned multimodal classifier (SigLIP2) to semantically validate candidate empty-shelf regions.

## 🎯 Project Core Objective

The companion evaluation tracks in this research (zero-shot Cosmos-Reason2-8B and Qwen2.5-VL) demonstrated that frozen, prompt-engineered Vision-Language Models plateau well below the YOLO-only baseline's recall and become actively unstable when explicit reasoning is enabled, never closing the gap on a highly domain-specific task like empty-shelf detection.

This repository addresses that gap directly: instead of relying on a frozen VLM, **SigLIP2 is adapted to the retail OOS domain through Parameter-Efficient Fine-Tuning (LoRA)** and combined with the YOLO26 detector in two complementary configurations — a sequential fine-tuned pipeline and a fully joint, end-to-end trained pipeline. The objective is to quantify how much domain adaptation recovers versus a frozen VLM, and what it costs in hardware/training complexity when the two modules are optimized jointly rather than sequentially.

## 🤖 Target Architecture

All pipelines in this repository are built around two components:

* **YOLO26** — single-stage geometric detector, used to extract candidate empty-shelf Region of Interest (ROI) boxes, prioritizing high Recall over precision.
* **SigLIP2 (Google)** — multimodal visual-language encoder, adapted to the retail domain via **LoRA (Low-Rank Adaptation)** rather than full fine-tuning, to avoid the Out-Of-Memory errors observed under full fine-tuning on the available hardware.

Two architectural variants of the semantic validation head are evaluated:

* **Full SigLIP** — retains both the vision and text encoders, validating ROI crops by aligning visual embeddings with positive/negative textual prompts. Maximizes semantic flexibility.
* **SigLIP + MLP Head** — replaces the text encoder with a lightweight Multi-Layer Perceptron classifier, trading semantic flexibility for a faster, binary-classifier architecture optimized solely for the OOS task.

**Negative Mining** is used during training in both variants: complex visual distractors (shadows, dark shelf grids, price tags) are deliberately injected to force the model to learn strict semantic boundaries and reduce False Positives.

## 📊 Dataset & Privacy Restrictions

> ⚠️ **Important Notice regarding Data Availability:** The dataset consists of **11,519 images** in total. Of these, 5,932 come from public and private datasets (SKU-110K, Grocery Products, WebMarket, Roboflow, and private MIVIA acquisitions) under ideal lighting and framing conditions. The remaining **5,587 frames are real-world, private surveillance/robot acquisitions** captured by the **Pepper mobile robot** as part of the research conducted at the **MIVIA Lab (University of Salerno)**, affected by motion blur, occlusions and realistic noise. **Because this dataset is the exclusive property of the MIVIA Lab and is subject to strict corporate non-disclosure agreements (NDAs) and privacy regulations, the raw images and ground-truth bounding box annotations for the private portion cannot be uploaded or distributed within this repository.**

The complete training configurations, evaluation metrics, rendered comparison tables and demo visualizations are maintained inside the project's `results/` directory footprint.

## 📑 Experimental Framework & Test Specifications

The codebase evaluates the YOLO26 + SigLIP2 synergy through **three progressive phases**, each analyzing a different level of interaction and training between the geometric and semantic modules.

### Phase 1: Zero-Shot Baseline
* **Objective:** Establish a baseline using the frozen YOLO26 detector paired with the pre-trained SigLIP2 VLM, with no domain-specific adaptation on either module.
* **Input Structure:** Candidate ROI boxes from YOLO26, validated by SigLIP2 using generic positive/negative text prompts.
* **Main Result:** Confirms the same structural weakness observed in the companion zero-shot VLM tracks (Cosmos-Reason2, Qwen2.5-VL) — a non-adapted semantic filter does not reliably distinguish genuinely empty shelf gaps from hard distractors (shadows, price tags, dark shelf grids).

### Phase 2: Sequential Fine-Tuning (Best Configuration)
* **Script:** `fase2_siglip_frozen.py` (training), `evaluate_fase2_siglip.py` (evaluation), `analyze_errors.py` (error decomposition).
* **Objective:** Recover the sensitivity lost in the zero-shot configuration by domain-adapting SigLIP2 via LoRA, while keeping YOLO26 frozen — a sequential, modular pipeline.
* **Training Techniques:** LoRA (Parameter-Efficient Fine-Tuning) + Negative Mining with injected hard distractors.
* **Main Result:** **Best configuration found across the whole project.** Peak F1-Score of **0.9374** (with Data Augmentation), preserving the robustness of the base detector while substantially filtering out false positives.
* **Additional output:** per-model prediction comparison visualizations (`visualize_predictions_comparison.py`, `visualize_fase2_siglip.py`).

### Phase 3: Joint Training (End-to-End)
* **Script:** `scripts/train.py`, built on `src/training/trainer.py`, `src/models/joint_pipeline.py`.
* **Objective:** Jointly train both the geometric (YOLO26) and semantic (SigLIP2) modules end-to-end in a single unified optimization loop, to assess the feasibility and hardware constraints of a fully joint approach versus the sequential pipeline of Phase 2.
* **Input Structure:** Shared forward pass — YOLO26 feature maps are cropped via RoI Align and fed directly into the SigLIP2 classifier within the same computational graph (`src/models/joint_pipeline.py`).
* **Main Result:** F1-Score of **0.8588** (with Data Augmentation) — measurably lower than Phase 2, **heavily penalized by hardware bottlenecks during joint backpropagation** (batch size and gradient accumulation constraints) rather than by a fundamental limitation of the joint formulation itself.

## 📈 Cross-Phase Summary

| Configuration | Detector Only | Fine-Tuning (Phase 2) | Joint (Phase 3) |
|---|---|---|---|
| **Augmentation** | 0.9368 | 0.9354 – **0.9374** | 0.8588 |
| **No Augmentation** | 0.8362 | 0.8361 – 0.8369 | 0.7541 |

*(All figures: F1-Score, detector aug/no_aug variants)*

Across all configurations, Data Augmentation provides an average metric improvement of **~10%**. The modular, sequentially fine-tuned pipeline (Phase 2) is the optimal approach overall — it preserves the baseline detector's robustness while reaching the highest F1-Score of the project. The fully joint approach (Phase 3) remains competitive and architecturally interesting, but its current hardware-constrained training setup prevents it from matching Phase 2, mirroring — at a much smaller performance gap — the same plateau-then-cost pattern observed when pushing the companion frozen-VLM tracks beyond their stable operating point.

## 📁 Repository Directory Structure

```text
Progetto_tesi_SigLIP/
├── src/
│   ├── utils/
│   │   └── config.py                      # Centralized hyperparameters
│   ├── data/
│   │   └── dataset.py                     # ShelfDataset (PyTorch Dataset, YOLO-format labels)
│   ├── models/
│   │   ├── detector.py                    # YOLO26 wrapper (final grad_fn-safe version)
│   │   ├── siglip_module.py               # SigLIP2 + LoRA (visual + text encoders)
│   │   └── joint_pipeline.py              # Unified YOLO26 + SigLIP2 forward pass (RoI Align)
│   ├── training/
│   │   ├── losses.py                      # SigLIPSigmoidLoss, YOLOLossWrapper, JointLoss
│   │   └── trainer.py                     # Phase 3 joint training loop
│   └── evaluation/
│       ├── fase2_siglip_frozen.py         # Phase 2 — sequential LoRA fine-tuning
│       ├── evaluate_fase2_siglip.py       # Phase 2 evaluation routine
│       ├── analyze_errors.py              # False positive/negative error decomposition
│       ├── visualize_fase2_siglip.py      # Phase 2 plot/table rendering
│       └── visualize_predictions_comparison.py  # Cross-phase prediction comparison
└── scripts/
    ├── train.py                           # Phase 3 entry point (python scripts/train.py [--resume ...])
    ├── render_pipeline_videos.py          # Per-phase demo video rendering
    ├── render_pipeline_video_combined.py  # Combined/segmented multi-phase demo video
    ├── inspect_yolo26.py                  # YOLO26 internal structure inspection (debug)
    └── debug_yolo_loss2.py                # model.loss() expected output format (debug)
```

> **Note:** the filenames under `src/evaluation/` and `scripts/render_*` reflect the working names used during development of Phases 1–2 and the demo material; please double-check them against your actual working tree before committing this README, since this project went through several iterations (v1/v2, aug/no_aug) and at least one naming conflict (`detector.py` vs `detector_fixed.py`) was still unresolved as of the last check. The `src/models/`, `src/training/`, `src/data/`, `src/utils/` entries for Phase 3 and `scripts/train.py`, `scripts/inspect_yolo26.py`, `scripts/debug_yolo_loss2.py` are confirmed against the current codebase.
