"""
src/utils/config.py
====================
Joint pipeline configuration YOLO26 + SigLIPv2.

CHANGES compared to your version (the one with MLP only):
  - siglip.variant: NEW — "vision_mlp" (the one you had, unchanged)
    or "completa" (vision+text with LoRA, like train_fase3_full_lora.py,
    restored as a selectable option — see discussion on
    why you want to be able to test both).
  - siglip.lora_r_text / lora_alpha_text: NEW — used only if
    variant="completa" (for the text encoder).
  - siglip.positive_prompts / negative_prompts: NEW — used only if
    variant="completa".
  - negative_mining.neg_ratio: NEW — replaces the fixed integer
    n_neg_synthetic you had in JointPipeline. Same meaning as
    neg_ratio in Phase 3 (proportional to the number of GTs in the image),
    so you can test 0.25/0.5 here too with direct comparison.
  - training.alpha/beta: UNCHANGED as default values, but now must be
    recalculated for each SigLIP variant with
    scripts/calibrate_loss_weights.py (the scale of the single-logit
    MLP loss and the dual-logit pos/neg one are different — a good alpha/beta
    for one variant might not be good for the other).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""
    data_dir:     Path = Path("dataset_finale")
    dataset_yaml: Path = Path("dataset_finale/data.yaml")
    img_size:     int  = 640
    roi_size:     int  = 384   # must match siglip.model_name — 384 for "giant-opt-patch16-384"

    # Padding around the box BEFORE RoI Align — same meaning as
    # --padding in train_fase3_lora.py/train_fase3_full_lora.py (extra
    # context around the box, not just the exact box). The Phase 3 checkpoint
    # you load as pretrained weights was trained with padding=0.50
    # ("pad50") — if it stays different here, SigLIP sees different crops from those
    # on which it specialized, despite the correct starting weights.
    roi_padding:  float = 0.50
    num_workers:  int  = 2
    batch_size:   int  = 4


@dataclass
class DetectorConfig:
    """Configuration for the YOLO26 detector component."""
    model_name:     str   = "weights/cluster_training/phase1_yolo_final/weights/best.pt"
    conf_threshold: float = 0.25
    iou_threshold:  float = 0.45

    # Confidence threshold used ONLY in train mode (to build candidate
    # ROIs on which to perform IoU matching — see JointPipeline._assign_roi_labels).
    # Historical default 0.1, designed for YOLO not yet calibrated (init from
    # scratch). With Phase 1 weights already well trained (option A, the case you
    # use) a higher value reduces "noise" ROIs passed to SigLIP without
    # losing correct predictions — less memory, lighter training.
    train_conf_threshold: float = 0.2

    # Maximum limit of ROIs (real predictions, not synthetic) passed to
    # SigLIP per image — keeps only the N most confident ones,
    # discards the rest. Necessary because the number of ROIs per image is
    # highly variable (5-30+): without a cap, peak memory and
    # CUDA fragmentation grow unpredictably over time (OOM
    # after hundreds of "good" iterations, not immediately — typical symptom).
    # Disabled by default (0 = no-op): experimentation showed
    # that few images exceed 20 ROIs, so the cap cut
    # rarely without changing the number of completed epochs per job —
    # does not justify the training signal lost on those few images
    # where it triggered. Can be re-enabled via --max_rois_per_image if needed
    # again in the future (e.g., different dataset with many more predictions/image).
    max_rois_per_image: int = 0


@dataclass
class SigLIPConfig:
    """Configuration for the SigLIP verification module and its variants."""
    # Same checkpoint used in Phase 2/3 if you want to be able to load those weights
    # (LoRA and MLP head are specific to the backbone dimension —
    # "base-224" and "giant-384" are not interchangeable).
    # Same checkpoint used in Phase 2/3 — REQUIRED to be able to load the
    # fine-tuned LoRA weights of Phase 3 (different backbone = incompatible weights,
    # shape mismatch on adapters). If you are using pretrained init (option
    # A) this MUST match the checkpoint on which train_fase3_*.py was
    # trained — verified: it is "giant-opt-patch16-384", not "base-224"
    # (that one was used in Phase 2, different).
    model_name: str = "google/siglip2-giant-opt-patch16-384"

    # Which architecture to use in Phase 4:
    #   "vision_mlp" -> vision only + LoRA + MLP head (the one you already had)
    #   "completa"   -> vision+text + LoRA on both, cosine sim to prompts
    #                   (like train_fase3_full_lora.py)
    variant: Literal["vision_mlp", "completa"] = "vision_mlp"

    # LoRA visual (used by both variants)
    lora_r_visual:     int   = 4
    lora_alpha_visual: int   = 8
    lora_dropout:      float = 0.30

    # LoRA text — used ONLY by variant="completa"
    lora_r_text:     int   = 4
    lora_alpha_text: int   = 8

    # MLP classifier — used ONLY by variant="vision_mlp"
    mlp_hidden:  list  = field(default_factory=lambda: [256, 64])
    mlp_dropout: float = 0.50

    # Prompts — used ONLY by variant="completa"
    positive_prompts: list = field(default_factory=lambda: [
        "an empty shelf",
        "a retail shelf with no products",
        "an out-of-stock shelf section",
    ])
    negative_prompts: list = field(default_factory=lambda: [
        "a shelf full of products",
        "a well-stocked retail shelf",
        "a shelf with items on it",
    ])

    # Folder of Phase 3 weights to reload (optional). If None, it is derived
    # automatically from neg_ratio/rank/padding via pretrained_weights_dir()
    # below — consistent with the naming convention of your SLURM .sh scripts
    # (e.g. "fase3_siglip_lora_full_r16_neg025_pad50").
    pretrained_weights_dir: str = None

    # Used ONLY by evaluation script: when True, build_siglip_module
    # skips loading Phase 3 weights (useless in that context — evaluation
    # directly loads the state_dict of the joint-trained checkpoint,
    # which would overwrite the just-loaded Phase 3 weights anyway).
    skip_pretrained_load: bool = False

    # DISABLED by default: causes "Trying to backward through the graph
    # a second time" in combination with PEFT/LoRA on a frozen base,
    # even with use_reentrant=False + enable_input_require_grads(). The
    # reduction of train_conf_threshold (see DetectorConfig) alone has
    # already cut ROIs per image by ~3x — try that, not this,
    # if you still go OOM. Re-enable it only for a targeted test if necessary.
    use_gradient_checkpointing: bool = False

    # Fixed parameters for automatic path derivation — rank 16 is
    # the only one you kept after discarding rank 8; padding 0.50
    # is the one from your .sh script. If you also have checkpoints with different
    # padding (e.g. 0.15, mentioned in Phase 3), update here or pass
    # --padding from CLI.
    rank:    int   = 16
    padding: float = 0.50
    # WARNING: root of the "vision_mlp" variant NOT CONFIRMED — I only have
    # your .sh for "completa" (fase3_siglip_lora_full). Verify/correct
    # this path before launching with variant="vision_mlp".
    weights_root_completa:   str = "weights/fase3_siglip_lora_full"
    weights_root_vision_mlp: str = "weights/fase3_siglip_lora"

    def resolve_pretrained_weights_dir(self, neg_ratio: float) -> str:
        """
        Derives the path of the Phase 3 checkpoint from variant/rank/neg_ratio/padding,
        following the convention of your SLURM .sh scripts. Used by train.py when
        pretrained_weights_dir is not explicitly set. `neg_ratio`
        must be passed explicitly (it comes from cfg.negative_mining.neg_ratio —
        SigLIPConfig does not have direct access to other sections of cfg).
        """
        if self.pretrained_weights_dir:
            return self.pretrained_weights_dir
        root = (self.weights_root_completa if self.variant == "completa"
                 else self.weights_root_vision_mlp)
        prefix = "fase3_siglip_lora_full" if self.variant == "completa" else "fase3_siglip_lora"
        neg_str = str(neg_ratio).replace("0.", "0").replace(".", "")
        pad_str = str(int(self.padding * 100))
        return f"{root}/{prefix}_r{self.rank}_neg{neg_str}_pad{pad_str}/best"


@dataclass
class NegativeMiningConfig:
    """
    Replaces the old JointPipeline.n_neg_synthetic (fixed integer).
    neg_ratio has the same meaning as Phase 3: synthetic_negatives
    per image = round(image_GTs * neg_ratio). With this you can
    re-run Phase 4 with --neg_ratio 0.25 and --neg_ratio 0.5 and compare
    directly with Phase 3 results.
    """
    neg_ratio:   float = 0.5
    neg_iou_max: float = 0.10   # unchanged from original JointPipeline


@dataclass
class TrainingConfig:
    """Configuration for the joint training process."""
    num_epochs:  int   = 50
    lr_detector: float = 5e-6
    lr_siglip:   float = 5e-6

    weight_decay_yolo:   float = 1e-4
    weight_decay_siglip: float = 1e-2

    use_amp: bool = False

    # If True, an OOM on a batch makes the job fail instead of skipping it and
    # continuing — useful ONLY for targeted diagnostic tests (e.g. "up to what
    # batch can this config hold?"), where you want to see the exact crash instead
    # of training continuing silently losing that batch. Leave
    # False in "real" runs you want to complete.
    raise_on_oom: bool = False

    # Joint loss weights — see scripts/calibrate_loss_weights.py to
    # recalculate them when changing siglip.variant.
    alpha: float = 0.49
    beta:  float = 0.51

    save_dir:            Path = Path("weights/cluster_training/phase2_joint")
    save_every:          int  = 5
    early_stop_patience: int  = 5
    device:              str  = "cuda"


PHASE1_WEIGHTS     = Path("weights/cluster_training/phase1_yolo_final/weights/best.pt")
PHASE1_WEIGHTS_AUG = Path("weights/cluster_training/phase1_yolo_final_aug/weights/best.pt")
PHASE2_WEIGHTS     = Path("weights/cluster_training/phase2_joint/best_model.pt")


@dataclass
class Config:
    """Global configuration object aggregating all sub-configs."""
    data:            DataConfig            = field(default_factory=DataConfig)
    detector:        DetectorConfig        = field(default_factory=DetectorConfig)
    siglip:          SigLIPConfig          = field(default_factory=SigLIPConfig)
    negative_mining: NegativeMiningConfig  = field(default_factory=NegativeMiningConfig)
    training:        TrainingConfig        = field(default_factory=TrainingConfig)


CFG = Config()