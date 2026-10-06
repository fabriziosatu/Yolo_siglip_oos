"""
src/training/train.py
=======================
Entry point for the joint pipeline training.

Usage:
    python src/training/train.py --siglip_variant vision_mlp --neg_ratio 0.5
    python src/training/train.py --siglip_variant completa --neg_ratio 0.25
    python src/training/train.py --resume weights/checkpoint_epoch010.pt
"""

import sys
import os
import argparse
from pathlib import Path

# Calculates the project root from the FILE position (src/training/train.py
# -> go up two levels), not from the launch folder. Before it used
# sys.path.insert(0, ".") which only works if you launch the script after having
# done an exact "cd $HPC_ROOT" — if you launch it from another folder (e.g. a
# manual test), "import src..." fails with ModuleNotFoundError.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# Explicit os.chdir: also relative paths USED AT RUNTIME (dataset_dir,
# detector.model_name, etc. in config.py) are relative to the working
# directory, not just the imports — without this, the import would work
# but dataset/weights loading would fail anyway if launched from
# a folder other than $HPC_ROOT.
os.chdir(PROJECT_ROOT)

from src.training.trainer import Trainer
from src.utils.config     import CFG


def parse_args():
    """Parses command line arguments, allowing runtime overrides for global configuration."""
    parser = argparse.ArgumentParser(description="Training joint pipeline YOLO26 + SigLIPv2")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint from which to resume training")
    parser.add_argument("--epochs", type=int, default=None,
                         help="Number of epochs (overwrites config)")
    parser.add_argument("--batch", type=int, default=None,
                         help="Batch size (overwrites config)")
    parser.add_argument("--yolo_weights", type=str, default=None,
                         help="Path to YOLO weights (Phase 1 fine-tuned for option A, "
                              "base COCO checkpoint like yolo26n.pt for option B/scratch — "
                              "overwrites detector.model_name)")
    parser.add_argument("--skip_pretrained_load", action="store_true",
                         help="Option B — SigLIP starts from random-init LoRA on the pretrained "
                              "backbone instead of Phase 3 weights (the SigLIP2 backbone "
                              "remains the one downloaded from HuggingFace, it's not pure "
                              "random — only the LoRA adapters are, see discussion on the "
                              "meaning of 'from scratch')")
    parser.add_argument("--train_conf_threshold", type=float, default=None,
                         help="Confidence threshold for candidate ROIs in train mode "
                              "(overwrites detector.train_conf_threshold — default 0.2 "
                              "for pretrained weights, try 0.1 if using init from scratch)")
    parser.add_argument("--roi_padding", type=float, default=None,
                         help="Padding around the box before RoI Align, same "
                              "meaning as --padding in Phase 3 (overwrites "
                              "data.roi_padding, default 0.50 — consistent with the "
                              "'pad50' checkpoints you load as pretrained weights)")
    parser.add_argument("--max_rois_per_image", type=int, default=None,
                         help="Maximum limit of ROIs per image passed to SigLIP "
                              "(overwrites detector.max_rois_per_image, default 20 — "
                              "reduces OOM/CUDA fragmentation from variable ROI counting)")
    parser.add_argument("--raise_on_oom", action="store_true",
                         help="Diagnostic mode: an OOM fails the job instead of "
                              "skipping the batch and continuing. Useful for tests like 'up to "
                              "what batch can this config hold?' — DO NOT use it in runs you "
                              "want to complete, you'd lose the whole job for a "
                              "single unlucky batch.")
    parser.add_argument("--save_dir", type=str, default=None,
                         help="Output folder for checkpoint/history of this run "
                              "(overwrites training.save_dir — use it to separate "
                              "experiments, otherwise they overwrite each other)")

    parser.add_argument("--siglip_variant", type=str, default=None,
                         choices=["vision_mlp", "completa"],
                         help="Which SigLIP architecture to use")
    parser.add_argument("--neg_ratio", type=float, default=None,
                         help="Synthetic negatives/GT ratio per image — guides BOTH the "
                              "Phase 3 checkpoint path to load AND the Phase 4 mining "
                              "(same meaning as Phase 3 — 0.25 or 0.5)")
    parser.add_argument("--siglip_weights_dir", type=str, default=None,
                         help="Manual override of the Phase 3 weights path (bypasses the "
                              "automatic derivation from --neg_ratio)")
    parser.add_argument("--alpha", type=float, default=None,
                         help="L_YOLO weight in the joint loss (overwrites config)")
    parser.add_argument("--beta", type=float, default=None,
                         help="L_SigLIP weight in the joint loss (overwrites config)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.epochs:
        CFG.training.num_epochs = args.epochs
    if args.batch:
        CFG.data.batch_size = args.batch
    if args.yolo_weights:
        CFG.detector.model_name = args.yolo_weights
    if args.skip_pretrained_load:
        CFG.siglip.skip_pretrained_load = True
    if args.train_conf_threshold is not None:
        CFG.detector.train_conf_threshold = args.train_conf_threshold
    if args.roi_padding is not None:
        CFG.data.roi_padding = args.roi_padding
    if args.max_rois_per_image is not None:
        CFG.detector.max_rois_per_image = args.max_rois_per_image
    if args.raise_on_oom:
        CFG.training.raise_on_oom = True
    if args.save_dir:
        CFG.training.save_dir = Path(args.save_dir)
    if args.siglip_variant:
        CFG.siglip.variant = args.siglip_variant
    if args.neg_ratio is not None:
        CFG.negative_mining.neg_ratio = args.neg_ratio
    if args.siglip_weights_dir:
        CFG.siglip.pretrained_weights_dir = args.siglip_weights_dir
    if args.alpha is not None or args.beta is not None:
        if args.alpha is None or args.beta is None:
            raise ValueError("--alpha and --beta must be passed together (alpha+beta must be 1)")
        CFG.training.alpha = args.alpha
        CFG.training.beta  = args.beta

    print(f"\n[Config] siglip.variant={CFG.siglip.variant}  neg_ratio={CFG.negative_mining.neg_ratio}")
    print(f"[Config] siglip.skip_pretrained_load={CFG.siglip.skip_pretrained_load} "
          f"({'OPTION B — SigLIP from scratch' if CFG.siglip.skip_pretrained_load else 'OPTION A — SigLIP from Phase 3'})")
    print(f"[Config] alpha={CFG.training.alpha}  beta={CFG.training.beta}")
    print(f"[Config] detector.model_name={CFG.detector.model_name}")
    print(f"[Config] detector.train_conf_threshold={CFG.detector.train_conf_threshold}")
    print(f"[Config] data.roi_padding={CFG.data.roi_padding}")
    print(f"[Config] detector.max_rois_per_image={CFG.detector.max_rois_per_image}")
    print(f"[Config] training.raise_on_oom={CFG.training.raise_on_oom} "
          f"({'DIAGNOSTIC — the job fails on OOM' if CFG.training.raise_on_oom else 'parachute active'})")
    print(f"[Config] training.save_dir={CFG.training.save_dir}\n")

    trainer = Trainer(cfg=CFG)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    history = trainer.train(start_epoch=start_epoch)