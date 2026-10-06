"""
src/training/trainer.py
========================
Training loop — differentiable joint loss with fixed weights (alpha/beta).

CHANGES compared to your version (everything else is UNCHANGED —
same separated optimizers scheme, same separated gradient clipping,
same saving scheme):

  - JointPipeline is now built with `cfg` (needed to choose
    the SigLIP variant) instead of just loose parameters.
  - encode_prompts() is called after building the pipeline
    (no-op if siglip.variant="vision_mlp").
  - self.siglip_loss_fn now encompasses BOTH SigLIP losses
    (bce_loss_fn for vision_mlp, sigmoid_loss_fn for completa) and the
    choice of which to use goes through compute_siglip_loss().
  - output["logits_pos"] -> output["siglip_out"] (variant-aware dict).
"""

import torch
import torch.nn as nn
import os
from pathlib import Path
from tqdm import tqdm
import json
import time

from src.models.joint_pipeline import JointPipeline
from src.training.losses       import SigLIPBCELoss, SigLIPSigmoidLoss, YOLOLossWrapper, JointLoss, compute_siglip_loss
from src.data.dataset          import build_dataloaders
from src.utils.config          import CFG


class Trainer:
    """Manages the training loop for the end-to-end joint pipeline."""
    def __init__(self, cfg=None):
        self.cfg    = cfg or CFG
        self.device = torch.device(self.cfg.training.device
                                   if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*60}")
        print(f"  Trainer — Differentiable Joint Loss")
        print(f"  Device        : {self.device}")
        print(f"  Batch         : {self.cfg.data.batch_size}")
        print(f"  siglip.variant: {self.cfg.siglip.variant}")
        print(f"  neg_ratio     : {self.cfg.negative_mining.neg_ratio}")
        print(f"{'='*60}")

        # ── Pipeline — now built from cfg ──────────────────────────────────
        self.pipeline = JointPipeline(cfg=self.cfg).to(self.device)

        # ── Precompute prompts (no-op if variant="vision_mlp") ─────────────
        self.pipeline.encode_prompts(
            positive_prompts = self.cfg.siglip.positive_prompts,
            negative_prompts = self.cfg.siglip.negative_prompts,
        )

        # ── Loss functions ────────────────────────────────────────────────────
        self.siglip_bce_loss     = SigLIPBCELoss()
        self.siglip_sigmoid_loss = SigLIPSigmoidLoss()
        self.yolo_loss_fn        = YOLOLossWrapper()
        self.joint_loss          = JointLoss(
            alpha = self.cfg.training.alpha,
            beta  = self.cfg.training.beta,
        )

        # ── Optimizers ─────────────────────────────────────────────────────
        yolo_params   = list(self.pipeline.detector.parameters()) + \
                        list(self.pipeline.feature_proj.parameters())
        siglip_params = list(self.pipeline.siglip.parameters())

        self.optimizer_yolo = torch.optim.AdamW(
            yolo_params, lr=self.cfg.training.lr_detector,
            weight_decay=self.cfg.training.weight_decay_yolo,
        )
        self.optimizer_siglip = torch.optim.AdamW(
            siglip_params, lr=self.cfg.training.lr_siglip,
            weight_decay=self.cfg.training.weight_decay_siglip,
        )

        self.scheduler_yolo = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_yolo, T_max=self.cfg.training.num_epochs,
            eta_min=self.cfg.training.lr_detector * 0.01,
        )
        self.scheduler_siglip = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_siglip, T_max=self.cfg.training.num_epochs,
            eta_min=self.cfg.training.lr_siglip * 0.01,
        )

        print("\nLoading datasets...")
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            img_size   = self.cfg.data.img_size,
            batch_size = self.cfg.data.batch_size,
            mode       = "phase2",
            data_dir   = str(self.cfg.data.data_dir),
        )

        self.save_dir = Path(self.cfg.training.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss":      [], "train_loss_yolo": [], "train_loss_sig": [],
            "val_loss":        [], "val_loss_yolo":   [], "val_loss_sig":   [],
            "alpha": self.cfg.training.alpha,
            "beta":  self.cfg.training.beta,
            "siglip_variant": self.cfg.siglip.variant,
            "neg_ratio": self.cfg.negative_mining.neg_ratio,
        }
        self.best_val_loss    = float("inf")
        self.patience_counter = 0

    def _compute_siglip_loss(self, output):
        """Dispatcher that selects the correct SigLIP loss based on configuration."""
        if output["n_rois"] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return compute_siglip_loss(
            self.cfg.siglip.variant, output["siglip_out"], output["roi_labels_gt"],
            self.siglip_bce_loss, self.siglip_sigmoid_loss,
        )

    def _train_epoch(self, epoch: int) -> dict:
        """Executes a single training epoch."""
        self.pipeline.train()

        if hasattr(self.train_loader.dataset, 'resample_negatives'):
            self.train_loader.dataset.resample_negatives()

        total_loss = total_yolo = total_sig = 0.0
        n_batches  = 0

        pbar = tqdm(
            self.train_loader,
            desc  = f"Epoch {epoch+1:3d}/{self.cfg.training.num_epochs} [train]",
            leave = False, ncols = 115,
        )

        n_skipped_oom = 0

        for batch in pbar:
            images = batch["images"].to(self.device)
            boxes  = batch["boxes"]
            labels = batch["labels"]

            self.optimizer_yolo.zero_grad()
            self.optimizer_siglip.zero_grad()

            try:
                output = self.pipeline(images, gt_boxes=boxes, gt_labels=labels)

                loss_yolo = self.yolo_loss_fn.compute(
                    gt_boxes=boxes, gt_labels=labels, images=images,
                    device=self.device, predictions=output["predictions"],
                )
                loss_siglip = self._compute_siglip_loss(output)

                loss_joint, _, _ = self.joint_loss(loss_yolo, loss_siglip)

                loss_joint.backward()
                nn.utils.clip_grad_norm_(
                    list(self.pipeline.detector.parameters()) +
                    list(self.pipeline.feature_proj.parameters()),
                    max_norm=10.0
                )
                nn.utils.clip_grad_norm_(list(self.pipeline.siglip.parameters()), max_norm=10.0)
                self.optimizer_yolo.step()
                self.optimizer_siglip.step()

            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if not (isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()):
                    raise  # it wasn't an OOM — don't hide it, fail the job as before
                if self.cfg.training.raise_on_oom:
                    raise  # diagnostic mode: fail the job instead of skipping the batch
                # Parachute: an unlucky batch (more ROIs than usual) must not
                # crash the entire job — especially with only 7h available.
                # Free partially accumulated gradients and CUDA cache, skip
                # this batch, continue with the next. If it happens OFTEN (not just
                # once in a while), it means that batch/max_rois_per_image
                # must be lowered anyway — it's not a fix for a structural problem,
                # just to not lose an entire run for a single extreme batch.
                n_skipped_oom += 1
                print(f"\n  ⚠ OOM on current batch (skipped so far: {n_skipped_oom}) — "
                      f"freeing memory and continuing. If it happens often, lower --batch "
                      f"or --max_rois_per_image.")
                self.optimizer_yolo.zero_grad(set_to_none=True)
                self.optimizer_siglip.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

            total_loss += loss_joint.item()
            total_yolo += loss_yolo.item()
            total_sig  += loss_siglip.item()
            n_batches  += 1

            n_pos = int((output["roi_labels_gt"] == 1).sum().item()) if output["n_rois"] > 0 else 0
            n_neg = output["n_rois"] - n_pos
            pbar.set_postfix({
                "Ly": f"{loss_yolo.item():.3f}", "Ls": f"{loss_siglip.item():.3f}",
                "L": f"{loss_joint.item():.3f}", "ROIs": f"+{n_pos}/-{n_neg}",
                "OOM_skip": n_skipped_oom,
            })

        if n_skipped_oom > 0:
            print(f"  ⚠ Epoch completed with {n_skipped_oom} batches skipped due to OOM "
                  f"out of {n_skipped_oom + n_batches} total.")

        n = max(n_batches, 1)
        return {"loss": total_loss/n, "loss_yolo": total_yolo/n, "loss_sig": total_sig/n}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
        """Executes a single validation epoch."""
        self.pipeline.eval()
        total_loss = total_yolo = total_sig = 0.0
        n_batches  = 0

        for batch in tqdm(
            self.val_loader,
            desc  = f"Epoch {epoch+1:3d}/{self.cfg.training.num_epochs} [val]  ",
            leave = False, ncols = 115,
        ):
            images = batch["images"].to(self.device)
            boxes  = batch["boxes"]
            labels = batch["labels"]

            output    = self.pipeline(images, gt_boxes=None)
            loss_yolo = self.yolo_loss_fn.compute(
                gt_boxes=boxes, gt_labels=labels, images=images,
                device=self.device, predictions=output["predictions"],
            )
            loss_siglip = self._compute_siglip_loss(output)

            loss_joint = self.joint_loss.alpha * loss_yolo + self.joint_loss.beta * loss_siglip
            total_loss += loss_joint.item()
            total_yolo += loss_yolo.item()
            total_sig  += loss_siglip.item()
            n_batches  += 1

        n = max(n_batches, 1)
        return {"loss": total_loss/n, "loss_yolo": total_yolo/n, "loss_sig": total_sig/n}

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves model state, optimizers, schedulers, and history."""
        checkpoint = {
            "epoch":               epoch,
            "pipeline_state_dict": self.pipeline.state_dict(),
            "optimizer_yolo":      self.optimizer_yolo.state_dict(),
            "optimizer_siglip":    self.optimizer_siglip.state_dict(),
            "scheduler_yolo":      self.scheduler_yolo.state_dict(),
            "scheduler_siglip":    self.scheduler_siglip.state_dict(),
            "best_val_loss":       self.best_val_loss,
            "history":             self.history,
            "alpha":               self.joint_loss.alpha,
            "beta":                self.joint_loss.beta,
            "siglip_variant":      self.cfg.siglip.variant,
        }
        self._atomic_save(checkpoint, self.save_dir / f"checkpoint_epoch{epoch+1:03d}.pt")
        if is_best:
            self._atomic_save(checkpoint, self.save_dir / "best_model.pt")
            print(f"    ✓ New best model → {self.save_dir}/best_model.pt")

    @staticmethod
    def _atomic_save(obj, final_path):
        """
        Atomic save — writes to a temporary file (same filesystem,
        to ensure the rename is atomic) and renames it only when
        writing is complete. If SLURM kills the job during torch.save(),
        the final file (final_path) is NEVER touched — the one written
        at the previous iteration remains valid, instead of being
        partially overwritten with a corrupted and unreadable .pt
        ("failed finding central directory").
        """
        final_path = Path(final_path)
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        torch.save(obj, tmp_path)
        os.replace(tmp_path, final_path)   # atomic on Linux (same filesystem)

    def train(self, start_epoch: int = 0):
        """Main training loop iterating over epochs."""
        a, b = self.joint_loss.alpha, self.joint_loss.beta
        if start_epoch > 0:
            print(f"\nResuming training from epoch {start_epoch+1} "
                  f"up to {self.cfg.training.num_epochs}")
        else:
            print(f"\nStarting training for {self.cfg.training.num_epochs} epochs")
        print(f"  L_joint = {a}*L_YOLO(CIoU+BCE) + {b}*L_SigLIP({self.cfg.siglip.variant})")
        print(f"  neg_ratio = {self.cfg.negative_mining.neg_ratio}\n")
        start_time = time.time()

        for epoch in range(start_epoch, self.cfg.training.num_epochs):
            epoch_start = time.time()
            train_m     = self._train_epoch(epoch)
            val_m       = self._val_epoch(epoch)

            self.scheduler_yolo.step()
            self.scheduler_siglip.step()

            for k, v in [
                ("train_loss", train_m["loss"]), ("train_loss_yolo", train_m["loss_yolo"]),
                ("train_loss_sig", train_m["loss_sig"]), ("val_loss", val_m["loss"]),
                ("val_loss_yolo", val_m["loss_yolo"]), ("val_loss_sig", val_m["loss_sig"]),
            ]:
                self.history[k].append(v)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{self.cfg.training.num_epochs} [{epoch_time:.0f}s] | "
                f"train L={train_m['loss']:.4f} (Ly={train_m['loss_yolo']:.4f} Ls={train_m['loss_sig']:.4f}) | "
                f"val L={val_m['loss']:.4f} (Ly={val_m['loss_yolo']:.4f} Ls={val_m['loss_sig']:.4f})"
            )

            is_best = val_m["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss    = val_m["loss"]
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch + 1) % self.cfg.training.save_every == 0 or is_best:
                self._save_checkpoint(epoch, is_best=is_best)

            if self.patience_counter >= self.cfg.training.early_stop_patience:
                print(f"\n⚠ Early stopping after {epoch+1} epochs")
                break

        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Completed in {total_time/60:.1f} min")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        return self.history

    def load_checkpoint(self, path: str):
        """Loads state from a checkpoint file."""
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.pipeline.load_state_dict(ck["pipeline_state_dict"])
        self.optimizer_yolo.load_state_dict(ck["optimizer_yolo"])
        self.optimizer_siglip.load_state_dict(ck["optimizer_siglip"])
        self.scheduler_yolo.load_state_dict(ck["scheduler_yolo"])
        self.scheduler_siglip.load_state_dict(ck["scheduler_siglip"])
        self.best_val_loss = ck["best_val_loss"]
        self.history       = ck.get("history", self.history)
        print(f"  ✓ Checkpoint loaded — epoch {ck['epoch']+1}")
        return ck["epoch"] + 1