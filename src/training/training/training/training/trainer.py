"""
src/training/trainer.py
========================
Training loop — Fase 2: joint loss differenziabile con pesi fissi.

L_joint = alpha * L_YOLO(CIoU+BCE) + beta * L_SigLIP(BCE)
con alpha=0.49, beta=0.51 — un solo backward aggiorna YOLO e SigLIP insieme.

Negativi sintetici generati on-the-fly dalla JointPipeline
tramite _sample_negative_rois() — nessun dataset esterno necessario.
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import json
import time

from src.models.joint_pipeline import JointPipeline
from src.training.losses       import SigLIPBCELoss, YOLOLossWrapper, JointLoss
from src.data.dataset          import build_dataloaders
from src.utils.config          import CFG


class Trainer:

    def __init__(self, cfg=None):
        self.cfg    = cfg or CFG
        self.device = torch.device(self.cfg.training.device
                                   if torch.cuda.is_available() else "cpu")

        print(f"\n{'='*60}")
        print(f"  Trainer — Fase 2: Joint Loss Differenziabile")
        print(f"  Device   : {self.device}")
        print(f"  Batch    : {self.cfg.data.batch_size}")
        print(f"  Workers  : {self.cfg.data.num_workers}")
        print(f"{'='*60}")

        # ── Pipeline ──────────────────────────────────────────────────────────
        # Nota: mlp_hidden e mlp_dropout sono parametri di SigLIPModule,
        # non di JointPipeline — vengono letti internamente dalla pipeline
        self.pipeline = JointPipeline(
            yolo_weights      = self.cfg.detector.model_name,
            siglip_model_name = self.cfg.siglip.model_name,
            conf_threshold    = self.cfg.detector.conf_threshold,
            lora_r_visual     = self.cfg.siglip.lora_r_visual,
            lora_alpha_visual = self.cfg.siglip.lora_alpha_visual,
            lora_dropout      = self.cfg.siglip.lora_dropout,
            roi_size          = self.cfg.data.roi_size,
        ).to(self.device)

        # ── Loss functions ────────────────────────────────────────────────────
        self.siglip_loss_fn = SigLIPBCELoss()
        self.yolo_loss_fn   = YOLOLossWrapper()
        self.joint_loss     = JointLoss(
            alpha = self.cfg.training.alpha,
            beta  = self.cfg.training.beta,
        )

        # ── Ottimizzatori ─────────────────────────────────────────────────────
        yolo_params   = list(self.pipeline.detector.parameters()) + \
                        list(self.pipeline.feature_proj.parameters())
        siglip_params = list(self.pipeline.siglip.parameters())

        self.optimizer_yolo = torch.optim.AdamW(
            yolo_params,
            lr           = self.cfg.training.lr_detector,
            weight_decay = self.cfg.training.weight_decay_yolo,
        )
        self.optimizer_siglip = torch.optim.AdamW(
            siglip_params,
            lr           = self.cfg.training.lr_siglip,
            weight_decay = self.cfg.training.weight_decay_siglip,
        )

        self.scheduler_yolo = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_yolo,
            T_max   = self.cfg.training.num_epochs,
            eta_min = self.cfg.training.lr_detector * 0.01,
        )
        self.scheduler_siglip = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_siglip,
            T_max   = self.cfg.training.num_epochs,
            eta_min = self.cfg.training.lr_siglip * 0.01,
        )

        # ── DataLoader — phase2 per training con negativi sintetici ──────────
        print("\nCarico i dataset...")
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
        }
        self.best_val_loss    = float("inf")
        self.patience_counter = 0

    def _train_epoch(self, epoch: int) -> dict:
        self.pipeline.train()

        # Ricampiona i negativi all'inizio di ogni epoca (1:1 con i positivi)
        if hasattr(self.train_loader.dataset, 'resample_negatives'):
            self.train_loader.dataset.resample_negatives()

        total_loss = total_yolo = total_sig = 0.0
        n_batches  = 0

        pbar = tqdm(
            self.train_loader,
            desc  = f"Epoch {epoch+1:3d}/{self.cfg.training.num_epochs} [train]",
            leave = False, ncols = 115,
        )

        for batch in pbar:
            images = batch["images"].to(self.device)
            boxes  = batch["boxes"]
            labels = batch["labels"]

            self.optimizer_yolo.zero_grad()
            self.optimizer_siglip.zero_grad()

            # La pipeline genera i negativi sintetici internamente
            output = self.pipeline(images, gt_boxes=boxes, gt_labels=labels)

            # L_YOLO — CIoU+BCE custom differenziabile
            loss_yolo = self.yolo_loss_fn.compute(
                gt_boxes    = boxes,
                gt_labels   = labels,
                images      = images,
                device      = self.device,
                predictions = output["predictions"],
            )

            # L_SigLIP — BCE sul logit MLP (output["logits_pos"])
            if output["n_rois"] > 0:
                loss_siglip = self.siglip_loss_fn(
                    output["logits_pos"],
                    output["roi_labels_gt"],
                )
            else:
                loss_siglip = torch.tensor(
                    0.0, device=self.device, requires_grad=True
                )

            # Un solo backward — aggiorna YOLO e SigLIP insieme
            loss_joint, _, _ = self.joint_loss(loss_yolo, loss_siglip)

            loss_joint.backward()
            nn.utils.clip_grad_norm_(
                list(self.pipeline.detector.parameters()) +
                list(self.pipeline.feature_proj.parameters()),
                max_norm=10.0
            )
            nn.utils.clip_grad_norm_(
                list(self.pipeline.siglip.parameters()),
                max_norm=10.0
            )
            self.optimizer_yolo.step()
            self.optimizer_siglip.step()

            total_loss += loss_joint.item()
            total_yolo += loss_yolo.item()
            total_sig  += loss_siglip.item()
            n_batches  += 1

            n_pos = int((output["roi_labels_gt"] == 1).sum().item()) \
                    if output["n_rois"] > 0 else 0
            n_neg = output["n_rois"] - n_pos
            pbar.set_postfix({
                "Ly":   f"{loss_yolo.item():.3f}",
                "Ls":   f"{loss_siglip.item():.3f}",
                "L":    f"{loss_joint.item():.3f}",
                "ROIs": f"+{n_pos}/-{n_neg}",
            })

        n = max(n_batches, 1)
        return {"loss": total_loss/n, "loss_yolo": total_yolo/n, "loss_sig": total_sig/n}

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> dict:
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
                gt_boxes    = boxes,
                gt_labels   = labels,
                images      = images,
                device      = self.device,
                predictions = output["predictions"],
            )
            if output["n_rois"] > 0:
                loss_siglip = self.siglip_loss_fn(
                    output["logits_pos"],
                    output["roi_labels_gt"],
                )
            else:
                loss_siglip = torch.tensor(0.0, device=self.device)

            loss_joint = (self.joint_loss.alpha * loss_yolo +
                          self.joint_loss.beta  * loss_siglip)
            total_loss += loss_joint.item()
            total_yolo += loss_yolo.item()
            total_sig  += loss_siglip.item()
            n_batches  += 1

        n = max(n_batches, 1)
        return {"loss": total_loss/n, "loss_yolo": total_yolo/n, "loss_sig": total_sig/n}

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
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
        }
        torch.save(checkpoint, self.save_dir / f"checkpoint_epoch{epoch+1:03d}.pt")
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_model.pt")
            print(f"    ✓ Nuovo best model → {self.save_dir}/best_model.pt")

    def train(self):
        a, b = self.joint_loss.alpha, self.joint_loss.beta
        print(f"\nInizio training per {self.cfg.training.num_epochs} epoche")
        print(f"  L_joint = {a}*L_YOLO(CIoU+BCE) + {b}*L_SigLIP(BCE)")
        print(f"  Negativi sintetici: generati on-the-fly dalla JointPipeline")
        print(f"  Un solo backward — YOLO e SigLIP aggiornati insieme\n")
        start_time = time.time()

        for epoch in range(self.cfg.training.num_epochs):
            epoch_start = time.time()
            train_m     = self._train_epoch(epoch)
            val_m       = self._val_epoch(epoch)

            self.scheduler_yolo.step()
            self.scheduler_siglip.step()

            for k, v in [
                ("train_loss",      train_m["loss"]),
                ("train_loss_yolo", train_m["loss_yolo"]),
                ("train_loss_sig",  train_m["loss_sig"]),
                ("val_loss",        val_m["loss"]),
                ("val_loss_yolo",   val_m["loss_yolo"]),
                ("val_loss_sig",    val_m["loss_sig"]),
            ]:
                self.history[k].append(v)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch+1:3d}/{self.cfg.training.num_epochs} "
                f"[{epoch_time:.0f}s] | "
                f"train L={train_m['loss']:.4f} "
                f"(Ly={train_m['loss_yolo']:.4f} Ls={train_m['loss_sig']:.4f}) | "
                f"val L={val_m['loss']:.4f} "
                f"(Ly={val_m['loss_yolo']:.4f} Ls={val_m['loss_sig']:.4f})"
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
                print(f"\n⚠ Early stopping dopo {epoch+1} epoche")
                break

        with open(self.save_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Completato in {total_time/60:.1f} min")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        return self.history

    def load_checkpoint(self, path: str):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.pipeline.load_state_dict(ck["pipeline_state_dict"])
        self.optimizer_yolo.load_state_dict(ck["optimizer_yolo"])
        self.optimizer_siglip.load_state_dict(ck["optimizer_siglip"])
        self.scheduler_yolo.load_state_dict(ck["scheduler_yolo"])
        self.scheduler_siglip.load_state_dict(ck["scheduler_siglip"])
        self.best_val_loss = ck["best_val_loss"]
        self.history       = ck.get("history", self.history)
        print(f"  ✓ Checkpoint caricato — epoca {ck['epoch']+1}")
        return ck["epoch"] + 1