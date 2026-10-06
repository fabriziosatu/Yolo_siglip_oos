"""
src/training/train_fase3_lora.py
===================================
PHASE 3 — Fine-tuning of SigLIP2 via LoRA (vision tower only) + MLP
classifier on top, for "empty shelf yes/no" verification.

Architecture (consistent with DRIVE paper and with what's required for Phase 4):
    crop (15% padding, like Phase 2)
        -> SigLIP2 vision tower (giant-opt-patch16-384), FROZEN
        -> LoRA adapter on WQ/WV (rank r=8), TRAINABLE
        -> pooled embedding
        -> MLP (1 hidden layer), TRAINABLE
        -> logit -> sigmoid -> P(empty shelf)

NO text tower: unlike Phase 2 (zero-shot, text-image comparison is needed),
here there is real training, so the text comparison is replaced by a head
trained directly on data — as per the professor's directive for Phase 4,
and as in the DRIVE paper (Sec. 3.2).

Training data:
    - POSITIVES (empty shelf, label=1): crops from dataset GT boxes,
      with 15% padding (same configuration as Phase 2).
    - NEGATIVES (full shelf, label=0): synthetic negative mining — boxes
      randomly generated in the same image, NOT overlapping with any GT
      (IoU < 0.1), with dimensions comparable to GT boxes of the same
      image. Same technique already used in the first joint training
      attempt to avoid the classifier always collapsing to "empty" (see
      initial project discussion).

Usage:
    python src/training/train_fase3_lora.py \
        --train_images dataset_finale/train/images \
        --train_labels dataset_finale/train/labels \
        --val_images   dataset_finale/val/images \
        --val_labels   dataset_finale/val/labels \
        --out_dir weights/fase3_siglip_lora \
        --epochs 30 --batch 16 --lr 1e-4 --padding 0.15 --neg_ratio 1.0

Output in --out_dir:
    lora_adapters/          - ONLY the LoRA adapters (peft save_pretrained),
                              NOT the entire frozen backbone (saves space,
                              see discussion on cluster disk space)
    mlp_head.pt             - weights of the MLP classifier
    results.csv             - loss/metrics per epoch (same format as
                              Ultralytics results.csv, for direct reuse
                              of analyze_yolo_loss_magnitude.py on
                              L_VLM magnitude in Phase 3)
    training_history.json   - same info as results.csv, in JSON
    best_epoch_info.json    - which epoch has the best val_loss
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model


SIGLIP_CKPT = "google/siglip2-giant-opt-patch16-384"
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # consistent with DRIVE paper (WQ, WV)
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ══════════════════════════════════════════════════════════════════════
# Geometric utilities (identical to Phase 2, for consistency)
# ══════════════════════════════════════════════════════════════════════

def load_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    """Reads GT boxes in YOLO format (class 0 = empty_shelf), absolute pixels."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        if cls != 0:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def iou(a, b):
    """Calculates intersection over union for two boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def crop_with_padding(image: Image.Image, box, padding_frac: float):
    """Extracts a crop from the image using the provided bounding box and adds a padding fraction."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * padding_frac, h * padding_frac
    x1p = max(0, x1 - pad_w)
    y1p = max(0, y1 - pad_h)
    x2p = min(image.width, x2 + pad_w)
    y2p = min(image.height, y2 + pad_h)
    return image.crop((x1p, y1p, x2p, y2p))


def generate_negative_box(img_w, img_h, ref_boxes, gt_boxes, max_tries=20):
    """
    Generates a random box with dimensions comparable to `ref_boxes` (the GTs
    of the same image), which does not overlap (IoU<0.1) with any GT.

    Position constraint (horizontal shelf band): the vertical coordinate
    of the negative is limited to the band [y_min, y_max] covered
    by the GT boxes of the same image (with a 20% margin), not left
    free over the whole image. Reason: shelves typically occupy a
    horizontal band of the photo (the camera is roughly at their height);
    outside that band there is typically ceiling, floor, or aisle —
    areas that are not "full shelf" and would pollute the negative
    label with false examples. The horizontal coordinate x remains free over
    the entire width, because along the shelf it is normal to find both
    empty areas and full areas.

    Returns None if it doesn't find a valid position within max_tries.
    """
    if not ref_boxes:
        return None

    # Vertical band covered by GTs, with 20% margin
    y_min = min(b[1] for b in gt_boxes)
    y_max = max(b[3] for b in gt_boxes)
    band_h = y_max - y_min
    margin = band_h * 0.2
    band_y_min = max(0, y_min - margin)
    band_y_max = min(img_h, y_max + margin)

    for _ in range(max_tries):
        ref = random.choice(ref_boxes)
        w = (ref[2] - ref[0]) * random.uniform(0.8, 1.2)
        h = (ref[3] - ref[1]) * random.uniform(0.8, 1.2)
        w = min(w, img_w - 1)
        h = min(h, max(1, band_y_max - band_y_min) - 1) if band_y_max > band_y_min else min(h, img_h - 1)
        h = max(h, 1)

        x1 = random.uniform(0, img_w - w)
        # y1 constrained within the shelf band, not over the whole image
        y_lo = band_y_min
        y_hi = max(band_y_min, band_y_max - h)
        y1 = random.uniform(y_lo, y_hi) if y_hi > y_lo else band_y_min

        candidate = [x1, y1, x1 + w, y1 + h]
        if all(iou(candidate, gt) < 0.1 for gt in gt_boxes):
            return candidate
    return None


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class SigLIPCropDataset(Dataset):
    """
    Precomputes the list of (image_path, box, label) at init — positives
    from GTs, negatives from synthetic negative mining — then crops and
    processes on-the-fly in __getitem__.
    """

    def __init__(self, images_dir: Path, labels_dir: Path, processor,
                 padding: float = 0.15, neg_ratio: float = 1.0, seed: int = 42):
        self.processor = processor
        self.padding = padding
        random.seed(seed)

        images_dir = Path(images_dir)
        labels_dir = Path(labels_dir)
        image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))

        self.samples = []  # (image_path, box, label)
        n_pos = n_neg = 0

        for img_path in image_paths:
            label_path = labels_dir / (img_path.stem + ".txt")
            with Image.open(img_path) as im:
                w, h = im.size
            gt_boxes = load_yolo_boxes(label_path, w, h)

            for box in gt_boxes:
                self.samples.append((img_path, box, 1))
                n_pos += 1

            n_neg_target = round(len(gt_boxes) * neg_ratio) if gt_boxes else 1
            for _ in range(n_neg_target):
                neg_box = generate_negative_box(w, h, gt_boxes, gt_boxes)
                if neg_box is not None:
                    self.samples.append((img_path, neg_box, 0))
                    n_neg += 1

        print(f"[Dataset] {images_dir.parent.name}: {n_pos} positives, {n_neg} negatives "
              f"({len(self.samples)} total samples)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Loads and crops the image for a given sample index."""
        img_path, box, label = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            crop = crop_with_padding(im, box, self.padding).copy()
        # We NO LONGER call the processor here: we do it only once
        # for the entire batch inside ProcessorCollate (much more efficient
        # than processing one image at a time, especially with
        # num_workers>0 where each worker would do the same thing in series).
        return crop, torch.tensor(label, dtype=torch.float32)


class ProcessorCollate:
    """
    collate_fn that groups an entire batch of PIL crops and passes them to the
    SigLIP processor ONLY ONCE (instead of one at a time inside
    __getitem__) — much more efficient, and necessary to use
    num_workers>0 in a sensible way.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        crops = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        inputs = self.processor(images=crops, return_tensors="pt")
        return inputs["pixel_values"], labels


# ══════════════════════════════════════════════════════════════════════
# Model: SigLIP2 vision tower + LoRA + MLP head
# ══════════════════════════════════════════════════════════════════════

class SigLIPLoRAClassifier(nn.Module):
    """
    Vision-only SigLIP2 fine-tuned with LoRA and an MLP classification head.
    """
    def __init__(self, ckpt: str = SIGLIP_CKPT, use_bf16: bool = True,
                 lora_r: int = LORA_RANK, lora_alpha: int = LORA_ALPHA,
                 lora_dropout: float = LORA_DROPOUT):
        """Initializes the model, applies LoRA, and sets up the MLP head."""
        super().__init__()
        # bf16: precision natively supported by the cluster's H100s —
        # halves the memory occupied by the frozen backbone and speeds up
        # forward pass, without significant loss of numerical stability
        # (unlike fp16, bf16 has the same exponential range as
        # fp32, so loss scaling is not needed).
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        full_model = AutoModel.from_pretrained(ckpt, torch_dtype=dtype)

        # Vision tower only — no text encoder (consistent with Phase 4 /
        # DRIVE paper: text is no longer needed once there is training).
        vision_model = full_model.vision_model
        hidden_size = full_model.config.vision_config.hidden_size

        # NOTE on lora_alpha: by convention lora_alpha/r remains constant
        # when rank changes (here the default is 16/8 = ratio 2).
        # If you raise --lora_rank to 16, also pass --lora_alpha 32 to keep
        # the same "effective scaling" of LoRA — otherwise just doubling
        # the rank without scaling alpha silently halves the intensity
        # of the LoRA update compared to the original rank 8.
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        self.vision_model = get_peft_model(vision_model, lora_config)

        # MLP classifier (frozen backbone + LoRA -> embedding -> logit)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, pixel_values):
        """
        Forward pass. Extracts features and computes the classification logit.
        """
        # The processor always returns float32; the backbone might
        # be in bf16 (see __init__) -> an explicit cast is needed before
        # forward, otherwise dtype mismatch error.
        backbone_dtype = next(self.vision_model.parameters()).dtype
        pixel_values = pixel_values.to(dtype=backbone_dtype)
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # [B, hidden_size]
        # The MLP head stays in fp32 for numerical stability of BCE loss,
        # even if the frozen backbone runs in bf16 (saves memory/speed)
        logits = self.mlp_head(pooled.float()).squeeze(-1)  # [B]
        return logits

    def trainable_parameters(self):
        """Returns parameters that require gradients."""
        # LoRA (inside vision_model, already filtered by peft: only adapters
        # require grad) + entire MLP head
        params = [p for p in self.vision_model.parameters() if p.requires_grad]
        params += list(self.mlp_head.parameters())
        return params

    def save(self, out_dir: Path):
        """Saves LoRA adapters and the MLP head."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # ONLY LoRA adapters (a few tens of MBs), NOT the frozen backbone
        self.vision_model.save_pretrained(str(out_dir / "lora_adapters"))
        torch.save(self.mlp_head.state_dict(), out_dir / "mlp_head.pt")


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              progress_every: int = 50, epoch_label: str = ""):
    """Executes a single training or validation epoch."""
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = total = 0
    n_total_batches = len(loader)

    t_start = time.time()
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, (pixel_values, labels) in enumerate(loader, start=1):
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(pixel_values)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Progress print: without this, a long epoch (thousands of
            # batches) gives no signal until it finishes completely —
            # impossible to distinguish "working slowly" from "stuck".
            if progress_every and (batch_idx % progress_every == 0 or batch_idx == n_total_batches):
                elapsed = time.time() - t_start
                batch_per_sec = batch_idx / elapsed if elapsed > 0 else 0
                eta_sec = (n_total_batches - batch_idx) / batch_per_sec if batch_per_sec > 0 else 0
                print(f"    [{epoch_label}] batch {batch_idx}/{n_total_batches} "
                      f"({batch_per_sec:.2f} batch/s, ETA {eta_sec/60:.1f} min) "
                      f"current_loss={total_loss/n_batches:.4f}", flush=True)

    avg_loss = total_loss / n_batches if n_batches else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc


def export_preview_samples(images_dir, labels_dir, out_dir, padding, neg_ratio, n_samples=60, seed=42):
    """
    Exports N positive crops and N negative crops as PNG images (without
    passing through the SigLIP processor — here we want to see them with our own eyes,
    no need to normalize them), with filenames indicating label and source box.
    Useful to visually check the quality of negative mining BEFORE trusting it
    and launching the real training.
    """
    out_dir = Path(out_dir)
    (out_dir / "positivi").mkdir(parents=True, exist_ok=True)
    (out_dir / "negativi").mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    random.shuffle(image_paths)

    n_pos_done = n_neg_done = 0
    for img_path in image_paths:
        if n_pos_done >= n_samples and n_neg_done >= n_samples:
            break
        label_path = labels_dir / (img_path.stem + ".txt")
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            gt_boxes = load_yolo_boxes(label_path, w, h)
            if not gt_boxes:
                continue

            if n_pos_done < n_samples:
                box = random.choice(gt_boxes)
                crop = crop_with_padding(im, box, padding)
                crop.save(out_dir / "positivi" / f"pos_{n_pos_done:03d}_{img_path.stem}.png")
                n_pos_done += 1

            if n_neg_done < n_samples:
                neg_box = generate_negative_box(w, h, gt_boxes, gt_boxes)
                if neg_box is not None:
                    crop = crop_with_padding(im, neg_box, padding)
                    crop.save(out_dir / "negativi" / f"neg_{n_neg_done:03d}_{img_path.stem}.png")
                    n_neg_done += 1

    print(f"[Preview] Saved {n_pos_done} positives in {out_dir / 'positivi'}")
    print(f"[Preview] Saved {n_neg_done} negatives in {out_dir / 'negativi'}")
    print(f"[Preview] Check the negatives by eye before launching the real training!")


def main(args):
    """Main execution function handling setup, training loop, and checkpointing."""
    if args.preview_dir:
        export_preview_samples(
            args.train_images, args.train_labels, args.preview_dir,
            padding=args.padding, neg_ratio=args.neg_ratio,
            n_samples=args.preview_n,
        )
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Device: {device}")
    print(f"[Setup] Loading processor/model: {SIGLIP_CKPT}")
    processor = AutoProcessor.from_pretrained(SIGLIP_CKPT)

    print("[Data] Building training dataset...")
    train_ds = SigLIPCropDataset(
        args.train_images, args.train_labels, processor,
        padding=args.padding, neg_ratio=args.neg_ratio, seed=42,
    )
    print("[Data] Building validation dataset...")
    val_ds = SigLIPCropDataset(
        args.val_images, args.val_labels, processor,
        padding=args.padding, neg_ratio=args.neg_ratio, seed=42,
    )

    collate_fn = ProcessorCollate(processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=collate_fn, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=collate_fn, persistent_workers=(args.num_workers > 0))

    print("[Model] Building SigLIP2 + LoRA + MLP...")
    print(f"[Model] LoRA rank={args.lora_rank}, alpha={args.lora_alpha} "
          f"(alpha/r ratio={args.lora_alpha/args.lora_rank:.1f})")
    model = SigLIPLoRAClassifier(lora_r=args.lora_rank, lora_alpha=args.lora_alpha,
                                  lora_dropout=args.lora_dropout).to(device)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.vision_model.parameters()) + \
        sum(p.numel() for p in model.mlp_head.parameters())
    print(f"[Model] Trainable parameters: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.2f}%)")

    criterion = nn.BCEWithLogitsLoss()

    # ── Differentiated LR: LoRA adapters start close to the identity of the
    # pretrained backbone (we want them to move slowly, low lr), the MLP
    # head starts from random weights and must learn from scratch (we want it
    # to move faster, high lr). A single lr for both either slows down
    # the head too much or accelerates the adapters too much — two separate param_groups
    # in the same optimizer resolve the compromise without making the
    # training sequential: there remains a single forward/backward/step per batch,
    # only the update step size changes for each group.
    lora_params = [p for p in model.vision_model.parameters() if p.requires_grad]
    head_params = list(model.mlp_head.parameters())
    lr_head = args.lr_head if args.lr_head is not None else 1e-3
    print(f"[Optimizer] lr LoRA adapter={args.lr}  |  lr MLP head={lr_head}")
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": lr_head, "weight_decay": args.weight_decay},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    # ── We open results.csv IMMEDIATELY, before training, and write a row
    # per epoch as it finishes — this way if the job is interrupted midway
    # (scancel, SLURM timeout, crash) data for already completed epochs are NOT
    # lost, unlike before when the file was written
    # all together only at the end of training (after loop break).
    results_csv_path = out_dir / "results.csv"
    csv_file = open(results_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
                          "val/box_loss", "val/cls_loss", "val/dfl_loss",
                          "train_acc", "val_acc"])
    csv_file.flush()

    print(f"\n[Training] Start — {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True,
                                            epoch_label=f"epoch {epoch+1}/{args.epochs} train")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False,
                                        epoch_label=f"epoch {epoch+1}/{args.epochs} val")
        scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Write this epoch's row immediately and force flush to disk
        # (flush) — without flush, Python/OS buffering might still
        # hold the data in memory and lose it in case of a sudden kill.
        csv_writer.writerow([epoch + 1, 0.0, train_loss, 0.0, 0.0, val_loss, 0.0, train_acc, val_acc])
        csv_file.flush()
        with open(out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(out_dir / "best")
            with open(out_dir / "best_epoch_info.json", "w") as f:
                json.dump({"epoch": epoch + 1, "val_loss": val_loss, "val_acc": val_acc}, f, indent=2)
        else:
            patience_counter += 1

        if (epoch + 1) % args.save_every == 0:
            model.save(out_dir / f"epoch_{epoch+1:03d}")

        if patience_counter >= args.early_stop_patience:
            print(f"\n[Early stopping] No improvement for {args.early_stop_patience} epochs")
            break

    # Close the incremental csv — all rows are already on disk,
    # written epoch by epoch during the loop above.
    csv_file.close()

    print(f"\n[Completed] Best val_loss={best_val_loss:.4f}")
    print(f"[Output] {out_dir / 'results.csv'}  (compatible with analyze_yolo_loss_magnitude.py)")
    print(f"[Output] {out_dir / 'best/'}  (best LoRA adapter + MLP head)")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3: LoRA training of SigLIP2 (vision-only) + MLP")
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_labels", type=str, required=True)
    parser.add_argument("--val_images", type=str, default=None,
                         help="Required only if not using --preview_dir")
    parser.add_argument("--val_labels", type=str, default=None,
                         help="Required only if not using --preview_dir")
    parser.add_argument("--out_dir", type=str, default="weights/fase3_siglip_lora")
    parser.add_argument("--epochs", type=int, default=50,
                         help="Maximum limit — early stopping usually stops earlier if it converges")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4,
                         help="Learning rate of LoRA adapters on the vision encoder")
    parser.add_argument("--lr_head", type=float, default=None,
                         help="Learning rate of the MLP head (random weights, must learn from "
                              "scratch — usually better to be higher than --lr). Default: 1e-3, "
                              "independent of --lr (it is not a multiplier).")
    parser.add_argument("--padding", type=float, default=0.15,
                         help="Relative crop padding, consistent with Phase 2 (0.15 = 15%%)")
    parser.add_argument("--neg_ratio", type=float, default=1.0,
                         help="Synthetic negatives to positive ratio, per image")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK,
                         help="LoRA rank (default 8, like DRIVE paper). Extra cost "
                              "negligible on H100 even at 16 or 32")
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA,
                         help="LoRA scaling — keep alpha/rank constant if you change the rank "
                              "(default 16, ratio 2 with rank=8)")
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT,
                         help="Dropout on LoRA adapters (default 0.05). Raise it (e.g. 0.15-0.2) "
                              "to slow down overfitting.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help="Weight decay of AdamW optimizer (default 1e-4). Raise it "
                              "(e.g. 1e-3) to slow down overfitting.")
    parser.add_argument("--num_workers", type=int, default=6,
                         help="Parallel workers for DataLoader (uses the 8 CPUs allocated "
                              "by the SLURM job). 0 = all serial in main process "
                              "(much slower, not recommended)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--preview_dir", type=str, default=None,
                         help="If set, DOES NOT train: only saves N positive/negative crops "
                              "in this folder for visual check, then exits")
    parser.add_argument("--preview_n", type=int, default=60,
                         help="Number of crops to export per class in --preview_dir mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.preview_dir and (not args.val_images or not args.val_labels):
        raise ValueError("--val_images and --val_labels are mandatory if not using --preview_dir")
    main(args)