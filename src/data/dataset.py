"""
src/data/dataset.py
====================
PyTorch Dataset for loading images and labels.

Supports two operational modes:

  mode='phase1'  — positives only (annotated empty_shelves)
                   Used for standalone YOLO training (Phase 1)
                   and for val/test in both phases.

  mode='phase2'  — positives + dynamically balanced negatives
                   Used for training the joint pipeline (Phase 2).
                   At each epoch, a subset of negatives equal to the
                   number of positives is sampled (1:1 ratio).
                   Negatives rotate across epochs.

Each sample always returns at least:
  - image:        Tensor [3, H, W] normalized in [0,1]
  - boxes:        Tensor [N, 4] YOLO format (xc, yc, w, h)
  - labels:       Tensor [N]   all 0 (empty_shelf)
  - image_path:   str
  - is_negative:  bool

In phase2 it adds:
  - neg_image:       Tensor [3, H, W]  paired negative image
  - neg_boxes:       Tensor [M, 4]     product boxes (x1n y1n x2n y2n)
  - neg_image_path:  str
"""

import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ── Constants ─────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data/processed_clean")
IMG_SUFFIX = {".jpg", ".jpeg", ".png"}

NEG_CONFIRMED_FILE = DATA_DIR / "negatives" / "confirmed.txt"
NEG_SKU_DIR        = DATA_DIR / "negatives" / "sku110k"
NEG_WEB_DIR        = DATA_DIR / "negatives" / "webmarket"


# ── Negative pool loading ─────────────────────────────────────────────────────

def _load_negative_pool(sku_img_dir: Path, web_img_dir: Path) -> list:
    """
    Loads the complete pool of negatives from all sources.
    Each element is a dict containing:
      'img_path'  : Path to the image (or None if resolved lazily)
      'boxes_path': Path to the .txt with product boxes (or None)
      'source'    : 'confirmed' | 'sku110k' | 'webmarket'
    """
    pool = []

    # 1. Confirmed negatives (Empty XMLs from video_oos_pepper)
    if NEG_CONFIRMED_FILE.exists():
        for line in NEG_CONFIRMED_FILE.read_text().splitlines():
            p = Path(line.strip())
            if p.exists():
                pool.append({
                    "img_path":   p,
                    "boxes_path": None,
                    "source":     "confirmed",
                })

    # 2. Hard negatives SKU110K
    if NEG_SKU_DIR.exists():
        for txt_path in sorted(NEG_SKU_DIR.rglob("*.txt")):
            img_path = sku_img_dir / f"{txt_path.stem}.jpg"
            pool.append({
                "img_path":   img_path if img_path.exists() else None,
                "boxes_path": txt_path,
                "source":     "sku110k",
            })

    # 3. Hard negatives WebMarket
    if NEG_WEB_DIR.exists():
        for txt_path in sorted(NEG_WEB_DIR.glob("*.txt")):
            img_path = web_img_dir / f"{txt_path.stem}.jpg"
            pool.append({
                "img_path":   img_path if img_path.exists() else None,
                "boxes_path": txt_path,
                "source":     "webmarket",
            })

    return pool


def _load_neg_boxes(boxes_path) -> list:
    """
    Reads a .txt file containing negatives.
    Format: x1_norm y1_norm x2_norm y2_norm (one box per line).
    Returns a list of tuples representing the boxes.
    """
    if boxes_path is None or not Path(boxes_path).exists():
        return []
    boxes = []
    with open(boxes_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                try:
                    boxes.append(tuple(float(p) for p in parts))
                except ValueError:
                    continue
    return boxes


# ── Dataset ───────────────────────────────────────────────────────────────────

class ShelfDataset(Dataset):
    """
    Dataset for the detection of empty spaces on shelves.

    Args:
        split:       'train', 'val' or 'test'
        img_size:    square input size (default 640)
        augment:     augmentations (train only)
        mode:        'phase1' or 'phase2'
        data_dir:    dataset root folder (default: data/processed_clean)
        sku_img_dir: SKU110K images folder
        web_img_dir: WebMarket images folder
    """

    def __init__(
        self,
        split,
        img_size=640,
        augment=False,
        mode="phase1",
        data_dir=DATA_DIR,
        sku_img_dir=Path("data/raw/sku110k/images"),
        web_img_dir=Path("data/raw/WebMarket/images"),
    ):
        assert mode in ("phase1", "phase2"), \
            f"mode must be 'phase1' or 'phase2', received: '{mode}'"

        self.img_size    = img_size
        self.augment     = augment
        self.mode        = mode
        self.sku_img_dir = Path(sku_img_dir)
        self.web_img_dir = Path(web_img_dir)

        data_dir = Path(data_dir)
        # Supports two structures:
        #   A) data_dir/images/<split>/  (standard structure)
        #   B) data_dir/<split>/images/  (HPC cluster structure)
        if (data_dir / "images" / split).exists():
            img_dir = data_dir / "images" / split
            lbl_dir = data_dir / "labels" / split
        else:
            img_dir = data_dir / split / "images"
            lbl_dir = data_dir / split / "labels"

        # Load positives
        self.positives = []
        for img_path in sorted(img_dir.glob("*")):
            if img_path.suffix.lower() not in IMG_SUFFIX:
                continue
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if lbl_path.exists():
                self.positives.append((img_path, lbl_path))

        if not self.positives:
            raise RuntimeError(
                f"No positive found in {img_dir}. "
                f"Did you run prepare_dataset.py?"
            )

        # Load negative pool (phase2 only and train only)
        self._neg_pool    = []
        self._neg_sampled = []

        if mode == "phase2" and split == "train":
            self._neg_pool = _load_negative_pool(self.sku_img_dir, self.web_img_dir)
            if not self._neg_pool:
                print("  ⚠ ShelfDataset phase2: no negative found.")
            else:
                self._resample_negatives()

        print(f"  ShelfDataset [{split}|{mode}]: "
              f"{len(self.positives)} positives, "
              f"{len(self._neg_sampled)} negatives per epoch")

    # ── Public API ────────────────────────────────────────────────────────────

    def resample_negatives(self):
        """
        Resamples the negatives for the new epoch (1:1 ratio).
        Call from the trainer at the beginning of each epoch:
            train_loader.dataset.resample_negatives()
        """
        if self.mode == "phase2" and self._neg_pool:
            self._resample_negatives()

    def __len__(self):
        return len(self.positives) + len(self._neg_sampled)

    def __getitem__(self, idx):
        if idx < len(self.positives):
            return self._load_positive(idx)
        else:
            return self._load_negative(idx - len(self.positives))

    # ── Private methods ───────────────────────────────────────────────────────

    def _resample_negatives(self):
        n = len(self.positives)
        if len(self._neg_pool) <= n:
            self._neg_sampled = list(self._neg_pool)
        else:
            self._neg_sampled = random.sample(self._neg_pool, n)

    def _load_positive(self, idx):
        img_path, lbl_path = self.positives[idx]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img, pad_info  = self._letterbox(img)

        boxes = self._load_yolo_labels(lbl_path)
        if boxes:
            boxes = self._adjust_boxes_letterbox(boxes, orig_w, orig_h, pad_info)
        if self.augment and boxes:
            img, boxes = self._augment(img, boxes)

        img_t    = TF.to_tensor(img)
        boxes_t  = torch.tensor(boxes, dtype=torch.float32) if boxes \
                   else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros(len(boxes), dtype=torch.long)

        sample = {
            "image":       img_t,
            "boxes":       boxes_t,
            "labels":      labels_t,
            "image_path":  str(img_path),
            "is_negative": False,
        }

        if self.mode == "phase2":
            sample["neg_image"]      = torch.zeros_like(img_t)
            sample["neg_boxes"]      = torch.zeros((0, 4), dtype=torch.float32)
            sample["neg_image_path"] = ""

        return sample

    def _load_negative(self, idx):
        neg      = self._neg_sampled[idx]
        img_path = neg["img_path"]

        img_t = None
        if img_path is not None and Path(img_path).exists():
            try:
                from PIL import ImageFile
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                img = Image.open(img_path).convert("RGB")
                img, _ = self._letterbox(img)
                img_t  = TF.to_tensor(img)
            except Exception:
                img_t = None

        if img_t is None:
            img_t = torch.zeros(3, self.img_size, self.img_size)

        neg_boxes   = _load_neg_boxes(neg.get("boxes_path"))
        neg_boxes_t = torch.tensor(neg_boxes, dtype=torch.float32) if neg_boxes \
                      else torch.zeros((0, 4), dtype=torch.float32)

        return {
            "image":         img_t,
            "boxes":         torch.zeros((0, 4), dtype=torch.float32),
            "labels":        torch.zeros(0, dtype=torch.long),
            "image_path":    str(img_path) if img_path else "",
            "is_negative":   True,
            "neg_image":     img_t,
            "neg_boxes":     neg_boxes_t,
            "neg_image_path": str(img_path) if img_path else "",
        }

    def _load_yolo_labels(self, lbl_path):
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        xc, yc, w, h = (float(p) for p in parts[1:])
                        if w > 0 and h > 0:
                            boxes.append((xc, yc, w, h))
                    except ValueError:
                        continue
        return boxes

    def _letterbox(self, img):
        target = self.img_size
        orig_w, orig_h = img.size
        scale    = min(target / orig_w, target / orig_h)
        new_w    = int(orig_w * scale)
        new_h    = int(orig_h * scale)
        img      = img.resize((new_w, new_h), Image.BILINEAR)
        pad_left = (target - new_w) // 2
        pad_top  = (target - new_h) // 2
        canvas   = Image.new("RGB", (target, target), (114, 114, 114))
        canvas.paste(img, (pad_left, pad_top))
        return canvas, (scale, pad_left, pad_top)

    def _adjust_boxes_letterbox(self, boxes, orig_w, orig_h, pad_info):
        scale, pad_left, pad_top = pad_info
        target   = self.img_size
        adjusted = []
        for (xc, yc, w, h) in boxes:
            xc_new = (xc * orig_w * scale + pad_left) / target
            yc_new = (yc * orig_h * scale + pad_top)  / target
            w_new  = (w  * orig_w * scale)             / target
            h_new  = (h  * orig_h * scale)             / target
            xc_new = min(max(xc_new, 0.0), 1.0)
            yc_new = min(max(yc_new, 0.0), 1.0)
            w_new  = min(max(w_new,  0.0), 1.0)
            h_new  = min(max(h_new,  0.0), 1.0)
            if w_new > 0 and h_new > 0:
                adjusted.append((xc_new, yc_new, w_new, h_new))
        return adjusted

    def _augment(self, img, boxes):
        if random.random() < 0.5:
            img   = TF.hflip(img)
            boxes = [(1.0 - xc, yc, w, h) for (xc, yc, w, h) in boxes]
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
        return img, boxes


# ── Collate functions ─────────────────────────────────────────────────────────

def collate_fn(batch):
    """Standard collate for phase1 — backward compatible with existing code."""
    return {
        "images":      torch.stack([b["image"]      for b in batch]),
        "boxes":       [b["boxes"]      for b in batch],
        "labels":      [b["labels"]     for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }


def collate_fn_phase2(batch):
    """
    Collate for phase2.
    Adds is_negative, neg_images, and neg_boxes to the standard batch.
    The JointPipeline uses is_negative to distinguish samples
    and assign the correct GT labels to the SigLIP ROIs.
    """
    return {
        # Standard fields (compatible with YOLO and analyze_losses)
        "images":      torch.stack([b["image"]  for b in batch]),
        "boxes":       [b["boxes"]  for b in batch],
        "labels":      [b["labels"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],

        # Negatives mask
        "is_negative": torch.tensor([b["is_negative"] for b in batch]),

        # Negative images and boxes for SigLIP
        "neg_images":      torch.stack([b["neg_image"]  for b in batch]),
        "neg_boxes":       [b["neg_boxes"] for b in batch],
        "neg_image_paths": [b["neg_image_path"] for b in batch],
    }


# ── Factory function ──────────────────────────────────────────────────────────

def build_dataloaders(
    img_size    = 640,
    batch_size  = 8,
    mode        = "phase1",
    data_dir    = DATA_DIR,
    sku_img_dir = Path("data/raw/sku110k/images"),
    web_img_dir = Path("data/raw/WebMarket/images"),
):
    """
    Builds the DataLoaders for train, val, and test.

    Args:
        mode: 'phase1' — standalone YOLO training
              'phase2' — joint pipeline training with negatives

    Usage in Phase 2 trainer (at the beginning of each epoch):
        train_loader.dataset.resample_negatives()
    """
    assert mode in ("phase1", "phase2")

    common = dict(
        img_size    = img_size,
        data_dir    = data_dir,
        sku_img_dir = sku_img_dir,
        web_img_dir = web_img_dir,
    )

    # val and test always use only positives for correct metrics
    train_ds = ShelfDataset("train", augment=True,  mode=mode,     **common)
    val_ds   = ShelfDataset("val",   augment=False, mode="phase1", **common)
    test_ds  = ShelfDataset("test",  augment=False, mode="phase1", **common)

    _collate_train = collate_fn_phase2 if mode == "phase2" else collate_fn

    n_workers = min(4, torch.multiprocessing.cpu_count())

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=n_workers, collate_fn=_collate_train,
        pin_memory=True, persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=(n_workers > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, collate_fn=collate_fn,
        pin_memory=True, persistent_workers=(n_workers > 0),
    )

    return train_loader, val_loader, test_loader


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Test phase1...")
    tl, vl, _ = build_dataloaders(batch_size=4, mode="phase1")
    b = next(iter(tl))
    print(f"  images: {b['images'].shape}  boxes[0]: {b['boxes'][0].shape}")

    print("\nTest phase2...")
    tl2, _, _ = build_dataloaders(batch_size=4, mode="phase2")
    b2 = next(iter(tl2))
    print(f"  images:      {b2['images'].shape}")
    print(f"  is_negative: {b2['is_negative'].tolist()}")
    print(f"  neg_images:  {b2['neg_images'].shape}")

    print("\nTest resample_negatives (simulates beginning of a new epoch)...")
    tl2.dataset.resample_negatives()
    print(f"  ✓ {len(tl2.dataset._neg_sampled)} negatives resampled")

    print("\n✓ Dataset OK!")