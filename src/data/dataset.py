"""
src/data/dataset.py
====================
Dataset PyTorch per il caricamento di immagini e label in formato YOLO.

Ogni campione restituisce:
  - image:      Tensor [3, H, W] normalizzato in [0,1]
  - boxes:      Tensor [N, 4]  in formato YOLO (xc, yc, w, h) normalizzato
                Solo box di classe 0 (empty_shelf) — full_shelf ignorati
  - labels:     Tensor [N]     tutti 0 (unica classe: empty_shelf)
  - image_path: str            percorso all'immagine originale
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
import random


class ShelfDataset(Dataset):
    def __init__(self, split: str, img_size: int = 640, augment: bool = False,
                 data_dir: str = "data/processed"):
        """
        Args:
            split:    'train', 'val' o 'test'
            img_size: dimensione a cui ridimensionare le immagini (quadrata)
            augment:  se True, applica augmentazioni durante il training
            data_dir: percorso alla cartella del dataset
        """
        self.img_size = img_size
        self.augment  = augment

        base    = Path(data_dir)
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split

        # Raccoglie coppie (immagine, label) valide
        self.samples = []
        for img_path in sorted(img_dir.glob("*.jpg")):
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nessuna immagine trovata in {img_dir}. "
                f"Hai eseguito prepare_datasets.py?"
            )

        print(f"  ShelfDataset [{split}]: {len(self.samples)} campioni caricati.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]

        # ── Carica immagine ───────────────────────────────────────────────────
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Ridimensiona mantenendo proporzioni con padding (letterbox)
        img, pad_info = self._letterbox(img, self.img_size)

        # ── Carica bounding box ───────────────────────────────────────────────
        boxes = self._load_labels(lbl_path)

        # Adatta le box al letterbox
        if len(boxes) > 0:
            boxes = self._adjust_boxes_letterbox(boxes, orig_w, orig_h, pad_info)

        # ── Augmentazioni (solo train) ────────────────────────────────────────
        if self.augment and len(boxes) > 0:
            img, boxes = self._augment(img, boxes)

        # ── Converti in tensor ────────────────────────────────────────────────
        img_tensor = TF.to_tensor(img)  # [3, H, W] in [0,1]

        if len(boxes) > 0:
            boxes_tensor  = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.zeros(len(boxes), dtype=torch.long)
        else:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros(0, dtype=torch.long)

        return {
            "image":      img_tensor,
            "boxes":      boxes_tensor,
            "labels":     labels_tensor,
            "image_path": str(img_path),
        }

    # ── Metodi privati ────────────────────────────────────────────────────────

    def _letterbox(self, img: Image.Image, target: int):
        """
        Ridimensiona l'immagine in un quadrato target x target
        aggiungendo padding grigio sui lati corti.
        Restituisce (immagine_ridimensionata, pad_info).
        pad_info = (scale, pad_left, pad_top)
        """
        orig_w, orig_h = img.size
        scale = min(target / orig_w, target / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        img = img.resize((new_w, new_h), Image.BILINEAR)

        # Padding per arrivare a target x target
        pad_left = (target - new_w) // 2
        pad_top  = (target - new_h) // 2

        new_img = Image.new("RGB", (target, target), (114, 114, 114))
        new_img.paste(img, (pad_left, pad_top))

        return new_img, (scale, pad_left, pad_top)

    def _adjust_boxes_letterbox(self, boxes, orig_w, orig_h, pad_info):
        """
        Riadatta le coordinate YOLO (normalizzate su orig_w/orig_h)
        alla nuova immagine letterboxed.
        """
        scale, pad_left, pad_top = pad_info
        target = self.img_size
        adjusted = []

        for (xc, yc, w, h) in boxes:
            # Converti in pixel assoluti rispetto all'originale
            xc_px = xc * orig_w
            yc_px = yc * orig_h
            w_px  = w  * orig_w
            h_px  = h  * orig_h

            # Scala e aggiungi padding
            xc_new = (xc_px * scale + pad_left) / target
            yc_new = (yc_px * scale + pad_top)  / target
            w_new  = (w_px  * scale)             / target
            h_new  = (h_px  * scale)             / target

            # Clamp
            xc_new = min(max(xc_new, 0.0), 1.0)
            yc_new = min(max(yc_new, 0.0), 1.0)
            w_new  = min(max(w_new,  0.0), 1.0)
            h_new  = min(max(h_new,  0.0), 1.0)

            if w_new > 0 and h_new > 0:
                adjusted.append((xc_new, yc_new, w_new, h_new))

        return adjusted

    def _load_labels(self, lbl_path: Path) -> list:
        """Legge il file .txt e restituisce lista di (xc, yc, w, h).
        Carica solo le box di classe 0 (empty_shelf) — full_shelf (classe 1) ignorati.
        """
        boxes = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls = int(parts[0])
                        if cls != 0:
                            continue
                        xc, yc, w, h = (float(p) for p in parts[1:])
                        if w > 0 and h > 0:
                            boxes.append((xc, yc, w, h))
                    except ValueError:
                        continue
        return boxes

    def _augment(self, img: Image.Image, boxes: list):
        """Augmentazioni leggere compatibili con le bounding box."""

        # Flip orizzontale con probabilità 0.5
        if random.random() < 0.5:
            img = TF.hflip(img)
            boxes = [(1.0 - xc, yc, w, h) for (xc, yc, w, h) in boxes]

        # Color jitter leggero (non modifica le box)
        if random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
        if random.random() < 0.5:
            img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
        if random.random() < 0.3:
            img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))

        return img, boxes


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch):
    """
    Funzione custom per il DataLoader.
    Necessaria perché ogni immagine ha un numero diverso di box.
    """
    images      = torch.stack([item["image"] for item in batch])
    image_paths = [item["image_path"] for item in batch]

    # Le box hanno dimensioni diverse per ogni immagine:
    # le teniamo come lista di tensor
    boxes  = [item["boxes"]  for item in batch]
    labels = [item["labels"] for item in batch]

    return {
        "images":      images,        # [B, 3, H, W]
        "boxes":       boxes,         # lista di B tensor [Ni, 4]
        "labels":      labels,        # lista di B tensor [Ni]
        "image_paths": image_paths,   # lista di B stringhe
    }


# ── Factory function ──────────────────────────────────────────────────────────

def build_dataloaders(img_size: int = 640, batch_size: int = 8,
                      data_dir: str = "data/processed"):
    """
    Costruisce i DataLoader per train, val e test.
    Uso:
        train_loader, val_loader, test_loader = build_dataloaders()
        train_loader, val_loader, test_loader = build_dataloaders(data_dir="data/oos_ridotto")
    """
    train_ds = ShelfDataset("train", img_size=img_size, augment=True,  data_dir=data_dir)
    val_ds   = ShelfDataset("val",   img_size=img_size, augment=False, data_dir=data_dir)
    test_ds  = ShelfDataset("test",  img_size=img_size, augment=False, data_dir=data_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,       # 0 obbligatorio su Windows con multiprocessing
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ── Test rapido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Test caricamento dataset...")
    train_loader, val_loader, test_loader = build_dataloaders(
        img_size=640, batch_size=4
    )

    batch = next(iter(train_loader))
    print(f"\nBatch di test:")
    print(f"  images shape : {batch['images'].shape}")
    print(f"  num box img0 : {len(batch['boxes'][0])}")
    print(f"  box img0[0]  : {batch['boxes'][0][0] if len(batch['boxes'][0]) > 0 else 'nessuna'}")
    print(f"  paths[0]     : {batch['image_paths'][0]}")
    print("\n✓ Dataset caricato correttamente!")