"""
src/training/train_fase3_lora_no_bandvincolo.py
==================================================
*** VARIANTE DI CONFRONTO — negative mining SENZA vincolo di fascia
    verticale, per confronto diretto con train_fase3_lora.py (versione
    con vincolo). Vedi generate_negative_box() per la differenza esatta. ***

FASE 3 — Fine-tuning di SigLIP2 tramite LoRA (solo vision tower) + MLP
classificatore in testa, per la verifica "scaffale vuoto sì/no".

Architettura (coerente con paper DRIVE e con quanto richiesto per la Fase 4):
    crop (padding 15%, come Fase 2)
        -> SigLIP2 vision tower (giant-opt-patch16-384), FROZEN
        -> adapter LoRA su WQ/WV (rank r=8), TRAINABILI
        -> pooled embedding
        -> MLP (1 hidden layer), TRAINABILE
        -> logit -> sigmoid -> P(scaffale vuoto)

NIENTE text tower: a differenza della Fase 2 (zero-shot, serve il confronto
testo-immagine), qui c'e' training vero, quindi si sostituisce il confronto
testuale con un head allenato direttamente sui dati — come da direttiva
prof per la Fase 4, e come nel paper DRIVE (Sez. 3.2).

Dati di training:
    - POSITIVI (scaffale vuoto, label=1): crop dalle GT box del dataset,
      con padding 15% (stessa configurazione della Fase 2).
    - NEGATIVI (scaffale pieno, label=0): negative mining sintetico — box
      generate casualmente nella stessa immagine, che NON si sovrappongono
      a nessuna GT (IoU < 0.1), con dimensioni comparabili alle GT box
      della stessa immagine. Stessa tecnica gia' usata nel primo tentativo
      di joint training per evitare che il classificatore collassi
      sempre "vuoto" (vedi discussione iniziale del progetto).

Uso:
    python src/training/train_fase3_lora.py \
        --train_images dataset_finale/train/images \
        --train_labels dataset_finale/train/labels \
        --val_images   dataset_finale/val/images \
        --val_labels   dataset_finale/val/labels \
        --out_dir weights/fase3_siglip_lora \
        --epochs 30 --batch 16 --lr 1e-4 --padding 0.15 --neg_ratio 1.0

Output in --out_dir:
    lora_adapters/          - SOLO gli adapter LoRA (peft save_pretrained),
                              NON l'intero backbone frozen (risparmio spazio,
                              vedi discussione sullo spazio disco del cluster)
    mlp_head.pt             - pesi dell'MLP classificatore
    results.csv             - loss/metriche per epoca (stesso formato di
                              results.csv di Ultralytics, per riuso diretto
                              di analyze_yolo_loss_magnitude.py sulla
                              magnitudo di L_VLM in Fase 3)
    training_history.json   - stessa info di results.csv, in JSON
    best_epoch_info.json    - quale epoca ha il miglior val_loss
"""

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor
from peft import LoraConfig, get_peft_model


SIGLIP_CKPT = "google/siglip2-giant-opt-patch16-384"
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # coerente col paper DRIVE (WQ, WV)
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ══════════════════════════════════════════════════════════════════════
# Utility geometriche (identiche a Fase 2, per coerenza)
# ══════════════════════════════════════════════════════════════════════

def load_yolo_boxes(label_path: Path, img_w: int, img_h: int):
    """Legge le GT box in formato YOLO (classe 0 = empty_shelf), pixel assoluti."""
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
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def crop_with_padding(image: Image.Image, box, padding_frac: float):
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
    VERSIONE SENZA VINCOLO DI FASCIA (posizione completamente libera).

    Genera una box casuale con dimensioni comparabili a `ref_boxes` (le GT
    della stessa immagine), che non si sovrapponga (IoU<0.1) a nessuna GT.
    La posizione (x, y) e' libera su tutta l'immagine — nessun vincolo che
    la tenga nella fascia orizzontale dello scaffale. Puo' quindi capitare
    su soffitto, pavimento, corridoio o altri elementi estranei alla scena
    di scaffale. Tenuta come riferimento per confronto con la versione con
    vincolo di fascia (--> train_fase3_lora.py).

    Ritorna None se non trova una posizione valida entro max_tries.
    """
    if not ref_boxes:
        return None
    for _ in range(max_tries):
        ref = random.choice(ref_boxes)
        w = (ref[2] - ref[0]) * random.uniform(0.8, 1.2)
        h = (ref[3] - ref[1]) * random.uniform(0.8, 1.2)
        w = min(w, img_w - 1)
        h = min(h, img_h - 1)
        x1 = random.uniform(0, img_w - w)
        y1 = random.uniform(0, img_h - h)
        candidate = [x1, y1, x1 + w, y1 + h]
        if all(iou(candidate, gt) < 0.1 for gt in gt_boxes):
            return candidate
    return None


# ══════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════

class SigLIPCropDataset(Dataset):
    """
    Precalcola all'init la lista di (image_path, box, label) — positivi
    dalle GT, negativi da negative mining sintetico — poi ritaglia e
    processa on-the-fly in __getitem__.
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

        print(f"[Dataset] {images_dir.parent.name}: {n_pos} positivi, {n_neg} negativi "
              f"({len(self.samples)} campioni totali)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box, label = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            crop = crop_with_padding(im, box, self.padding)
            # Il processor ridimensiona/normalizza; lo eseguiamo qui per
            # restituire gia' un tensore pronto (piu' comodo da collate-are)
            inputs = self.processor(images=[crop], return_tensors="pt")
        return inputs["pixel_values"][0], torch.tensor(label, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════════════
# Modello: SigLIP2 vision tower + LoRA + MLP head
# ══════════════════════════════════════════════════════════════════════

class SigLIPLoRAClassifier(nn.Module):
    def __init__(self, ckpt: str = SIGLIP_CKPT):
        super().__init__()
        full_model = AutoModel.from_pretrained(ckpt)

        # Solo la vision tower — niente text encoder (coerente con Fase 4 /
        # paper DRIVE: il testo non serve piu' una volta che c'e' training).
        vision_model = full_model.vision_model
        hidden_size = full_model.config.vision_config.hidden_size

        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
        )
        self.vision_model = get_peft_model(vision_model, lora_config)

        # MLP classificatore (frozen backbone + LoRA -> embedding -> logit)
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # [B, hidden_size]
        logits = self.mlp_head(pooled).squeeze(-1)  # [B]
        return logits

    def trainable_parameters(self):
        # LoRA (dentro vision_model, gia' filtrati da peft: solo gli adapter
        # richiedono grad) + MLP head per intero
        params = [p for p in self.vision_model.parameters() if p.requires_grad]
        params += list(self.mlp_head.parameters())
        return params

    def save(self, out_dir: Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Solo gli adapter LoRA (poche decine di MB), non il backbone frozen
        self.vision_model.save_pretrained(str(out_dir / "lora_adapters"))
        torch.save(self.mlp_head.state_dict(), out_dir / "mlp_head.pt")


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    n_batches = 0
    correct = total = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for pixel_values, labels in loader:
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

    avg_loss = total_loss / n_batches if n_batches else 0.0
    acc = correct / total if total else 0.0
    return avg_loss, acc


def export_preview_samples(images_dir, labels_dir, out_dir, padding, neg_ratio, n_samples=60, seed=42):
    """
    Esporta N crop positivi e N crop negativi come immagini PNG (senza
    passare dal processor SigLIP — qui vogliamo vederli con i nostri occhi,
    non serve normalizzarli), con nome file che indica etichetta e box di
    origine. Serve per controllare a occhio la qualita' del negative mining
    PRIMA di fidarsi e lanciare il training vero.
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

    print(f"[Preview] Salvati {n_pos_done} positivi in {out_dir / 'positivi'}")
    print(f"[Preview] Salvati {n_neg_done} negativi in {out_dir / 'negativi'}")
    print(f"[Preview] Controlla a occhio i negativi prima di lanciare il training vero!")


def main(args):
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
    print(f"[Setup] Carico processor/modello: {SIGLIP_CKPT}")
    processor = AutoProcessor.from_pretrained(SIGLIP_CKPT)

    print("[Dati] Costruisco dataset di training...")
    train_ds = SigLIPCropDataset(
        args.train_images, args.train_labels, processor,
        padding=args.padding, neg_ratio=args.neg_ratio, seed=42,
    )
    print("[Dati] Costruisco dataset di validazione...")
    val_ds = SigLIPCropDataset(
        args.val_images, args.val_labels, processor,
        padding=args.padding, neg_ratio=args.neg_ratio, seed=42,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=0, pin_memory=True)

    print("[Modello] Costruisco SigLIP2 + LoRA + MLP...")
    model = SigLIPLoRAClassifier().to(device)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.vision_model.parameters()) + \
        sum(p.numel() for p in model.mlp_head.parameters())
    print(f"[Modello] Parametri allenabili: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.2f}%)")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\n[Training] Inizio — {args.epochs} epoche, batch={args.batch}, lr={args.lr}")
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

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
            print(f"\n[Early stopping] Nessun miglioramento da {args.early_stop_patience} epoche")
            break

    # ── Salva log finale — stesso schema di results.csv di Ultralytics,
    # cosi' analyze_yolo_loss_magnitude.py puo' essere riusato per L_VLM ──
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
                          "val/box_loss", "val/cls_loss", "val/dfl_loss",
                          "train_acc", "val_acc"])
        # Mappiamo la loss BCE interamente su "cls_loss" (box/dfl = 0, non
        # applicabili a un classificatore) per riusare lo stesso script di
        # analisi magnitudo gia' scritto per Fase 1, senza doverne fare uno
        # nuovo da zero.
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i], 0.0, history["train_loss"][i], 0.0,
                0.0, history["val_loss"][i], 0.0,
                history["train_acc"][i], history["val_acc"][i],
            ])

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Completato] Best val_loss={best_val_loss:.4f}")
    print(f"[Output] {out_dir / 'results.csv'}  (compatibile con analyze_yolo_loss_magnitude.py)")
    print(f"[Output] {out_dir / 'best/'}  (adapter LoRA + MLP head migliori)")


def parse_args():
    parser = argparse.ArgumentParser(description="Fase 3: training LoRA di SigLIP2 (vision-only) + MLP")
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_labels", type=str, required=True)
    parser.add_argument("--val_images", type=str, default=None,
                         help="Richiesto solo se non usi --preview_dir")
    parser.add_argument("--val_labels", type=str, default=None,
                         help="Richiesto solo se non usi --preview_dir")
    parser.add_argument("--out_dir", type=str, default="weights/fase3_siglip_lora")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--padding", type=float, default=0.15,
                         help="Padding relativo del crop, coerente con Fase 2 (0.15 = 15%%)")
    parser.add_argument("--neg_ratio", type=float, default=1.0,
                         help="Rapporto negativi sintetici per positivo, per immagine")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=8)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--preview_dir", type=str, default=None,
                         help="Se impostato, NON allena: salva solo N crop positivi/negativi "
                              "in questa cartella per controllo visivo, poi esce")
    parser.add_argument("--preview_n", type=int, default=60,
                         help="Numero di crop da esportare per classe in modalita' --preview_dir")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.preview_dir and (not args.val_images or not args.val_labels):
        raise ValueError("--val_images e --val_labels sono obbligatori se non usi --preview_dir")
    main(args)