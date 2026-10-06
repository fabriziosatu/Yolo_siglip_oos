"""
src/training/train_fase3_full_lora.py
=======================================
FASE 3 (variante B) — Fine-tuning di SigLIP2 COMPLETO (vision + text tower)
tramite LoRA su entrambe le torri, per la verifica "scaffale vuoto si/no".

Architettura (contrastiva, coerente con Fase 2 zero-shot ma qui ALLENATA):
    crop (padding 15%, come Fase 2/3-MLP)
        -> SigLIP2 vision tower, FROZEN + adapter LoRA (rank r_visual)
        -> pooled embedding visivo v, normalizzato
    prompt testuali ("an empty shelf" / "a shelf full of products")
        -> SigLIP2 text tower, FROZEN + adapter LoRA (rank r_text)
        -> embedding testuali precomputati (ma non piu' frozen: gli
           adapter LoRA sul text encoder si allenano anch'essi)
    logit_pos = v . text_pos * logit_scale + logit_bias
    logit_neg = v . text_neg * logit_scale + logit_bias
        -> SigLIPSigmoidLoss(logit_pos, logit_neg, label)

Rispetto alla variante MLP (train_fase3_lora.py):
    - NON sostituisce il text tower con un MLP: lo mantiene, allenandolo
      anch'esso con LoRA (rank piu' piccolo, r_text = r_visual/2 di default,
      stessa convenzione gia' usata in src/models/siglip_module.py per la
      Fase 4).
    - La loss e' la SigLIPSigmoidLoss di src/training/losses.py (identica
      nella forma a quella usata nella pipeline joint), non una BCE su MLP.
    - Serve per avere pesi pronti anche per questa architettura, nel caso
      la prof preferisca tenere il text tower invece di sostituirlo con
      un MLP (come attualmente e' impostato joint_pipeline.py/Fase 4).

Dati di training: IDENTICI alla variante MLP (stesso positive/negative
mining, stesso padding, stesso --neg_ratio) — riusa SigLIPCropDataset,
ProcessorCollate, crop_with_padding, generate_negative_box, load_yolo_boxes
senza alcuna modifica: sono agnostiche rispetto al modello.

Uso:
    python src/training/train_fase3_full_lora.py \
        --train_images dataset_finale/train/images \
        --train_labels dataset_finale/train/labels \
        --val_images   dataset_finale/val/images \
        --val_labels   dataset_finale/val/labels \
        --out_dir weights/fase3_siglip_lora_full \
        --epochs 30 --batch 16 --lr 1e-4 --padding 0.15 --neg_ratio 0.5 \
        --lora_rank 8 --lora_alpha 16

Output in --out_dir:
    lora_adapters_visual/    - adapter LoRA del visual encoder
    lora_adapters_text/      - adapter LoRA del text encoder
    scale_bias.pt            - logit_scale/logit_bias (allenabili anch'essi)
    results.csv               - stesso schema della variante MLP, per riuso
                                diretto di analyze_yolo_loss_magnitude.py
    training_history.json
    best_epoch_info.json
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from peft import LoraConfig, get_peft_model


SIGLIP_CKPT = "google/siglip2-giant-opt-patch16-384"
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # coerente col paper DRIVE (WQ, WV)
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Prompt "no_context" — stessa coppia usata in Fase 2 (quella che si e'
# comportata bene; "context" con distrattori aveva collassato, vedi
# discussione precedente sul crollo di TPR).
PROMPT_POSITIVE = ["an empty shelf", "a retail shelf with no products",
                    "an out-of-stock shelf section"]
PROMPT_NEGATIVE = ["a shelf full of products", "a well-stocked retail shelf",
                    "a shelf with items on it"]


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
    Genera una box casuale con dimensioni comparabili a `ref_boxes` (le GT
    della stessa immagine), che non si sovrapponga (IoU<0.1) a nessuna GT.

    Vincolo di posizione (fascia orizzontale dello scaffale): la coordinata
    verticale del negativo viene limitata alla fascia [y_min, y_max] coperta
    dalle GT box della stessa immagine (con un margine del 20%), non lasciata
    libera su tutta l'immagine. Motivo: gli scaffali occupano tipicamente una
    fascia orizzontale della foto (la telecamera e' circa alla loro altezza);
    fuori da quella fascia c'e' tipicamente soffitto, pavimento o corridoio —
    zone che non sono "scaffale pieno" e andrebbero a inquinare l'etichetta
    negativa con falsi esempi. La coordinata orizzontale x resta libera su
    tutta la larghezza, perche' lungo lo scaffale e' normale trovare sia
    zone vuote sia zone piene.

    Ritorna None se non trova una posizione valida entro max_tries.
    """
    if not ref_boxes:
        return None

    # Fascia verticale coperta dalle GT, con margine 20%
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
        # y1 vincolata dentro la fascia scaffale, non su tutta l'immagine
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
            crop = crop_with_padding(im, box, self.padding).copy()
        # NON chiamiamo piu' il processor qui: lo facciamo una volta sola
        # per l'intero batch dentro ProcessorCollate (molto piu' efficiente
        # che processare un'immagine alla volta, specialmente con
        # num_workers>0 dove ogni worker farebbe la stessa cosa in serie).
        return crop, torch.tensor(label, dtype=torch.float32)


class ProcessorCollate:
    """
    collate_fn che raggruppa un intero batch di crop PIL e li passa al
    processor SigLIP UNA SOLA VOLTA (invece che uno alla volta dentro
    __getitem__) — molto piu' efficiente, e necessario per usare
    num_workers>0 in modo sensato.
    """
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):
        crops = [item[0] for item in batch]
        labels = torch.stack([item[1] for item in batch])
        inputs = self.processor(images=crops, return_tensors="pt")
        return inputs["pixel_values"], labels


# ══════════════════════════════════════════════════════════════════════
# Modello: SigLIP2 completo (vision + text) + LoRA su entrambe le torri
# ══════════════════════════════════════════════════════════════════════

class SigLIPSigmoidLoss(nn.Module):
    """Identica a src/training/losses.py — reimplementata qui per rendere
    lo script autosufficiente (nessuna dipendenza da src/ se lanciato da
    una cartella diversa sul cluster)."""

    def forward(self, logits_pos, logits_neg, roi_labels):
        labels = roi_labels.float().unsqueeze(1)
        loss_pos = F.binary_cross_entropy_with_logits(logits_pos, labels, reduction="mean")
        loss_neg = F.binary_cross_entropy_with_logits(logits_neg, 1.0 - labels, reduction="mean")
        return (loss_pos + loss_neg) / 2.0


class SigLIPFullLoRAModel(nn.Module):
    """
    SigLIP2 completo (vision + text), LoRA su entrambe le torri.
    Stessa logica di src/models/siglip_module.py, ma con:
      - checkpoint 'giant-opt-patch16-384' (coerente con Fase 2/3-MLP,
        invece del 'base-patch16-224' di default in siglip_module.py)
      - supporto bf16 per il backbone (stessa scelta della variante MLP)
      - rank del text encoder derivato da quello visivo (r_text = r/2),
        stessa convenzione gia' in uso in src/models/siglip_module.py
    """

    def __init__(self, ckpt: str = SIGLIP_CKPT, use_bf16: bool = True,
                 lora_r: int = LORA_RANK, lora_alpha: int = LORA_ALPHA,
                 lora_r_text: int = None, lora_alpha_text: int = None,
                 lora_dropout: float = LORA_DROPOUT):
        super().__init__()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        full_model = AutoModel.from_pretrained(ckpt, torch_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt)

        lora_r_text = lora_r_text if lora_r_text is not None else max(1, lora_r // 2)
        lora_alpha_text = lora_alpha_text if lora_alpha_text is not None else max(1, lora_alpha // 2)

        lora_cfg_visual = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=LORA_TARGET_MODULES, bias="none",
        )
        self.visual_encoder = get_peft_model(full_model.vision_model, lora_cfg_visual)

        lora_cfg_text = LoraConfig(
            r=lora_r_text, lora_alpha=lora_alpha_text, lora_dropout=lora_dropout,
            target_modules=LORA_TARGET_MODULES, bias="none",
        )
        self.text_encoder = get_peft_model(full_model.text_model, lora_cfg_text)

        # logit_scale/logit_bias: parametri pretrainati di SigLIP, qui
        # TRAINABILI (a differenza della Fase 2 frozen) — fanno parte
        # dell'adattamento fine del training contrastivo.
        self.logit_scale = nn.Parameter(full_model.logit_scale.detach().clone().float())
        self.logit_bias = nn.Parameter(full_model.logit_bias.detach().clone().float())

        self.text_embeds_pos = None
        self.text_embeds_neg = None

    @torch.no_grad()
    def encode_prompts(self, positive_prompts, negative_prompts, device):
        """Precomputa gli embedding testuali INIZIALI. Vengono ricalcolati
        (non solo precomputati una volta) ad ogni epoca nel training loop,
        perche' a differenza della Fase 2 il text encoder qui e' allenabile
        — i suoi embedding cambiano man mano che gli adapter si aggiornano."""
        return self._encode(positive_prompts, device), self._encode(negative_prompts, device)

    def _encode(self, prompts, device):
        tokens = self.tokenizer(prompts, return_tensors="pt", padding="max_length",
                                 truncation=True, max_length=64).to(device)
        text_dtype = next(self.text_encoder.parameters()).dtype
        out = self.text_encoder(**{k: v for k, v in tokens.items()})
        embeds = out.pooler_output.float()
        embeds = F.normalize(embeds, dim=-1)
        return F.normalize(embeds.mean(dim=0, keepdim=True), dim=-1)  # (1, 768)

    def forward(self, pixel_values, positive_prompts, negative_prompts):
        device = pixel_values.device
        backbone_dtype = next(self.visual_encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=backbone_dtype)

        vis_out = self.visual_encoder(pixel_values=pixel_values)
        v = F.normalize(vis_out.pooler_output.float(), dim=-1)  # (B, 768)

        # Il text encoder e' allenabile: ricalcoliamo i suoi embedding ad
        # ogni forward (niente cache statica come in Fase 2 frozen).
        text_pos = self._encode(positive_prompts, device)  # (1, 768)
        text_neg = self._encode(negative_prompts, device)  # (1, 768)

        sim_pos = v @ text_pos.T  # (B, 1)
        sim_neg = v @ text_neg.T  # (B, 1)

        logits_pos = sim_pos * self.logit_scale.exp() + self.logit_bias
        logits_neg = sim_neg * self.logit_scale.exp() + self.logit_bias
        return logits_pos, logits_neg

    def trainable_parameters(self):
        params = [p for p in self.visual_encoder.parameters() if p.requires_grad]
        params += [p for p in self.text_encoder.parameters() if p.requires_grad]
        params += [self.logit_scale, self.logit_bias]
        return params

    def save(self, out_dir: Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.visual_encoder.save_pretrained(str(out_dir / "lora_adapters_visual"))
        self.text_encoder.save_pretrained(str(out_dir / "lora_adapters_text"))
        torch.save({"logit_scale": self.logit_scale.detach().cpu(),
                     "logit_bias": self.logit_bias.detach().cpu()},
                    out_dir / "scale_bias.pt")


# ══════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, train: bool,
              progress_every: int = 50, epoch_label: str = ""):
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

            logits_pos, logits_neg = model(pixel_values, PROMPT_POSITIVE, PROMPT_NEGATIVE)
            loss = criterion(logits_pos, logits_neg, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            # Decisione: "vuoto" vince se il logit positivo batte il negativo
            # (stessa regola no_context gia' usata e validata in Fase 2)
            preds = (logits_pos.squeeze(-1) > logits_neg.squeeze(-1)).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if progress_every and (batch_idx % progress_every == 0 or batch_idx == n_total_batches):
                elapsed = time.time() - t_start
                batch_per_sec = batch_idx / elapsed if elapsed > 0 else 0
                eta_sec = (n_total_batches - batch_idx) / batch_per_sec if batch_per_sec > 0 else 0
                print(f"    [{epoch_label}] batch {batch_idx}/{n_total_batches} "
                      f"({batch_per_sec:.2f} batch/s, ETA {eta_sec/60:.1f} min) "
                      f"loss_corrente={total_loss/n_batches:.4f}", flush=True)

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

    collate_fn = ProcessorCollate(processor)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               collate_fn=collate_fn, persistent_workers=(args.num_workers > 0))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             collate_fn=collate_fn, persistent_workers=(args.num_workers > 0))

    print("[Modello] Costruisco SigLIP2 completo (vision+text) + LoRA su entrambe le torri...")
    lora_r_text = args.lora_rank_text if args.lora_rank_text is not None else max(1, args.lora_rank // 2)
    lora_alpha_text = args.lora_alpha_text if args.lora_alpha_text is not None else max(1, args.lora_alpha // 2)
    print(f"[Modello] LoRA visual: rank={args.lora_rank}, alpha={args.lora_alpha} "
          f"(rapporto={args.lora_alpha/args.lora_rank:.1f})")
    print(f"[Modello] LoRA text  : rank={lora_r_text}, alpha={lora_alpha_text}")
    model = SigLIPFullLoRAModel(
        lora_r=args.lora_rank, lora_alpha=args.lora_alpha,
        lora_r_text=lora_r_text, lora_alpha_text=lora_alpha_text,
        lora_dropout=args.lora_dropout,
    ).to(device)

    n_trainable = sum(p.numel() for p in model.trainable_parameters())
    n_total = (sum(p.numel() for p in model.visual_encoder.parameters()) +
               sum(p.numel() for p in model.text_encoder.parameters()))
    print(f"[Modello] Parametri allenabili: {n_trainable:,} / {n_total:,} "
          f"({100*n_trainable/n_total:.2f}%)")

    criterion = SigLIPSigmoidLoss()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    # ── results.csv scritto epoca per epoca (non solo a fine training) —
    # cosi' se il job viene interrotto a meta' (scancel, timeout, crash) le
    # epoche gia' completate restano su disco invece di andare perse.
    results_csv_path = out_dir / "results.csv"
    csv_file = open(results_csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
                          "val/box_loss", "val/cls_loss", "val/dfl_loss",
                          "train_acc", "val_acc"])
    csv_file.flush()

    print(f"\n[Training] Inizio — {args.epochs} epoche, batch={args.batch}, lr={args.lr}")
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
            print(f"\n[Early stopping] Nessun miglioramento da {args.early_stop_patience} epoche")
            break

    csv_file.close()

    print(f"\n[Completato] Best val_loss={best_val_loss:.4f}")
    print(f"[Output] {out_dir / 'results.csv'}  (compatibile con analyze_yolo_loss_magnitude.py)")
    print(f"[Output] {out_dir / 'best/'}  (adapter LoRA visual+text, logit_scale/bias migliori)")


def parse_args():
    parser = argparse.ArgumentParser(description="Fase 3 (variante B): training LoRA di SigLIP2 completo (vision+text)")
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_labels", type=str, required=True)
    parser.add_argument("--val_images", type=str, default=None,
                         help="Richiesto solo se non usi --preview_dir")
    parser.add_argument("--val_labels", type=str, default=None,
                         help="Richiesto solo se non usi --preview_dir")
    parser.add_argument("--out_dir", type=str, default="weights/fase3_siglip_lora_full")
    parser.add_argument("--epochs", type=int, default=50,
                         help="Tetto massimo — l'early stopping di solito ferma prima se converge")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--padding", type=float, default=0.15,
                         help="Padding relativo del crop, coerente con Fase 2 (0.15 = 15%%)")
    parser.add_argument("--neg_ratio", type=float, default=1.0,
                         help="Rapporto negativi sintetici per positivo, per immagine")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=12)
    parser.add_argument("--lora_rank", type=int, default=LORA_RANK,
                         help="Rank LoRA del VISUAL encoder (default 8)")
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA,
                         help="Scaling LoRA del visual encoder (default 16, rapporto 2 con rank=8)")
    parser.add_argument("--lora_rank_text", type=int, default=None,
                         help="Rank LoRA del TEXT encoder (default: lora_rank/2, "
                              "stessa convenzione di src/models/siglip_module.py)")
    parser.add_argument("--lora_alpha_text", type=int, default=None,
                         help="Scaling LoRA del text encoder (default: lora_alpha/2)")
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT,
                         help="Dropout sugli adapter LoRA, sia visual che text (default 0.05)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                         help="Weight decay dell'ottimizzatore AdamW (default 1e-4)")
    parser.add_argument("--num_workers", type=int, default=6,
                         help="Worker paralleli per il DataLoader (usa gli 8 CPU allocati "
                              "dal job SLURM). 0 = tutto in serie nel processo principale "
                              "(molto piu' lento, non consigliato)")
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