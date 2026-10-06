"""
src/evaluation/evaluate_fase2_siglip.py
==========================================
Valutazione completa della pipeline Fase 2 (YOLO frozen + SigLIP2 frozen
zero-shot), replicando lo schema di valutazione gia' usato per Cosmos
(evaluate_yolo_cosmos_v2.py), adattato a SigLIP2.

Valuta 4 pipeline in un colpo solo:
    2 detector (aug / noaug) x 2 prompt set (no_context / context)
su una griglia di:
    conf     = [0.25, 0.5]
    iou_thr  = [0.0, 0.25, 0.5, 0.75]

Differenze rispetto alla versione Cosmos (dichiarate esplicitamente, vedi
discussione in chat):
  - Input a SigLIP2: SOLO il crop con padding 15% (non l'immagine intera
    con box disegnato + coordinate testuali) — per SigLIP2 il crop e' la
    configurazione che funziona (vedi paper DRIVE), l'overlay peggiora
    drasticamente le prestazioni di un modello contrastivo.
  - "context" per SigLIP2 NON e' un'istruzione step-by-step (SigLIP non
    "ragiona"): e' un confronto multi-classe fra la didascalia "empty" e
    piu' didascalie candidate (full + distrattori strutturali), la
    decisione e' "empty vince su tutte le altre", niente negazione
    testuale (i modelli contrastivi la gestiscono male).
  - SigLIP risponde con un punteggio di similarita' (sigmoid), non con
    testo libero "yes"/"no" — qui "yes"/"no" sono etichette derivate
    (yes = SigLIP conferma "empty", no = SigLIP la rigetta).

Metodologia di conteggio (identica alla versione Cosmos):
  - multi-match: una pred "matcha" una GT se IoU >= soglia (o IoU>0 se
    soglia=0), nessun vincolo 1-a-1.
  - TPR/FPR/FNR normalizzate su #GT (non su #pred).
  - Casistiche YES/NO incrociate con TP/FP reali di YOLO.

Uso:
    python src/evaluation/evaluate_fase2_siglip.py \
        --aug_weights weights/fase1_baseline/aug/best.pt \
        --noaug_weights weights/fase1_baseline/no_aug/best.pt \
        --test_images dataset_finale/test/images \
        --test_labels dataset_finale/test/labels \
        --out_dir output/fase2_evaluation_full \
        --conf 0.25 0.5 \
        --iou_thr 0.0 0.25 0.5 0.75

Output in --out_dir:
    metrics_full.json           - tutte le combinazioni conf x iou x det x prompt
    metrics_full_summary.txt    - stesso contenuto in formato tabellare leggibile

NOTA: questo script produce SOLO le metriche grezze (JSON + txt). Per generare
le tabelle/grafici (identici in stile a quelli del progetto Cosmos), lancia
in un secondo momento:
    python src/evaluation/visualize_fase2_siglip.py \
        --json_path <out_dir>/metrics_full.json \
        --out_dir <out_dir>/vis \
        --prompt context
"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoModel, AutoProcessor


# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

SIGLIP_CKPT = "google/siglip2-giant-opt-patch16-384"
CROP_PADDING = 0.15  # coerente con fase2_siglip_frozen.py

CONF_THRESHOLDS_DEFAULT = [0.25, 0.5]
IOU_THRESHOLDS_DEFAULT = [0.0, 0.25, 0.5, 0.75]

# Stessi PROMPT_SETS di fase2_siglip_frozen.py — se li modifichi la',
# aggiorna anche qui (o importa da fase2_siglip_frozen invece di duplicare,
# se preferisci un'unica fonte di verita').
PROMPT_SETS = {
    "no_context": {
        "empty": "an empty shelf with no products",
        "full": "a shelf filled with products",
        "distractors": [],
    },
    "context": {
        "empty": "a bare empty supermarket shelf, visible back wall and shadow, no items",
        "full": "a supermarket shelf stocked with visible products",
        "distractors": [
            "a metallic shelf edge or divider, no products visible",
            "a closed freezer or refrigerator glass door",
            "a blurry out of focus photo of a shelf",
            "a dark, poorly lit shelf far from the camera",
        ],
    },
}
PROMPT_NAMES = list(PROMPT_SETS.keys())

IOU_MATCH_THRESHOLD = 0.5  # non usato direttamente (multi_match gestisce le soglie), tenuto per compatibilita'


# ══════════════════════════════════════════════════════════════════════
# Utility geometriche (identiche a evaluate_yolo_cosmos_v2.py)
# ══════════════════════════════════════════════════════════════════════

def load_test_pairs(img_dir: Path, lbl_dir: Path):
    """Carica coppie immagine/label. Ogni GT box e' [x1,y1,x2,y2] in pixel."""
    pairs = []
    paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    for img_path in paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        gt_boxes = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = map(float, parts[:5])
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            gt_boxes.append([x1, y1, x2, y2])
        pairs.append({"path": img_path, "image": img, "gt": gt_boxes, "W": W, "H": H})
    return pairs


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def multi_match(pred_boxes, gt_boxes, iou_thr):
    """
    Multi-matching (stessa convenzione del progetto Cosmos):
      is_tp_pred[i] = True se pred i matcha almeno una GT
      is_fn_gt[j]   = True se GT j non e' matchata da nessuna pred
    """
    mat_pred = [[] for _ in range(len(pred_boxes))]
    mat_gt = [[] for _ in range(len(gt_boxes))]
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            v = iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi].append(gi)
                mat_gt[gi].append(pi)
    is_tp_pred = [len(lst) > 0 for lst in mat_pred]
    is_fn_gt = [len(lst) == 0 for lst in mat_gt]
    return is_tp_pred, is_fn_gt


def crop_with_padding(image: Image.Image, box, padding_frac: float):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    pad_w = w * padding_frac
    pad_h = h * padding_frac
    x1p = max(0, x1 - pad_w)
    y1p = max(0, y1 - pad_h)
    x2p = min(image.width, x2 + pad_w)
    y2p = min(image.height, y2 + pad_h)
    return image.crop((x1p, y1p, x2p, y2p))


# ══════════════════════════════════════════════════════════════════════
# SigLIP2 — classificatore zero-shot multi-prompt-set
# (carica il modello UNA sola volta, precalcola i testi per ENTRAMBI i
#  prompt set — evita di caricare due volte ~1B parametri in GPU)
# ══════════════════════════════════════════════════════════════════════

class SigLIP2MultiPromptClassifier:
    def __init__(self, ckpt: str = SIGLIP_CKPT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SigLIP2] Carico {ckpt} su {self.device}...")
        self.model = AutoModel.from_pretrained(ckpt).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(ckpt)

        # Precalcola gli input testuali tokenizzati per ogni prompt set
        self.text_inputs_by_set = {}
        for name, cfg in PROMPT_SETS.items():
            texts = [cfg["empty"], cfg["full"]] + list(cfg.get("distractors", []))
            with torch.no_grad():
                self.text_inputs_by_set[name] = {
                    "texts": texts,
                    "inputs": self.processor(
                        text=texts, padding="max_length", max_length=64,
                        return_tensors="pt",
                    ).to(self.device),
                }
            print(f"[SigLIP2] prompt_set '{name}': {len(texts)} didascalie -> {texts}")

    @torch.no_grad()
    def classify(self, crop: Image.Image, prompt_name: str):
        """
        Restituisce (is_empty: bool, winning_distractor: str | None).

        Decisione (is_empty): SOLO empty vs full (indici 0 e 1), esattamente
        come in "no_context" — i distrattori (se presenti) NON influenzano
        piu' la decisione. Motivo: con l'argmax su tutte le didascalie,
        alcuni distrattori (es. "shelf edge, no products visible")
        condividono lessico con "empty" ("no products") e vincono spesso
        anche su crop davvero vuoti, causando un crollo di TPR (vedi analisi
        in chat: FN saliti all'84% con la vecchia regola).

        winning_distractor: SOLO diagnostico (non usato per la decisione) —
        il testo del distrattore con score piu' alto sia di "empty" che di
        "full", oppure None se nessun distrattore batte entrambi. Serve a
        verificare empiricamente se e' sempre lo stesso distrattore a
        "vincere" (bias/artefatto lessicale) o se la vittoria e' distribuita
        in base al contenuto visivo reale (selettivita' genuina).
        """
        text_data = self.text_inputs_by_set[prompt_name]
        texts = text_data["texts"]
        image_inputs = self.processor(images=[crop], return_tensors="pt").to(self.device)

        outputs = self.model(
            input_ids=text_data["inputs"]["input_ids"],
            attention_mask=text_data["inputs"].get("attention_mask"),
            pixel_values=image_inputs["pixel_values"],
        )
        logits = outputs.logits_per_image  # [1, n_testi]
        probs = torch.sigmoid(logits).squeeze(0)

        score_empty = probs[0].item()
        score_full = probs[1].item()
        is_empty = score_empty > score_full

        winning_distractor = None
        if probs.numel() > 2:
            distractor_scores = probs[2:]
            best_idx = int(torch.argmax(distractor_scores).item())
            best_score = distractor_scores[best_idx].item()
            if best_score > max(score_empty, score_full):
                winning_distractor = texts[2 + best_idx]

        return is_empty, winning_distractor


# ══════════════════════════════════════════════════════════════════════
# Metriche (identiche nello spirito a evaluate_yolo_cosmos_v2.py,
# "cosmos_*" rinominato "siglip_*")
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(all_results, iou_thr, detector, prompt):
    total_gt = total_tp = total_fp = total_fn = 0
    yolo_total = siglip_yes_total = siglip_no_total = 0
    siglip_yes_were_tp = siglip_yes_were_fp = 0
    siglip_no_were_tp = siglip_no_were_fp = 0
    fn_missed_by_yolo = 0  # FN "irriducibili": GT che NESSUNA box YOLO (anche scartate) copre

    for r in all_results:
        gt_boxes = r["gt"]
        pred_boxes = r["preds"][detector]
        responses = r["responses"][detector][prompt]  # lista di bool (True=yes/empty)

        n_gt = len(gt_boxes)
        total_gt += n_gt
        yolo_total += len(pred_boxes)

        # Baseline: GT non coperta da NESSUNA box YOLO (indipendentemente da SigLIP) —
        # questa e' la parte di FN di cui SigLIP non puo' essere responsabile.
        _, is_fn_gt_baseline = multi_match(pred_boxes, gt_boxes, iou_thr)
        fn_missed_by_yolo += sum(is_fn_gt_baseline)

        is_tp_real, _ = multi_match(pred_boxes, gt_boxes, iou_thr)

        for resp, is_real_tp in zip(responses, is_tp_real):
            if resp:  # "yes" / empty confermato
                siglip_yes_total += 1
                if is_real_tp:
                    siglip_yes_were_tp += 1
                else:
                    siglip_yes_were_fp += 1
            else:  # "no" / rigettato
                siglip_no_total += 1
                if is_real_tp:
                    siglip_no_were_tp += 1
                else:
                    siglip_no_were_fp += 1

        kept_boxes = [b for b, resp in zip(pred_boxes, responses) if resp]

        if kept_boxes:
            is_tp_kept, is_fn_gt = multi_match(kept_boxes, gt_boxes, iou_thr)
            total_tp += sum(is_tp_kept)
            total_fp += sum(not tp for tp in is_tp_kept)
            total_fn += sum(is_fn_gt)
        else:
            total_fn += n_gt

    fpr = total_fp / total_gt if total_gt > 0 else 0.0
    fnr = total_fn / total_gt if total_gt > 0 else 0.0
    tpr = total_tp / total_gt if total_gt > 0 else 0.0

    # Delta FN attribuibile specificamente al filtro SigLIP (non a YOLO):
    # entrambi i termini sono conteggi "per GT", quindi la sottrazione e' valida
    # (a differenza dei conteggi per-box come siglip_yes/no_were_*, dove il
    # multi-match puo' generare doppi conteggi se piu' box matchano la stessa GT).
    fn_rejected_by_siglip = total_fn - fn_missed_by_yolo

    return {
        "FPR": fpr, "FNR": fnr, "TPR": tpr,
        "TP": total_tp, "FP": total_fp, "FN": total_fn, "GT": total_gt,
        "yolo_total": yolo_total,
        "siglip_yes_total": siglip_yes_total,
        "siglip_no_total": siglip_no_total,
        "siglip_yes_were_tp": siglip_yes_were_tp,
        "siglip_yes_were_fp": siglip_yes_were_fp,
        "siglip_no_were_tp": siglip_no_were_tp,
        "siglip_no_were_fp": siglip_no_were_fp,
        "fn_missed_by_yolo": fn_missed_by_yolo,
        "fn_rejected_by_siglip": fn_rejected_by_siglip,
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aug_weights", required=True)
    parser.add_argument("--noaug_weights", required=True)
    parser.add_argument("--test_images", required=True)
    parser.add_argument("--test_labels", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--conf", nargs="+", type=float, default=CONF_THRESHOLDS_DEFAULT)
    parser.add_argument("--iou_thr", nargs="+", type=float, default=IOU_THRESHOLDS_DEFAULT)
    parser.add_argument("--padding", type=float, default=CROP_PADDING)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("  Valutazione pipeline YOLO + SigLIP2 (Fase 2) — griglia completa")
    print("=" * 70)

    print(f"[1/3] Carico detector aug   <- {args.aug_weights}")
    yolo_aug = YOLO(args.aug_weights)
    print(f"[2/3] Carico detector noaug <- {args.noaug_weights}")
    yolo_noaug = YOLO(args.noaug_weights)
    detectors = {"aug": yolo_aug, "noaug": yolo_noaug}

    print(f"[3/3] Carico SigLIP2 (entrambi i prompt set precalcolati)")
    siglip = SigLIP2MultiPromptClassifier(device=args.device)

    pairs = load_test_pairs(Path(args.test_images), Path(args.test_labels))
    if args.max_images:
        pairs = pairs[:args.max_images]
    print(f"\n  Immagini test : {len(pairs)}")
    print(f"  CONF          : {args.conf}")
    print(f"  IOU           : {args.iou_thr}")
    print(f"  PROMPT SETS   : {PROMPT_NAMES}")

    # ── YOLO inference (una volta per conf x detector) ───────────────────
    print("\nFase 1/2: YOLO inference...")
    all_results = {}
    for conf in args.conf:
        results_conf = []
        for sample in pairs:
            entry = {"path": sample["path"], "gt": sample["gt"], "image": sample["image"],
                      "preds": {}, "responses": {}}
            for det_name, yolo in detectors.items():
                res = yolo(sample["path"], conf=conf, verbose=False)
                boxes = res[0].boxes.xyxy.cpu().numpy().tolist() if res and res[0].boxes is not None else []
                entry["preds"][det_name] = boxes
                entry["responses"][det_name] = {}
            results_conf.append(entry)
        all_results[conf] = results_conf
        print(f"  conf={conf}: YOLO completato su {len(results_conf)} immagini")

    # ── SigLIP2 inference (per ogni conf x detector x prompt) ────────────
    print("\nFase 2/2: SigLIP2 inference (crop con padding)...")
    PROGRESS_EVERY = 50
    from collections import Counter
    distractor_stats = {}  # (conf, det, prompt) -> (Counter per etichetta, n_boxes_totali)
    for conf in args.conf:
        for det_name in detectors:
            for prompt_name in PROMPT_NAMES:
                print(f"  det={det_name}  conf={conf}  prompt='{prompt_name}'")
                yes_count = no_count = boxes_done = 0
                distractor_win_counter = Counter()
                n_images = len(all_results[conf])
                for img_idx, entry in enumerate(all_results[conf], start=1):
                    boxes = entry["preds"][det_name]
                    image = entry["image"]
                    resps = []
                    for box in boxes:
                        crop = crop_with_padding(image, box, args.padding)
                        is_empty, winning_distractor = siglip.classify(crop, prompt_name)
                        resps.append(is_empty)
                        if is_empty:
                            yes_count += 1
                        else:
                            no_count += 1
                        if winning_distractor is not None:
                            distractor_win_counter[winning_distractor] += 1
                    entry["responses"][det_name][prompt_name] = resps
                    boxes_done += len(boxes)
                    if img_idx % PROGRESS_EVERY == 0 or img_idx == n_images:
                        print(f"    [{img_idx}/{n_images}] box_tot={boxes_done} "
                              f"YES={yes_count} NO={no_count}", flush=True)
                distractor_stats[(conf, det_name, prompt_name)] = (distractor_win_counter, boxes_done)
                total_distractor_wins = sum(distractor_win_counter.values())
                if boxes_done > 0 and prompt_name != "no_context":
                    pct = 100 * total_distractor_wins / boxes_done
                    print(f"    [diagnostico, non usato per la decisione] "
                          f"un distrattore avrebbe vinto su {total_distractor_wins}/{boxes_done} "
                          f"box ({pct:.1f}%). Distribuzione per etichetta:")
                    for label, count in distractor_win_counter.most_common():
                        label_pct = 100 * count / total_distractor_wins if total_distractor_wins else 0
                        print(f"        {count:5d} ({label_pct:5.1f}% delle vittorie distrattore) -> \"{label}\"")

    # ── Metriche su tutta la griglia ──────────────────────────────────────
    print("\nCalcolo metriche...")
    output = {}
    for conf in args.conf:
        for iou_t in args.iou_thr:
            for det_name in detectors:
                for prompt_name in PROMPT_NAMES:
                    key = f"conf={conf}_iou={iou_t}_{det_name}_{prompt_name}"
                    m = compute_metrics(all_results[conf], iou_t, det_name, prompt_name)
                    m.update({"conf": conf, "iou": iou_t, "detector": det_name, "prompt": prompt_name})
                    output[key] = m
                    print(f"  {det_name:6s} {prompt_name:12s} conf={conf} iou={iou_t} | "
                          f"TPR={m['TPR']:.3f} FPR={m['FPR']:.3f} FNR={m['FNR']:.3f}")

    with open(out_dir / "metrics_full.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Output] {out_dir / 'metrics_full.json'}")

    # Diagnostica distrattori (non usata dalla pipeline, solo per capire quanto
    # e QUALE distrattore avrebbe interferito se fosse stato considerato nella
    # decisione — utile per distinguere bias lessicale sistematico (sempre la
    # stessa etichetta a vincere) da selettivita' genuina distribuita sul
    # contenuto visivo).
    distractor_diag = {
        f"conf={c}_{det}_{p}": {
            "n_boxes": n,
            "total_distractor_wins": sum(counter.values()),
            "wins_by_label": dict(counter.most_common()),
        }
        for (c, det, p), (counter, n) in distractor_stats.items()
    }
    with open(out_dir / "distractor_diagnostics.json", "w") as f:
        json.dump(distractor_diag, f, indent=2)
    print(f"[Output] {out_dir / 'distractor_diagnostics.json'} (solo diagnostico)")

    with open(out_dir / "metrics_full_summary.txt", "w") as f:
        f.write(f"{'detector':8s} | {'prompt':12s} | {'conf':4s} | {'iou':4s} | "
                f"{'TPR':5s} | {'FPR':5s} | {'FNR':5s} | {'YOLO_tot':8s} | "
                f"{'YES':5s} | {'NO':5s}\n")
        f.write("-" * 100 + "\n")
        for key, m in output.items():
            f.write(f"{m['detector']:8s} | {m['prompt']:12s} | {m['conf']:.2f} | {m['iou']:.2f} | "
                    f"{m['TPR']:.3f} | {m['FPR']:.3f} | {m['FNR']:.3f} | {m['yolo_total']:8d} | "
                    f"{m['siglip_yes_total']:5d} | {m['siglip_no_total']:5d}\n")
    print(f"[Output] {out_dir / 'metrics_full_summary.txt'}")

    print("\nCompletato. Metriche grezze salvate — per generare tabelle e grafici, lancia:")
    print(f"  python src/evaluation/visualize_fase2_siglip.py --json_path {out_dir / 'metrics_full.json'} --out_dir {out_dir}/vis")


if __name__ == "__main__":
    main()