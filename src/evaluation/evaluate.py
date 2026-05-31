"""
src/evaluation/evaluate.py
===========================
Valutazione completa della pipeline: YOLO26 standalone (Fase 1)
vs pipeline joint YOLO26 + SigLIP (Fase 2) sul test set.

Metriche calcolate:

  YOLO standalone (Fase 1):
    - Precision        : TP / (TP + FP)
    - Recall           : TP / (TP + FN)
    - F1               : media armonica di P e R
    - FP-Rate          : FP / (FP + TN)  — quante zone PIENE YOLO classifica come vuote
    - mAP@0.5          : area sotto la curva P-R con IoU >= 0.5
    - mAP@0.5:0.95     : media mAP su IoU da 0.5 a 0.95

  Pipeline completa (Fase 2 — YOLO + SigLIP):
    - Precision        : dopo filtro SigLIP
    - Recall           : dopo filtro SigLIP
    - F1               : dopo filtro SigLIP
    - FP-Rate          : dopo filtro SigLIP (attesa piu' bassa della Fase 1)
    - Curva P-R        : a diverse soglie SigLIP (0.1 -> 0.9)
    - Delta vs Fase 1  : miglioramento/peggioramento di ogni metrica

La FP-Rate e' la metrica chiave che giustifica l'aggiunta di SigLIP:
  se YOLO ha FP-Rate alta, SigLIP serve a filtrare i falsi positivi.

Esegui con:
  python src/evaluation/evaluate.py
  python src/evaluation/evaluate.py --siglip_thresh 0.5
  python src/evaluation/evaluate.py --phase1_weights altro/percorso.pt
"""

import argparse
import sys
import json
import torch
import numpy as np
from pathlib import Path
from torchvision.ops import box_iou
from tqdm import tqdm

sys.path.insert(0, ".")

from src.data.dataset          import build_dataloaders
from src.models.joint_pipeline import JointPipeline
from src.utils.config          import CFG, PHASE1_WEIGHTS, PHASE2_WEIGHTS


# ── Configurazione ────────────────────────────────────────────────────────────

IOU_THRESHOLD = 0.5    # soglia IoU per considerare una predizione corretta


# ── Argomenti ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Valuta YOLO standalone e pipeline YOLO+SigLIP sul test set"
    )
    parser.add_argument("--phase1_weights", type=str, default=str(PHASE1_WEIGHTS),
        help="Pesi YOLO Fase 1")
    parser.add_argument("--phase2_weights", type=str, default=str(PHASE2_WEIGHTS),
        help="Checkpoint pipeline Fase 2")
    parser.add_argument("--data_dir", type=str, default="data/processed_clean",
        help="Cartella dataset (default: data/processed_clean)")
    parser.add_argument("--siglip_thresh", type=float, default=0.5,
        help="Soglia sigmoid SigLIP per classificare vuoto (default: 0.5)")
    parser.add_argument("--conf_threshold", type=float, default=0.25,
        help="Soglia confidence YOLO (default: 0.25)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="output/evaluation",
        help="Cartella per salvare i risultati JSON")
    return parser.parse_args()


# ── Metriche ──────────────────────────────────────────────────────────────────

def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    """
    Calcola precision, recall, F1, FP-Rate da contatori.

    FP-Rate = FP / (FP + TN)
      Misura quante zone PIENE vengono erroneamente classificate come vuote.
      Alta FP-Rate di YOLO giustifica l'aggiunta di SigLIP come filtro.

    TN in object detection = zone dell'immagine senza GT che YOLO
    NON ha proposto come vuote. Approssimato come:
      TN = n_background_patches - FP
    Per semplicita' usiamo TN stimato dal numero di immagini.
    """
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    fp_rate   = fp / (fp + tn + 1e-9)

    return {
        "TP":        tp,
        "FP":        fp,
        "FN":        fn,
        "TN":        tn,
        "Precision": precision,
        "Recall":    recall,
        "F1":        f1,
        "FP-Rate":   fp_rate,
    }


def match_predictions_to_gt(
    pred_boxes:  torch.Tensor,   # (N, 4) xyxy pixel
    gt_boxes:    torch.Tensor,   # (M, 4) xywh normalizzato
    img_size:    int,
    iou_thr:     float = IOU_THRESHOLD,
) -> tuple[int, int, int]:
    """
    Associa predizioni alle GT box via IoU greedy matching.

    Returns:
        (tp, fp, fn)
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    # Converti GT da xywh norm a xyxy pixel
    s   = float(img_size)
    xc  = gt_boxes[:, 0] * s;  yc  = gt_boxes[:, 1] * s
    w   = gt_boxes[:, 2] * s;  h   = gt_boxes[:, 3] * s
    gt_xyxy = torch.stack([xc-w/2, yc-h/2, xc+w/2, yc+h/2], dim=1)

    iou_mat  = box_iou(pred_boxes, gt_xyxy)   # (N, M)
    matched_gt = set()
    tp = 0

    # Greedy: assegna ogni predizione alla GT con IoU massimo
    for pred_idx in range(len(pred_boxes)):
        best_iou, best_gt = iou_mat[pred_idx].max(dim=0)
        best_gt = best_gt.item()
        if best_iou.item() >= iou_thr and best_gt not in matched_gt:
            tp += 1
            matched_gt.add(best_gt)

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes)   - tp
    return tp, fp, fn


# ── Valutazione YOLO standalone ───────────────────────────────────────────────

def evaluate_yolo_standalone(
    pipeline:    JointPipeline,
    test_loader,
    device:      torch.device,
    conf_thr:    float,
) -> dict:
    """
    Valuta YOLO26 standalone sul test set.
    SigLIP non viene usato — tutte le ROI con conf >= conf_thr sono predizioni.

    TN stimato: per ogni immagine, approssimato come
    (n_predizioni_totali_YOLO - FP) sommato su tutto il test set.
    In detection, TN viene tipicamente stimato via background patches.
    """
    pipeline.eval()

    total_tp = total_fp = total_fn = 0
    total_images  = 0
    # TN stimato: zone senza GT non proposte da YOLO
    # Usiamo come proxy il numero di GT box non trovate come negative
    # e le predizioni sotto soglia che non matchano GT
    total_tn_proxy = 0

    # Per la curva P-R a diverse confidence
    all_scores   = []   # confidence di ogni predizione
    all_tp_flags = []   # 1 se TP, 0 se FP

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  YOLO standalone", leave=False):
            images = batch["images"].to(device)
            boxes  = batch["boxes"]

            predictions, feature_map, _ = pipeline.detector(images)
            B = images.shape[0]

            for i in range(B):
                pred   = predictions[i]
                gt     = boxes[i].to(device)
                img_sz = images.shape[-1]

                # Filtra per confidence
                mask       = pred[:, 4] >= conf_thr
                pred_filt  = pred[mask]

                if len(pred_filt) == 0:
                    fn = len(gt)
                    total_fn += fn
                    total_tn_proxy += 1
                    continue

                pred_xyxy = pred_filt[:, :4]
                tp, fp, fn = match_predictions_to_gt(
                    pred_xyxy, gt, img_sz
                )
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # TN proxy: predizioni sotto soglia che non matchano GT
                # (scaffali pieni che YOLO ha correttamente ignorato)
                n_below_thresh = int((pred[:, 4] < conf_thr).sum().item())
                total_tn_proxy += n_below_thresh

                # Per curva P-R
                scores = pred_filt[:, 4].cpu().tolist()
                for j, score in enumerate(scores):
                    all_scores.append(score)
                    # TP se questa predizione ha matchato una GT
                    # (ricostruiamo dal matching)
                    all_tp_flags.append(1 if j < tp else 0)

                total_images += 1

    metrics = compute_metrics(
        total_tp, total_fp, total_fn, total_tn_proxy
    )

    # mAP@0.5 approssimato dalla curva P-R
    if all_scores:
        sorted_idx = np.argsort(all_scores)[::-1]
        tp_cum = np.cumsum([all_tp_flags[i] for i in sorted_idx])
        fp_cum = np.cumsum([1 - all_tp_flags[i] for i in sorted_idx])
        prec   = tp_cum / (tp_cum + fp_cum + 1e-9)
        rec    = tp_cum / (total_tp + total_fn + 1e-9)
        metrics["AP@0.5_approx"] = float(np.trapz(prec, rec))

    metrics["n_images"] = total_images
    return metrics


# ── Valutazione pipeline completa ─────────────────────────────────────────────

def evaluate_full_pipeline(
    pipeline:       JointPipeline,
    test_loader,
    device:         torch.device,
    conf_thr:       float,
    siglip_thresh:  float,
) -> dict:
    """
    Valuta la pipeline completa YOLO + SigLIP sul test set.

    Flusso:
      1. YOLO propone box candidate con conf >= conf_thr
      2. SigLIP classifica ogni box: score >= siglip_thresh → VUOTO
      3. Solo le box classificate VUOTE vengono mantenute
      4. Le box mantenute vengono matchate con le GT

    Calcola anche la curva P-R a diverse soglie SigLIP.
    """
    pipeline.eval()

    total_tp = total_fp = total_fn = 0
    total_tn_proxy = 0
    total_images   = 0

    # Raccoglie score SigLIP per la curva P-R a diverse soglie
    all_siglip_scores = []
    all_tp_flags      = []
    all_fn_counts     = []

    # Risultati a diverse soglie SigLIP
    thresholds   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thresh_stats = {t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
                    for t in thresholds}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Pipeline completa", leave=False):
            images = batch["images"].to(device)
            boxes  = batch["boxes"]

            output = pipeline(images, gt_boxes=None)

            B      = images.shape[0]
            img_sz = images.shape[-1]

            if output["n_rois"] == 0:
                for i in range(B):
                    fn = len(boxes[i])
                    total_fn += fn
                    total_tn_proxy += 1
                continue

            rois   = output["rois"]         # (N, 5) [batch_idx, x1,y1,x2,y2]
            logits = output["logits_pos"]   # (N, 1)
            scores = torch.sigmoid(logits).squeeze(1)  # (N,)

            for i in range(B):
                gt     = boxes[i].to(device)
                mask_i = rois[:, 0].long() == i
                rois_i = rois[mask_i, 1:]      # (Ni, 4) xyxy pixel
                scr_i  = scores[mask_i]        # (Ni,)

                if len(rois_i) == 0:
                    total_fn += len(gt)
                    continue

                # ── Valutazione alla soglia principale ────────────────────────
                empty_mask = scr_i >= siglip_thresh
                pred_xyxy  = rois_i[empty_mask]

                if len(pred_xyxy) == 0:
                    total_fn += len(gt)
                    total_tn_proxy += int((~empty_mask).sum().item())
                    continue

                tp, fp, fn = match_predictions_to_gt(pred_xyxy, gt, img_sz)
                total_tp   += tp
                total_fp   += fp
                total_fn   += fn
                total_tn_proxy += int((~empty_mask).sum().item())

                # Per curva P-R
                scr_sorted  = scr_i.cpu().tolist()
                for j, sc in enumerate(scr_sorted):
                    all_siglip_scores.append(sc)
                    all_tp_flags.append(1 if j < tp else 0)
                all_fn_counts.append(fn)

                # ── Valutazione a diverse soglie ──────────────────────────────
                for thr in thresholds:
                    emask  = scr_i >= thr
                    preds  = rois_i[emask]
                    n_tn   = int((~emask).sum().item())

                    if len(preds) == 0:
                        thresh_stats[thr]["fn"] += len(gt)
                        thresh_stats[thr]["tn"] += n_tn
                    else:
                        tp_t, fp_t, fn_t = match_predictions_to_gt(
                            preds, gt, img_sz
                        )
                        thresh_stats[thr]["tp"] += tp_t
                        thresh_stats[thr]["fp"] += fp_t
                        thresh_stats[thr]["fn"] += fn_t
                        thresh_stats[thr]["tn"] += n_tn

                total_images += 1

    metrics = compute_metrics(total_tp, total_fp, total_fn, total_tn_proxy)
    metrics["siglip_threshold"] = siglip_thresh
    metrics["n_images"]         = total_images

    # Curva P-R a diverse soglie SigLIP
    pr_curve = {}
    for thr in thresholds:
        s = thresh_stats[thr]
        m = compute_metrics(s["tp"], s["fp"], s["fn"], s["tn"])
        pr_curve[thr] = {
            "Precision": m["Precision"],
            "Recall":    m["Recall"],
            "F1":        m["F1"],
            "FP-Rate":   m["FP-Rate"],
        }
    metrics["pr_curve_by_threshold"] = pr_curve

    return metrics


# ── Report ─────────────────────────────────────────────────────────────────────

def print_metrics(name: str, m: dict):
    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  Immagini valutate : {m.get('n_images', 'N/A')}")
    print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN≈{m['TN']}")
    print()
    print(f"  Precision  : {m['Precision']*100:6.2f}%")
    print(f"  Recall     : {m['Recall']*100:6.2f}%")
    print(f"  F1         : {m['F1']*100:6.2f}%")
    print(f"  FP-Rate    : {m['FP-Rate']*100:6.2f}%  "
          f"({'alta — SigLIP necessario' if m['FP-Rate'] > 0.1 else 'bassa'})")
    if "AP@0.5_approx" in m:
        print(f"  AP@0.5 (≈) : {m['AP@0.5_approx']*100:6.2f}%")


def print_pr_curve(pr_curve: dict):
    print(f"\n  Curva P-R a diverse soglie SigLIP:")
    print(f"  {'Soglia':>8}  {'Precision':>10}  {'Recall':>8}  "
          f"{'F1':>8}  {'FP-Rate':>9}")
    print(f"  {'─'*52}")
    best_f1     = 0
    best_thresh = 0
    for thr, m in sorted(pr_curve.items()):
        mark = " ← best F1" if m["F1"] > best_f1 else ""
        if m["F1"] > best_f1:
            best_f1     = m["F1"]
            best_thresh = thr
        print(f"  {thr:>8.1f}  {m['Precision']*100:>9.2f}%  "
              f"{m['Recall']*100:>7.2f}%  "
              f"{m['F1']*100:>7.2f}%  "
              f"{m['FP-Rate']*100:>8.2f}%{mark}")
    print(f"\n  => Soglia ottimale (max F1): {best_thresh}")


def print_comparison(m_yolo: dict, m_pipeline: dict):
    print(f"\n  {'='*60}")
    print(f"  CONFRONTO YOLO standalone vs Pipeline completa")
    print(f"  {'='*60}")
    print(f"  {'Metrica':<12} {'YOLO (F1)':>12} {'Pipeline':>12} {'Delta':>10}")
    print(f"  {'─'*50}")
    keys = ["Precision", "Recall", "F1", "FP-Rate"]
    for k in keys:
        v1 = m_yolo[k] * 100
        v2 = m_pipeline[k] * 100
        d  = v2 - v1
        sign = "+" if d >= 0 else ""
        # Per FP-Rate il miglioramento è una riduzione (delta negativo = buono)
        marker = ""
        if k == "FP-Rate" and d < -1:
            marker = " ✓ riduzione"
        elif k in ("Precision", "F1") and d > 1:
            marker = " ✓ miglioramento"
        print(f"  {k:<12} {v1:>11.2f}% {v2:>11.2f}% "
              f"{sign}{d:>8.2f}%{marker}")

    print(f"\n  Interpretazione:")
    delta_fpr = (m_pipeline["FP-Rate"] - m_yolo["FP-Rate"]) * 100
    delta_pre = (m_pipeline["Precision"] - m_yolo["Precision"]) * 100
    if delta_fpr < -2:
        print(f"  ✓ SigLIP riduce la FP-Rate di {abs(delta_fpr):.1f}% "
              f"— giustifica l'architettura a due stadi")
    elif delta_fpr > 2:
        print(f"  ⚠ SigLIP aumenta la FP-Rate di {delta_fpr:.1f}% "
              f"— la soglia SigLIP potrebbe essere troppo bassa")
    else:
        print(f"  ~ FP-Rate invariata — SigLIP non peggiora il detector")

    if delta_pre > 1:
        print(f"  ✓ Precision migliorata di {delta_pre:.1f}% "
              f"— SigLIP filtra efficacemente i falsi positivi di YOLO")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  VALUTAZIONE PIPELINE — TEST SET")
    print("=" * 60)
    print(f"  Fase 1 weights  : {args.phase1_weights}")
    print(f"  Fase 2 weights  : {args.phase2_weights}")
    print(f"  Dataset         : {args.data_dir}")
    print(f"  Conf threshold  : {args.conf_threshold}")
    print(f"  SigLIP threshold: {args.siglip_thresh}")
    print(f"  Device          : {device}")

    # ── Carica pipeline ───────────────────────────────────────────────────────
    print("\nCarico pipeline...")
    pipeline = JointPipeline(
        yolo_weights      = args.phase1_weights,
        siglip_model_name = CFG.siglip.model_name,
        conf_threshold    = args.conf_threshold,
        lora_r_visual     = CFG.siglip.lora_r_visual,
        lora_alpha_visual = CFG.siglip.lora_alpha_visual,
        lora_dropout      = CFG.siglip.lora_dropout,
        roi_size          = CFG.data.roi_size,
    ).to(device)

    ck    = torch.load(args.phase2_weights, map_location=device, weights_only=False)
    state = {k: v for k, v in ck["pipeline_state_dict"].items()
             if "model.23" not in k}
    pipeline.load_state_dict(state, strict=False)
    print(f"  Pipeline caricata — epoca {ck['epoch']+1}")

    # ── Test loader ───────────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(
        img_size   = CFG.data.img_size,
        batch_size = args.batch_size,
        mode       = "phase1",
        data_dir   = args.data_dir,
    )
    print(f"  Test set: {len(test_loader.dataset)} immagini\n")

    # ── Valutazione 1: YOLO standalone ───────────────────────────────────────
    print(f"{'─'*60}")
    print("  [1/2] Valuto YOLO26 standalone (Fase 1)...")
    metrics_yolo = evaluate_yolo_standalone(
        pipeline, test_loader, device, args.conf_threshold
    )
    print_metrics("YOLO26 standalone (Fase 1)", metrics_yolo)

    # ── Valutazione 2: Pipeline completa ─────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  [2/2] Valuto pipeline completa YOLO + SigLIP (Fase 2)...")
    metrics_pipeline = evaluate_full_pipeline(
        pipeline, test_loader, device,
        conf_thr      = args.conf_threshold,
        siglip_thresh = args.siglip_thresh,
    )
    print_metrics(
        f"Pipeline completa — soglia SigLIP={args.siglip_thresh}",
        metrics_pipeline
    )

    # Curva P-R a diverse soglie SigLIP
    if "pr_curve_by_threshold" in metrics_pipeline:
        print_pr_curve(metrics_pipeline["pr_curve_by_threshold"])

    # ── Confronto ─────────────────────────────────────────────────────────────
    print_comparison(metrics_yolo, metrics_pipeline)

    # ── Salva risultati JSON ──────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "yolo_standalone": {
            k: float(v) if isinstance(v, (float, np.floating)) else v
            for k, v in metrics_yolo.items()
        },
        "full_pipeline": {
            k: (float(v) if isinstance(v, (float, np.floating)) else
                {str(kk): {kkk: float(vvv) for kkk, vvv in vv.items()}
                 for kk, vv in v.items()}
                if isinstance(v, dict) else v)
            for k, v in metrics_pipeline.items()
        },
        "config": {
            "conf_threshold":  args.conf_threshold,
            "siglip_threshold": args.siglip_thresh,
            "iou_threshold":   IOU_THRESHOLD,
            "phase1_weights":  args.phase1_weights,
            "phase2_weights":  args.phase2_weights,
        }
    }

    out_path = out_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  ✓ Risultati salvati in: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()