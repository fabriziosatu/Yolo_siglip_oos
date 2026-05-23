"""
src/evaluation/evaluate.py
===========================
Valutazione YOLO26 standalone (Fase 1) vs pipeline joint (Fase 2).

Esegui con:
  python src/evaluation/evaluate.py
  python src/evaluation/evaluate.py --phase1_weights altro/percorso.pt
"""

import argparse
import sys
import shutil
import torch
from pathlib import Path
sys.path.insert(0, ".")

from src.utils.config import CFG, PHASE1_WEIGHTS, PHASE2_WEIGHTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_weights", type=str, default=str(PHASE1_WEIGHTS))
    parser.add_argument("--phase2_weights", type=str, default=str(PHASE2_WEIGHTS))
    parser.add_argument("--data",  type=str, default=str(CFG.data.dataset_yaml))
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def eval_yolo(weights_path: str, name: str, data: str, split: str):
    from ultralytics import YOLO
    print(f"\n  Valuto: {name}")
    model  = YOLO(weights_path)
    result = model.val(data=data, split=split, verbose=True)
    p  = result.results_dict.get("metrics/precision(B)", 0)
    r  = result.results_dict.get("metrics/recall(B)",    0)
    m5 = result.results_dict.get("metrics/mAP50(B)",     0)
    m95= result.results_dict.get("metrics/mAP50-95(B)",  0)
    f1 = 2*p*r/(p+r+1e-9)
    return {"Precision":p,"Recall":r,"F1":f1,"mAP@0.5":m5,"mAP@0.5:95":m95}


def extract_yolo_from_joint(phase2_path: str, phase1_path: str) -> str:
    """Estrae i pesi YOLO aggiornati dal checkpoint della pipeline joint."""
    out = Path(phase2_path).parent / "yolo_phase2.pt"
    print(f"  Estraggo pesi YOLO da: {phase2_path}")

    ck   = torch.load(phase2_path, map_location="cpu", weights_only=False)
    base = torch.load(phase1_path, map_location="cpu", weights_only=False)

    # Esclude il Detect layer (model.23) — ha shape COCO incompatibile
    yolo_state = {k.replace("detector.model.", ""): v
                  for k, v in ck["pipeline_state_dict"].items()
                  if k.startswith("detector.model.")
                  and "model.23" not in k}

    if not yolo_state:
        print("  ⚠ Nessun peso YOLO trovato — uso i pesi Fase 1 come base")
        shutil.copy(phase1_path, out)
        return str(out)

    if "model" in base:
        base["model"].load_state_dict(yolo_state, strict=False)
    torch.save(base, out)
    print(f"  ✓ Pesi YOLO Fase 2 salvati in: {out}")
    return str(out)


def print_results(name, metrics):
    print(f"\n  {'─'*40}")
    print(f"  {name}")
    print(f"  {'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v*100:.1f}%")


def main():
    args = parse_args()

    print("=" * 60)
    print("  VALUTAZIONE NATIVA ULTRALYTICS — FASE 1 vs FASE 2")
    print("=" * 60)
    print(f"  Dataset: {args.data}")
    print(f"  Split  : {args.split}")

    # Fase 1
    print(f"\n{'─'*60}")
    print(f"  [1/2] YOLO26 standalone (Fase 1)...")
    m1 = eval_yolo(args.phase1_weights, "Fase 1", args.data, args.split)
    print_results("YOLO26 standalone (Fase 1)", m1)

    # Fase 2
    print(f"\n{'─'*60}")
    print(f"  [2/2] YOLO26 dalla pipeline joint (Fase 2)...")
    yolo2 = extract_yolo_from_joint(args.phase2_weights, args.phase1_weights)
    m2    = eval_yolo(yolo2, "Fase 2", args.data, args.split)
    print_results("YOLO26 dalla pipeline joint (Fase 2)", m2)

    # Confronto
    print(f"\n{'='*60}")
    print(f"  CONFRONTO FASE 1 vs FASE 2")
    print(f"{'='*60}")
    print(f"  {'Metrica':<16} {'Fase 1':>8} {'Fase 2':>8} {'Delta':>8}")
    print(f"  {'─'*44}")
    for k in m1:
        v1 = m1[k]*100; v2 = m2[k]*100; d = v2-v1
        sign = "+" if d >= 0 else ""
        print(f"  {k:<16} {v1:>7.1f}% {v2:>7.1f}% {sign}{d:>6.1f}%")
    print(f"\n  Nota: la Fase 2 valuta solo i pesi YOLO aggiornati")
    print(f"        dal joint training, non include il contributo di SigLIP.")
    print("=" * 60)


if __name__ == "__main__":
    main()