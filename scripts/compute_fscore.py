"""
src/evaluation/compute_fscore.py
====================================
Calcola F1-score (oltre a Precision e Recall) per la BASELINE (YOLO-only,
prima del filtro SigLIP) e per la PIPELINE (dopo il filtro SigLIP).

NOVITA': non serve piu' elencare a mano i metrics_full.json — lo script
li trova DA SOLO cercando ricorsivamente dentro le cartelle indicate
(--search_dirs, default "results" e "output"). Funziona sia con i file
prodotti da evaluate_fase2_siglip.py sia da evaluate_fase3_siglip.py,
perche' condividono lo stesso schema di campi.

In piu', per ogni riga calcola il **Delta vs Baseline** (F1 pipeline -
F1 baseline) con un'etichetta MIGLIORA/PEGGIORA — risponde direttamente
alla domanda "stiamo migliorando o peggiorando rispetto a YOLO da solo?"
senza dover confrontare i numeri a mano.

Formule:
    Precision = TP / (TP + FP)
    Recall    = TP / GT                      (= TPR)
    F1        = 2*TP / (2*TP + FP + FN)      (equivalente a 2PR/(P+R),
                                               ma senza rischio di
                                               divisione per zero se
                                               P+R=0)

BASELINE (YOLO-only, prima di SigLIP):
    TP_base = n_real_tp_total
    FP_base = n_real_fp_total
    FN_base = fn_missed_by_yolo
    GT_base = GT (stesso GT della pipeline, e' lo stesso test set)

PIPELINE (dopo il filtro SigLIP):
    TP, FP, FN, GT presi direttamente dall'entry.

Uso (scansione automatica, il caso comune):
    python src/evaluation/compute_fscore.py \
        --search_dirs results output \
        --out_csv results/fscore_tutte_le_fasi.csv \
        --conf 0.25 --iou 0.0

Uso (file specifici, se preferisci controllo manuale):
    python src/evaluation/compute_fscore.py \
        --json_paths results/fase2_evaluation_full_v3/metrics_full.json \
        --out_csv results/fscore_fase2.csv
"""

import argparse
import csv
import json
from pathlib import Path


def safe_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def safe_precision(tp, fp):
    denom = tp + fp
    return (tp / denom) if denom > 0 else 0.0


def safe_recall(tp, gt):
    return (tp / gt) if gt > 0 else 0.0


def discover_json_files(search_dirs):
    """Cerca ricorsivamente tutti i metrics_full.json sotto le cartelle indicate."""
    found = []
    for d in search_dirs:
        root = Path(d)
        if not root.exists():
            print(f"[Attenzione] Cartella non trovata: {root} — salto.")
            continue
        found.extend(sorted(root.rglob("metrics_full.json")))
    return found


def process_file(json_path: Path, conf_filter, iou_filter):
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for key, m in data.items():
        conf = m.get("conf")
        iou = m.get("iou")
        if conf_filter is not None and conf not in conf_filter:
            continue
        if iou_filter is not None and iou not in iou_filter:
            continue

        detector = m.get("detector", "?")
        # Fase2 usa "prompt", Fase3 usa "arch" — supportiamo entrambi
        config_name = m.get("arch", m.get("prompt", "?"))

        gt = m.get("GT", 0)
        tp, fp, fn = m.get("TP", 0), m.get("FP", 0), m.get("FN", 0)

        row = {
            "file": json_path.name,
            "cartella": json_path.parent.name,
            "config": config_name,
            "detector": detector,
            "conf": conf,
            "iou": iou,
            "GT": gt,
        }

        # Pipeline (dopo SigLIP)
        row["pipeline_TP"] = tp
        row["pipeline_FP"] = fp
        row["pipeline_FN"] = fn
        row["pipeline_Precision"] = round(safe_precision(tp, fp), 4)
        row["pipeline_Recall"] = round(safe_recall(tp, gt), 4)
        row["pipeline_F1"] = round(safe_f1(tp, fp, fn), 4)

        # Baseline (YOLO-only, prima di SigLIP) — se i campi sono presenti
        if all(k in m for k in ("n_real_tp_total", "n_real_fp_total", "fn_missed_by_yolo")):
            tp_b = m["n_real_tp_total"]
            fp_b = m["n_real_fp_total"]
            fn_b = m["fn_missed_by_yolo"]
            row["baseline_TP"] = tp_b
            row["baseline_FP"] = fp_b
            row["baseline_FN"] = fn_b
            row["baseline_Precision"] = round(safe_precision(tp_b, fp_b), 4)
            row["baseline_Recall"] = round(safe_recall(tp_b, gt), 4)
            baseline_f1 = safe_f1(tp_b, fp_b, fn_b)
            row["baseline_F1"] = round(baseline_f1, 4)

            # Delta vs Baseline: risponde direttamente a "meglio o peggio di YOLO da solo?"
            delta = row["pipeline_F1"] - row["baseline_F1"]
            row["delta_vs_baseline"] = round(delta, 4)
            row["verdetto"] = "MIGLIORA" if delta > 0.0005 else ("PEGGIORA" if delta < -0.0005 else "invariato")
        else:
            row["baseline_TP"] = row["baseline_FP"] = row["baseline_FN"] = "n/d"
            row["baseline_Precision"] = row["baseline_Recall"] = row["baseline_F1"] = "n/d"
            row["delta_vs_baseline"] = "n/d"
            row["verdetto"] = "n/d"

        rows.append(row)

    return rows


def main(args):
    if args.json_paths:
        json_files = [Path(p) for p in args.json_paths]
        print(f"[Info] Uso i {len(json_files)} file passati esplicitamente con --json_paths")
    else:
        json_files = discover_json_files(args.search_dirs)
        print(f"[Scansione] Trovati {len(json_files)} metrics_full.json sotto {args.search_dirs}:")
        for f in json_files:
            print(f"    {f}")

    all_rows = []
    for path in json_files:
        if not path.exists():
            print(f"[Attenzione] Non trovato: {path} — salto.")
            continue
        rows = process_file(path, args.conf, args.iou)
        print(f"[Dati] {path}: {len(rows)} righe estratte")
        all_rows.extend(rows)

    if not all_rows:
        print("[Attenzione] Nessuna riga estratta da nessun file.")
        return

    # Ordina per cartella (tende a seguire l'ordine cronologico fase2_* poi
    # fase3_*, dato lo schema di naming usato nel progetto), poi per detector
    all_rows.sort(key=lambda r: (r["cartella"], r["config"], r["detector"]))

    fieldnames = ["file", "cartella", "config", "detector", "conf", "iou", "GT",
                  "baseline_TP", "baseline_FP", "baseline_FN",
                  "baseline_Precision", "baseline_Recall", "baseline_F1",
                  "pipeline_TP", "pipeline_FP", "pipeline_FN",
                  "pipeline_Precision", "pipeline_Recall", "pipeline_F1",
                  "delta_vs_baseline", "verdetto"]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[Output] {out_csv}  ({len(all_rows)} righe totali)")

    # Anteprima leggibile a schermo, con il verdetto in evidenza
    print(f"\n{'Cartella':30s} {'Config':22s} {'Det':6s} | "
          f"{'F1 base':8s} | {'F1 pipe':8s} | {'Delta':7s} | Verdetto")
    print("-" * 100)
    for r in all_rows:
        print(f"{r['cartella']:30s} {r['config']:22s} {r['detector']:6s} | "
              f"{r['baseline_F1']:>8} | {r['pipeline_F1']:>8} | "
              f"{r['delta_vs_baseline']:>7} | {r['verdetto']}")

    # Riepilogo finale: quante configurazioni migliorano vs peggiorano, per detector
    print("\n" + "=" * 50)
    print("RIEPILOGO — la pipeline SigLIP sta aiutando o no?")
    print("=" * 50)
    for det in sorted({r["detector"] for r in all_rows}):
        subset = [r for r in all_rows if r["detector"] == det and r["verdetto"] != "n/d"]
        n_migliora = sum(1 for r in subset if r["verdetto"] == "MIGLIORA")
        n_peggiora = sum(1 for r in subset if r["verdetto"] == "PEGGIORA")
        n_invariato = sum(1 for r in subset if r["verdetto"] == "invariato")
        print(f"  {det:8s}: {n_migliora} migliorano, {n_peggiora} peggiorano, "
              f"{n_invariato} invariate (su {len(subset)} configurazioni confrontate)")
        if subset:
            best = max(subset, key=lambda r: r["pipeline_F1"])
            print(f"             -> Miglior F1 pipeline: {best['config']} "
                  f"({best['cartella']}) = {best['pipeline_F1']}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_paths", nargs="+", default=None,
                    help="Se specificato, usa SOLO questi file (disabilita la scansione "
                         "automatica). Se omesso, cerca in --search_dirs.")
    p.add_argument("--search_dirs", nargs="+", default=["results", "output"],
                    help="Cartelle in cui cercare ricorsivamente metrics_full.json "
                         "(default: results, output)")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--conf", nargs="+", type=float, default=None,
                    help="Filtra solo questi valori di conf (default: tutti)")
    p.add_argument("--iou", nargs="+", type=float, default=None,
                    help="Filtra solo questi valori di iou (default: tutti)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())