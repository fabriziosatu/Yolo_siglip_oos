"""
scripts/analyze_yolo_loss_magnitude.py
========================================
Estrae la magnitudo di L_YOLO dai log di training della Fase 1, per poterla
confrontare con la magnitudo di L_VLM (vedi output di
src/evaluation/fase2_siglip_frozen.py e, in seguito, i log di training
della Fase 3) e calibrare alpha/beta della loss congiunta di Fase 4.

Ultralytics salva automaticamente, per ogni run di training, un file
`results.csv` dentro la cartella del run (es. runs/detect/train/results.csv
oppure weights/fase1_baseline/aug/results.csv a seconda di dove hai
spostato i pesi). Contiene colonne tipo:
    train/box_loss, train/cls_loss, train/dfl_loss,
    val/box_loss,   val/cls_loss,   val/dfl_loss

Uso:
    python scripts/analyze_yolo_loss_magnitude.py \
        --results_csv weights/fase1_baseline/aug/results.csv \
        --out_dir output/fase1_loss_magnitude \
        --last_n_epochs 20

--last_n_epochs: quante epoche finali considerare come "regime convergente"
(la loss delle prime epoche e' rumorosa e alta, non rappresentativa per la
calibrazione — quello che serve e' la magnitudo a convergenza).
"""

import argparse
import csv
import statistics as stats
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOSS_COLUMNS = [
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
]


def load_results_csv(path: Path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # Ultralytics a volte mette spazi extra nei nomi colonna
    rows = [{k.strip(): v for k, v in row.items()} for row in rows]
    return rows


def to_float_series(rows, column):
    series = []
    for row in rows:
        val = row.get(column)
        if val is None or val == "":
            continue
        try:
            series.append(float(val))
        except ValueError:
            continue
    return series


def summarize(series):
    if not series:
        return None
    return {
        "n": len(series),
        "mean": stats.mean(series),
        "median": stats.median(series),
        "std": stats.stdev(series) if len(series) > 1 else 0.0,
        "min": min(series),
        "max": max(series),
    }


def analyze_one(results_csv: Path, label: str, out_dir: Path, args, f):
    """Analizza un singolo results.csv e scrive le statistiche nel file f.
    Ritorna (available_cols, train_total_series) per il plot combinato finale."""
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Non trovo {results_csv}. Controlla il path — deve puntare al "
            f"'results.csv' del run (non a un file di pesi .pt)."
        )

    rows = load_results_csv(results_csv)
    print(f"[Dati] '{label}': {len(rows)} epoche trovate in {results_csv}")

    available_cols = [c for c in LOSS_COLUMNS if c in rows[0]]
    if not available_cols:
        print(f"[Attenzione] '{label}': nessuna colonna attesa trovata. "
              f"Colonne disponibili: {list(rows[0].keys())}")
        return [], []

    # Loss totale = somma delle componenti disponibili (box+cls+dfl per YOLO;
    # per Fase 3 SigLIP di solito box/dfl sono 0 e conta solo cls_loss)
    train_total = []
    for row in rows:
        try:
            total = (
                float(row.get("train/box_loss", 0) or 0)
                + float(row["train/cls_loss"])
                + float(row.get("train/dfl_loss", 0) or 0)
            )
            train_total.append(total)
        except (KeyError, ValueError):
            pass

    f.write(f"=== Magnitudo loss — {label} ===\n")
    f.write(f"File sorgente: {results_csv}\n")
    f.write(f"Epoche totali: {len(rows)}\n\n")

    for col in available_cols:
        series = to_float_series(rows, col)
        full_stats = summarize(series)
        last_n = series[-args.last_n_epochs:] if len(series) >= args.last_n_epochs else series
        conv_stats = summarize(last_n)
        first_n = series[:args.first_n_epochs] if args.first_n_epochs else []
        first_stats = summarize(first_n) if first_n else None

        f.write(f"--- {col} ---\n")
        if full_stats:
            f.write(f"  Intero training : mean={full_stats['mean']:.4f}  "
                    f"median={full_stats['median']:.4f}  std={full_stats['std']:.4f}  "
                    f"min={full_stats['min']:.4f}  max={full_stats['max']:.4f}\n")
        if first_stats:
            f.write(f"  Prime {len(first_n)} epoche (regime iniziale): "
                    f"mean={first_stats['mean']:.4f}  median={first_stats['median']:.4f}\n")
        if conv_stats:
            f.write(f"  Ultime {len(last_n)} epoche (convergenza): "
                    f"mean={conv_stats['mean']:.4f}  median={conv_stats['median']:.4f}  "
                    f"std={conv_stats['std']:.4f}\n")
        f.write("\n")

    if train_total:
        total_stats = summarize(train_total)
        last_n_total = train_total[-args.last_n_epochs:] if len(train_total) >= args.last_n_epochs else train_total
        conv_total_stats = summarize(last_n_total)
        f.write(f"--- Loss totale (train) — {label} ---\n")
        f.write(f"  Intero training : mean={total_stats['mean']:.4f}  "
                f"median={total_stats['median']:.4f}  std={total_stats['std']:.4f}\n")
        f.write(f"  Ultime {len(last_n_total)} epoche (convergenza): "
                f"mean={conv_total_stats['mean']:.4f}  median={conv_total_stats['median']:.4f}\n\n")

    print(f"=== '{label}' a convergenza (ultime {len(last_n) if available_cols else 0} epoche) ===")
    if train_total:
        print(f"mean={stats.mean(last_n_total):.4f}  median={stats.median(last_n_total):.4f}")

    return available_cols, train_total


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in args.results_csv]
    if args.labels:
        if len(args.labels) != len(csv_paths):
            raise ValueError(
                f"--labels deve avere lo stesso numero di elementi di --results_csv "
                f"({len(args.labels)} vs {len(csv_paths)})"
            )
        labels = args.labels
    else:
        labels = [p.stem for p in csv_paths]

    summary_txt = out_dir / "loss_magnitude.txt"
    all_totals = {}
    with open(summary_txt, "w") as f:
        f.write(">>> Confronto magnitudo loss tra run — utile per calibrare "
                "alpha/beta della loss congiunta di Fase 4.\n\n")
        for path, label in zip(csv_paths, labels):
            _, train_total = analyze_one(path, label, out_dir, args, f)
            if train_total:
                all_totals[label] = train_total

    print(f"[Output] Statistiche salvate in {summary_txt}")

    # ── Plot combinato delle curve di loss totale, una linea per run ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, series in all_totals.items():
        ax.plot(range(1, len(series) + 1), series, label=label, linewidth=2)
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Loss totale (train)")
    ax.set_title("Confronto magnitudo loss tra run")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curves_comparison.png", dpi=150)
    print(f"[Output] Grafico salvato in {out_dir / 'loss_curves_comparison.png'}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estrae e confronta la magnitudo di loss tra uno o piu' run "
                     "(Fase 1 YOLO oppure Fase 3 SigLIP+LoRA)"
    )
    parser.add_argument("--results_csv", type=str, nargs="+", required=True,
                         help="Uno o piu' path a results.csv (es. r8 ed r16 della Fase 3)")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                         help="Etichette per ciascun results_csv, stesso ordine e stessa "
                              "lunghezza (es. r8 r16). Se omesso, usa il nome del file.")
    parser.add_argument("--out_dir", type=str, default="output/loss_magnitude")
    parser.add_argument("--last_n_epochs", type=int, default=20,
                         help="Numero di epoche finali da considerare come regime convergente")
    parser.add_argument("--first_n_epochs", type=int, default=0,
                         help="Numero di epoche iniziali da riportare come regime iniziale "
                              "(0 = non calcolarlo)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)