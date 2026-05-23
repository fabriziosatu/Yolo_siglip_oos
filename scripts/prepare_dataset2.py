"""
integra_mivia.py
================
Integra il dataset MIVIA OOS (store + video) nel dataset esistente
data/processed/ senza toccarne i file già presenti.

COSA FA:
  1. Store MIVIA (store1–10):
       - Prende solo le immagini che hanno un .txt YOLO nella cartella labels/
       - Applica filtro blur (soglia 300)
       - Copia in data/processed/ assegnando split 70/15/15

  2. Video MIVIA (video1–23):
       - video1: usa i .txt YOLO già presenti in Final/
       - video2–23: converte XML → YOLO (scarta XML senza <object>)
       - Applica filtro blur (soglia 100) sulle ImmaginiOriginali
       - Applica stride temporale (1 ogni 3 frame) per ridurre ridondanza
       - Copia in data/processed/ assegnando split 70/15/15

  3. Tutti i nomi file vengono prefissati con "mivia_" per evitare
     collisioni con i file Roboflow/WebMarket/SKU già presenti.

  4. Aggiorna data/processed/dataset.yaml con i nuovi conteggi.

USO:
    python integra_mivia.py ^
        --oos_root   "C:\\Users\\186337\\Desktop\\oos1\\OOS dataset" ^
        --video_root "C:\\Users\\186337\\Desktop\\oos1\\video pepper\\video_oos_pepper\\Video" ^
        --processed  "C:\\Users\\186337\\Desktop\\Progetto tesi\\data\\processed"

DIPENDENZE:
    pip install opencv-python tqdm
"""

import argparse
import math
import os
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm


# ── Parametri ─────────────────────────────────────────────────────────────────

TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
SEED          = 42

SOGLIA_BLUR_STORE = 200   # soglia bilanciata per immagini store compresse
SOGLIA_BLUR_VIDEO = 100

VIDEO_STRIDE  = 3         # 1 frame ogni 3 (riduce ridondanza temporale)

STORE_DIRS = {
    "store1":  ("Grocery Products",          "store1"),
    "store2":  ("Grocery Products",          "store2"),
    "store3":  ("Grocery Products",          "store3"),
    "store4":  ("Grocery Products",          "store4"),
    "store5":  ("Grocery Products",          "store5"),
    "store6":  ("Web Market",                "store6"),
    "store7":  ("Supermarket1 - Agropoli",   "store7"),
    "store8":  ("Supermarket2 - Gragnano",   "store8"),
    "store9":  ("Supermarket3 - Salerno",    "store9"),
    "store10": ("SKU110K",                   "store10"),
}

VIDEO_NAMES = [
    "video1","video2","video3","video4","video5","video6","video7",
    "video9","video10","video11","video12","video14","video15",
    "video17","video18","video19","video20","video21","video22","video23",
]


# ── Utility ───────────────────────────────────────────────────────────────────

random.seed(SEED)

def assign_split() -> str:
    r = random.random()
    if r < TRAIN_RATIO:          return "train"
    elif r < TRAIN_RATIO + VAL_RATIO: return "val"
    else:                         return "test"


def blur_score(path: Path) -> float | None:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def safe_stem(name: str) -> str:
    """Rimuove caratteri non sicuri per nomi file."""
    return re.sub(r'[^\w\-]', '_', name)


def copia_campione(img_src: Path, lbl_lines: list[str],
                   processed: Path, stem: str) -> bool:
    """
    Copia immagine e scrive label YOLO in data/processed/{split}/.
    Restituisce True se ok.
    """
    if not img_src.exists() or not lbl_lines:
        return False

    split   = assign_split()
    dst_img = processed / "images" / split / f"{stem}.jpg"
    dst_lbl = processed / "labels" / split / f"{stem}.txt"

    try:
        shutil.copy2(img_src, dst_img)
    except Exception as e:
        print(f"  ⚠ Copia fallita {img_src.name}: {e}")
        return False

    with open(dst_lbl, "w") as f:
        f.writelines(lbl_lines)

    return True


def xml_to_yolo_lines(xml_path: Path) -> list[str]:
    """
    Converte un XML Pascal VOC in righe YOLO (classe 0).
    Restituisce lista vuota se non ci sono <object>.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    size  = root.find("size")
    if size is None:
        return []
    img_w = int(size.findtext("width",  "0"))
    img_h = int(size.findtext("height", "0"))
    if img_w <= 0 or img_h <= 0:
        return []

    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            x1 = float(bb.findtext("xmin"))
            y1 = float(bb.findtext("ymin"))
            x2 = float(bb.findtext("xmax"))
            y2 = float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            continue

        x1 = max(0.0, x1); y1 = max(0.0, y1)
        x2 = min(float(img_w), x2); y2 = min(float(img_h), y2)
        w  = x2 - x1; h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        xc = (x1 + w / 2) / img_w
        yc = (y1 + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h
        lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

    return lines


# ── Step 1: Store MIVIA ───────────────────────────────────────────────────────

def integra_store(oos_root: Path, processed: Path) -> dict:
    print("\n── STORE MIVIA (store1–10) ────────────────────────────────")
    stats = defaultdict(int)

    for store_id, (parent_folder, store_name) in STORE_DIRS.items():
        store_dir = oos_root / parent_folder / store_name
        img_dir   = store_dir / "images"
        lbl_dir   = store_dir / "labels"

        if not img_dir.exists():
            print(f"  [SKIP] {store_id}: cartella non trovata")
            continue

        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg")) + \
               sorted(img_dir.glob("*.png"))

        ok = blur = no_lbl = 0
        for img_path in tqdm(imgs, desc=f"  {store_id:8s}", ncols=80, leave=False):
            # Controlla label
            lbl_path = lbl_dir / img_path.with_suffix(".txt").name
            if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                no_lbl += 1
                continue

            # Filtro blur
            score = blur_score(img_path)
            if score is None or score < SOGLIA_BLUR_STORE:
                blur += 1
                continue

            # Leggi label esistente
            with open(lbl_path) as f:
                lbl_lines = f.readlines()
            if not lbl_lines:
                no_lbl += 1
                continue

            stem = f"mivia_{store_id}__{safe_stem(img_path.stem)}"
            if copia_campione(img_path, lbl_lines, processed, stem):
                ok += 1

        print(f"  {store_id:8s}: {ok:4d} aggiunte | {blur:3d} blur scartate | "
              f"{no_lbl:3d} senza label")
        stats["ok"] += ok
        stats["blur"] += blur
        stats["no_lbl"] += no_lbl

    print(f"\n  Store TOTALE: {stats['ok']} immagini aggiunte")
    return stats


# ── Step 2: Video MIVIA ───────────────────────────────────────────────────────

def integra_video(video_root: Path, processed: Path) -> dict:
    print("\n── VIDEO MIVIA (video1–23) ────────────────────────────────")
    stats = defaultdict(int)

    for video_name in VIDEO_NAMES:
        video_dir = video_root / video_name
        if not video_dir.exists():
            print(f"  [SKIP] {video_name}: cartella non trovata")
            continue

        orig_dir = video_dir / "ImmaginiOriginali"
        xml_dir  = video_dir / "XML"
        final_dir = video_dir / "Final"

        # ── video1: usa i .txt YOLO già in Final/ ─────────────────────────
        if video_name == "video1" and final_dir.exists():
            imgs = sorted(final_dir.glob("*.jpg"))
            # Stride temporale
            def num_key(p):
                ns = re.findall(r'\d+', p.stem)
                return int(ns[-1]) if ns else 0
            imgs = sorted(imgs, key=num_key)[::VIDEO_STRIDE]

            ok = blur = no_lbl = 0
            for img_path in tqdm(imgs, desc=f"  {video_name:8s}", ncols=80, leave=False):
                lbl_path = final_dir / img_path.with_suffix(".txt").name
                if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                    no_lbl += 1
                    continue
                score = blur_score(img_path)
                if score is None or score < SOGLIA_BLUR_VIDEO:
                    blur += 1
                    continue
                with open(lbl_path) as f:
                    lbl_lines = f.readlines()
                if not lbl_lines:
                    no_lbl += 1
                    continue
                stem = f"mivia_{video_name}__{safe_stem(img_path.stem)}"
                if copia_campione(img_path, lbl_lines, processed, stem):
                    ok += 1

            print(f"  {video_name:8s}: {ok:4d} aggiunte | {blur:3d} blur | "
                  f"{no_lbl:3d} senza label  [YOLO]")
            stats["ok"] += ok; stats["blur"] += blur; stats["no_lbl"] += no_lbl
            continue

        # ── video2–23: converte XML → YOLO ────────────────────────────────
        if not orig_dir.exists() or not xml_dir.exists():
            print(f"  [SKIP] {video_name}: ImmaginiOriginali o XML non trovati")
            continue

        # Raccoglie immagini con stride temporale
        imgs = sorted(orig_dir.glob("*.jpg")) + sorted(orig_dir.glob("*.jpeg")) + \
               sorted(orig_dir.glob("*.png"))
        def num_key(p):
            ns = re.findall(r'\d+', p.stem)
            return int(ns[-1]) if ns else 0
        imgs = sorted(imgs, key=num_key)[::VIDEO_STRIDE]

        ok = blur = no_xml = no_obj = 0
        for img_path in tqdm(imgs, desc=f"  {video_name:8s}", ncols=80, leave=False):
            xml_path = xml_dir / img_path.with_suffix(".xml").name
            if not xml_path.exists():
                no_xml += 1
                continue

            # Converti XML → YOLO (scarta se 0 oggetti)
            lbl_lines = xml_to_yolo_lines(xml_path)
            if not lbl_lines:
                no_obj += 1
                continue

            # Filtro blur
            score = blur_score(img_path)
            if score is None or score < SOGLIA_BLUR_VIDEO:
                blur += 1
                continue

            stem = f"mivia_{video_name}__{safe_stem(img_path.stem)}"
            if copia_campione(img_path, lbl_lines, processed, stem):
                ok += 1

        print(f"  {video_name:8s}: {ok:4d} aggiunte | {blur:3d} blur | "
              f"{no_xml:3d} senza XML | {no_obj:3d} XML vuoti  [XML→YOLO]")
        stats["ok"] += ok; stats["blur"] += blur
        stats["no_xml"] += no_xml; stats["no_obj"] += no_obj

    print(f"\n  Video TOTALE: {stats['ok']} immagini aggiunte")
    return stats


# ── Step 3: Aggiorna dataset.yaml ─────────────────────────────────────────────

def aggiorna_yaml(processed: Path):
    totali = {}
    for split in ["train", "val", "test"]:
        n = len(list((processed / "images" / split).glob("*.jpg")))
        totali[split] = n

    yaml_path = processed / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {processed.resolve().as_posix()}\n\n")
        f.write(f"train: images/train   # {totali['train']} immagini\n")
        f.write(f"val:   images/val     # {totali['val']} immagini\n")
        f.write(f"test:  images/test    # {totali['test']} immagini\n\n")
        f.write("nc: 1\n")
        f.write("names: ['empty_shelf']\n")

    print(f"\n  dataset.yaml aggiornato → {yaml_path}")
    return totali


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Integra dataset MIVIA OOS in data/processed/ esistente"
    )
    parser.add_argument("--oos_root",   required=True,
                        help="Root OOS dataset (contiene Grocery Products, Web Market...)")
    parser.add_argument("--video_root", required=True,
                        help="Root video (contiene video1, video2...)")
    parser.add_argument("--processed",  required=True,
                        help="Path a data/processed/ del progetto")
    args = parser.parse_args()

    oos_root   = Path(args.oos_root)
    video_root = Path(args.video_root)
    processed  = Path(args.processed)

    # Diagnostica: mostra i video trovati
    if video_root.exists():
        found = [d.name for d in sorted(video_root.iterdir()) if d.is_dir()]
        print(f"  Video trovati in {video_root.name}/: {found[:5]}{'...' if len(found)>5 else ''}")
    else:
        print(f"❌ video_root non trovato: {video_root}")
        print("   Controlla il path --video_root")
        return

    # Verifica che data/processed/ esista già
    if not (processed / "images" / "train").exists():
        print(f"❌ {processed} non trovato o non ha la struttura attesa.")
        print("   Esegui prima prepare_datasets.py per creare il dataset base.")
        return

    # Conta file esistenti prima
    pre = {s: len(list((processed/"images"/s).glob("*.jpg")))
           for s in ["train","val","test"]}
    print(f"\n{'='*60}")
    print(f"  INTEGRAZIONE MIVIA → data/processed/")
    print(f"{'='*60}")
    print(f"  Dataset esistente: train={pre['train']} | val={pre['val']} | test={pre['test']}")
    print(f"  Soglia blur store : {SOGLIA_BLUR_STORE}")
    print(f"  Soglia blur video : {SOGLIA_BLUR_VIDEO}")
    print(f"  Stride video      : 1 ogni {VIDEO_STRIDE} frame")

    # Integra
    integra_store(oos_root, processed)
    integra_video(video_root, processed)

    # Aggiorna yaml e stampa riepilogo finale
    post = aggiorna_yaml(processed)

    print(f"\n{'='*60}")
    print(f"  ✅ COMPLETATO")
    print(f"{'='*60}")
    print(f"  {'split':6s} | {'prima':>6s} | {'dopo':>6s} | {'aggiunte':>8s}")
    print(f"  {'-'*36}")
    tot_pre = tot_post = 0
    for s in ["train","val","test"]:
        agg = post[s] - pre[s]
        print(f"  {s:6s} | {pre[s]:6d} | {post[s]:6d} | {agg:8d}")
        tot_pre += pre[s]; tot_post += post[s]
    print(f"  {'-'*36}")
    print(f"  {'TOT':6s} | {tot_pre:6d} | {tot_post:6d} | {tot_post-tot_pre:8d}")
    print(f"\n  Dataset pronto in: {processed}")


if __name__ == "__main__":
    main()