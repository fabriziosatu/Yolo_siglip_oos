"""
src/evaluation/visualize_fase2_siglip_v2.py  —  adapted from visualize_yolo_cosmos_v2_FN.py
------------------------------------------------------------------------------------------
Generates tables and plots for the Phase 2 pipeline (YOLO frozen + SigLIP2 frozen
zero-shot), reusing the PIL rendering engine already proven in the Cosmos
project — same visual style (tables 2a/2b/2c, bar_metrics, cases_table,
SigLIP2 vs YOLO table), adapted to read the JSON produced by
evaluate_fase2_siglip.py instead of evaluate_yolo_cosmos_v2.py.

Differences compared to the original:
  - Reads "metrics_full.json" (keys conf=..._iou=..._{detector}_{prompt},
    same scheme as the Cosmos project but with "siglip_*" fields instead of
    "cosmos_*" — SigLIP2 is a zero-shot contrastive model, not a generative VLM,
    but the YES/NO vs TP/FP accounting is conceptually identical).
  - Added the baseline FN (YOLO only) vs pipeline (with SigLIP2) summary
    in the cases table, using "fn_missed_by_yolo"/"fn_rejected_by_siglip"
    now also computed by evaluate_fase2_siglip.py.
"""

import argparse, json, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════
#  CONFIG — default paths/parameters (modify only these)
# ══════════════════════════════════════════════════════════════
_PROJECT_ROOT = os.path.expanduser("~/Progetto_tesi_SigLIP")

JSON_PATH_DEFAULT = f"{_PROJECT_ROOT}/output/fase2_evaluation_full/metrics_full.json"
OUT_DIR_DEFAULT   = f"{_PROJECT_ROOT}/output/fase2_evaluation_full/vis"
MODE_DEFAULT      = "siglip"
PROMPT_DEFAULT    = "context"
CONF_DEFAULT      = [0.25, 0.5]
IOU_DEFAULT       = [0.0, 0.25, 0.5, 0.75]
BAR_CONF_DEFAULT  = 0.25
# ══════════════════════════════════════════════════════════════
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ─────────────────────────────────────────────────────────────────────────────
#  FONT
# ─────────────────────────────────────────────────────────────────────────────
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
FONT_REG  = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def _font(path, size):
    """Loads a font from the path, with a default fallback."""
    try:    return ImageFont.truetype(path, size)
    except: return ImageFont.load_default()

def _fmt(v, d=3):
    """Formats a number to d decimal places, handles None."""
    return f"{v:.{d}f}" if v is not None else "—"

def _text_w(text, font):
    """Calculates text width, with a heuristic fallback."""
    try:
        bb = font.getbbox(str(text))
        return bb[2] - bb[0]
    except:
        return len(str(text)) * 7


# ─────────────────────────────────────────────────────────────────────────────
#  THEME COLORS (inspired by attached tables)
# ─────────────────────────────────────────────────────────────────────────────
C_BG       = (255, 252, 248)
C_HDR_GRP  = (180, 140, 100)
C_HDR_SUB  = (210, 180, 150)
C_GRP_BG   = (235, 220, 200)
C_ROW1     = (255, 252, 248)
C_ROW2     = (248, 242, 235)
C_SECTION  = (220, 205, 185)
C_TEXT     = (40,  40,  40)
C_BEST     = (0,  140,   0)
C_LINE     = (200, 185, 170)
C_SEP      = (150, 110,  70)
C_WHITE    = (255, 255, 255)
C_TITLE    = (80,  50,  20)


# ─────────────────────────────────────────────────────────────────────────────
#  IoU UTILITY
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    """Calculates intersection over union for two boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_pairs(img_dir: Path, lbl_dir: Path):
    """Loads image and label pairs from directories."""
    pairs = []
    for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists(): continue
        from PIL import Image as _I
        img = _I.open(img_path)
        W, H = img.size
        gt = []
        for line in lbl_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) < 5: continue
            _, cx, cy, bw, bh = map(float, p[:5])
            gt.append([(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H])
        pairs.append({"path": img_path, "gt": gt})
    return pairs


def _multi_match(preds, gts, iou_thr):
    """
    Multi-matching identical to evaluate_yolo_cosmos.py (professor directives).
    A pred is TP if it matches at least one GT with sufficient IoU.
    A GT is FN if it is not matched by any pred.
    Returns (is_tp_pred, matched_gt_indices).
    """
    n_preds = len(preds)
    n_gts   = len(gts)
    mat_pred = [[] for _ in range(n_preds)]
    mat_gt   = [[] for _ in range(n_gts)]

    for pi, pb in enumerate(preds):
        for gi, gb in enumerate(gts):
            v = _iou(pb, gb)
            if (iou_thr > 0 and v >= iou_thr) or (iou_thr == 0 and v > 0):
                mat_pred[pi].append(gi)
                mat_gt[gi].append(pi)

    is_tp_pred = [len(lst) > 0 for lst in mat_pred]
    matched    = {gi for gi, lst in enumerate(mat_gt) if len(lst) > 0}
    return is_tp_pred, matched


def _classify_fp(box):
    """Classifies FP errors."""
    x1,y1,x2,y2 = box; bw=x2-x1; bh=y2-y1; area=bw*bh
    if bw<40 or bh<20 or area<800:   return "dark_region"
    if area<1600:                     return "occupied_shelf"
    return "other_fp"

def _classify_fn(box):
    """Classifies FN errors."""
    x1,y1,x2,y2 = box; bw=x2-x1; bh=y2-y1; area=bw*bh
    if bw<32 or bh<32 or area<1024:  return "small_box"
    return "other_fn"


def evaluate_yolo_run(yolo, pairs, conf_vals, iou_vals):
    """Evaluates a YOLO detector using the same formulas as evaluate_yolo_cosmos.py."""
    from ultralytics import YOLO as _Y
    results = {}
    for conf in conf_vals:
        samples = []
        for s in pairs:
            res   = yolo(s["path"], conf=conf, verbose=False)
            boxes = res[0].boxes.xyxy.cpu().numpy().tolist() \
                    if res and res[0].boxes is not None else []
            samples.append({"gt": s["gt"], "preds": boxes})

        for iou_t in iou_vals:
            total_gt=0; total_tp=0; total_fp=0; total_fn=0
            fp_cats=defaultdict(int); fn_cats=defaultdict(int)
            for s in samples:
                gt=s["gt"]; preds=s["preds"]
                is_tp, matched = _multi_match(preds, gt, iou_t)
                total_gt += len(gt)
                for pb, tp in zip(preds, is_tp):
                    if tp: total_tp+=1
                    else:  total_fp+=1; fp_cats[_classify_fp(pb)]+=1
                for j,gb in enumerate(gt):
                    if j not in matched:
                        total_fn+=1; fn_cats[_classify_fn(gb)]+=1

            results[(conf,iou_t)] = {
                "GT":total_gt,"TP":total_tp,"FP":total_fp,"FN":total_fn,
                "TPR": total_tp/total_gt if total_gt>0 else 0.0,
                "FPR": total_fp/total_gt if total_gt>0 else 0.0,
                "FNR": total_fn/total_gt if total_gt>0 else 0.0,
                "fp_cats": dict(fp_cats),
                "fn_cats": dict(fn_cats),
            }

    # Ultralytics for Table 1
    try:
        uv = yolo.val(verbose=False)
        rd = uv.results_dict
        p  = float(rd.get("metrics/precision(B)",0))
        r  = float(rd.get("metrics/recall(B)",   0))
        results["ultralytics"] = {
            "precision": p, "recall": r,
            "f1":        2*p*r/(p+r) if (p+r)>0 else 0.0,
            "map50":     float(rd.get("metrics/mAP50(B)",    0)),
            "map50_95":  float(rd.get("metrics/mAP50-95(B)", 0)),
        }
    except Exception as e:
        print(f"  [WARN] Ultralytics val: {e}")
        results["ultralytics"] = {}

    return results


def load_siglip_results(json_path, conf_vals, iou_vals, prompt_thrs_unused, prompt):
    """
    Loads results from JSON v2 (evaluate_yolo_cosmos.py).
    v2 keys format: conf={conf}_iou={iou}_{detector}_{prompt} (without siglip_thr_unused).
    prompt_thrs_unused is ignored (kept for compatibility) — returns {None: all_results}.
    """
    with open(json_path) as f:
        data = json.load(f)
    run_names = sorted({v["detector"] for v in data.values()})
    all_results = {}
    for det in run_names:
        res = {}
        for conf in conf_vals:
            for iou_t in iou_vals:
                k = f"conf={conf}_iou={iou_t}_{det}_{prompt}"
                if k not in data:
                    continue
                m = data[k]
                res[(conf, iou_t)] = {
                    "GT":  m["GT"],  "TP": m["TP"],
                    "FP":  m["FP"],  "FN": m["FN"],
                    "TPR": m["TPR"],
                    "FPR": m["FP"] / m["GT"] if m["GT"] > 0 else 0.0,
                    "FNR": m["FNR"],
                    "fp_cats": m.get("fp_cats", {}),
                    "fn_cats": m.get("fn_cats", {}),
                    # SigLIP2 vs YOLO metrics (new in v2)
                    "yolo_total":       m.get("yolo_total", 0),
                    "siglip_yes_total":   m.get("siglip_yes_total", 0),
                    "siglip_no_total":    m.get("siglip_no_total", 0),
                    "siglip_yes_were_tp": m.get("siglip_yes_were_tp", 0),
                    "siglip_yes_were_fp": m.get("siglip_yes_were_fp", 0),
                    "siglip_no_were_tp":  m.get("siglip_no_were_tp", 0),
                    "siglip_no_were_fp":  m.get("siglip_no_were_fp", 0),
                    # FN cause breakdown, per GT (v4)
                    "fn_missed_by_yolo":     m.get("fn_missed_by_yolo", 0),
                    "fn_rejected_by_siglip": m.get("fn_rejected_by_siglip", 0),
                    # Distractors crossed with real TP/FP (v5)
                    "n_real_tp_total": m.get("n_real_tp_total", 0),
                    "n_real_fp_total": m.get("n_real_fp_total", 0),
                    "distractor_wins_by_tp": m.get("distractor_wins_by_tp", {}),
                    "distractor_wins_by_fp": m.get("distractor_wins_by_fp", {}),
                }
        res["ultralytics"] = {}
        all_results[det] = res
    return run_names, {None: all_results}


# ─────────────────────────────────────────────────────────────────────────────
#  1. BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_bars(all_results, run_names, conf_val, iou_vals, title, out_path):
    """Generates the bar chart for TPR, FPR, FNR."""
    fig, axes = plt.subplots(1, len(iou_vals), figsize=(6*len(iou_vals), 6),
                             sharey=False)
    if len(iou_vals) == 1: axes = [axes]

    bar_colors = {"TPR": "#2ecc71", "FPR": "#e74c3c", "FNR": "#f39c12"}
    w = 0.22

    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, iou_t in zip(axes, iou_vals):
        x = np.arange(len(run_names))
        for mi, (metric, mc) in enumerate(bar_colors.items()):
            vals = [all_results[rn].get((conf_val, iou_t), {}).get(metric, 0)
                    for rn in run_names]
            offset = (mi - 1) * w
            rects  = ax.bar(x + offset, vals, w, color=mc, alpha=0.88,
                            edgecolor="white", linewidth=0.5, label=metric)
            for r, v in zip(rects, vals):
                if v > 0.02:
                    ax.text(r.get_x()+r.get_width()/2, r.get_height()+0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"IoU={iou_t}", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace("_"," ").title() for n in run_names], fontsize=9)
        ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.18)
        ax.grid(axis="y", alpha=0.3)
        if iou_t == iou_vals[0]:
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Bar chart → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  2. PNG TABLE — engine
# ─────────────────────────────────────────────────────────────────────────────

def _make_png_table(title_text, cols, col_widths, all_rows, get_val_fn,
                    best_cols, lower_is_better_cols, int_cols,
                    footer_text, out_path, conf_thresholds):
    """
    Generates a PNG table with a two-level header for grouped columns
    (columns with '/c' in the name are grouped).
    """
    ROW_H  = 30; GRP_H = 26
    HDR_R1 = 20; HDR_R2 = 20; HDR_H = HDR_R1 + HDR_R2
    PAD    = 14; TITLE_H = 26

    n_data = sum(1 for r in all_rows if r[0] != "__group__")
    n_grp  = sum(1 for r in all_rows if r[0] == "__group__")
    total_h = TITLE_H + HDR_H + n_data*ROW_H + n_grp*GRP_H + 44
    total_w = sum(col_widths) + PAD*2 + 4

    F_T   = _font(FONT_BOLD, 13); F_H1 = _font(FONT_BOLD, 11)
    F_H2  = _font(FONT_REG,  10); F_C  = _font(FONT_REG,  11)
    F_M   = _font(FONT_BOLD, 11)

    img  = Image.new("RGB", (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    def x_col(ci): return PAD + sum(col_widths[:ci])
    def is_grp(col): return "/c" in col

    def draw_centered(txt, x, y, w, h, font, color):
        tw = _text_w(txt, font)
        draw.text((x+(w-tw)//2, y+(h-13)//2), str(txt), fill=color, font=font)

    def hline(y, width=1, color=None):
        draw.line([(PAD,y),(total_w-PAD,y)], fill=color or C_LINE, width=width)

    def vline(x, y1, y2, width=1, color=None):
        draw.line([(x,y1),(x,y2)], fill=color or C_SEP, width=width)

    # Title
    draw.text((PAD, 4), title_text, fill=C_TITLE, font=F_T)

    # Header
    y_hdr = TITLE_H
    draw.rectangle([PAD, y_hdr, total_w-PAD, y_hdr+HDR_H], fill=C_HDR_SUB)
    n_conf = len(conf_thresholds)
    grp_ci = 0
    for ci, (col, w) in enumerate(zip(cols, col_widths)):
        x = x_col(ci)
        if not is_grp(col):
            draw.rectangle([x, y_hdr, x+w, y_hdr+HDR_H], fill=C_HDR_GRP)
            ty = y_hdr+(HDR_H-13)//2
            tx = x+6 if ci<=1 else x+(w-_text_w(col,F_H1))//2
            draw.text((tx,ty), col, fill=C_WHITE, font=F_H1)
        else:
            ci_in = grp_ci % n_conf
            if ci_in == 0:
                gw = sum(col_widths[ci:ci+n_conf])
                draw.rectangle([x,y_hdr,x+gw,y_hdr+HDR_R1], fill=C_HDR_GRP)
                lbl = col.split("/c")[0].replace("_"," ")
                tw = _text_w(lbl, F_H1)
                draw.text((x+(gw-tw)//2, y_hdr+(HDR_R1-13)//2), lbl, fill=C_WHITE, font=F_H1)
                vline(x, y_hdr, y_hdr+HDR_H)
            sub_lbl = "c"+col.split("/c")[1]
            draw.rectangle([x, y_hdr+HDR_R1, x+w, y_hdr+HDR_H], fill=C_HDR_SUB)
            draw.text((x+(w-_text_w(sub_lbl,F_H2))//2, y_hdr+HDR_R1+(HDR_R2-11)//2),
                      sub_lbl, fill=C_WHITE, font=F_H2)
            draw.line([(x,y_hdr+HDR_R1),(x+w,y_hdr+HDR_R1)], fill=C_SEP, width=1)
            grp_ci += 1

    hline(y_hdr+HDR_H, width=2)
    y = y_hdr+HDR_H

    # Calculate best (only data rows)
    data_rows = [(l,s) for l,s in all_rows if l != "__group__"]
    def best_val(col):
        vals = [get_val_fn(l,s,col) for l,s in data_rows]
        vals = [v for v in vals if v is not None]
        if not vals: return None
        return min(vals) if col in lower_is_better_cols else max(vals)
    bests = {col: best_val(col) for col in best_cols}

    # Rows
    di = 0
    for label, split in all_rows:
        if label == "__group__":
            draw.rectangle([PAD,y,total_w-PAD,y+GRP_H], fill=C_GRP_BG)
            draw.text((PAD+8, y+(GRP_H-13)//2), split, fill=C_TITLE, font=F_M)
            hline(y+GRP_H, width=2); y+=GRP_H; continue

        bg = C_ROW1 if di%2==0 else C_ROW2
        draw.rectangle([PAD,y,total_w-PAD,y+ROW_H], fill=bg)
        draw.text((x_col(0)+6, y+(ROW_H-13)//2), label, fill=C_TEXT, font=F_M)
        draw.text((x_col(1)+6, y+(ROW_H-13)//2), split, fill=C_TEXT, font=F_C)

        grp_c2 = 0
        for ci, col in enumerate(cols[2:], start=2):
            val = get_val_fn(label, split, col)
            if col in int_cols and val is not None:
                txt = str(int(val))
            else:
                txt = _fmt(val) if val is not None else "—"
            best  = bests.get(col)
            isbest = (val is not None and best is not None and abs(val-best)<1e-6)
            draw_centered(txt, x_col(ci), y, col_widths[ci], ROW_H,
                          F_C, C_BEST if isbest else C_TEXT)
            if is_grp(col):
                if grp_c2 % n_conf == 0:
                    vline(x_col(ci), y, y+ROW_H, color=C_LINE)
                grp_c2 += 1

        hline(y+ROW_H); y+=ROW_H; di+=1

    draw.rectangle([PAD,TITLE_H,total_w-PAD,y+4], outline=C_LINE, width=1)
    draw.text((PAD, y+8), footer_text, fill=(120,100,80), font=F_C)
    img.save(out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
#  4. TABLES 2a (FPR) and 2c (TPR)
# ─────────────────────────────────────────────────────────────────────────────

def make_table2(all_results, run_names, conf_vals, iou_vals,
                split_label, out_dir, conf_thresholds):
    """Creates and saves the summary tables for FPR, FNR, and TPR."""
    fpr_cols = [f"FPR_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    tpr_cols = [f"TPR_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    fp_cols  = [f"#FP_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    tp_cols  = [f"#TP_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    W = 78

    all_rows = [("__group__", split_label)]
    for rn in run_names:
        all_rows.append((rn, split_label))

    def get_val(label, split, col):
        for iou_t in iou_vals:
            for conf_t in conf_vals:
                m = all_results[label].get((conf_t, iou_t), {})
                if col == f"FPR_IoU@{iou_t}/c{conf_t}": return m.get("FPR")
                if col == f"TPR_IoU@{iou_t}/c{conf_t}": return m.get("TPR")
                if col == f"#FP_IoU@{iou_t}/c{conf_t}": return m.get("FP")
                if col == f"#TP_IoU@{iou_t}/c{conf_t}": return m.get("TP")
                if col == "#GT": return m.get("GT")
        return None

    # 2a FPR
    _make_png_table(
        title_text=f"Table 2a — FPR (#pred with empty list / #GT)\nGrouped by IoU  |  conf: {conf_vals}",
        cols=["Run","Split","#GT",*fpr_cols,*fp_cols],
        col_widths=[160,55,52,*[W]*len(fpr_cols),*[W]*len(fp_cols)],
        all_rows=all_rows, get_val_fn=get_val,
        best_cols=fpr_cols, lower_is_better_cols=fpr_cols,
        int_cols={*fp_cols,"#GT"},
        footer_text=f"Green=lowest value  |  FPR=FP/GT  |  conf: {conf_vals}  |  IoU: {iou_vals} (Multi-match)",
        out_path=str(out_dir/"tabella2a_fpr.png"),
        conf_thresholds=conf_thresholds,
    )
    print(f"  [OK] Table 2a → {out_dir/'tabella2a_fpr.png'}")


    # 2b FNR
    fnr_cols = [f"FNR_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]
    fn_cols  = [f"#FN_IoU@{iou}/c{conf}" for iou in iou_vals for conf in conf_vals]

    def get_val_fnr(label, split, col):
        for iou_t in iou_vals:
            for conf_t in conf_vals:
                m = all_results[label].get((conf_t, iou_t), {})
                if col == f"FNR_IoU@{iou_t}/c{conf_t}": return m.get("FNR")
                if col == f"#FN_IoU@{iou_t}/c{conf_t}": return m.get("FN")
                if col == "#GT": return m.get("GT")
        return None

    _make_png_table(
        title_text=f"Table 2b — FNR (#GT with empty list / #GT)\nGrouped by IoU  |  conf: {conf_vals}",
        cols=["Run","Split","#GT",*fnr_cols,*fn_cols],
        col_widths=[160,55,52,*[W]*len(fnr_cols),*[W]*len(fn_cols)],
        all_rows=all_rows, get_val_fn=get_val_fnr,
        best_cols=fnr_cols, lower_is_better_cols=fnr_cols,
        int_cols={*fn_cols,"#GT"},
        footer_text=f"Green=lowest value  |  FNR=FN/GT  |  conf: {conf_vals}  |  IoU: {iou_vals} (Multi-match)",
        out_path=str(out_dir/"tabella2b_fnr.png"),
        conf_thresholds=conf_thresholds,
    )
    print(f"  [OK] Table 2b → {out_dir/'tabella2b_fnr.png'}")

    # 2c TPR
    _make_png_table(
        title_text=f"Table 2c — TPR (#match pred↔GT / #GT)\nGrouped by IoU  |  conf: {conf_vals}",
        cols=["Run","Split","#GT",*tpr_cols,*tp_cols],
        col_widths=[160,55,52,*[W]*len(tpr_cols),*[W]*len(tp_cols)],
        all_rows=all_rows, get_val_fn=get_val, best_cols=tpr_cols, lower_is_better_cols=[],
        int_cols={*tp_cols,"#GT"},
        footer_text=f"Green=highest value  |  TPR=TP/GT  |  conf: {conf_vals}  |  IoU: {iou_vals} (Multi-match)",
        out_path=str(out_dir/"tabella2c_tpr.png"),
        conf_thresholds=conf_thresholds,
    )
    print(f"  [OK] Table 2c → {out_dir/'tabella2c_tpr.png'}")


# ─────────────────────────────────────────────────────────────────────────────
#  6. SIGLIP2 vs YOLO TABLE — numerical YES/NO count
# ─────────────────────────────────────────────────────────────────────────────

def make_siglip_vs_yolo_table(all_results, run_names, conf_vals, iou_vals,
                             split_label, out_path):
    """Generates a table specifically comparing SigLIP2 confirmations vs YOLO predictions."""
    cols = ["Run", "Split", "#GT",
            *[f"YOLO_tot/c{c}" for c in conf_vals],
            *[f"C_YES/c{c}"    for c in conf_vals],
            *[f"C_NO/c{c}"     for c in conf_vals],
            *[f"YES_TP/c{c}"   for c in conf_vals],
            *[f"YES_FP/c{c}"   for c in conf_vals],
            *[f"NO_FP/c{c}"    for c in conf_vals],
            *[f"NO_TP/c{c}"    for c in conf_vals]]
    W        = 72
    cw       = [160, 55, 52] + [W] * (len(cols) - 3)
    iou_ref  = iou_vals[0]
    all_rows = [("__group__", split_label)] + [(rn, split_label) for rn in run_names]

    def get_val(label, split, col):
        for c in conf_vals:
            m = all_results[label].get((c, iou_ref), {})
            if col == f"YOLO_tot/c{c}": return m.get("yolo_total")
            if col == f"C_YES/c{c}":    return m.get("siglip_yes_total")
            if col == f"C_NO/c{c}":     return m.get("siglip_no_total")
            if col == f"YES_TP/c{c}":   return m.get("siglip_yes_were_tp")
            if col == f"YES_FP/c{c}":   return m.get("siglip_yes_were_fp")
            if col == f"NO_FP/c{c}":    return m.get("siglip_no_were_fp")
            if col == f"NO_TP/c{c}":    return m.get("siglip_no_were_tp")
            if col == "#GT":
                return all_results[label].get((conf_vals[0], iou_ref), {}).get("GT")
        return None

    int_cols = (
        {f"YOLO_tot/c{c}" for c in conf_vals} |
        {f"C_YES/c{c}"    for c in conf_vals} |
        {f"C_NO/c{c}"     for c in conf_vals} |
        {f"YES_TP/c{c}"   for c in conf_vals} |
        {f"YES_FP/c{c}"   for c in conf_vals} |
        {f"NO_FP/c{c}"    for c in conf_vals} |
        {f"NO_TP/c{c}"    for c in conf_vals} | {"#GT"}
    )
    best_cols       = [f"NO_FP/c{c}" for c in conf_vals]
    lower_is_better = ([f"YES_FP/c{c}" for c in conf_vals] +
                       [f"NO_TP/c{c}"  for c in conf_vals])

    _make_png_table(
        title_text=(f"SigLIP2 vs YOLO Table — YES/NO per detector"
                    f"  (IoU={iou_ref}, conf: {conf_vals})"),
        cols=cols, col_widths=cw, all_rows=all_rows,
        get_val_fn=get_val, best_cols=best_cols,
        lower_is_better_cols=lower_is_better, int_cols=int_cols,
        footer_text=("Green=better value  |  YES_TP: confirms real empty  |  "
                     "YES_FP: confirms YOLO FP  |  NO_FP: corrects FP  |  "
                     "NO_TP: discards real empty"),
        out_path=str(out_path), conf_thresholds=conf_vals,
    )
    print(f"  [OK] SigLIP2 vs YOLO Table -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  7. CASES TABLE — SigLIP2 vs YOLO decision flow
# ─────────────────────────────────────────────────────────────────────────────

def make_casistiche_table(all_results, run_names, conf_val, iou_val, out_path):
    """Flowchart-style table with the 4 SigLIP2 vs YOLO cases."""
    C_HDR_BG   = (60,  60,  80)
    C_HDR_FG   = (255, 255, 255)
    C_SEC_OK   = (210, 240, 210)
    C_SEC_ERR  = (245, 215, 215)
    C_ROW1_OK  = (240, 255, 240)
    C_ROW1_ERR = (255, 242, 242)
    C_ROW2_OK  = (225, 245, 225)
    C_ROW2_ERR = (250, 230, 230)
    C_TEXT     = (40,  40,  40)
    C_GREEN    = (0,   130,  0)
    C_RED      = (180,  20, 20)
    C_LINE     = (200, 185, 170)
    C_BG       = (255, 252, 248)
    C_TITLE    = (80,  50,  20)
    C_ESITO_OK = (0,   110,  0)
    C_ESITO_ER = (160,  10, 10)

    FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_REG  = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    F_T = _font(FONT_BOLD, 13)
    F_H = _font(FONT_BOLD, 11)
    F_C = _font(FONT_REG,  11)
    F_M = _font(FONT_BOLD, 11)

    TITLE_H = 30; HDR_H = 40; ROW_H = 38; SEC_H = 26; FOOT_H = 28; PAD = 14
    SUMMARY_H = 34 * 3 + 30  # 3 summary rows + section title
    col_w_cas  = 170; col_w_desc = 270; col_w_run = 140; col_w_esit = 130
    n_runs  = len(run_names)
    total_w = PAD*2 + col_w_cas + col_w_desc + col_w_run * n_runs + col_w_esit
    total_h = TITLE_H + HDR_H + 3 * SEC_H + 3 * ROW_H + SUMMARY_H + FOOT_H + 10

    img  = Image.new("RGB", (total_w, total_h), C_BG)
    draw = ImageDraw.Draw(img)

    def hline(y, color=None, width=1):
        draw.line([(PAD, y), (total_w - PAD, y)], fill=color or C_LINE, width=width)

    def col_x(ci):
        xs = [PAD,
              PAD + col_w_cas,
              PAD + col_w_cas + col_w_desc]
        for _ in range(n_runs):
            xs.append(xs[-1] + col_w_run)
        xs.append(xs[-1] + col_w_esit)
        return xs[ci]

    def center_text(text, x, y, w, h, font, color):
        tw = _text_w(text, font)
        draw.text((x + (w - tw) // 2, y + (h - 13) // 2), text, fill=color, font=font)

    # Title
    draw.text((PAD, 6),
              f"SigLIP2 vs YOLO Cases  (conf={conf_val}, IoU={iou_val})",
              fill=C_TITLE, font=F_T)

    # Header
    y = TITLE_H
    draw.rectangle([PAD, y, total_w - PAD, y + HDR_H], fill=C_HDR_BG)
    draw.text((col_x(0) + 6, y + (HDR_H - 13) // 2), "Case",   fill=C_HDR_FG, font=F_H)
    draw.text((col_x(1) + 6, y + (HDR_H - 13) // 2), "Description", fill=C_HDR_FG, font=F_H)
    for ri, rn in enumerate(run_names):
        m        = all_results[rn].get((conf_val, iou_val), {})
        yolo_tot = m.get("yolo_total", 0)
        cx       = col_x(2 + ri)
        tw1 = _text_w(rn, F_H)
        draw.text((cx + (col_w_run - tw1) // 2, y + 5),          rn,                 fill=C_HDR_FG,       font=F_H)
        sub = f"YOLO tot={yolo_tot}"
        tw2 = _text_w(sub, F_C)
        draw.text((cx + (col_w_run - tw2) // 2, y + HDR_H - 17), sub,                fill=(210, 200, 190), font=F_C)
    center_text("Final outcome", col_x(2 + n_runs), y, col_w_esit, HDR_H, F_H, C_HDR_FG)
    hline(y + HDR_H, color=(100, 100, 120), width=2)
    y += HDR_H

    casistiche = [
        ("YOLO YES / was TP",  "SigLIP2 confirms a real empty space",            "siglip_yes_were_tp", "\u2713  final TP",  True),
        ("YOLO YES / was FP",  "SigLIP2 confirms a YOLO false positive",       "siglip_yes_were_fp", "\u2717  final FP",  False),
        ("YOLO NO  / was FP",  "SigLIP2 corrects a YOLO false positive",       "siglip_no_were_fp",  "\u2713  reduces FP",  True),
    ]

    prev_ok = None
    di = 0
    for cas_label, desc, key, esito, is_ok in casistiche:
        if prev_ok != is_ok:
            sec_bg  = C_SEC_OK  if is_ok else C_SEC_ERR
            sec_lbl = "CASES WITH CORRECT OUTCOME" if is_ok else "CASES WITH WRONG OUTCOME"
            sec_col = C_ESITO_OK if is_ok else C_ESITO_ER
            draw.rectangle([PAD, y, total_w - PAD, y + SEC_H], fill=sec_bg)
            draw.text((PAD + 8, y + (SEC_H - 13) // 2), sec_lbl, fill=sec_col, font=F_M)
            hline(y + SEC_H)
            y += SEC_H
            prev_ok = is_ok

        row_bg = (C_ROW1_OK if is_ok else C_ROW1_ERR) if di % 2 == 0 else (C_ROW2_OK if is_ok else C_ROW2_ERR)
        draw.rectangle([PAD, y, total_w - PAD, y + ROW_H], fill=row_bg)
        draw.text((col_x(0) + 6, y + (ROW_H - 13) // 2), cas_label, fill=C_TEXT, font=F_M)
        draw.text((col_x(1) + 6, y + (ROW_H - 13) // 2), desc,      fill=C_TEXT, font=F_C)

        for ri, rn in enumerate(run_names):
            m        = all_results[rn].get((conf_val, iou_val), {})
            yolo_tot = m.get("yolo_total", 1) or 1
            n        = m.get(key, 0)
            pct      = n / yolo_tot * 100
            txt      = f"{n}  ({pct:.1f}%)"
            col      = C_GREEN if is_ok else C_RED
            cx       = col_x(2 + ri)
            tw       = _text_w(txt, F_M)
            draw.text((cx + (col_w_run - tw) // 2, y + (ROW_H - 13) // 2), txt, fill=col, font=F_M)

        esito_col = C_ESITO_OK if is_ok else C_ESITO_ER
        center_text(esito, col_x(2 + n_runs), y, col_w_esit, ROW_H, F_M, esito_col)
        hline(y + ROW_H)
        y  += ROW_H
        di += 1

    # ── Summary: FN baseline (YOLO only) vs FN pipeline (with SigLIP2) ───────
    # Both calculated PER GT (same unit of measurement, subtractable from each other
    # properly — unlike the per-box counts above, here there is no
    # double counting risk from multi-match).
    SUM_BG_HDR = (50, 70, 50)
    y += 6
    draw.rectangle([PAD, y, total_w - PAD, y + 26], fill=SUM_BG_HDR)
    draw.text((PAD + 8, y + 6), "FN — baseline (YOLO only) vs pipeline (with SigLIP2) comparison, per GT",
               fill=(255, 255, 255), font=F_H)
    y += 26

    summary_rows = [
        ("Baseline FN (YOLO alone, no filter)",   "fn_missed_by_yolo", None),
        ("Pipeline FN (after SigLIP2 filter)",             "FN",                None),
        ("Δ FN caused by SigLIP2 (pipeline − baseline)", "fn_rejected_by_siglip", "delta"),
    ]
    for label, key, mark in summary_rows:
        row_bg = (245, 248, 240)
        draw.rectangle([PAD, y, total_w - PAD, y + 34], fill=row_bg)
        draw.text((col_x(0) + 6, y + 10), label, fill=C_TEXT, font=F_C)
        for ri, rn in enumerate(run_names):
            m   = all_results[rn].get((conf_val, iou_val), {})
            n   = m.get(key, 0)
            txt = f"+{n}" if mark == "delta" else str(n)
            col = C_RED if mark == "delta" else C_TEXT
            cx  = col_x(2 + ri)
            tw  = _text_w(txt, F_M)
            draw.text((cx + (col_w_run - tw) // 2, y + 10), txt, fill=col, font=F_M)
        hline(y + 34)
        y += 34

    draw.rectangle([PAD, TITLE_H, total_w - PAD, y + 4], outline=C_LINE, width=1)
    draw.text((PAD, y + 8),
              f"% of total YOLO BBs  |  conf={conf_val}  IoU={iou_val}  Green=correct  Red=wrong  |  "
              f"FN Summary: same unit (per GT), valid subtraction",
              fill=(120, 100, 80), font=F_C)
    img.save(str(out_path))
    print(f"  [OK] Cases table -> {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def make_distractor_table(all_results, run_names, conf_val, iou_val, out_path):
    """
    Table: rows = distractor labels (ordered by descending total TP+FP count),
    columns grouped by detector with two sub-columns each: "% on TP" and "% on FP" 
    — crosses distractor wins with the REAL state of the box (was it a real TP or a real
    FP?), instead of just giving the indistinct total.

    Why it matters: if a distractor (e.g. "blurry"/"dark") wins a lot both
    on boxes that were real TPs and on those that were FPs, in the same
    proportion, it's likely just image quality noise (lexical bias/artifact). 
    If instead it wins much more often on one type than the other, it might 
    reflect a real and systematic visual characteristic of that category (e.g. 
    "real empty spaces are often in darker areas of the store").
    """
    C_HDR_BG = (60, 60, 80)
    C_HDR_FG = (255, 255, 255)
    C_SUBHDR_BG = (150, 140, 160)
    C_ROW1 = (255, 252, 248)
    C_ROW2 = (248, 242, 235)
    C_TEXT = (40, 40, 40)
    C_TITLE = (80, 50, 20)

    FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    F_T = _font(FONT_BOLD, 13)
    F_H = _font(FONT_BOLD, 10)
    F_C = _font(FONT_REG, 9)
    F_M = _font(FONT_BOLD, 10)

    # Collects labels seen in at least one detector (TP or FP)
    labels = set()
    per_det = {}
    for rn in run_names:
        d = all_results.get(rn, {}).get((conf_val, iou_val))
        if d is None:
            continue
        per_det[rn] = d
        labels.update(d.get("distractor_wins_by_tp", {}).keys())
        labels.update(d.get("distractor_wins_by_fp", {}).keys())

    if not labels:
        print(f"  [SKIP] No distractor TP/FP data for conf={conf_val} iou={iou_val} — table not generated")
        return

    def total_count(label):
        return sum(per_det[rn].get("distractor_wins_by_tp", {}).get(label, 0)
                   + per_det[rn].get("distractor_wins_by_fp", {}).get(label, 0)
                   for rn in per_det)
    labels_sorted = sorted(labels, key=total_count, reverse=True)

    PAD = 14
    TITLE_H = 46
    HDR_H = 26
    SUBHDR_H = 24
    ROW_H = 30
    col_w_label = 400
    col_w_sub = 95  # one column for TP%, one for FP%
    total_w = PAD * 2 + col_w_label + col_w_sub * 2 * len(per_det)
    total_h = TITLE_H + HDR_H + SUBHDR_H + ROW_H * len(labels_sorted) + 30

    img = Image.new("RGB", (total_w, total_h), (255, 252, 248))
    draw = ImageDraw.Draw(img)

    draw.text((PAD, 6),
              f"Distractors crossed with real TP/FP (diagnostic ONLY)  conf={conf_val}  IoU={iou_val}",
              fill=C_TITLE, font=F_T)

    def col_x(det_idx, sub_idx):
        # sub_idx: 0=TP, 1=FP
        return PAD + col_w_label + (det_idx * 2 + sub_idx) * col_w_sub

    y = TITLE_H
    draw.rectangle([PAD, y, total_w - PAD, y + HDR_H], fill=C_HDR_BG)
    draw.text((PAD + 6, y + (HDR_H - 11) // 2), "Distractor", fill=C_HDR_FG, font=F_H)
    for di, rn in enumerate(per_det):
        label_w = col_w_sub * 2
        x0 = col_x(di, 0)
        tw = _text_w(rn, F_H)
        draw.text((x0 + (label_w - tw) // 2, y + (HDR_H - 11) // 2), rn, fill=C_HDR_FG, font=F_H)
    y += HDR_H

    draw.rectangle([PAD, y, total_w - PAD, y + SUBHDR_H], fill=C_SUBHDR_BG)
    for di, rn in enumerate(per_det):
        for si, sub_label in enumerate(["% on TP", "% on FP"]):
            x0 = col_x(di, si)
            tw = _text_w(sub_label, F_C)
            draw.text((x0 + (col_w_sub - tw) // 2, y + (SUBHDR_H - 10) // 2),
                       sub_label, fill=(255, 255, 255), font=F_C)
    y += SUBHDR_H

    for li, label in enumerate(labels_sorted):
        bg = C_ROW1 if li % 2 == 0 else C_ROW2
        draw.rectangle([PAD, y, total_w - PAD, y + ROW_H], fill=bg)
        display_label = label if _text_w(label, F_C) <= col_w_label - 12 else label[:65] + "..."
        draw.text((PAD + 6, y + (ROW_H - 10) // 2), display_label, fill=C_TEXT, font=F_C)
        for di, rn in enumerate(per_det):
            d = per_det[rn]
            n_tp = d.get("n_real_tp_total", 0)
            n_fp = d.get("n_real_fp_total", 0)
            c_tp = d.get("distractor_wins_by_tp", {}).get(label, 0)
            c_fp = d.get("distractor_wins_by_fp", {}).get(label, 0)
            pct_tp = 100 * c_tp / n_tp if n_tp else 0.0
            pct_fp = 100 * c_fp / n_fp if n_fp else 0.0
            for si, txt in enumerate([f"{pct_tp:.1f}%", f"{pct_fp:.1f}%"]):
                x0 = col_x(di, si)
                tw = _text_w(txt, F_M)
                draw.text((x0 + (col_w_sub - tw) // 2, y + (ROW_H - 11) // 2),
                           txt, fill=C_TEXT, font=F_M)
        y += ROW_H

    draw.rectangle([PAD, TITLE_H, total_w - PAD, y], outline=(200, 185, 170), width=1)
    draw.text((PAD, y + 6),
              "% calculated separately on real-TP #boxes and real-FP #boxes for that detector  |  "
              "if similar to each other, the pattern is probably just image quality noise",
              fill=(120, 100, 80), font=F_C)
    img.save(str(out_path))
    print(f"  [OK] TP/FP distractors table -> {out_path}")


def run_one_config(json_path, config_name, conf_vals, iou_vals, bar_conf, out_dir):
    """Runs the visualization pipeline for a single configuration."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_names, all_results = load_fase3_results(json_path, conf_vals, iou_vals, config_name)
    split_label = "TEST"
    bar_title   = (f"YOLO-only vs fine-tuned YOLO+SigLIP2 ({config_name}) — "
                   f"conf={bar_conf}  split=test\n"
                   f"{len(run_names)} detectors x {len(iou_vals)} IoU thresholds")

    print(f"\n{'='*50}\n  config: {config_name}\n  output: {out_dir}\n{'='*50}")
    print("  [1/5] Bar chart...")
    plot_bars(all_results, run_names, bar_conf, iou_vals,
              bar_title, out_dir/"bar_metrics.png")
    print("  [2/5] Tables 2a/2b/2c...")
    make_table2(all_results, run_names, conf_vals, iou_vals,
                split_label, out_dir, conf_vals)
    print("  [3/5] SigLIP2 vs YOLO Table (numeric)...")
    make_siglip_vs_yolo_table(all_results, run_names, conf_vals, iou_vals,
                            split_label, out_dir/"tabella_siglip_vs_yolo.png")
    print("  [4/5] Cases table (decision flow)...")
    make_casistiche_table(all_results, run_names, bar_conf, iou_vals[0],
                          out_dir/"tabella_casistiche.png")
    print("  [5/5] Distractors table (diagnostic, crossed with real TP/FP)...")
    make_distractor_table(all_results, run_names, bar_conf, iou_vals[0],
                           out_dir/"tabella_distrattori.png")


def main():
    """Main execution entry point: parses arguments and controls workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", default=JSON_PATH_DEFAULT,
                        help="metrics_full.json produced by evaluate_fase3_siglip.py")
    parser.add_argument("--config",    default=None,
                        help="If specified, generates ONLY this config in --out_dir "
                             "(must exactly match the 'arch' field in the json, e.g. "
                             "'mlp_neg05'). If omitted (default), automatically detects ALL "
                             "configs present in the json and generates one per "
                             "subfolder (--out_dir/<config>).")
    parser.add_argument("--conf",      nargs="+", type=float, default=CONF_DEFAULT)
    parser.add_argument("--iou",       nargs="+", type=float, default=IOU_DEFAULT)
    parser.add_argument("--bar_conf",  type=float, default=BAR_CONF_DEFAULT)
    parser.add_argument("--out_dir",   default=OUT_DIR_DEFAULT)
    args = parser.parse_args()

    if args.config:
        # Explicit behavior: a single configuration, directly in --out_dir
        run_one_config(args.json_path, args.config, args.conf, args.iou,
                       args.bar_conf, args.out_dir)
        print(f"\nEverything saved in: {args.out_dir}")
    else:
        # Default: detects configurations present in the json ('arch' field),
        # one subfolder per configuration — no longer a fixed list,
        # so it works with any label (mlp_neg05, full_neg025_context, etc.)
        with open(args.json_path) as f:
            data = json.load(f)
        config_names = sorted({v["arch"] for v in data.values()})
        if not config_names:
            print(f"[Warning] No configurations found in {args.json_path} "
                  f"('arch' field missing or empty file).")
            return
        print(f"Configurations detected in the json: {config_names}")
        for config_name in config_names:
            sub_out_dir = Path(args.out_dir) / config_name
            run_one_config(args.json_path, config_name, args.conf, args.iou,
                           args.bar_conf, sub_out_dir)
        print(f"\nEverything saved in: {args.out_dir}/{{{','.join(config_names)}}}")


if __name__ == "__main__":
    main()