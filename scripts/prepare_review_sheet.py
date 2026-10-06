"""
src/evaluation/prepare_review_sheet.py
=========================================
Genera un foglio Excel per la categorizzazione manuale dei "nuovi FP"
(crop salvati da compare_new_fp_by_distractor.py con --dump_dir).

Legge direttamente i nomi dei file nella cartella dei crop — non serve
incollare nulla a mano. Ogni riga = un crop, con detector e distrattore
automatico gia' estratti dal nome file; la colonna "Categoria" resta
vuota, con un menu a tendina con le 3 categorie individuate a occhio
(vedi discussione in chat) + una quarta per i casi non classificabili.

Schema del nome file (deciso da compare_new_fp_by_distractor.py):
    <detector>_<indice>_<distrattore_o_nessuno>_<nome_immagine_originale>.jpg
    es: aug_0018_nessuno_video14_image1824_jpg_rf_....jpg
        noaug_0001_a_blurry_video14_image1105_jpg_rf_....jpg

Uso:
    python src/evaluation/prepare_review_sheet.py \
        --crops_dir output/fase3_new_fp_analysis/crops \
        --out_xlsx output/fase3_new_fp_analysis/revisione_manuale.xlsx
"""

import argparse
import re
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter

CATEGORIES = [
    "1 - Scaffale pieno reale (errore genuino)",
    "2 - Griglia/fondo che indica davvero vuoto (possibile problema di annotazione)",
    "3 - Prodotti della fila adiacente (confusione da prospettiva laterale)",
    "4 - Altro / non chiaro",
]


def parse_filename(fname: str):
    """Ritorna (detector, indice, distrattore_auto, immagine_originale)."""
    stem = fname[:-4] if fname.lower().endswith(".jpg") else fname
    parts = stem.split("_")
    detector = parts[0]
    idx = parts[1]
    if len(parts) > 2 and parts[2] == "nessuno":
        distrattore = "nessuno"
        resto = parts[3:]
    else:
        # il tag distrattore e' sempre di 2 parole (winning_distractor.split()[:2])
        distrattore = "_".join(parts[2:4]) if len(parts) > 3 else "?"
        resto = parts[4:]
    immagine_originale = "_".join(resto)
    return detector, idx, distrattore, immagine_originale


def main(args):
    crops_dir = Path(args.crops_dir)
    files = sorted([f for f in crops_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not files:
        print(f"[Attenzione] Nessun crop trovato in {crops_dir}")
        return
    print(f"[Dati] {len(files)} crop trovati in {crops_dir}")

    FONT = "Arial"
    bold = Font(name=FONT, bold=True, size=10, color="FFFFFF")
    normal = Font(name=FONT, size=10)
    header_fill = PatternFill(start_color="1E2761", end_color="1E2761", fill_type="solid")
    thin = Side(style="thin", color="DDDDDD")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    wb = openpyxl.Workbook()

    # ── Foglio Istruzioni ──
    ws0 = wb.active
    ws0.title = "Istruzioni"
    ws0["A1"] = "Come compilare questo foglio"
    ws0["A1"].font = Font(name=FONT, bold=True, size=14)
    istruzioni = (
        "Per ogni riga del foglio 'Dati', apri il file corrispondente (cartella crop sul cluster) "
        "e assegna una categoria dal menu a tendina in colonna E, in base a cosa mostra l'immagine:\n\n"
        "1 - Scaffale pieno reale: il crop mostra chiaramente prodotti, il modello ha sbagliato.\n\n"
        "2 - Griglia/fondo che indica davvero vuoto: l'immagine mostra una struttura/fondo scaffale "
        "vuoto, ma la ground truth non copre quella zona — possibile problema di annotazione, non "
        "necessariamente un errore del modello.\n\n"
        "3 - Prodotti della fila adiacente: foto scattata lateralmente, i prodotti visibili nel crop "
        "appartengono fisicamente a un altro scaffale/corsia, non a quello della box.\n\n"
        "4 - Altro / non chiaro: qualunque caso che non rientra chiaramente nelle prime tre.\n\n"
        "Il foglio 'Riepilogo' calcola automaticamente le percentuali per categoria e per detector "
        "man mano che compili — non serve toccarlo."
    )
    ws0["A3"] = istruzioni
    ws0["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws0.column_dimensions["A"].width = 110
    ws0.row_dimensions[3].height = 260

    # ── Foglio Dati ──
    ws = wb.create_sheet("Dati")
    headers = ["File", "Detector", "Distrattore automatico", "Immagine originale", "Categoria (manuale)", "Note"]
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.font = bold
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = border
    ws.row_dimensions[1].height = 26

    for r, f in enumerate(files, start=2):
        detector, idx, distrattore, immagine = parse_filename(f.name)
        ws.cell(row=r, column=1, value=f.name).font = normal
        ws.cell(row=r, column=2, value=detector).font = normal
        ws.cell(row=r, column=3, value=distrattore.replace("_", " ")).font = normal
        ws.cell(row=r, column=4, value=immagine).font = normal
        ws.cell(row=r, column=5, value="").font = normal
        ws.cell(row=r, column=6, value="").font = normal
        for c in range(1, 7):
            ws.cell(row=r, column=c).border = border

    # Menu a tendina per la colonna Categoria
    dv = DataValidation(type="list", formula1='"' + ",".join(CATEGORIES) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"E2:E{len(files)+1}")

    widths = [55, 10, 30, 35, 55, 30]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"

    # ── Foglio Riepilogo (formule, si aggiorna da solo) ──
    ws2 = wb.create_sheet("Riepilogo")
    ws2["A1"] = "Riepilogo per categoria e detector"
    ws2["A1"].font = Font(name=FONT, bold=True, size=13)
    ws2["A3"] = "Categoria"
    ws2["B3"] = "aug"
    ws2["C3"] = "noaug"
    ws2["D3"] = "Totale"
    ws2["E3"] = "% sul totale"
    for c in range(1, 6):
        cell = ws2.cell(row=3, column=c)
        cell.font = bold
        cell.fill = header_fill
        cell.border = border

    n_files = len(files)
    for i, cat in enumerate(CATEGORIES, start=4):
        ws2.cell(row=i, column=1, value=cat).font = normal
        ws2.cell(row=i, column=2, value=f'=COUNTIFS(Dati!$B:$B,"aug",Dati!$E:$E,$A{i})').font = normal
        ws2.cell(row=i, column=3, value=f'=COUNTIFS(Dati!$B:$B,"noaug",Dati!$E:$E,$A{i})').font = normal
        ws2.cell(row=i, column=4, value=f'=B{i}+C{i}').font = normal
        ws2.cell(row=i, column=5, value=f'=IF({n_files}=0,0,D{i}/{n_files})').font = normal
        ws2.cell(row=i, column=5).number_format = "0.0%"
        for c in range(1, 6):
            ws2.cell(row=i, column=c).border = border

    last_row = 3 + len(CATEGORIES)
    ws2.cell(row=last_row+1, column=1, value="Totale crop analizzati").font = bold_black = Font(name=FONT, bold=True, size=10)
    ws2.cell(row=last_row+1, column=4, value=n_files).font = bold_black

    ws2.column_dimensions["A"].width = 65
    for col in "BCDE":
        ws2.column_dimensions[col].width = 12

    wb.save(args.out_xlsx)
    print(f"[Output] {args.out_xlsx}  ({len(files)} righe da compilare)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--crops_dir", required=True)
    p.add_argument("--out_xlsx", required=True)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())