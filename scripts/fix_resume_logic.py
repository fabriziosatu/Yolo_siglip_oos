"""
fix_resume_logic.py
=====================
Esegui questo UNA VOLTA sul cluster, dalla root del progetto
($HPC_ROOT), per sostituire in tutti gli .sh la vecchia logica di
auto-resume (che sceglieva ciecamente il checkpoint piu' recente per
nome file, anche se corrotto) con una versione robusta che prova a
caricare davvero ogni checkpoint, dal piu' recente al piu' vecchio, e
salta quelli che risultano corrotti (es. il job ucciso da SLURM a meta'
scrittura).

Uso:
    cd /home/F.SATURNINO/Progetto_tesi_SigLIP
    python3 fix_resume_logic.py

Non serve indicare i path degli .sh — cerca da solo tutti i file .sh
sotto jobs/ (ricorsivamente) che contengono il blocco RESUME_ARG da
sostituire, e mostra quali ha aggiornato.
"""

import glob

OLD_BLOCK = '''RESUME_ARG=""
if [ -d "$HPC_ROOT/$SAVE_DIR" ]; then
    LATEST_CKPT=$(ls -1 "$HPC_ROOT/$SAVE_DIR"/checkpoint_epoch*.pt 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$LATEST_CKPT" ]; then
        RESUME_ARG="--resume $LATEST_CKPT"
        echo "  Trovato checkpoint precedente -> riprendo da: $LATEST_CKPT"
    fi
fi'''

NEW_BLOCK = '''RESUME_ARG=""
if [ -d "$HPC_ROOT/$SAVE_DIR" ]; then
    # Prova i checkpoint dal piu' recente al piu' vecchio, saltando quelli
    # corrotti (es. il job ucciso da SLURM a meta' scrittura prima del fix
    # del salvataggio atomico) invece di prendere ciecamente il piu' recente
    # per nome file.
    for CKPT in $(ls -1 "$HPC_ROOT/$SAVE_DIR"/checkpoint_epoch*.pt 2>/dev/null | sort -Vr); do
        if python3 -c "import torch; torch.load('$CKPT', map_location='cpu', weights_only=False)" >/dev/null 2>&1; then
            RESUME_ARG="--resume $CKPT"
            echo "  Trovato checkpoint valido -> riprendo da: $CKPT"
            break
        else
            echo "  \u26a0 Checkpoint corrotto, salto: $CKPT"
        fi
    done
    if [ -z "$RESUME_ARG" ]; then
        echo "  Nessun checkpoint valido trovato in $SAVE_DIR -- riparto da zero."
    fi
fi'''


def main():
    files = glob.glob("jobs/**/*.sh", recursive=True)
    if not files:
        print("Nessun .sh trovato sotto jobs/ — sei nella cartella giusta "
              "($HPC_ROOT)? Lancia questo script da li'.")
        return

    n_fixed = 0
    n_already_new = 0
    n_no_resume = 0

    for f in files:
        with open(f, encoding="utf-8") as fh:
            content = fh.read()

        if OLD_BLOCK in content:
            content = content.replace(OLD_BLOCK, NEW_BLOCK)
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(content)
            n_fixed += 1
            print(f"  ✓ Aggiornato: {f}")
        elif "for CKPT in" in content and "RESUME_ARG" in content:
            n_already_new += 1  # gia' con la versione nuova, salta silenziosamente
        elif "RESUME_ARG" in content:
            print(f"  ⚠ ATTENZIONE — '{f}' ha RESUME_ARG ma il blocco non combacia "
                  f"esattamente col vecchio template: controllalo a mano.")
        else:
            n_no_resume += 1  # script senza logica di resume (es. valutazione), normale

    print(f"\nRiepilogo: {n_fixed} aggiornati, {n_already_new} gia' aggiornati, "
          f"{n_no_resume} senza logica di resume (ignorati).")


if __name__ == "__main__":
    main()