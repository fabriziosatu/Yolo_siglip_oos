#!/bin/bash
#SBATCH --job-name=oos_fase4_joint
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --array=0-7
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs4/fase4_%a_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs4/fase4_%a_%j.err"

# ============================================================
# Job array — 8 combinazioni Fase 4 in un unico script.
# SLURM lancia 8 job separati (uno per indice 0..7, --array=0-7),
# ognuno legge la propria riga dall'array COMBOS sotto in base a
# $SLURM_ARRAY_TASK_ID. Ogni job ha risorse/log separati come se
# fosse uno script a se stante — non e' un job "unico" che gira
# 8 volte in sequenza, sono 8 job indipendenti schedulati insieme.
#
# Lancio: sbatch fase4_joint_array.sh
# Lancio di UNA sola combinazione (utile per il primo test, come
# discusso — verifica caricamento pesi/tempi prima di consumare
# ore GPU sulle altre 7): sbatch --array=0 fase4_joint_array.sh
# ============================================================

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HF_HOME="/home/F.SATURNINO/Progetto_tesi_SigLIP/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"
cd "$HPC_ROOT"
mkdir -p "$HPC_ROOT/jobs/logs4"

# ── Le 8 combinazioni: yolo_tag|yolo_path|variant|neg_ratio|alpha|beta ────────
# (alpha/beta calcolati con la formula "pura" sulle magnitudo misurate)
COMBOS=(
  "aug|weights/fase1_baseline/aug/best.pt|completa|0.25|0.0089|0.9911"
  "aug|weights/fase1_baseline/aug/best.pt|completa|0.5|0.0059|0.9941"
  "aug|weights/fase1_baseline/aug/best.pt|vision_mlp|0.25|0.0053|0.9947"
  "aug|weights/fase1_baseline/aug/best.pt|vision_mlp|0.5|0.0048|0.9952"
  "no_aug|weights/fase1_baseline/no_aug/best.pt|completa|0.25|0.0171|0.9829"
  "no_aug|weights/fase1_baseline/no_aug/best.pt|completa|0.5|0.0114|0.9886"
  "no_aug|weights/fase1_baseline/no_aug/best.pt|vision_mlp|0.25|0.0101|0.9899"
  "no_aug|weights/fase1_baseline/no_aug/best.pt|vision_mlp|0.5|0.0092|0.9908"
)

IFS='|' read -r YOLO_TAG YOLO_PATH VARIANT NEG_RATIO ALPHA BETA <<< "${COMBOS[$SLURM_ARRAY_TASK_ID]}"
NEG_TAG=$(echo "$NEG_RATIO" | sed 's/^0\.//; s/\.//')
RUN_NAME="${YOLO_TAG}__${VARIANT}__neg${NEG_TAG}"
SAVE_DIR="weights/fase4_joint/$RUN_NAME"

# ── Auto-resume ──────────────────────────────────────────────────────────────
# Il limite della partition e' 7h fisso, non alzabile — un joint training di
# 50 epoche potrebbe non starci in una singola sessione. Trainer salva un
# checkpoint ogni 5 epoche (save_every=5) + ad ogni nuovo best, quindi se
# questo job viene ucciso da SLURM al time limit, al prossimo rilancio dello
# STESSO comando riprende da li' invece che da zero — cerchiamo qui il
# checkpoint piu' recente in SAVE_DIR (se esiste) e lo passiamo a --resume.
RESUME_ARG=""
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
            echo "  ⚠ Checkpoint corrotto, salto: $CKPT"
        fi
    done
    if [ -z "$RESUME_ARG" ]; then
        echo "  Nessun checkpoint valido trovato in $SAVE_DIR -- riparto da zero."
    fi
fi

echo "=================================================="
echo "Job array task: $SLURM_ARRAY_TASK_ID  ($RUN_NAME)"
echo "  YOLO      : $YOLO_TAG -> $YOLO_PATH"
echo "  SigLIP    : $VARIANT  (Fase 3, rank 16, pad50, neg_ratio=$NEG_RATIO)"
echo "  alpha/beta: $ALPHA / $BETA"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# NOTA: --time=07:00:00 (stesso limite usato nei tuoi job di Fase 3 — la
# partition "aiq" sembra avere un tetto massimo a 7h, verificato con
# l'errore PartitionTimeLimit su un tentativo con 8h). Il joint training
# e' comunque piu' pesante di un training singolo: se un task va in
# timeout prima di completare le epoche previste, riduci
# training.num_epochs o --batch invece di alzare --time (non puoi
# superare il limite della partition).

python -u src/training/train.py \
    --yolo_weights "$HPC_ROOT/$YOLO_PATH" \
    --siglip_variant "$VARIANT" \
    --neg_ratio "$NEG_RATIO" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --batch 4 \
    --save_dir "weights/fase4_joint/$RUN_NAME"

echo "=================================================="
echo "Job array task $SLURM_ARRAY_TASK_ID completato: $(date)"
echo "=================================================="
