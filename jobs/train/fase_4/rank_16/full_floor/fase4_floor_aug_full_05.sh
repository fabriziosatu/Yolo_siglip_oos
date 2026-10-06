#!/bin/bash
#SBATCH --job-name=oos_fase4_floor_aug_completa_neg05
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs4/fase4_floor_aug_completa_neg05_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs4/fase4_floor_aug_completa_neg05_%j.err"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HF_HOME="/home/F.SATURNINO/Progetto_tesi_SigLIP/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"
cd "$HPC_ROOT"

echo "=================================================="
echo "Job: oos_fase4_floor_aug_completa_neg05"
echo "  RESTART DA EPOCA 0 (pesi Fase 3 puliti, non riprende dai run 'pura')"
echo "  alpha/beta: 0.3 / 0.7  (strategia FLOOR, non 'pura')"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$HPC_ROOT/jobs/logs4"

SAVE_DIR="weights/fase4_joint/aug__completa__neg05_floor"

# ── Auto-resume (solo per RIPRENDERE questo specifico esperimento floor
#    se interrotto — non ha nulla a che fare con i run 'pura' precedenti,
#    SAVE_DIR e' nuovo apposta) ──────────────────────────────────────────
RESUME_ARG=""
if [ -d "$HPC_ROOT/$SAVE_DIR" ]; then
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

python -u src/training/train.py \
    --yolo_weights /home/F.SATURNINO/Progetto_tesi_SigLIP/weights/fase1_baseline/aug/best.pt \
    --siglip_variant completa \
    --neg_ratio 0.5 \
    --roi_padding 0.5 \
    --max_rois_per_image 0 \
    --alpha 0.3 \
    --beta 0.7 \
    --batch 2 \
    --save_dir "$SAVE_DIR" \
    $RESUME_ARG

echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="