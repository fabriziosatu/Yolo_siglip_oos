#!/bin/bash
#SBATCH --job-name=oos_fase4_aug_vision_mlp_neg025
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs4/fase4_aug_vision_mlp_neg025_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs4/fase4_aug_vision_mlp_neg025_%j.err"

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
echo "Job: oos_fase4_aug_vision_mlp_neg025"
echo "  YOLO      : Augmentation (Fase 1, pretrained)"
echo "  SigLIP    : vision_mlp (Fase 3, rank 16, pad50, neg_ratio=0.25)"
echo "  alpha/beta: 0.0053 / 0.9947  (bilanciamento 'pura')"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$HPC_ROOT/jobs/logs4"

SAVE_DIR="weights/fase4_joint/aug__vision_mlp__neg025_pad50_nocap"

# ── Auto-resume ────────────────────────────────────────────────────────────
# La partition "aiq" ha un tetto di 7h fisso, non alzabile. Un joint
# training potrebbe non starci in una sessione sola. Trainer salva un
# checkpoint ogni 5 epoche (save_every=5) + ad ogni nuovo best: se questo
# job viene ucciso da SLURM al time limit, rilancia lo STESSO comando
# (sbatch di questo stesso file) — riprende da solo dall'ultimo checkpoint
# invece che da zero.
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

python -u src/training/train.py \
    --yolo_weights /home/F.SATURNINO/Progetto_tesi_SigLIP/weights/fase1_baseline/aug/best.pt \
    --siglip_variant vision_mlp \
    --neg_ratio 0.25 \
    --roi_padding 0.5 \
    --max_rois_per_image 0 \
    --alpha 0.0053 \
    --beta 0.9947 \
    --batch 2 \
    --save_dir "$SAVE_DIR" \
    $RESUME_ARG

echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="