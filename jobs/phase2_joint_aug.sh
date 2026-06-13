#!/bin/bash
#SBATCH --job-name=oos_joint_fase2_aug
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto tesi/logs/fase2_aug_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto tesi/logs/fase2_aug_%j.err"

# ── Setup ambiente ────────────────────────────────────────────────
source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HPC_ROOT="/home/F.SATURNINO/Progetto tesi"
cd "$HPC_ROOT"

# ── Info job ──────────────────────────────────────────────────────
echo "=================================================="
echo "Job: oos_joint_fase2_aug"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "Cartella: $(pwd)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Crea cartella logs se non esiste ─────────────────────────────
mkdir -p "$HPC_ROOT/logs"

# ── Verifica pesi Fase 1 ──────────────────────────────────────────
PHASE1_WEIGHTS="$HPC_ROOT/weights/cluster_training/phase1_yolo_final_aug/weights/best.pt"
if [ ! -f "$PHASE1_WEIGHTS" ]; then
    echo "ERRORE: pesi Fase 1 (aug) non trovati in $PHASE1_WEIGHTS"
    exit 1
fi
echo "Pesi Fase 1 (aug) trovati: $PHASE1_WEIGHTS"

# ── Fase 2 — aug ─────────────────────────────────────────────────
python -u src/training/train_phase2_joint.py \
    --yolo_weights "$PHASE1_WEIGHTS" \
    --epochs 50 \
    --batch  16

echo "=================================================="
echo "Fase 2 aug completata: $(date)"
echo "=================================================="