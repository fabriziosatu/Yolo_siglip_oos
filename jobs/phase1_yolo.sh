#!/bin/bash
#SBATCH --job-name=oos_yolo_fase1
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto tesi/logs/fase1_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto tesi/logs/fase1_%j.err"

# ── Setup ambiente ────────────────────────────────────────────────
source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HPC_ROOT="/home/F.SATURNINO/Progetto tesi"

# Spostati nella cartella del progetto — necessario per trovare src/
cd "$HPC_ROOT"

# ── Info job ──────────────────────────────────────────────────────
echo "=================================================="
echo "Job: oos_yolo_fase1"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "Cartella: $(pwd)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Crea cartella logs se non esiste ─────────────────────────────
mkdir -p "$HPC_ROOT/logs"

# ── Fase 1 ────────────────────────────────────────────────────────
python -u src/training/train_phase1_yolo.py \
    --epochs 80 \
    --batch  16  \
    --imgsz  640 \
    --device 0

echo "=================================================="
echo "Fase 1 Completata: $(date)"
echo "=================================================="