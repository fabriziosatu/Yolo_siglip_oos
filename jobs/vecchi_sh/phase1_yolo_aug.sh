#!/bin/bash
#SBATCH --job-name=oos_yolo_fase1_aug
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto tesi/logs/fase1_aug_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto tesi/logs/fase1_aug_%j.err"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0
export HPC_ROOT="/home/F.SATURNINO/Progetto tesi"
cd "$HPC_ROOT"

echo "=================================================="
echo "Job: oos_yolo_fase1_aug"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "Cartella: $(pwd)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$HPC_ROOT/logs"

python -u src/training/train_phase1_yolo.py \
    --epochs  150  \
    --batch   32  \
    --imgsz   640 \
    --device  0   \
    --name    phase1_yolo_merged_aug \
    --augment

echo "=================================================="
echo "Fase 1 Aug Completata: $(date)"
echo "=================================================="
