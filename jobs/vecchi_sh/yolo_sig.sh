#!/bin/bash
#SBATCH --job-name=oos_fase3_loss_magnitude
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase3_loss_magnitude_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase3_loss_magnitude_%j.err"

# ── Setup ambiente ────────────────────────────────────────────────
source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"

# Spostati nella cartella del progetto — necessario per trovare src/
cd "$HPC_ROOT"

# ── Info job ──────────────────────────────────────────────────────
echo "=================================================="
echo "Job: oos_fase3_loss_magnitude"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "Cartella: $(pwd)"
echo "Python: $(which python)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Crea cartella logs se non esiste ─────────────────────────────
mkdir -p "$HPC_ROOT/jobs/logs"

# ── Analisi magnitudo L_VLM — confronto Fase 3 LoRA rank 8 vs rank 16 ──
# NOTA: adegua questi due path ai results.csv reali salvati dai due
# training (quelli con le colonne train/cls_loss, val/cls_loss, ecc.),
# non ai file di pesi .pt.
python -u src/evaluation/analyze_yolo_loss_magnitude.py \
    --results_csv weights/fase3_siglip_lora/fase3_siglip_lora_r8/results.csv weights/fase3_siglip_lora/fase3_siglip_lora_r16/results.csv \
    --labels r8 r16 \
    --out_dir results/fase3_loss_magnitude \
    --last_n_epochs 5 \
    --first_n_epochs 3

echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="