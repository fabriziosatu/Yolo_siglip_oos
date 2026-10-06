#!/bin/bash
#SBATCH --job-name=oos_fase3_r8
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase3_r8_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase3_r8_%j.err"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HF_HOME="/home/F.SATURNINO/Progetto_tesi_SigLIP/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"
cd "$HPC_ROOT"

echo "=================================================="
echo "Job: oos_fase3_r8 (LoRA rank=8, alpha=16 — redo neg_ratio=1.0 con parametri anti-overfitting)"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$HPC_ROOT/jobs/logs"

# ── Fase 3 — SigLIP + LoRA (rank 8), neg_ratio=1.0 ──────
# RILANCIATO con parametri anti-overfitting (il run originale usava
# lr=1e-4, nessun dropout/weight_decay dedicato, patience=12 — qui
# aggiornato per essere coerente con gli altri due blocchi sotto).
#python -u src/training/train_fase3_lora.py \
#    --train_images dataset_finale/train/images \
#    --train_labels dataset_finale/train/labels \
#    --val_images dataset_finale/val/images \
#    --val_labels dataset_finale/val/labels \
#    --out_dir weights/fase3_siglip_lora/fase3_siglip_lora_r8_neg1 \
#    --epochs 50 \
#    --batch 16 \
#    --lr 3e-5 \
#    --lr_head 1e-3 \
#    --lora_dropout 0.15 \
#    --weight_decay 1e-3 \
#    --padding 0.15 \
#    --neg_ratio 1.0 \
#    --save_every 5 \
#    --early_stop_patience 8 \
#    --lora_rank 8 \
#    --lora_alpha 16

# ── Fase 3 — SigLIP + LoRA (rank 8), neg_ratio=0 ──────
# GIA' ALLENATO — commentato per non ripeterlo.
# python -u src/training/train_fase3_lora.py \
#     --train_images dataset_finale/train/images \
#     --train_labels dataset_finale/train/labels \
#     --val_images dataset_finale/val/images \
#     --val_labels dataset_finale/val/labels \
#     --out_dir weights/fase3_siglip_lora/fase3_siglip_lora_r8_neg0 \
#     --epochs 50 \
#     --batch 16 \
#     --lr 1e-4 \
#     --padding 0.15 \
#     --neg_ratio 0 \
#     --save_every 5 \
#     --early_stop_patience 12 \
#     --lora_rank 8 \
#     --lora_alpha 16

# ── Fase 3 — SigLIP + LoRA (rank 8), neg_ratio=0.25 ──────
# Da rilanciare con LR differenziato per renderlo confrontabile con
# neg_ratio=1.0 — lasciato commentato per ora, su richiesta.
 python -u src/training/train_fase3_lora.py \
     --train_images dataset_finale/train/images \
     --train_labels dataset_finale/train/labels \
     --val_images dataset_finale/val/images \
     --val_labels dataset_finale/val/labels \
     --out_dir weights/fase3_siglip_lora/fase3_siglip_lora_r8_neg025_v2 \
     --epochs 50 \
     --batch 16 \
     --lr 3e-5 \
     --lr_head 1e-3 \
     --lora_dropout 0.15 \
     --weight_decay 1e-3 \
     --padding 0.15 \
     --neg_ratio 0.25 \
     --save_every 5 \
     --early_stop_patience 8 \
     --lora_rank 8 \
     --lora_alpha 16
echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="