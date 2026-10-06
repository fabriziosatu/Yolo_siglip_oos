#!/bin/bash
#SBATCH --job-name=oos_fase3_mlp_r8_pad50
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs2/fase3_mlp_r8_pad50_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs2/fase3_mlp_r8_pad50_%j.err"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HF_HOME="/home/F.SATURNINO/Progetto_tesi_SigLIP/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"
cd "$HPC_ROOT"

echo "=================================================="
echo "Job: oos_fase3_mlp_r8_pad50 (MLP, rank=8, padding=0.50, neg_ratio 0.25 e 0.5)"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$HPC_ROOT/jobs/logs2"

# ── MLP — rank 8, neg_ratio=0.25, padding=0.50 ──────
#python -u src/training/train_fase3_lora.py \
#    --train_images dataset_finale/train/images \
#    --train_labels dataset_finale/train/labels \
#    --val_images dataset_finale/val/images \
#    --val_labels dataset_finale/val/labels \
#    --out_dir weights/fase3_siglip_lora/fase3_siglip_lora_r8_neg025_pad50 \
#    --epochs 50 \
#    --batch 16 \
#    --lr 3e-5 \
#    --lr_head 1e-3 \
#    --lora_dropout 0.15 \
#    --weight_decay 1e-3 \
#    --padding 0.50 \
#    --neg_ratio 0.25 \
#    --save_every 5 \
#    --early_stop_patience 8 \
#    --lora_rank 8 \
#    --lora_alpha 16

# ── MLP — rank 8, neg_ratio=0.5, padding=0.50 ──────
python -u src/training/train_fase3_lora.py \
    --train_images dataset_finale/train/images \
    --train_labels dataset_finale/train/labels \
    --val_images dataset_finale/val/images \
    --val_labels dataset_finale/val/labels \
    --out_dir weights/fase3_siglip_lora/fase3_siglip_lora_r8_neg05_pad50 \
    --epochs 50 \
    --batch 16 \
   --lr 3e-5 \
    --lr_head 1e-3 \
    --lora_dropout 0.15 \
    --weight_decay 1e-3 \
    --padding 0.50 \
    --neg_ratio 0.5 \
    --save_every 5 \
    --early_stop_patience 8 \
    --lora_rank 8 \
    --lora_alpha 16

echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="