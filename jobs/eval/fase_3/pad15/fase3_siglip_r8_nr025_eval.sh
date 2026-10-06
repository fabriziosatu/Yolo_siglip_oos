#!/bin/bash
#SBATCH --job-name=oos_fase3_eval_r8_neg025
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=07:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs2/fase3_eval_r8_neg025_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs2/fase3_eval_r8_neg025_%j.err"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HF_HOME="/home/F.SATURNINO/Progetto_tesi_SigLIP/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"
cd "$HPC_ROOT"

echo "=================================================="
echo "Job: oos_fase3_eval_r8_neg025 (rank=8, padding=0.15, MLP:neg025_v2 + Completo:neg025)"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$HPC_ROOT/jobs/logs2"

echo "--- [1/2] evaluate_fase3_siglip.py ---"
python -u src/evaluation/evaluate_fase3_siglip.py \
    --aug_weights weights/fase1_baseline/aug/best.pt \
    --noaug_weights weights/fase1_baseline/no_aug/best.pt \
    --mlp_dirs  weights/fase3_siglip_lora/fase3_siglip_lora_r8_neg025_v2/best \
    --mlp_labels r8_neg025 \
    --full_dirs  weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r8_neg025/best \
    --full_labels r8_neg025 \
    --test_images dataset_finale/test/images \
    --test_labels dataset_finale/test/labels \
    --out_dir results/fase3_evaluation_rank8_neg025 \
    --conf 0.25 0.5 \
    --iou_thr 0.0 0.25 0.5 0.75

echo "=================================================="
echo "Valutazione completata, genero le visualizzazioni: $(date)"
echo "=================================================="

echo "--- [2/2] visualize_fase3_siglip.py ---"
python -u src/evaluation/visualize_fase3_siglip.py \
    --json_path results/fase3_evaluation_rank8_neg025/metrics_full.json \
    --out_dir results/fase3_evaluation_rank8_neg025/vis \
    --conf 0.25 0.5 \
    --iou 0.0 0.25 0.5 0.75 \
    --bar_conf 0.25

echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="