#!/bin/bash
#SBATCH --job-name=oos_fase3_new_fp
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase3_new_fp_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase3_new_fp_%j.err"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HF_HOME="/home/F.SATURNINO/Progetto_tesi_SigLIP/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"
cd "$HPC_ROOT"

echo "=================================================="
echo "Job: oos_fase3_new_fp (analisi nuovi FP + dump crop per ispezione manuale)"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$HPC_ROOT/jobs/logs"

python -u src/evaluation/compare_new_fp_by_distractor.py \
    --aug_weights weights/fase1_baseline/aug/best.pt \
    --noaug_weights weights/fase1_baseline/no_aug/best.pt \
    --full_dir weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r16_neg025/best \
    --test_images dataset_finale/test/images \
    --test_labels dataset_finale/test/labels \
    --out_dir output/fase3_new_fp_analysis \
    --conf 0.25 \
    --iou_thr 0.0 \
    --dump_dir output/fase3_new_fp_analysis/crops

echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="