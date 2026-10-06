#!/bin/bash
#SBATCH --job-name=oos_fase3_neg025_full
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase3_neg025_full_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase3_neg025_full_%j.err"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HF_HOME="/home/F.SATURNINO/Progetto_tesi_SigLIP/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"
cd "$HPC_ROOT"

echo "=================================================="
echo "Job: oos_fase3_neg025_full (rank=16, neg_ratio=0.25 — rieseguito con script aggiornato)"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$HPC_ROOT/jobs/logs"

# ── 1/3 — Valutazione (versione aggiornata, con incrocio TP/FP dei distrattori) ──
echo "--- [1/3] evaluate_fase3_siglip.py ---"
python -u src/evaluation/evaluate_fase3_siglip.py \
    --aug_weights weights/fase1_baseline/aug/best.pt \
    --noaug_weights weights/fase1_baseline/no_aug/best.pt \
    --mlp_dirs  weights/fase3_siglip_lora/fase3_siglip_lora_r16_neg025_v2/best \
    --mlp_labels neg025 \
    --full_dirs  weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r16_neg025/best \
    --full_labels neg025 \
    --test_images dataset_finale/test/images \
    --test_labels dataset_finale/test/labels \
    --out_dir results/fase3_evaluation_full \
    --conf 0.25 0.5 \
    --iou_thr 0.0 0.25 0.5 0.75

# ── 2/3 — Visualizzazione (tutte le configurazioni rilevate in automatico) ──
echo "--- [2/3] visualize_fase3_siglip.py ---"
python -u src/evaluation/visualize_fase3_siglip.py \
    --json_path results/fase3_evaluation_full/metrics_full.json \
    --out_dir results/fase3_evaluation_full/vis \
    --conf 0.25 0.5 \
    --iou 0.0 0.25 0.5 0.75 \
    --bar_conf 0.25

# ── 3/3 — Analisi nuovi FP vs zero-shot 'Versione 3', per distrattore ──
echo "--- [3/3] compare_new_fp_by_distractor.py ---"
python -u src/evaluation/compare_new_fp_by_distractor.py \
    --aug_weights weights/fase1_baseline/aug/best.pt \
    --noaug_weights weights/fase1_baseline/no_aug/best.pt \
    --full_dir weights/fase3_siglip_lora_full/fase3_siglip_lora_full_r16_neg025/best \
    --test_images dataset_finale/test/images \
    --test_labels dataset_finale/test/labels \
    --out_dir results/fase3_evaluation_full/new_fp_analysis \
    --conf 0.25 \
    --iou_thr 0.0

echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="