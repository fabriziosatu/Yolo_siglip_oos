#!/bin/bash
#SBATCH --job-name=oos_fase2_eval_full
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase2_eval_full_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs/fase2_eval_full_%j.err"

# ── Setup ambiente ────────────────────────────────────────────────
source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HF_HOME="/home/F.SATURNINO/Progetto_tesi_SigLIP/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"
cd "$HPC_ROOT"

echo "=================================================="
echo "Job: oos_fase2_eval_full"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$HPC_ROOT/jobs/logs"

OUT_DIR="results/fase2_evaluation_full_v3"

# ── Step 1/2: valutazione completa (griglia detector x prompt x conf x IoU) ──
echo ""
echo "--- Inizio evaluate_fase2_siglip_v2.py ($(date)) ---"
python -u src/evaluation/evaluate_fase2_siglip_v2.py \
    --aug_weights weights/fase1_baseline/aug/best.pt \
    --noaug_weights weights/fase1_baseline/no_aug/best.pt \
    --test_images dataset_finale/test/images \
    --test_labels dataset_finale/test/labels \
    --out_dir $OUT_DIR \
    --conf 0.25 0.5 \
    --iou_thr 0.0 0.25 0.5 0.75
echo "--- Fine evaluate_fase2_siglip.py ($(date)) ---"

# ── Step 2/2: tabelle e grafici (per entrambi i prompt set) ──────────────────
echo ""
echo "--- Inizio visualize_fase2_siglip_v2.py — no_context ($(date)) ---"
python -u src/evaluation/visualize_fase2_siglip_v2.py \
    --mode siglip \
    --json_path $OUT_DIR/metrics_full.json \
    --out_dir $OUT_DIR/vis_no_context \
    --prompt no_context \
    --conf 0.25 0.5 \
    --iou 0.0 0.25 0.5 0.75 \
    --bar_conf 0.25
echo "--- Fine visualize_fase2_siglip_v2.py — no_context ($(date)) ---"

echo ""
echo "--- Inizio visualize_fase2_siglip_v2.py — context ($(date)) ---"
python -u src/evaluation/visualize_fase2_siglip_v2.py \
    --mode siglip \
    --json_path $OUT_DIR/metrics_full.json \
    --out_dir $OUT_DIR/vis_context \
    --prompt context \
    --conf 0.25 0.5 \
    --iou 0.0 0.25 0.5 0.75 \
    --bar_conf 0.25
echo "--- Fine visualize_fase2_siglip_v2.py — context ($(date)) ---"

echo ""
echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="