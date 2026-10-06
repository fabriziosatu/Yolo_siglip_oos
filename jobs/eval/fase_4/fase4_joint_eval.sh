#!/bin/bash
#SBATCH --job-name=oos_fase4_eval
#SBATCH --partition=aiq
#SBATCH --gres=gpu:h100:1
#SBATCH --account=did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_aiq_qos
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs4/fase4_eval_%j.out"
#SBATCH --error="/home/F.SATURNINO/Progetto_tesi_SigLIP/jobs/logs4/fase4_eval_%j.err"

source /hpc/apps/anaconda/anaconda3/etc/profile.d/conda.sh
conda activate oos
module load cuda12.8/toolkit/12.8.0

export HF_HOME="/home/F.SATURNINO/Progetto_tesi_SigLIP/hf_models"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export HPC_ROOT="/home/F.SATURNINO/Progetto_tesi_SigLIP"
cd "$HPC_ROOT"

echo "=================================================="
echo "Job: oos_fase4_eval — valutazione degli 8 checkpoint joint-trained"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "=================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p "$HPC_ROOT/jobs/logs4"
mkdir -p "$HPC_ROOT/results/fase4_eval"

# NOTA: --time=03:00:00 e' una stima — la valutazione non ha backward
# (meno pesante del training) ma comunque gira il forward completo di 8
# checkpoint giant sull'intero test set, in sequenza. Se non basta,
# valuta di dividere in 2 lanci separati (4 checkpoint ciascuno) invece
# di alzare il tempo oltre il limite della partition.

python -u src/evaluation/evaluate_fase4_joint.py \
    --checkpoints \
        weights/fase4_joint/aug__completa__neg025_pad50_nocap/best_model.pt \
        weights/fase4_joint/aug__completa__neg05_pad50_nocap/best_model.pt \
        weights/fase4_joint/no_aug__completa__neg025_pad50_nocap/best_model.pt \
        weights/fase4_joint/no_aug__completa__neg05_pad50_nocap/best_model.pt \
        weights/fase4_joint/aug__vision_mlp__neg025_pad50_nocap/best_model.pt \
        weights/fase4_joint/aug__vision_mlp__neg05_pad50_nocap/best_model.pt \
        weights/fase4_joint/no_aug__vision_mlp__neg025_pad50_nocap/best_model.pt \
        weights/fase4_joint/no_aug__vision_mlp__neg05_pad50_nocap/best_model.pt \
    --labels \
        aug_completa_neg025 aug_completa_neg05 \
        noaug_completa_neg025 noaug_completa_neg05 \
        aug_mlp_neg025 aug_mlp_neg05 \
        noaug_mlp_neg025 noaug_mlp_neg05 \
    --data_dir dataset_finale \
    --out_dir results/fase4_eval \
    --yolo_arch_ref weights/fase1_baseline/aug/best.pt

echo "=================================================="
echo "Job completato: $(date)"
echo "=================================================="