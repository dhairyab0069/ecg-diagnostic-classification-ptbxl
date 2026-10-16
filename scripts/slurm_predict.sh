#!/bin/bash
#SBATCH --job-name=ecg-predict
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-predict-%j.out

module load anaconda3/2024.06
source activate ecg
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

for run in runs/resnet1d_baseline runs/resnet1d_shuffle_trained; do
  python scripts/predict.py "$run"
done
