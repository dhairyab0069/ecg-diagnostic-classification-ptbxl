#!/bin/bash
#SBATCH --job-name=ecg-external
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-external-%j.out

# Usage: sbatch scripts/slurm_external.sh   (downloads Georgia once, then scores every model on it)
set -eo pipefail
module load anaconda3/2024.06
source activate ecg
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"   # use this checkout's code, not wherever the env was installed from

[ -f data/georgia/cache_100hz.npz ] || python scripts/download_external.py
python scripts/external_eval.py runs/resnet1d_baseline runs/resnet1d_shuffle_trained runs/resnet1d_leaky_relu runs/resnet1d_tanh
