#!/bin/bash
#SBATCH --job-name=ecg-setup
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-setup-%j.out
# One-time setup: conda env, PTB-XL (100 Hz) from the AWS mirror, signal cache, tests.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1

module load anaconda3/2024.06
if ! conda env list | grep -q '^ecg '; then
  conda create -y -n ecg python=3.11
fi
source activate ecg
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"

python scripts/download_data.py --rate 100
# build the signal cache once, so parallel training jobs don't each rebuild it
python -c "from ecg.data import load_ptbxl; X, Y, f = load_ptbxl('data/ptb-xl/1.0.3', 100); print(X.shape, Y.sum(0))"

pytest -q
