#!/bin/bash
#SBATCH --job-name=ecg-ptbxl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm-%j.out

# Usage: sbatch scripts/slurm_train.sh [config]   (default: configs/default.yaml)
CONFIG=${1:-configs/default.yaml}

module load anaconda3/2024.06
source activate ecg
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1   # so epoch lines show up in the log as they happen

NAME=$(python -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$CONFIG")
for seed in 0 1 2; do
  python scripts/train.py --config "$CONFIG" --seed $seed
done
python scripts/aggregate.py runs/$NAME
