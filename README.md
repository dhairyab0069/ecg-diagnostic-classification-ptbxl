# ECG Diagnostic Classification on PTB-XL

Multi-label classification of 12-lead ECGs into five diagnostic superclasses
(NORM, MI, STTC, CD, HYP) with a 1D ResNet in PyTorch, plus a temporal probe
that tests how much each diagnosis depends on rhythm and timing.

## Question

Does the model use temporal structure, or mostly local waveform shape?
The probe re-evaluates the trained model on test ECGs whose 1-second segments
are randomly shuffled. Beat shape survives, but rhythm and order do not.
The per-class AUROC drop shows which diagnoses depend on temporal structure.

Three checks make the probe interpretable:

- **Segment sweep** (2, 5, 10, 20, 50 segments): at 50 segments (0.2 s) the cuts split individual beats.
- **Roll control**: cuts at the same boundaries but keeps segment order, separating the effect of
  discontinuities at the cuts from the effect of lost temporal order.
- **Shuffle-trained model** (`configs/shuffle_trained.yaml`): if a model trained on shuffled ECGs matches
  the baseline on a class, that diagnosis does not need temporal order.

## Setup

```bash
pip install -e ".[dev]"
python scripts/download_data.py --rate 100   # fast AWS mirror, ~0.5 GB
pytest
```

## Run

```bash
python scripts/train.py --config configs/default.yaml --seed 0
python scripts/aggregate.py runs/resnet1d_baseline
```

Override any config value from the command line:

```bash
python scripts/train.py --set train.epochs=10 model.dropout=0.5
```

On SLURM:

```bash
sbatch scripts/slurm_setup.sh                                     # once: env, data, cache, tests
sbatch scripts/slurm_train.sh                                     # baseline, 3 seeds + aggregate
sbatch scripts/slurm_train.sh configs/shuffle_trained.yaml        # shuffle-trained variant
```

## Protocol

- Dataset: PTB-XL v1.0.3, 100 Hz, 10 s recordings
- Split: official stratified folds (1 to 8 train, 9 validation, 10 test)
- Normalization: per-lead z-score using training-set statistics only
- Model selection: best validation macro AUROC
- Metric: macro AUROC on the test fold, mean and std over 3 seeds

## Results

| Class | Test AUROC | Shuffled AUROC | Drop |
|-------|-----------|----------------|------|
| NORM  |           |                |      |
| MI    |           |                |      |
| STTC  |           |                |      |
| CD    |           |                |      |
| HYP   |           |                |      |
| Macro |           |                |      |

## Structure

```
configs/default.yaml     hyperparameters and paths
src/ecg/data.py          loading, label aggregation, splits
src/ecg/model.py         1D ResNet
src/ecg/transforms.py    augmentation, shuffle probe, roll control
src/ecg/engine.py        train and predict loops
src/ecg/metrics.py       AUROC
scripts/train.py         training entry point
scripts/aggregate.py     mean and std across seeds
scripts/download_data.py parallel download from the PhysioNet AWS mirror
scripts/slurm_setup.sh   one-time SLURM setup
scripts/slurm_train.sh   3-seed SLURM job (takes a config path)
tests/                   shape and probe sanity checks
```

## Reference

Wagner et al., PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 2020.
