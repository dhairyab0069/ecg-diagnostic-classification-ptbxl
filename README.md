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

Test fold, mean ± std over 3 seeds. Drops are test AUROC minus probed AUROC for the baseline model
(10 segments = 1 s each, 50 segments = 0.2 s each).

| Class | Baseline test AUROC | Shuffle-trained test AUROC | Drop, shuffle 10 seg | Drop, roll 10 seg | Drop, shuffle 50 seg | Drop, roll 50 seg |
|-------|------|------|------|------|------|------|
| NORM | 0.944 ± 0.001 | 0.943 ± 0.002 | 0.006 | 0.011 | 0.080 | 0.024 |
| MI | 0.922 ± 0.003 | 0.923 ± 0.004 | 0.002 | 0.006 | 0.036 | 0.021 |
| STTC | 0.932 ± 0.002 | 0.932 ± 0.001 | 0.006 | 0.014 | 0.153 | 0.043 |
| CD | 0.916 ± 0.003 | 0.920 ± 0.001 | 0.003 | 0.010 | 0.056 | 0.039 |
| HYP | 0.896 ± 0.003 | 0.894 ± 0.003 | 0.002 | 0.007 | 0.055 | 0.020 |
| **Macro** | 0.922 ± 0.001 | 0.923 ± 0.001 | 0.004 | 0.010 | 0.076 | 0.029 |

Macro AUROC drop for the baseline across the segment sweep:

| Segments | 2 | 5 | 10 | 20 | 50 |
|----------|---|---|----|----|----|
| Shuffle | 0.001 | 0.001 | 0.004 | 0.019 | 0.076 |
| Roll control | 0.001 | 0.002 | 0.010 | 0.032 | 0.030 |

Full tables, including the shuffle-trained model's probes, are printed by `scripts/aggregate.py`.

### Findings

- **Order at the 1 s scale and above is not needed.** At 2 to 20 segments the shuffle costs no more than the
  roll control, which keeps segment order, so the drop comes from the cuts rather than from lost order.
  A model trained on shuffled 1 s segments matches the baseline (0.923 vs 0.922 macro).
- **Order within a beat matters.** At 50 segments (0.2 s, shorter than one beat) the shuffle drop (0.076)
  is well above the control (0.030), because the P wave, QRS complex and T wave no longer appear in order.
- **STTC depends on it most** (0.153 drop vs 0.043 control at 50 segments), consistent with ST and T changes
  being defined relative to the QRS complex that precedes them.
- Caveat: the roll control adds a discontinuity inside each segment as well as at the boundaries, so it
  likely overstates the cost of the cuts; the conclusion that coarse order is unused does not depend on it,
  since the shuffle drop at 10 segments is only 0.004 in absolute terms.

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
