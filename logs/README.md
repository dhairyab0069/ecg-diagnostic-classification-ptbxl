# Run logs

SLURM job output from the runs reported in the main README (Northeastern Explorer, October 2026).
Per-seed metrics are in `runs/<model>/seed<N>/results.json`, epoch histories in `history.json`, the
resolved config in `config.yaml`, and the 3-seed means in `runs/<model>/summary.json`.

| Log | Job | What it contains |
|---|---|---|
| `setup_11138450.log` | `scripts/slurm_setup.sh` | Environment install, PTB-XL download (43,598 files), signal cache (21,388 records), `pytest` (6 passed) |
| `train_baseline_11138451.log` | `scripts/slurm_train.sh` | Baseline: 30 epochs × 3 seeds, test and probe AUROC per seed, then the aggregate tables |
| `train_shuffle_trained_11138452.log` | `scripts/slurm_train.sh configs/shuffle_trained.yaml` | Shuffle-trained model, same layout |
| `predict_11139237.log` | `scripts/slurm_predict.sh` | Saving per-record predictions for the confusion matrices |

Each training log ends with the 3-seed tables printed by `scripts/aggregate.py`. Hardware: one NVIDIA V100-SXM2
per training job, about 2 s per epoch.
