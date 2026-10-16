"""Save per-record test probabilities for each seed's best checkpoint, clean and under the probes.

Writes runs/<name>/seed<N>/preds.npz with val_prob, val_true, test_true and one test_<condition>
probability array per condition (clean, shuffle_<k>, roll_<k>). Used by plot_results.py
for confusion matrices.

Example:
    python scripts/predict.py runs/resnet1d_baseline
"""
import argparse
import glob
import os

import numpy as np
import torch

from ecg.data import build_loaders
from ecg.engine import predict
from ecg.model import build_model
from ecg.transforms import roll_segments, shuffle_segments
from ecg.utils import get_device

CONDITIONS = {"clean": None,
              **{f"{name}_{k}": (fn, k) for k in (10, 50) for name, fn in
                 [("shuffle", shuffle_segments), ("roll", roll_segments)]}}


def predict_seed(seed_dir, device):
    ckpt = torch.load(os.path.join(seed_dir, "best.pt"), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    loaders, _ = build_loaders(cfg, device)
    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    seed = int(os.path.basename(seed_dir).removeprefix("seed"))

    val_prob, val_true = predict(model, loaders["val"], device)
    out = {"val_prob": val_prob, "val_true": val_true}
    for name, probe in CONDITIONS.items():
        torch.manual_seed(seed)  # same perturbations as train.py's probes
        transform = None if probe is None else (lambda x, fn=probe[0], k=probe[1]: fn(x, k))
        out[f"test_{name}"], out["test_true"] = predict(model, loaders["test"], device, transform)
    np.savez(os.path.join(seed_dir, "preds.npz"), **out)
    print(f"wrote {seed_dir}/preds.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()
    device = get_device()
    for seed_dir in sorted(glob.glob(os.path.join(args.run_dir, "seed*"))):
        predict_seed(seed_dir, device)


if __name__ == "__main__":
    main()
