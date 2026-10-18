"""Score saved checkpoints on the Georgia database without retraining, clean and under the shuffle probes.

Each checkpoint is normalized with its own PTB-XL training statistics. Classes with fewer than
--min-positives positives in Georgia are left out (MI has 7). The PTB-XL test AUROC over the same
classes is reported alongside, so the two macros compare like for like.

Writes runs/<model>/seed<N>/external_georgia.json and runs/<model>/external_georgia_summary.json.

Example:
    python scripts/download_external.py
    python scripts/external_eval.py runs/resnet1d_baseline runs/resnet1d_tanh
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from ecg import CLASSES
from ecg.data import make_loader, normalize
from ecg.engine import predict
from ecg.model import build_model
from ecg.transforms import roll_segments, shuffle_segments
from ecg.utils import get_device, save_json

CONDITIONS = {"clean": None,
              **{f"{name}_{k}": (fn, k) for k in (10, 50) for name, fn in
                 [("shuffle", shuffle_segments), ("roll", roll_segments)]}}


def scores(probs, y, classes):
    per_class = {c: float(roc_auc_score(y[:, CLASSES.index(c)], probs[:, CLASSES.index(c)])) for c in classes}
    return {"macro": float(np.mean(list(per_class.values()))), "per_class": per_class}


def eval_seed(seed_dir, X, Y, classes, device):
    ckpt = torch.load(os.path.join(seed_dir, "best.pt"), map_location=device, weights_only=False)
    model = build_model(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    x = normalize(X, ckpt["norm"]["mean"], ckpt["norm"]["std"])
    loader = make_loader(x, torch.from_numpy(Y), 512, shuffle=False)
    seed = int(os.path.basename(seed_dir).removeprefix("seed"))
    out = {}
    for name, probe in CONDITIONS.items():
        torch.manual_seed(seed)  # same perturbations as train.py's probes
        transform = None if probe is None else (lambda b, fn=probe[0], k=probe[1]: fn(b, k))
        probs, _ = predict(model, loader, device, transform)
        out[name] = scores(probs, Y, classes)

    ptbxl = json.load(open(os.path.join(seed_dir, "results.json")))["test"]["per_class"]
    out["ptbxl_test"] = {"macro": float(np.mean([ptbxl[c] for c in classes])),
                         "per_class": {c: ptbxl[c] for c in classes}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--data", default="data/georgia/cache_100hz.npz")
    ap.add_argument("--min-positives", type=int, default=50)
    args = ap.parse_args()
    device = get_device()

    d = np.load(args.data)
    X, Y = d["X"], d["Y"]
    classes = [c for i, c in enumerate(CLASSES) if Y[:, i].sum() >= args.min_positives]
    print(f"Georgia: {len(X)} records; evaluating {classes}")

    for run_dir in args.run_dirs:
        per_seed = []
        for seed_dir in sorted(glob.glob(os.path.join(run_dir, "seed*"))):
            res = eval_seed(seed_dir, X, Y, classes, device)
            save_json(res, os.path.join(seed_dir, "external_georgia.json"))
            per_seed.append(res)
        summary = {cond: {c: {"mean": float(np.mean([r[cond]["per_class"][c] if c != "macro" else r[cond]["macro"]
                                                    for r in per_seed])),
                              "std": float(np.std([r[cond]["per_class"][c] if c != "macro" else r[cond]["macro"]
                                                  for r in per_seed]))}
                          for c in classes + ["macro"]}
                   for cond in per_seed[0]}
        summary["classes"], summary["records"] = classes, int(len(X))
        save_json(summary, os.path.join(run_dir, "external_georgia_summary.json"))

        print(f"\n{run_dir} ({len(per_seed)} seeds)")
        print(f"{'class':6s} {'PTB-XL test':>16s} {'Georgia':>16s} {'drop 1 s':>9s} {'drop 0.2 s':>10s}")
        for c in classes + ["macro"]:
            g = summary["clean"][c]["mean"]
            print(f"{c:6s} {summary['ptbxl_test'][c]['mean']:9.4f} ± {summary['ptbxl_test'][c]['std']:.4f}"
                  f" {g:9.4f} ± {summary['clean'][c]['std']:.4f}"
                  f" {g - summary['shuffle_10'][c]['mean']:9.4f} {g - summary['shuffle_50'][c]['mean']:10.4f}")


if __name__ == "__main__":
    main()
