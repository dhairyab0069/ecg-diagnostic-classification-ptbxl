"""End-to-end checks: training loop, and the train -> predict -> aggregate scripts on synthetic data."""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

from conftest import ROOT, make_dataset
from ecg import CLASSES
from ecg.engine import evaluate, predict, train_one_epoch
from ecg.model import ResNet1D

SMALL = {"widths": (8, 8, 16, 16), "kernel_size": 7, "dropout": 0.0}


def loader(n=64, seed=0):
    X, Y, _ = make_dataset(n=n, seed=seed)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X.transpose(0, 2, 1).copy()), torch.from_numpy(Y))
    return torch.utils.data.DataLoader(ds, batch_size=16)


def test_predict_returns_probabilities():
    probs, targets = predict(ResNet1D(**SMALL), loader(), torch.device("cpu"))
    assert probs.shape == targets.shape == (64, len(CLASSES))
    assert (probs >= 0).all() and (probs <= 1).all()


def test_predict_applies_the_transform():
    model = ResNet1D(**SMALL).eval()
    zeros, _ = predict(model, loader(), torch.device("cpu"), transform=torch.zeros_like)
    expected = torch.sigmoid(model(torch.zeros(1, 12, 1000))).detach().numpy()
    assert np.allclose(zeros, expected, atol=1e-5)


def test_training_learns_a_learnable_task():
    torch.manual_seed(0)
    model, data, device = ResNet1D(**SMALL), loader(n=128), torch.device("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    aug = {"scale_min": 1.0, "scale_max": 1.0, "noise_std": 0.0, "random_shift": False}
    losses = [train_one_epoch(model, data, opt, sched, scaler, nn.BCEWithLogitsLoss(), device, aug) for _ in range(8)]
    assert losses[-1] < 0.6 * losses[0]
    assert evaluate(model, data, device)["macro"] > 0.9


def run(script, *args):
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", script), *args],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


@pytest.mark.slow
def test_train_predict_aggregate_end_to_end(fake_ptbxl, tmp_path):
    root, _ = fake_ptbxl
    out_dir = tmp_path / "runs"
    overrides = [f"data.root={root}", f"out_dir={out_dir}", "name=tiny", "train.epochs=2",
                 "train.batch_size=32", "train.num_workers=0", "model.widths=[8,8,16,16]",
                 "probe.num_segments=[2,10]"]
    for seed in (0, 1):
        run("train.py", "--config", "configs/default.yaml", "--seed", str(seed), "--set", *overrides)

    run_dir = out_dir / "tiny"
    results = json.loads((run_dir / "seed0" / "results.json").read_text())
    assert set(results["test"]["per_class"]) == set(CLASSES)
    assert set(results["probe"]) == {"2", "10"} and set(results["probe"]["2"]) == {"shuffle", "roll"}
    assert len(json.loads((run_dir / "seed0" / "history.json").read_text())) == 2

    run("predict.py", str(run_dir))
    preds = np.load(run_dir / "seed0" / "preds.npz")
    assert preds["test_clean"].shape == preds["test_true"].shape

    run("aggregate.py", str(run_dir))
    summary = json.loads((run_dir / "summary.json").read_text())
    seeds = [json.loads((run_dir / f"seed{s}" / "results.json").read_text())["test"]["macro"] for s in (0, 1)]
    assert summary["macro"]["test"] == pytest.approx(np.mean(seeds))


@pytest.mark.parametrize("model", ["resnet1d_baseline", "resnet1d_shuffle_trained", "resnet1d_leaky_relu",
                                   "resnet1d_tanh"])
def test_committed_summaries_match_committed_results(model, tmp_path):
    # the README tables come from summary.json; recomputing them from the per-seed results must agree
    shutil.copytree(os.path.join(ROOT, "runs", model), tmp_path / model,
                    ignore=shutil.ignore_patterns("*.npz", "*.pt"))
    run("aggregate.py", str(tmp_path / model))
    fresh = json.loads((tmp_path / model / "summary.json").read_text())
    committed = json.loads(open(os.path.join(ROOT, "runs", model, "summary.json")).read())
    for cls in committed:
        for key in committed[cls]:
            assert fresh[cls][key] == pytest.approx(committed[cls][key], abs=1e-9), (cls, key)
