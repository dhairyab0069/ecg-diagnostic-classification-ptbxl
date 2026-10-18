import os
import sys

import numpy as np
import pytest

from ecg import CLASSES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))  # so tests can import plot_results etc.


def make_dataset(n=300, t=1000, seed=0):
    """Synthetic PTB-XL stand-in: signals (n, t, 12), multi-hot labels (n, 5), folds 1-10.

    Each class adds its own sine wave, so a model can learn the labels.
    """
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, len(CLASSES))) < 0.4).astype(np.float32)
    time = np.arange(t) / 100
    X = 0.1 * rng.standard_normal((n, t, 12)).astype(np.float32)
    for c in range(len(CLASSES)):
        X += Y[:, c, None, None] * np.sin(2 * np.pi * (c + 1) * time)[None, :, None].astype(np.float32)
    fold = np.arange(n) % 10 + 1
    return X, Y, fold


@pytest.fixture
def fake_ptbxl(tmp_path):
    """A data root holding only the signal cache, which load_ptbxl reads before anything else."""
    X, Y, fold = make_dataset()
    for f in (9, 10):  # AUROC needs positives and negatives in val and test
        assert 0 < Y[fold == f].mean(0).min() and Y[fold == f].mean(0).max() < 1
    root = tmp_path / "ptb-xl"
    root.mkdir()
    np.savez(root / "cache_100hz.npz", X=X, Y=Y, fold=fold)
    return root, (X, Y, fold)
