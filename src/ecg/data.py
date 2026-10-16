"""PTB-XL loading, label aggregation, normalization, and data loaders."""
import ast
import os

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, TensorDataset

from ecg import CLASSES


def load_ptbxl(root, sampling_rate=100):
    """Return signals (N, T, 12), multi-hot labels (N, 5), and stratified folds (N,)."""
    cache = os.path.join(root, f"cache_{sampling_rate}hz.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["Y"], d["fold"]

    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]

    def to_superclasses(codes):
        return {scp.loc[c, "diagnostic_class"] for c in codes if c in scp.index}

    labels = df.scp_codes.apply(to_superclasses)
    keep = labels.apply(len) > 0
    df, labels = df[keep], labels[keep]

    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    print(f"Reading {len(df)} records (first run only, then cached)...")
    X = np.stack([wfdb.rdsamp(os.path.join(root, f))[0] for f in df[col]]).astype(np.float32)
    Y = np.array([[c in lab for c in CLASSES] for lab in labels], dtype=np.float32)
    fold = df.strat_fold.values
    tmp = f"{cache}.{os.getpid()}.tmp.npz"  # atomic write, in case two jobs build the cache at once
    np.savez(tmp, X=X, Y=Y, fold=fold)
    os.replace(tmp, cache)
    return X, Y, fold


def normalize(X, mean, std):
    """Per-lead z-score, then reshape to (N, 12, T) for Conv1d."""
    return torch.from_numpy(((X - mean) / std).transpose(0, 2, 1).copy())


def build_splits(cfg):
    X, Y, fold = load_ptbxl(cfg["root"], cfg["sampling_rate"])
    tr = np.isin(fold, cfg["train_folds"])
    va = fold == cfg["val_fold"]
    te = fold == cfg["test_fold"]

    # statistics from the training split only, to avoid leakage
    mean = X[tr].mean(axis=(0, 1), keepdims=True)
    std = X[tr].std(axis=(0, 1), keepdims=True) + 1e-6

    splits = {name: (normalize(X[m], mean, std), torch.from_numpy(Y[m]))
              for name, m in [("train", tr), ("val", va), ("test", te)]}
    return splits, {"mean": mean, "std": std}


def make_loader(x, y, batch_size, shuffle, num_workers=0, pin_memory=False):
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      drop_last=shuffle, num_workers=num_workers, pin_memory=pin_memory)
