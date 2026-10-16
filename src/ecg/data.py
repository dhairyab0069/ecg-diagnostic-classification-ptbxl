"""PTB-XL loading, label aggregation, normalization, and data loaders."""
import ast
import os

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, TensorDataset

from ecg import CLASSES


def record_column(sampling_rate):
    """Column of ptbxl_database.csv holding record paths at this sampling rate."""
    return "filename_lr" if sampling_rate == 100 else "filename_hr"


def load_labels(root):
    """Return the metadata table and each record's set of diagnostic superclasses,
    dropping records with no diagnostic statement."""
    df = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]
    labels = df.scp_codes.apply(lambda codes: {scp.loc[c, "diagnostic_class"] for c in codes if c in scp.index})
    keep = labels.apply(len) > 0
    return df[keep], labels[keep]


def read_signals(root, paths):
    print(f"Reading {len(paths)} records (first run only, then cached)...")
    return np.stack([wfdb.rdsamp(os.path.join(root, f))[0] for f in paths]).astype(np.float32)


def save_cache(path, **arrays):
    tmp = f"{path}.{os.getpid()}.tmp.npz"  # atomic write, in case two jobs build the cache at once
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def load_ptbxl(root, sampling_rate=100):
    """Return signals (N, T, 12), multi-hot labels (N, 5), and stratified folds (N,)."""
    cache = os.path.join(root, f"cache_{sampling_rate}hz.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        return d["X"], d["Y"], d["fold"]

    df, labels = load_labels(root)
    X = read_signals(root, df[record_column(sampling_rate)])
    Y = np.array([[c in lab for c in CLASSES] for lab in labels], dtype=np.float32)
    fold = df.strat_fold.values
    save_cache(cache, X=X, Y=Y, fold=fold)
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


def build_loaders(cfg, device):
    splits, norm = build_splits(cfg["data"])
    tcfg, pin = cfg["train"], device.type == "cuda"
    loaders = {name: make_loader(*splits[name], tcfg["batch_size"] if name == "train" else 512,
                                 name == "train", tcfg["num_workers"], pin)
               for name in splits}
    print("  ".join(f"{name} {len(x)}" for name, (x, _) in splits.items()) + f"  device {device}")
    return loaders, norm
