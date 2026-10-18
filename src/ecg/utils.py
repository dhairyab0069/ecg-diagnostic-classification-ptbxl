import json
import os
import random

import numpy as np
import torch
import yaml


def load_config(path, overrides=()):
    """Load a YAML config and apply dotted overrides like train.epochs=10."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        value = yaml.safe_load(value)
        if isinstance(value, str):  # PyYAML reads "1e-4" (no decimal point) as a string
            try:
                value = float(value)
            except ValueError:
                pass
        node[leaf] = value
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
