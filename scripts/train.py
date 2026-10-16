"""Train a model, select the best epoch on validation, evaluate on test.

Example:
    python scripts/train.py --config configs/default.yaml --seed 0
    python scripts/train.py --config configs/default.yaml --set train.epochs=5 probe.enabled=false
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import yaml

from ecg import CLASSES
from ecg.data import build_loaders
from ecg.engine import evaluate, train_one_epoch
from ecg.metrics import score
from ecg.model import build_model
from ecg.transforms import roll_segments, shuffle_segments
from ecg.utils import get_device, load_config, save_json, set_seed

PROBES = {"shuffle": shuffle_segments, "roll": roll_segments}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.lr=0.001")
    return ap.parse_args()


def make_run_dir(cfg, seed):
    run_dir = os.path.join(cfg["out_dir"], cfg["name"], f"seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    return run_dir


def build_optimizer(model, tcfg, steps_per_epoch):
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tcfg["lr"], total_steps=tcfg["epochs"] * steps_per_epoch)
    return opt, sched


def fit(model, loaders, cfg, device, save_best):
    """Train for the configured epochs, calling save_best() whenever val macro AUROC improves."""
    tcfg = cfg["train"]
    opt, sched = build_optimizer(model, tcfg, len(loaders["train"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()

    best, history = -1.0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        loss = train_one_epoch(model, loaders["train"], opt, sched, scaler, loss_fn, device, cfg["augment"])
        val_auc = evaluate(model, loaders["val"], device)["macro"]
        history.append({"epoch": epoch, "train_loss": loss, "val_macro_auroc": val_auc})
        if val_auc > best:
            best = val_auc
            save_best()
        print(f"epoch {epoch:02d}  loss {loss:.4f}  val AUROC {val_auc:.4f}  best {best:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    return best, history


def run_probes(model, loader, device, num_segments, seed):
    """Evaluate under each probe transform for each segment count: {k: {probe: scores}}."""
    ks = num_segments if isinstance(num_segments, list) else [num_segments]
    out = {}
    for k in ks:
        out[str(k)] = {}
        for name, fn in PROBES.items():
            torch.manual_seed(seed)  # same permutations / shifts for every model and seed
            out[str(k)][name] = evaluate(model, loader, device, lambda x: fn(x, k))
    return out


def report(title, test, probed=None):
    """Print macro and per-class AUROC, plus the drop under each probe if given."""
    print(f"\n{title}")
    for c in ["macro"] + CLASSES:
        line = f"  {c:5s} {score(test, c):.4f}"
        for name, entry in (probed or {}).items():
            line += f"   {name} {score(entry, c):.4f} (drop {score(test, c) - score(entry, c):+.4f})"
        print(line)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.set)
    set_seed(args.seed)
    device = get_device()
    run_dir = make_run_dir(cfg, args.seed)
    ckpt_path = os.path.join(run_dir, "best.pt")

    loaders, norm = build_loaders(cfg, device)
    model = build_model(cfg["model"]).to(device)
    save_best = lambda: torch.save({"model": model.state_dict(), "config": cfg, "norm": norm}, ckpt_path)
    best, history = fit(model, loaders, cfg, device, save_best)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model"])
    test = evaluate(model, loaders["test"], device)
    results = {"seed": args.seed, "val_macro_auroc": best, "test": test}
    report("TEST AUROC", test)

    if cfg["probe"]["enabled"]:
        results["probe"] = run_probes(model, loaders["test"], device, cfg["probe"]["num_segments"], args.seed)
        for k, probed in results["probe"].items():
            report(f"{k} SEGMENTS", test, probed)

    save_json(results, os.path.join(run_dir, "results.json"))
    save_json(history, os.path.join(run_dir, "history.json"))


if __name__ == "__main__":
    main()
