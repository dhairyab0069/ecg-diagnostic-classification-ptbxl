"""Summarize results across seeds as mean +/- std.

Example:
    python scripts/aggregate.py runs/resnet1d_baseline
"""
import argparse
import glob
import json
import os

import numpy as np

from ecg import CLASSES
from ecg.metrics import score

ROWS = CLASSES + ["macro"]


def fmt(values):
    return f"{np.mean(values):.4f} +/- {np.std(values):.4f}"


def load_results(run_dir):
    files = sorted(glob.glob(os.path.join(run_dir, "seed*", "results.json")))
    if not files:
        raise SystemExit(f"no results.json files under {run_dir}")
    rs = []
    for path in files:
        with open(path) as f:
            rs.append(json.load(f))
    return rs


def across_seeds(rs, get_entry, cls):
    """AUROC for one class (or "macro") from every seed, as an array."""
    return np.array([score(get_entry(r), cls) for r in rs])


def print_table(title, columns):
    """columns: {header: fn(cls) -> per-seed values}."""
    print(f"\n{title}")
    print(f"{'class':6s}" + "".join(f" {h:>20s}" for h in columns))
    for c in ROWS:
        print(f"{c:6s}" + "".join(f" {fmt(fn(c)):>20s}" for fn in columns.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()

    rs = load_results(args.run_dir)
    print(f"{len(rs)} seeds")
    test = lambda c: across_seeds(rs, lambda r: r["test"], c)
    print_table("Test AUROC", {"test AUROC": test})

    if not all("probe" in r for r in rs):
        return
    ks = sorted(rs[0]["probe"], key=int)
    probed = lambda k, name: lambda c: across_seeds(rs, lambda r: r["probe"][k][name], c)
    probe_names = list(rs[0]["probe"][ks[0]])
    for name in probe_names:
        print_table(f"AUROC drop under {name} (test minus probed), by number of segments",
                    {f"{k} seg": lambda c, k=k: test(c) - probed(k, name)(c) for k in ks})

    summary = {c: {"test": float(test(c).mean()),
                   **{f"{name}_{k}": float(probed(k, name)(c).mean()) for name in probe_names for k in ks}}
               for c in ROWS}
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
