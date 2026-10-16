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


def fmt(values):
    return f"{np.mean(values):.4f} +/- {np.std(values):.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "seed*", "results.json")))
    if not files:
        raise SystemExit(f"no results.json files under {args.run_dir}")
    rs = []
    for path in files:
        with open(path) as f:
            rs.append(json.load(f))
    print(f"{len(rs)} seeds\n")

    def clean(c):
        return np.array([r["test_macro_auroc"] if c == "macro" else r["test_per_class"][c] for r in rs])

    def probed(c, k, kind):
        e = [r["probe"][k][kind] for r in rs]
        return np.array([x["macro"] if c == "macro" else x["per_class"][c] for x in e])

    print(f"{'class':6s} {'test AUROC':>20s}")
    for c in CLASSES + ["macro"]:
        print(f"{c:6s} {fmt(clean(c)):>20s}")

    if not all("probe" in r for r in rs):
        return
    ks = sorted(rs[0]["probe"], key=int)
    for kind in ["shuffle", "roll"]:
        print(f"\nAUROC drop under {kind} (test minus probed), by number of segments")
        print(f"{'class':6s}" + "".join(f" {k + ' seg':>20s}" for k in ks))
        for c in CLASSES + ["macro"]:
            print(f"{c:6s}" + "".join(f" {fmt(clean(c) - probed(c, k, kind)):>20s}" for k in ks))

    summary = {c: {"test": float(clean(c).mean()),
                   **{f"{kind}_{k}": float(probed(c, k, kind).mean()) for kind in ["shuffle", "roll"] for k in ks}}
               for c in CLASSES + ["macro"]}
    with open(os.path.join(args.run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
