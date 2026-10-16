"""Download PTB-XL v1.0.3 from PhysioNet's AWS mirror, in parallel.

Much faster than the recursive wget in download_data.sh, and fetches only the
sampling rate you need (100 Hz is about 0.5 GB).

Example:
    python scripts/download_data.py --rate 100
"""
import argparse
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

BASE = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"


def fetch(rel, out):
    path = os.path.join(out, rel)
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(BASE + rel, path + ".part")
    os.replace(path + ".part", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ptb-xl/1.0.3")
    ap.add_argument("--rate", type=int, choices=[100, 500], default=100)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    for f in ["ptbxl_database.csv", "scp_statements.csv"]:
        fetch(f, args.out)
    df = pd.read_csv(os.path.join(args.out, "ptbxl_database.csv"))
    col = "filename_lr" if args.rate == 100 else "filename_hr"
    files = [f"{r}.{ext}" for r in df[col] for ext in ("dat", "hea")]
    print(f"Downloading {len(files)} files...")
    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda f: fetch(f, args.out), files), 1):
            if i % 5000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    print(f"Done. Data is in {args.out}")


if __name__ == "__main__":
    main()
