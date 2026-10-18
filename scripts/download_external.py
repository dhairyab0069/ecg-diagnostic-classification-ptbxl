"""Download the Georgia 12-lead ECG database (PhysioNet/CinC Challenge 2021) and build a 100 Hz cache.

Keeps 10 s recordings whose codes give at least one diagnostic label (see src/ecg/external.py),
and writes data/georgia/cache_100hz.npz with X (N, 1000, 12) in mV, Y (N, 5) and record ids.

physionet.org serves these files at roughly 100 KB/s per client, so all ~10,000 signals take hours.
Headers are small, so they are all fetched first; then signals are fetched only for a seeded random
sample of the eligible records (--max-records, default 3,000; 0 = all of them).

Example:
    python scripts/download_external.py
"""
import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests  # installed with wfdb

from ecg import CLASSES
from ecg.external import LEADS, labels_from_dx, load_record, parse_header, to_100hz

BASE = "https://physionet.org/files/challenge-2021/1.0.3/training/georgia/"
FOLDERS = [f"g{i}" for i in range(1, 12)]


_local = threading.local()


def fetch(rel, out, tries=6):
    """Download one file; return False if the server doesn't have it."""
    path = os.path.join(out, rel)
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # one keep-alive session per thread: physionet.org throttles many new connections (HTTP 503)
    session = _local.__dict__.setdefault("session", requests.Session())
    for attempt in range(tries):
        try:
            r = session.get(BASE + rel, timeout=60)
            if r.status_code == 404:  # a few records listed in RECORDS were removed from the release
                return False
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)
    with open(path + ".part", "wb") as f:
        f.write(r.content)
    os.replace(path + ".part", path)
    return True


def download(records, ext, args):
    files = [f"{r}.{ext}" for r in records]
    print(f"Downloading {len(files)} .{ext} files...", flush=True)
    missing = []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (r, ok) in enumerate(zip(records, pool.map(lambda f: fetch(f, args.out), files)), 1):
            if not ok:
                missing.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(files)}", flush=True)
    if missing:
        print(f"  {len(missing)} not on the server, skipped: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/georgia")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-records", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for folder in FOLDERS:
        fetch(f"{folder}/RECORDS", args.out)
    records = [f"{folder}/{name.strip()}" for folder in FOLDERS
               for name in open(os.path.join(args.out, folder, "RECORDS")) if name.strip()]
    missing = download(records, "hea", args)
    records = [r for r in records if r not in set(missing)]

    eligible, skipped = {}, Counter()
    for r in records:
        fs, n, leads, gains, dx = parse_header(open(os.path.join(args.out, r + ".hea")).read())
        y = labels_from_dx(dx)
        if leads != LEADS:
            skipped["lead order differs"] += 1
        elif n != 10 * fs:
            skipped["not 10 s long"] += 1
        elif y is None:
            skipped["no diagnostic label"] += 1
        else:
            eligible[r] = (fs, gains, y)
    ids = sorted(eligible)
    if 0 < args.max_records < len(ids):
        ids = sorted(np.random.default_rng(args.seed).choice(ids, args.max_records, replace=False).tolist())
    if download(ids, "mat", args):
        raise SystemExit("some signal files are missing although their headers exist")

    X = np.stack([to_100hz(load_record(os.path.join(args.out, r + ".mat"), eligible[r][1]), eligible[r][0])
                  for r in ids])
    Y = np.stack([eligible[r][2] for r in ids])
    np.savez(os.path.join(args.out, "cache_100hz.npz"), X=X, Y=Y, ids=np.array(ids))

    counts = {"records": len(records), "missing_on_server": len(missing), "eligible": len(eligible), "sampled": len(ids), "seed": args.seed,
              "skipped": dict(skipped), "positives": {c: int(Y[:, i].sum()) for i, c in enumerate(CLASSES)}}
    with open(os.path.join(args.out, "counts.json"), "w") as f:
        json.dump(counts, f, indent=2)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
