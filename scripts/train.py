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
from ecg.data import build_splits, make_loader
from ecg.engine import predict, train_one_epoch
from ecg.metrics import auroc
from ecg.model import build_model
from ecg.transforms import roll_segments, shuffle_segments
from ecg.utils import get_device, load_config, save_json, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.lr=0.001")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    set_seed(args.seed)
    device = get_device()
    run_dir = os.path.join(cfg["out_dir"], cfg["name"], f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    splits, norm = build_splits(cfg["data"])
    tcfg = cfg["train"]
    pin = device.type == "cuda"
    train_dl = make_loader(*splits["train"], tcfg["batch_size"], True, tcfg["num_workers"], pin)
    val_dl = make_loader(*splits["val"], 512, False, tcfg["num_workers"], pin)
    test_dl = make_loader(*splits["test"], 512, False, tcfg["num_workers"], pin)
    print(f"train {len(splits['train'][0])}  val {len(splits['val'][0])}  "
          f"test {len(splits['test'][0])}  device {device}")

    model = build_model(cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, tcfg["lr"], total_steps=tcfg["epochs"] * len(train_dl))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    ckpt_path = os.path.join(run_dir, "best.pt")

    best, history = -1.0, []
    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        loss = train_one_epoch(model, train_dl, opt, sched, scaler, loss_fn, device, cfg["augment"])
        val_auc, _ = auroc(*predict(model, val_dl, device))
        history.append({"epoch": epoch, "train_loss": loss, "val_macro_auroc": val_auc})
        if val_auc > best:
            best = val_auc
            torch.save({"model": model.state_dict(), "config": cfg, "norm": norm}, ckpt_path)
        print(f"epoch {epoch:02d}  loss {loss:.4f}  val AUROC {val_auc:.4f}  best {best:.4f}"
              f"  ({time.time() - t0:.0f}s)", flush=True)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_auc, test_per = auroc(*predict(model, test_dl, device))
    results = {"seed": args.seed, "val_macro_auroc": best,
               "test_macro_auroc": test_auc, "test_per_class": test_per}
    print(f"\nTEST macro AUROC {test_auc:.4f}")
    for c in CLASSES:
        print(f"  {c:5s} {test_per[c]:.4f}")

    if cfg["probe"]["enabled"]:
        ks = cfg["probe"]["num_segments"]
        results["probe"] = {}
        for k in ks if isinstance(ks, list) else [ks]:
            entry = {}
            for name, fn in [("shuffle", shuffle_segments), ("roll", roll_segments)]:
                torch.manual_seed(args.seed)
                macro, per = auroc(*predict(model, test_dl, device, lambda x: fn(x, k)))
                entry[name] = {"macro": macro, "per_class": per}
            results["probe"][str(k)] = entry
            print(f"\n{k} SEGMENTS  shuffled {entry['shuffle']['macro']:.4f} "
                  f"(drop {test_auc - entry['shuffle']['macro']:+.4f})  "
                  f"roll control {entry['roll']['macro']:.4f} "
                  f"(drop {test_auc - entry['roll']['macro']:+.4f})")
            for c in CLASSES:
                sh, ro = entry["shuffle"]["per_class"][c], entry["roll"]["per_class"][c]
                print(f"  {c:5s} shuffled {sh:.4f} (drop {test_per[c] - sh:+.4f})  "
                      f"roll {ro:.4f} (drop {test_per[c] - ro:+.4f})")

    save_json(results, os.path.join(run_dir, "results.json"))
    save_json(history, os.path.join(run_dir, "history.json"))


if __name__ == "__main__":
    main()
