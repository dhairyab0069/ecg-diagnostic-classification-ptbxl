"""Make the report figures from results.json, history.json and preds.npz.

Example:
    python scripts/predict.py runs/resnet1d_baseline        # once per run, for the confusion matrices
    python scripts/plot_results.py --out figures
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.metrics import f1_score

from ecg import CLASSES
from ecg.metrics import score

RUNS = {"Baseline": "runs/resnet1d_baseline", "Shuffle-trained": "runs/resnet1d_shuffle_trained"}
KS = ["2", "5", "10", "20", "50"]
SEG_LEN = {k: 10 / int(k) for k in KS}  # seconds per segment in a 10 s recording
ROWS = CLASSES + ["macro"]

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#3987e5", "#86b6ef", "#f0efec", "#f2a3a2", "#e34948", "#a32a2a"])

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load_json(run_dir, name):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(run_dir, "seed*", name)))]


def drops(results, kind, cls):
    """(seeds, k) array of test AUROC minus probed AUROC."""
    return np.array([[score(r["test"], cls) - score(r["probe"][k][kind], cls) for k in KS] for r in results])


def ink_for(value, norm, cmap):
    r, g, b, _ = cmap(norm(value))
    return INK if 0.299 * r + 0.587 * g + 0.114 * b > 0.55 else SURFACE


def annotate(ax, data, norm, cmap, fmt="{:.3f}"):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = 0.0 if abs(data[i, j]) < 5e-4 else data[i, j]  # no "-0.000"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                    color=ink_for(data[i, j], norm, cmap))


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}/{name}")


def plot_sweep(results, out):
    """Macro AUROC drop vs segment length, shuffle and roll, one panel per model."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    x = [SEG_LEN[k] for k in KS]
    for ax, (model, rs) in zip(axes, results.items()):
        for kind, label, color in [("shuffle", "Shuffle", BLUE), ("roll", "Roll control", ORANGE)]:
            d = drops(rs, kind, "macro")
            mean, std = d.mean(0), d.std(0)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
            ax.annotate(label, (x[-1], mean[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK_2)
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:g} s" for v in x])
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_xlabel("Segment length")
        ax.set_title(model)
    axes[0].set_ylabel("Macro AUROC drop")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Order matters only below one heartbeat", x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig2_segment_sweep.png")


def plot_drop_heatmaps(results, out):
    """Class x segment-length maps of shuffle drop, roll drop and their difference (baseline)."""
    rs = results["Baseline"]
    shuffle = np.array([drops(rs, "shuffle", c).mean(0) for c in ROWS])
    roll = np.array([drops(rs, "roll", c).mean(0) for c in ROWS])
    diff = shuffle - roll
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    seq_norm = plt.Normalize(0, max(shuffle.max(), roll.max()))
    lim = np.abs(diff).max()
    div_norm = TwoSlopeNorm(0, -lim, lim)
    panels = [(shuffle, "Shuffle drop", SEQ, seq_norm), (roll, "Roll-control drop", SEQ, seq_norm),
              (diff, "Shuffle − roll (effect of order)", DIV, div_norm)]
    for ax, (data, title, cmap, norm) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        annotate(ax, data, norm, cmap)
        ax.set_xticks(range(len(KS)), [f"{SEG_LEN[k]:g} s" for k in KS])
        ax.set_yticks(range(len(ROWS)), [r if r != "macro" else "Macro" for r in ROWS])
        ax.set_xlabel("Segment length")
        ax.set_title(title)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(len(CLASSES) - 0.5, color=SURFACE, linewidth=3)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).outline.set_visible(False)
    fig.suptitle("Baseline AUROC drop by class (mean of 3 seeds)", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig3_drop_heatmaps.png")


def load_preds(run_dir):
    return [dict(np.load(f)) for f in sorted(glob.glob(os.path.join(run_dir, "seed*", "preds.npz")))]


def val_thresholds(p):
    """Per-class threshold that maximizes F1 on the validation fold."""
    grid = np.linspace(0.05, 0.95, 91)
    return np.array([grid[np.argmax([f1_score(p["val_true"][:, i], p["val_prob"][:, i] >= t) for t in grid])]
                     for i in range(len(CLASSES))])


def label_confusion(preds, condition):
    """5x5 multi-label confusion: row i = records with true label i, column j = share predicted as j."""
    counts, totals = np.zeros((5, 5)), np.zeros(5)
    for p in preds:
        y, yhat = p["test_true"].astype(bool), p[f"test_{condition}"] >= val_thresholds(p)
        counts += y.T.astype(float) @ yhat.astype(float)
        totals += y.sum(0)
    return counts / totals[:, None]


def binary_confusion(preds, condition):
    """(5, 2, 2) row-normalized per-class confusion [[TN, FP], [FN, TP]], pooled over seeds."""
    cm = np.zeros((5, 2, 2))
    for p in preds:
        y, yhat = p["test_true"].astype(bool), p[f"test_{condition}"] >= val_thresholds(p)
        for i in range(5):
            for a in (0, 1):
                for b in (0, 1):
                    cm[i, a, b] += np.sum((y[:, i] == a) & (yhat[:, i] == b))
    return cm / cm.sum(2, keepdims=True)


def plot_label_confusion(preds, out):
    """Clean vs 0.2 s-shuffle multi-label confusion and their difference (baseline)."""
    clean, shuffled = label_confusion(preds, "clean"), label_confusion(preds, "shuffle_50")
    diff = shuffled - clean
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    lim = np.abs(diff).max()
    panels = [(clean, "Clean test", SEQ, plt.Normalize(0, 1), "{:.2f}"),
              (shuffled, "Shuffled 0.2 s segments", SEQ, plt.Normalize(0, 1), "{:.2f}"),
              (diff, "Difference (shuffled − clean)", DIV, TwoSlopeNorm(0, -lim, lim), "{:+.2f}")]
    for ax, (data, title, cmap, norm, fmt) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, norm=norm)
        annotate(ax, data, norm, cmap, fmt)
        ax.set_xticks(range(5), CLASSES)
        ax.set_yticks(range(5), CLASSES)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).outline.set_visible(False)
    fig.suptitle("Baseline label confusion: share of records with each true label predicted as each label "
                 "(3 seeds pooled)", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig4_label_confusion.png")


def plot_binary_confusion(preds, out):
    """Per-class 2x2 confusion matrices, clean vs 0.2 s shuffle, and the difference (baseline)."""
    conds = [("clean", "Clean"), ("shuffle_50", "Shuffled 0.2 s")]
    cms = {c: binary_confusion(preds, c) for c, _ in conds}
    diff = cms["shuffle_50"] - cms["clean"]
    lim = np.abs(diff).max()
    fig, axes = plt.subplots(3, 5, figsize=(12, 7.6))
    for row, (data, label, cmap, norm, fmt) in enumerate([
            (cms["clean"], "Clean", SEQ, plt.Normalize(0, 1), "{:.2f}"),
            (cms["shuffle_50"], "Shuffled 0.2 s", SEQ, plt.Normalize(0, 1), "{:.2f}"),
            (diff, "Difference", DIV, TwoSlopeNorm(0, -lim, lim), "{:+.2f}")]):
        for i, c in enumerate(CLASSES):
            ax = axes[row, i]
            ax.imshow(data[i], cmap=cmap, norm=norm)
            annotate(ax, data[i], norm, cmap, fmt)
            ax.set_xticks([0, 1], ["neg", "pos"])
            ax.set_yticks([0, 1], ["neg", "pos"])
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(c, loc="center")
            if i == 0:
                ax.set_ylabel(f"{label}\ntrue", fontweight="bold")
            if row == 2:
                ax.set_xlabel("predicted")
    fig.suptitle("Per-class confusion at validation-F1 thresholds, row-normalized (baseline, 3 seeds pooled)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig5_class_confusion.png")


def plot_training_curves(out):
    """Train loss and validation macro AUROC per epoch, mean and range over seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for (model, run_dir), color in zip(RUNS.items(), [BLUE, ORANGE]):
        hs = load_json(run_dir, "history.json")
        epochs = [h["epoch"] for h in hs[0]]
        for ax, key in zip(axes, ["train_loss", "val_macro_auroc"]):
            v = np.array([[h[key] for h in run] for run in hs])
            ax.fill_between(epochs, v.min(0), v.max(0), color=color, alpha=0.15, linewidth=0)
            ax.plot(epochs, v.mean(0), color=color, linewidth=2, label=model)
    for ax, title in zip(axes, ["Training loss", "Validation macro AUROC"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig1_training_curves.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {model: load_json(d, "results.json") for model, d in RUNS.items()}
    plot_sweep(results, args.out)
    plot_drop_heatmaps(results, args.out)
    plot_training_curves(args.out)
    preds = load_preds(RUNS["Baseline"])
    if preds:
        plot_label_confusion(preds, args.out)
        plot_binary_confusion(preds, args.out)
    else:
        print("no preds.npz found; run scripts/predict.py for the confusion matrices")


if __name__ == "__main__":
    main()
