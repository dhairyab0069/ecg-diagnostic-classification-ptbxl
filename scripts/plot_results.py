"""Make the report figures from results.json, history.json and preds.npz.

Example:
    python scripts/predict.py runs/resnet1d_baseline        # once per run, for the confusion matrices
    python scripts/plot_results.py --out figures

Figures 5-7, the extra_activation figure and activation_metrics.csv compare the ReLU, LeakyReLU
and tanh models and are made only when all three runs exist. Figure 2 (ROC curves) needs preds.npz
for each of the three. extra_*.png are supporting figures that the README does not show.
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
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

from ecg import CLASSES
from ecg.metrics import score

RUNS = {"Baseline": "runs/resnet1d_baseline", "Shuffle-trained": "runs/resnet1d_shuffle_trained"}
ACT_RUNS = {"ReLU": "runs/resnet1d_baseline", "LeakyReLU": "runs/resnet1d_leaky_relu", "tanh": "runs/resnet1d_tanh"}
KS = ["2", "5", "10", "20", "50"]
SEG_LEN = {k: 10 / int(k) for k in KS}  # seconds per segment in a 10 s recording
ROWS = CLASSES + ["macro"]

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ACT_COLORS = {"ReLU": BLUE, "LeakyReLU": ORANGE, "tanh": AQUA}
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
    save(fig, out, "fig3_segment_sweep.png")


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
    save(fig, out, "extra_drop_heatmaps.png")


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
    save(fig, out, "extra_class_confusion.png")


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


def test_scores(results, cls):
    return np.array([score(r["test"], cls) for r in results])


def plot_activation_auroc(results, out):
    """Test AUROC per class for each activation, mean +/- std over seeds, as a dot plot."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    offsets = np.linspace(-0.22, 0.22, len(results))
    for off, (model, rs) in zip(offsets, results.items()):
        v = np.array([test_scores(rs, c) for c in ROWS])
        x = np.arange(len(ROWS)) + off
        ax.errorbar(x, v.mean(1), yerr=v.std(1), fmt="none", ecolor=ACT_COLORS[model], elinewidth=2, capsize=0)
        ax.plot(x, v.mean(1), "o", color=ACT_COLORS[model], markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=model)
    ax.axvline(len(CLASSES) - 0.5, color=GRID, linewidth=1)
    ax.set_xticks(range(len(ROWS)), [r if r != "macro" else "Macro" for r in ROWS])
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_ylabel("Test AUROC")
    ax.legend(frameon=False, loc="lower left", ncol=3)
    ax.set_title("Test AUROC by label (dot: mean of 3 seeds; line: ± std)")
    fig.suptitle("LeakyReLU matches ReLU; tanh is slightly lower", x=0.02, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig5_activation_auroc.png")


def plot_activation_sweep(results, out):
    """Macro AUROC drop vs segment length for each activation, shuffle and roll control side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    x = [SEG_LEN[k] for k in KS]
    for ax, (kind, title) in zip(axes, [("shuffle", "Shuffle"), ("roll", "Roll control")]):
        for model, rs in results.items():
            d = drops(rs, kind, "macro")
            mean, std = d.mean(0), d.std(0)
            color = ACT_COLORS[model]
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
            ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=model)
        ax.set_xscale("log")
        ax.set_xticks(x, [f"{v:g} s" for v in x])
        ax.minorticks_off()
        ax.invert_xaxis()
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_xlabel("Segment length")
        ax.set_title(title)
    axes[0].set_ylabel("Macro AUROC drop")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("All three activations rely on order only within a heartbeat", x=0.02, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig7_activation_sweep.png")


def plot_activation_class_drop(results, out):
    """Per-class AUROC drop under 0.2 s shuffle and roll control, for each activation."""
    k = KS[-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    offsets = np.linspace(-0.22, 0.22, len(results))
    for ax, (kind, title) in zip(axes, [("shuffle", "Shuffle, 0.2 s"), ("roll", "Roll control, 0.2 s")]):
        for off, (model, rs) in zip(offsets, results.items()):
            d = np.array([drops(rs, kind, c)[:, KS.index(k)] for c in ROWS])
            x = np.arange(len(ROWS)) + off
            ax.errorbar(x, d.mean(1), yerr=d.std(1), fmt="none", ecolor=ACT_COLORS[model], elinewidth=2)
            ax.plot(x, d.mean(1), "o", color=ACT_COLORS[model], markersize=7, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=model)
        ax.axvline(len(CLASSES) - 0.5, color=GRID, linewidth=1)
        ax.set_xticks(range(len(ROWS)), [r if r != "macro" else "Macro" for r in ROWS])
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_ylim(bottom=0)
        ax.set_title(title)
    axes[0].set_ylabel("AUROC drop")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("STTC depends most on order inside a beat, for every activation", x=0.02, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "extra_activation_class_drop.png")


def plot_activation_training(out):
    """Train loss and validation macro AUROC per epoch for each activation."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for model, run_dir in ACT_RUNS.items():
        hs = load_json(run_dir, "history.json")
        epochs = [h["epoch"] for h in hs[0]]
        for ax, key in zip(axes, ["train_loss", "val_macro_auroc"]):
            v = np.array([[h[key] for h in run] for run in hs])
            ax.fill_between(epochs, v.min(0), v.max(0), color=ACT_COLORS[model], alpha=0.12, linewidth=0)
            ax.plot(epochs, v.mean(0), color=ACT_COLORS[model], linewidth=2, label=model)
    for ax, title in zip(axes, ["Training loss", "Validation macro AUROC"]):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    axes[1].set_ylim(0.85, None)
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, out, "fig6_activation_training.png")


def plot_roc_curves(preds, out):
    """Test ROC curve per class for each activation, averaged over seeds on a common false-positive grid."""
    fpr_grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(13, 3.2), sharey=True)
    for i, (ax, c) in enumerate(zip(axes, CLASSES)):
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1, linestyle="--")
        for model, ps in preds.items():
            tprs, aucs = [], []
            for p in ps:
                fpr, tpr, _ = roc_curve(p["test_true"][:, i], p["test_clean"][:, i])
                tprs.append(np.interp(fpr_grid, fpr, tpr))
                aucs.append(roc_auc_score(p["test_true"][:, i], p["test_clean"][:, i]))
            ax.plot(fpr_grid, np.mean(tprs, 0), color=ACT_COLORS[model], linewidth=2,
                    label=f"{model} {np.mean(aucs):.3f}")
        ax.set_title(c, loc="center")
        ax.set_xlabel("False-positive rate")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(color=GRID, linewidth=0.8)
        ax.legend(frameon=False, loc="lower right", fontsize=8, title="AUROC", title_fontsize=8)
    axes[0].set_ylabel("True-positive rate")
    fig.suptitle("Test ROC curve for each label (mean of 3 seeds)", x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout()
    save(fig, out, "fig2_roc_curves.png")


def write_activation_metrics(results, out):
    """One row per activation and class: test AUROC, drops at 1 s and 0.2 s, best epoch."""
    header = ["activation", "label", "test_auroc_mean", "test_auroc_std",
              "shuffle_drop_1s", "roll_drop_1s", "shuffle_drop_0.2s", "roll_drop_0.2s",
              "order_effect_0.2s", "best_epoch_mean"]
    lines = [",".join(header)]
    for model, rs in results.items():
        hs = load_json(ACT_RUNS[model], "history.json")
        best_epoch = np.mean([np.argmax([h["val_macro_auroc"] for h in run]) + 1 for run in hs])
        for c in ROWS:
            t = test_scores(rs, c)
            sh, ro = drops(rs, "shuffle", c).mean(0), drops(rs, "roll", c).mean(0)
            i1, i02 = KS.index("10"), KS.index("50")
            row = [model, c, t.mean(), t.std(), sh[i1], ro[i1], sh[i02], ro[i02], sh[i02] - ro[i02], best_epoch]
            lines.append(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row))
    path = os.path.join(out, "activation_metrics.csv")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    results = {model: load_json(d, "results.json") for model, d in RUNS.items()}
    plot_sweep(results, args.out)
    plot_drop_heatmaps(results, args.out)
    plot_training_curves(args.out)
    if all(glob.glob(os.path.join(d, "seed*", "results.json")) for d in ACT_RUNS.values()):
        act = {model: load_json(d, "results.json") for model, d in ACT_RUNS.items()}
        plot_activation_auroc(act, args.out)
        plot_activation_sweep(act, args.out)
        plot_activation_class_drop(act, args.out)
        plot_activation_training(args.out)
        write_activation_metrics(act, args.out)
    act_preds = {model: load_preds(d) for model, d in ACT_RUNS.items()}
    if all(act_preds.values()):
        plot_roc_curves(act_preds, args.out)
    else:
        print("no preds.npz for every activation; run scripts/predict.py for the ROC curves")
    preds = act_preds["ReLU"]
    if preds:
        plot_label_confusion(preds, args.out)
        plot_binary_confusion(preds, args.out)
    else:
        print("no preds.npz found; run scripts/predict.py for the confusion matrices")


if __name__ == "__main__":
    main()
