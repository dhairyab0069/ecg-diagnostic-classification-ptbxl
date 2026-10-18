"""Make the animated figures for the README.

anim_shuffle.gif   one test ECG cut into smaller and smaller pieces and shuffled, next to the
                   baseline's macro AUROC drop at each piece length (from summary.json)
anim_training.gif  validation AUROC and training loss drawn epoch by epoch for ReLU, LeakyReLU, tanh

Example:
    python scripts/make_gifs.py --out figures
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import wfdb
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

from plot_results import ACT_COLORS, ACT_RUNS, BLUE, GRID, INK, INK_2, KS, ORANGE, SEG_LEN, SURFACE, load_json

RECORD = "records100/00000/00079_lr"  # clean normal ECG from the test fold
LEAD = 1  # lead II
PIECE_CMAP = LinearSegmentedColormap.from_list("pieces", ["#86b6ef", "#2a78d6", "#0d366b"])

# frames per phase of each piece length in the shuffle animation
CUT, MOVE, HOLD, BACK = 6, 14, 14, 8


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


def shuffle_frames(signal, summary, out, fps=20):
    t = np.arange(signal.size) / 100
    rng = np.random.default_rng(0)
    perms = {}
    for k in KS:
        perm = rng.permutation(int(k))
        while np.all(perm == np.arange(int(k))):
            perm = rng.permutation(int(k))
        perms[k] = perm
    drop = {kind: [summary["macro"]["test"] - summary["macro"][f"{kind}_{k}"] for k in KS]
            for kind in ("shuffle", "roll")}

    fig = plt.figure(figsize=(10, 4.2))
    ax_ecg = fig.add_axes([0.05, 0.2, 0.58, 0.62])
    ax_drop = fig.add_axes([0.72, 0.2, 0.26, 0.62])
    per_stage = CUT + MOVE + HOLD + BACK
    frames = [(s, f) for s in range(len(KS) - 1) for f in range(per_stage)]
    frames += [(len(KS) - 1, f) for f in range(CUT + MOVE)]
    frames += [(len(KS) - 1, CUT + MOVE)] * 40  # end on the 0.2 s shuffle and hold it before looping

    def draw(frame):
        stage, f = frame
        k = KS[stage]
        n = int(k)
        seg = signal.size // n
        perm = perms[k]
        if f < CUT:
            progress, show_cuts = 0.0, f / CUT
        elif f < CUT + MOVE:
            progress, show_cuts = ease((f - CUT) / MOVE), 1.0
        elif f < CUT + MOVE + HOLD:
            progress, show_cuts = 1.0, 1.0
        else:
            progress, show_cuts = 1.0 - ease((f - CUT - MOVE - HOLD) / BACK), 1.0

        ax_ecg.cla()
        # piece j (original position j) moves to the slot where it lands after shuffling
        dest = np.empty(n, int)
        dest[perm] = np.arange(n)
        for j in range(n):
            x0 = j * seg / 100
            x1 = dest[j] * seg / 100
            dx = (x1 - x0) * progress
            lift = 0.15 * np.sin(np.pi * progress) * (1 if j % 2 else -1) if 0 < progress < 1 else 0
            piece = signal[j * seg:(j + 1) * seg]
            ax_ecg.plot(t[j * seg:(j + 1) * seg] + dx, piece + lift, color=PIECE_CMAP(j / max(n - 1, 1)),
                        linewidth=1.6, solid_capstyle="round")
        if show_cuts:
            for j in range(1, n):
                ax_ecg.axvline(j * seg / 100, color=ORANGE, linewidth=0.8, alpha=0.6 * show_cuts)
        ax_ecg.set_xlim(0, 10)
        ax_ecg.set_ylim(signal.min() - 0.4, signal.max() + 0.4)
        ax_ecg.set_yticks([])
        ax_ecg.spines["left"].set_visible(False)
        ax_ecg.set_xlabel("Time (s)")
        state = "shuffled" if progress > 0.5 else "original order"
        ax_ecg.set_title(f"{n} pieces of {SEG_LEN[k]:g} s  ·  {state}")

        ax_drop.cla()
        x = [SEG_LEN[kk] for kk in KS]
        shown = stage + 1 if f >= CUT + MOVE else stage
        for kind, label, color in [("shuffle", "Shuffle", BLUE), ("roll", "Control", ORANGE)]:
            if shown:
                ax_drop.plot(x[:shown], drop[kind][:shown], color=color, linewidth=2, marker="o", markersize=6,
                             markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax_drop.set_xscale("log")
        ax_drop.set_xticks(x, [f"{v:g}" for v in x])
        ax_drop.minorticks_off()
        ax_drop.set_xlim(6.5, 0.15)
        ax_drop.set_ylim(-0.005, 0.09)
        ax_drop.axhline(0, color=INK_2, linewidth=0.8)
        ax_drop.grid(axis="y", color=GRID, linewidth=0.8)
        ax_drop.set_xlabel("Piece length (s)")
        ax_drop.set_title("Score drop (AUROC)")
        if shown:
            ax_drop.legend(frameon=False, loc="upper left")
            ax_drop.annotate(f"{drop['shuffle'][shown - 1]:.3f}", (x[shown - 1], drop["shuffle"][shown - 1]),
                             xytext=(-14, 6), textcoords="offset points", ha="center", fontsize=9, color=INK,
                             bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))
        fig.suptitle("Shuffling whole beats barely matters; breaking beats apart does", x=0.02, ha="left",
                     fontweight="bold", fontsize=12)

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    path = os.path.join(out, "anim_shuffle.gif")
    anim.save(path, writer=PillowWriter(fps=fps), dpi=90, savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    print(f"wrote {path}")


def training_frames(out, fps=8):
    hist = {m: load_json(d, "history.json") for m, d in ACT_RUNS.items()}
    epochs = np.array([h["epoch"] for h in next(iter(hist.values()))[0]])
    series = {m: {key: np.array([[h[key] for h in run] for run in hs]) for key in ("train_loss", "val_macro_auroc")}
              for m, hs in hist.items()}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    frames = list(range(1, len(epochs) + 1)) + [len(epochs)] * 16

    def draw(e):
        for ax, key, title in zip(axes, ["train_loss", "val_macro_auroc"], ["Training loss", "Validation AUROC"]):
            ax.cla()
            for m, s in series.items():
                v = s[key][:, :e]
                ax.fill_between(epochs[:e], v.min(0), v.max(0), color=ACT_COLORS[m], alpha=0.12, linewidth=0)
                ax.plot(epochs[:e], v.mean(0), color=ACT_COLORS[m], linewidth=2, label=m)
                ax.plot(epochs[e - 1], v.mean(0)[-1], "o", color=ACT_COLORS[m], markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.set_xlim(0, len(epochs) + 1)
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
        axes[0].set_ylim(0.1, 0.45)
        axes[1].set_ylim(0.85, 0.935)
        axes[1].legend(frameon=False, loc="lower right")
        fig.suptitle(f"Training, epoch {e} of {len(epochs)} (mean of 3 seeds)", x=0.02, ha="left",
                     fontweight="bold", fontsize=12)
        fig.tight_layout()

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps)
    path = os.path.join(out, "anim_training.gif")
    anim.save(path, writer=PillowWriter(fps=fps), dpi=90, savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--root", default="data/ptb-xl/1.0.3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    signal = wfdb.rdsamp(os.path.join(args.root, RECORD))[0][:, LEAD]
    with open("runs/resnet1d_baseline/summary.json") as f:
        summary = json.load(f)
    shuffle_frames(signal, summary, args.out)
    training_frames(args.out)


if __name__ == "__main__":
    main()
