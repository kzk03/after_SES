"""Multi-horizon stacking の AUC heatmap + gain heatmap (archetype × horizon)。

Panel (a): AUC by archetype × horizon, comparing models
Panel (b): Stack gain vs best single (heatmap)
Panel (c): feature importance heatmap (GB) target × feature
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 10

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
ARCHS = ["A1", "A2", "A3", "A4", "ALL"]
ARCH_LABELS = ["A1\nNewcomer", "A2\nStrict\nReviewer", "A3\nVeteran", "A4\nSaturated", "ALL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-arch", type=Path,
                    default=Path("outputs/analysis/multi_horizon_stacking_volat_ep150/per_archetype.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/analysis/multi_horizon_stacking_volat_ep150/fig_stacking.pdf"))
    args = ap.parse_args()

    df = pd.read_csv(args.per_arch)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # === Panel (a): AUC line plot per archetype ===
    ax = axes[0]
    for arch_idx, arch in enumerate(ARCHS):
        sub = df[df["archetype"] == arch].sort_values("target")
        if sub.empty: continue
        x = np.arange(len(HORIZONS))
        offset = (arch_idx - 2) * 0.02
        lw = 2.5 if arch == "ALL" else 1.5
        alpha = 1.0 if arch == "ALL" else 0.6
        ax.plot(x + offset, sub["auc_stack"], marker="o", color=f"C{arch_idx}",
                label=f"{arch} Stack", lw=lw, alpha=alpha)
    ax.set_xticks(np.arange(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS)
    ax.set_ylabel("AUC-ROC (Stacking, 5-fold CV)")
    ax.set_xlabel("Target horizon")
    ax.set_title("(a) Stacking AUC by archetype × horizon")
    ax.set_ylim(0.55, 0.90)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(lw=0.3, alpha=0.4)

    # === Panel (b): Heatmap of gain vs best single ===
    ax = axes[1]
    matrix = np.zeros((5, 4))
    for i, arch in enumerate(ARCHS):
        for j, h in enumerate(HORIZONS):
            row = df[(df["archetype"] == arch) & (df["target"] == h)]
            if not row.empty:
                matrix[i, j] = row["gain_vs_best"].iloc[0]
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-0.05, vmax=0.20, aspect="auto")
    for i in range(5):
        for j in range(4):
            val = matrix[i, j]
            color = "black" if abs(val) < 0.1 else "white"
            ax.text(j, i, f"{val*100:+.1f}", ha="center", va="center", fontsize=10,
                    fontweight="bold", color=color)
    ax.set_xticks(np.arange(4)); ax.set_xticklabels(HORIZONS)
    ax.set_yticks(np.arange(5)); ax.set_yticklabels(ARCH_LABELS, fontsize=9)
    ax.set_title("(b) AUC gain (pt) of Stacking vs best single model")
    ax.set_xlabel("Target horizon")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="gain (×100 pt)")

    # === Panel (c): AUC comparison bars, per horizon (ALL row) ===
    ax = axes[2]
    all_df = df[df["archetype"] == "ALL"].sort_values("target")
    x = np.arange(len(HORIZONS))
    w = 0.18
    ax.bar(x - 1.5*w, all_df["auc_irl"], w, label="IRL solo", color="#4A90D9")
    ax.bar(x - 0.5*w, all_df["auc_rf"], w, label="RF solo", color="#E07B39")
    ax.bar(x + 0.5*w, all_df["auc_avg"], w, label="Uniform (0.5)", color="#5BA85B")
    ax.bar(x + 1.5*w, all_df["auc_stack"], w, label="Stacking (GB)", color="#7B3F99")
    ax.set_xticks(x); ax.set_xticklabels(HORIZONS)
    ax.set_ylabel("AUC-ROC")
    ax.set_xlabel("Target horizon")
    ax.set_title("(c) Overall AUC by horizon × method (n=4,534)")
    ax.set_ylim(0.72, 0.88)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", lw=0.3, alpha=0.4)
    # gain annotations
    for i, row in all_df.reset_index().iterrows():
        ax.text(x[i] + 1.5*w, row["auc_stack"] + 0.005,
                f"+{(row['auc_stack'] - row['auc_avg'])*100:.1f}",
                ha="center", fontsize=8, color="#7B3F99", fontweight="bold")

    fig.suptitle("Multi-Horizon Stacking (GB meta-learner) — volat ep150 cold IRL × RF",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
