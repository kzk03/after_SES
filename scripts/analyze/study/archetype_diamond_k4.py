# -*- coding: utf-8 -*-
"""k=4 archetype α の Diamond 投影 (Eugster 2012 HIIT slides 風)。

4 頂点を rhombus の上下左右に配置し、各観測の α = (a1,a2,a3,a4) を
  x = a2 - a4
  y = a1 - a3
で 2D 投影。色は Cold IRL acc − RF acc (赤=IRL強 / 青=RF強)。

入力: k4_{dev|pair}_alpha_with_metrics.csv
出力: --out で指定する PDF
"""
from __future__ import annotations
import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEV_LABELS = {
    "A1": "A1: Sporadic Newcomer",
    "A2": "A2: Strict Reviewer-Only",
    "A3": "A3: Approving Veteran",
    "A4": "A4: Saturated Active",
}
PAIR_LABELS = {
    "A1": "A1: Unfamiliar Dir",
    "A2": "A2: Familiar / Recent Reject",
    "A3": "A3: Familiar / Consistent Accept",
    "A4": "A4: Mid-Engagement",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--metric", default="auc", choices=["acc", "auc"])
    p.add_argument("--style", default="hexbin", choices=["scatter", "hexbin"])
    p.add_argument("--gridsize", type=int, default=10)
    p.add_argument("--vmin", type=float, default=-0.15)
    p.add_argument("--vmax", type=float, default=0.15)
    p.add_argument("--labels", default="dev", choices=["dev", "pair"])
    args = p.parse_args()

    LABELS = DEV_LABELS if args.labels == "dev" else PAIR_LABELS

    df = pd.read_csv(args.alpha_csv)
    a1 = df["A1"].to_numpy()
    a2 = df["A2"].to_numpy()
    a3 = df["A3"].to_numpy()
    a4 = df["A4"].to_numpy()

    x = a2 - a4
    y = a1 - a3

    col_irl = f"{args.metric}_cold_irl"
    col_rf = f"{args.metric}_rf_dir"
    diff = df[col_irl].to_numpy() - df[col_rf].to_numpy()

    if args.metric == "acc":
        cbar_label = f"acc(Cold IRL) − acc(RF_Dir)\n+ → IRL 強   − → RF 強"
    else:
        cbar_label = f"AUC(Cold IRL) − AUC(RF_Dir)\n+ → IRL 強   − → RF 強"

    fig, ax = plt.subplots(figsize=(8, 7.5))
    diamond_x = [0, 1, 0, -1, 0]
    diamond_y = [1, 0, -1, 0, 1]
    ax.plot(diamond_x, diamond_y, "k-", linewidth=1.2)

    nan_mask = np.isnan(diff)
    val_mask = ~nan_mask

    if args.style == "scatter":
        if nan_mask.any():
            ax.scatter(x[nan_mask], y[nan_mask], c="lightgray", s=10,
                       edgecolor="gray", linewidth=0.1)
        sc = ax.scatter(x[val_mask], y[val_mask], c=diff[val_mask],
                        cmap="RdBu_r", vmin=args.vmin, vmax=args.vmax,
                        s=22, edgecolor="k", linewidth=0.2, alpha=0.85)
    else:
        sc = ax.hexbin(x[val_mask], y[val_mask], C=diff[val_mask],
                       reduce_C_function=np.mean,
                       gridsize=args.gridsize, cmap="RdBu_r",
                       vmin=args.vmin, vmax=args.vmax,
                       mincnt=2, extent=(-1.05, 1.05, -1.05, 1.05),
                       edgecolors="white", linewidths=0.4)

    # 頂点ラベル
    ax.text(0, 1.08, LABELS["A1"], ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.text(1.08, 0, LABELS["A2"], ha="left", va="center", fontsize=10, fontweight="bold")
    ax.text(0, -1.08, LABELS["A3"], ha="center", va="top", fontsize=10, fontweight="bold")
    ax.text(-1.08, 0, LABELS["A4"], ha="right", va="center", fontsize=10, fontweight="bold")
    for vx, vy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
        ax.plot(vx, vy, "o", markersize=13, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)

    n_unit = "dev" if args.labels == "dev" else "pair"
    title_metric = "accuracy" if args.metric == "acc" else "AUC"
    style_lbl = f"hexbin gridsize={args.gridsize}" if args.style == "hexbin" else "scatter"
    ax.set_title(f"k=4 Archetype Diamond ({n_unit}-level, {title_metric}, {style_lbl})\n"
                 f"n={val_mask.sum()} {n_unit} (有効 metric) / total {len(df)}",
                 fontsize=11)

    fig.tight_layout()
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"saved: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
