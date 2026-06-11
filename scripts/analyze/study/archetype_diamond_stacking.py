# -*- coding: utf-8 -*-
"""Multi-horizon Stacking の archetype Diamond 可視化。

dev-level α + per-dev (stack, avg, irl, rf) AUC を merge し、
diff_stack_avg または diff_stack_best を色として Diamond 投影。
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv"))
    p.add_argument("--stack-csv", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/multi_horizon_stacking_volat_ep150/dev_level_per_dev_aucs.csv"))
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--vs", default="avg", choices=["avg", "best", "irl", "rf"],
                   help="比較対象: stack - avg / stack - max(irl,rf) / stack - irl / stack - rf")
    p.add_argument("--gridsize", type=int, default=14)
    p.add_argument("--vmin", type=float, default=-0.3)
    p.add_argument("--vmax", type=float, default=0.3)
    args = p.parse_args()

    alpha = pd.read_csv(args.alpha_csv, index_col=0)
    alpha.index.name = "developer"
    alpha = alpha.reset_index()
    stk = pd.read_csv(args.stack_csv)
    m = alpha.merge(stk, on="developer", how="inner")
    print(f"merged: {len(m)} devs")

    if args.vs == "avg":
        diff = (m.auc_stack - m.auc_avg).to_numpy()
        cbar = "AUC(Stack) − AUC(Uniform Avg)\n+ → Stack 強"
        vs_lbl = "vs Uniform Avg"
    elif args.vs == "best":
        diff = (m.auc_stack - m[["auc_irl", "auc_rf"]].max(axis=1)).to_numpy()
        cbar = "AUC(Stack) − max(AUC(IRL), AUC(RF))\n+ → Stack > best single"
        vs_lbl = "vs Best Single"
    elif args.vs == "irl":
        diff = (m.auc_stack - m.auc_irl).to_numpy()
        cbar = "AUC(Stack) − AUC(IRL)\n+ → Stack 強"
        vs_lbl = "vs IRL"
    else:
        diff = (m.auc_stack - m.auc_rf).to_numpy()
        cbar = "AUC(Stack) − AUC(RF)\n+ → Stack 強"
        vs_lbl = "vs RF"

    a1, a2, a3, a4 = m.A1.to_numpy(), m.A2.to_numpy(), m.A3.to_numpy(), m.A4.to_numpy()
    x = a2 - a4
    y = a1 - a3

    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.plot([0, 1, 0, -1, 0], [1, 0, -1, 0, 1], "k-", linewidth=1.2)

    val_mask = ~np.isnan(diff)
    sc = ax.hexbin(x[val_mask], y[val_mask], C=diff[val_mask],
                   reduce_C_function=np.mean,
                   gridsize=args.gridsize, cmap="RdBu_r",
                   vmin=args.vmin, vmax=args.vmax,
                   mincnt=1, extent=(-1.05, 1.05, -1.05, 1.05),
                   edgecolors="white", linewidths=0.4)

    for arch, (vx, vy) in zip(["A1","A2","A3","A4"], [(0,1),(1,0),(0,-1),(-1,0)]):
        ax.plot(vx, vy, "o", markersize=13, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        offset_map = {(0,1):(0,0.08,"center","bottom"),
                      (1,0):(0.08,0,"left","center"),
                      (0,-1):(0,-0.08,"center","top"),
                      (-1,0):(-0.08,0,"right","center")}
        ox, oy, ha, va = offset_map[(vx,vy)]
        ax.text(vx+ox, vy+oy, DEV_LABELS[arch], ha=ha, va=va,
                fontsize=10, fontweight="bold")

    ax.set_xlim(-1.7, 1.7)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    cbar_obj = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cbar_obj.set_label(cbar, fontsize=9)
    ax.set_title(f"k=4 Archetype Diamond: Multi-Horizon Stacking {vs_lbl}\n"
                 f"per-dev AUC (n={val_mask.sum()}), gridsize={args.gridsize}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"saved: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
