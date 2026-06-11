# -*- coding: utf-8 -*-
"""Pair-level (dev × dir) per-pair AUC を pair archetype simplex (diamond) に投影.

各点 = 1 pair, 位置 = pair α の 2D 投影,
色 = pair の 4-horizon pool で計算した Δ = AUC_IRL − AUC_RF.
|Δ| ≤ threshold (≒ tied) は非表示。
"""
from __future__ import annotations

import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CORNERS = {"A1": (0.0, -1.0), "A2": (1.0, 0.0),
           "A3": (0.0, 1.0), "A4": (-1.0, 0.0)}


def draw_diamond(ax) -> None:
    pts = [CORNERS[a] for a in ["A1", "A2", "A3", "A4", "A1"]]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "k-", lw=1.2, zorder=2)
    for name, (cx, cy) in CORNERS.items():
        ax.scatter([cx], [cy], s=260, facecolor="white",
                   edgecolor="black", lw=1.2, zorder=4)
        ax.text(cx, cy, name[-1], ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=5)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",
                   default="outputs/analysis/archetype_pair_level_volat_ep150_coldonly/k4_pair_alpha_with_metrics.csv",
                   type=pathlib.Path)
    p.add_argument("--out",
                   default="outputs/analysis/multi_horizon_stacking_volat_ep150/fig_diamond_pair_pooled.pdf",
                   type=pathlib.Path)
    p.add_argument("--gray-threshold", type=float, default=0.05)
    p.add_argument("--vlim", type=float, default=0.4)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df["diff"] = df["auc_cold_irl"] - df["auc_rf_dir"]
    df = df.dropna(subset=["diff"]).copy()
    df["x"] = df["A2"] - df["A4"]
    df["y"] = df["A3"] - df["A1"]
    n_total = len(df)

    shown = df[df["diff"].abs() > args.gray_threshold]
    n_hidden = n_total - len(shown)
    irl_wins = (df["diff"] > args.gray_threshold).sum()
    rf_wins = (df["diff"] < -args.gray_threshold).sum()
    print(f"n_total={n_total}, hidden_tied={n_hidden}, "
          f"IRL_wins={irl_wins}, RF_wins={rf_wins}")

    fig, ax = plt.subplots(figsize=(6.8, 6.3))
    draw_diamond(ax)
    # IRL 優位 (赤) と RF 優位 (青) の 2 段描画 (重なり順を制御)
    shown_sorted = shown.reindex(shown["diff"].abs().sort_values().index)
    sc = ax.scatter(shown_sorted["x"], shown_sorted["y"],
                    c=shown_sorted["diff"],
                    cmap="RdBu_r", vmin=-args.vlim, vmax=args.vlim,
                    s=16, alpha=0.75, edgecolor="black", lw=0.1, zorder=3)
    ax.set_title(
        f"Per-pair Δ = AUC$_{{IRL}}$ − AUC$_{{RF}}$ on pair archetype simplex\n"
        f"n={n_total} pair (with AUC), "
        f"shown |Δ|>{args.gray_threshold}: "
        f"IRL wins {irl_wins} / RF wins {rf_wins} / tied {n_hidden} hidden",
        fontsize=10.5,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.03,
                        label="Δ = AUC$_{IRL}$ − AUC$_{RF}$ (per-pair, 4-horizon pool)")
    cbar.ax.tick_params(labelsize=9)
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
