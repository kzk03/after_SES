# -*- coding: utf-8 -*-
"""Eugster (2012) soccer archetype 風の k=4 diamond 散布図.

各 unit (dev または dev×dir pair) を α-simplex を 2D 投影した diamond 上に配置し、
per-unit AUC / ACC を色で表示する。IRL_cold と RF_dir、両者の差分を 3 panel で並列表示。

座標:
    x = α_A2 − α_A4
    y = α_A3 − α_A1

入力:
    dev-level: k4_dev_alpha_with_metrics.csv  (n=289 acc / n=103 auc)
    pair-level: k4_pair_alpha_with_metrics.csv (n=3677 acc / n=1651 auc)
    stacking: dev_per_dev_with_alpha.csv (n=103 with auc_stack)
"""
from __future__ import annotations

import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEV_LABELS = {
    "A1": "A1: Sporadic\nNewcomer",
    "A2": "A2: Strict\nReviewer-Only",
    "A3": "A3: Approving\nVeteran",
    "A4": "A4: Saturated\nActive",
}
PAIR_LABELS = {
    "A1": "A1", "A2": "A2", "A3": "A3", "A4": "A4",
}
CORNERS = {"A1": (0.0, -1.0), "A2": (1.0, 0.0),
           "A3": (0.0, 1.0), "A4": (-1.0, 0.0)}


def draw_diamond(ax, labels: dict) -> None:
    pts = [CORNERS["A1"], CORNERS["A2"], CORNERS["A3"], CORNERS["A4"], CORNERS["A1"]]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color="black", lw=1.2, zorder=2)
    for name, (cx, cy) in CORNERS.items():
        ax.scatter([cx], [cy], s=300, facecolor="white",
                   edgecolor="black", lw=1.2, zorder=4)
        ax.text(cx, cy, name[-1], ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=5)
        if name == "A1":
            ax.text(cx, cy - 0.12, labels[name], ha="center", va="top", fontsize=8)
        elif name == "A3":
            ax.text(cx, cy + 0.12, labels[name], ha="center", va="bottom", fontsize=8)
        elif name == "A2":
            ax.text(cx + 0.12, cy, labels[name], ha="left", va="center",
                    fontsize=8, rotation=270)
        elif name == "A4":
            ax.text(cx - 0.12, cy, labels[name], ha="right", va="center",
                    fontsize=8, rotation=90)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.30, 1.30)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot_scatter(ax, df: pd.DataFrame, value_col: str, title: str,
                 vmin: float, vmax: float, labels: dict,
                 cmap: str = "RdYlGn", unit: str = "dev",
                 point_size: float = 42) -> object:
    draw_diamond(ax, labels)
    sub = df.dropna(subset=[value_col]).copy()
    sc = ax.scatter(sub["x"], sub["y"], c=sub[value_col],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    s=point_size, alpha=0.7,
                    edgecolor="black", lw=0.3, zorder=3)
    med = sub[value_col].median()
    mean = sub[value_col].mean()
    ax.set_title(f"{title}\nn={len(sub)} {unit}s, median={med:.3f}, mean={mean:.3f}",
                 fontsize=10)
    return sc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=pathlib.Path,
                   help="alpha + metrics CSV (dev or pair level)")
    p.add_argument("--metric", choices=["auc", "acc"], default="auc")
    p.add_argument("--unit", choices=["dev", "pair"], default="dev")
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--include-stack", action="store_true",
                   help="add 4th panel for auc_stack (only available in stacking output)")
    p.add_argument("--point-size", type=float, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df["x"] = df["A2"] - df["A4"]
    df["y"] = df["A3"] - df["A1"]
    labels = DEV_LABELS if args.unit == "dev" else PAIR_LABELS

    if args.metric == "auc":
        irl_col, rf_col = "auc_cold_irl", "auc_rf_dir"
        stack_col = "auc_stack" if args.include_stack else None
        # stacking output uses different col names
        if "auc_irl" in df.columns and "auc_rf" in df.columns:
            irl_col, rf_col = "auc_irl", "auc_rf"
        vmin, vmax = 0.3, 1.0
        cbar_label = f"Per-{args.unit} AUC"
        unit_metric = "AUC"
    else:
        irl_col, rf_col = "acc_cold_irl", "acc_rf_dir"
        stack_col = None
        vmin, vmax = 0.3, 1.0
        cbar_label = f"Per-{args.unit} ACC"
        unit_metric = "ACC"

    df["diff"] = df[irl_col] - df[rf_col]

    panels = [
        (irl_col, f"IRL_cold ({unit_metric})", vmin, vmax, "RdYlGn"),
        (rf_col, f"RF_dir ({unit_metric})", vmin, vmax, "RdYlGn"),
        ("diff", f"IRL − RF ({unit_metric}) — comparison",
         -0.3, 0.3, "RdBu_r"),
    ]
    if stack_col and stack_col in df.columns:
        panels.append((stack_col, f"Stacking ({unit_metric})", vmin, vmax, "RdYlGn"))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6))
    if n == 1:
        axes = [axes]

    last_sc = []
    for ax, (col, title, vmn, vmx, cmap) in zip(axes, panels):
        sc = plot_scatter(ax, df, col, title, vmn, vmx, labels,
                          cmap=cmap, unit=args.unit, point_size=args.point_size)
        last_sc.append((ax, sc, cmap, col))

    # separate colorbars: one for AUC/ACC group (panels 1, 2, [4]) and one for diff
    # build position: shared cbar for green panels, separate for diff
    green_axes = [ax for ax, _, cmap, _ in last_sc if cmap == "RdYlGn"]
    diff_axes = [ax for ax, _, cmap, _ in last_sc if cmap == "RdBu_r"]
    green_sc = [sc for _, sc, cmap, _ in last_sc if cmap == "RdYlGn"][0]
    diff_sc = [sc for _, sc, cmap, _ in last_sc if cmap == "RdBu_r"][0]

    cbar1 = fig.colorbar(green_sc, ax=green_axes, shrink=0.7, pad=0.02,
                         label=cbar_label, orientation="horizontal", aspect=40)
    cbar1.ax.tick_params(labelsize=8)
    cbar2 = fig.colorbar(diff_sc, ax=diff_axes, shrink=0.7, pad=0.02,
                         label=f"IRL − RF Δ{unit_metric}",
                         orientation="horizontal", aspect=20)
    cbar2.ax.tick_params(labelsize=8)

    unit_full = "developer" if args.unit == "dev" else "(developer × directory) pair"
    fig.suptitle(
        f"k=4 archetype simplex (diamond) — per-{unit_full} "
        f"{unit_metric} distribution: IRL vs RF",
        fontsize=12, y=1.04,
    )
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
