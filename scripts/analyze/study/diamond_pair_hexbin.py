# -*- coding: utf-8 -*-
"""Pair-level diamond の hex-bin ヒートマップ版.

各 hex に入る pair の (label, prob) を pool して
hex 内 pooled AUC を IRL / RF 別に計算し、Δ = AUC_IRL − AUC_RF を色表示.
"""
from __future__ import annotations

import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon
from sklearn.metrics import roc_auc_score

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
CORNERS = {"A1": (0.0, -1.0), "A2": (1.0, 0.0),
           "A3": (0.0, 1.0), "A4": (-1.0, 0.0)}


def load_pair_long(pred_root: pathlib.Path) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        p = pred_root / f"lstm_baseline/train_{h}/eval_{h}/pair_predictions.csv"
        d = pd.read_csv(p)
        d["horizon"] = h
        rows.append(d[["developer", "directory", "horizon", "label",
                       "irl_dir_prob_calibrated", "rf_dir_prob"]])
    return pd.concat(rows, ignore_index=True)


def make_hex_grid(gridsize: int):
    extent = 1.1
    r = extent / gridsize
    dx = r * np.sqrt(3)
    dy = r * 1.5
    centers = []
    row = 0
    cy = -extent
    while cy <= extent + r:
        offset = (r * np.sqrt(3) / 2) if row % 2 else 0
        cx = -extent + offset
        while cx <= extent + r:
            centers.append((cx, cy))
            cx += dx
        cy += dy
        row += 1
    return centers, r


def draw_corners(ax) -> None:
    pts = [CORNERS[a] for a in ["A1", "A2", "A3", "A4", "A1"]]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "k-", lw=1.2, zorder=4)
    for name, (cx, cy) in CORNERS.items():
        ax.scatter([cx], [cy], s=260, facecolor="white",
                   edgecolor="black", lw=1.2, zorder=5)
        ax.text(cx, cy, name[-1], ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=6)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", default="outputs/main32_mce_volat_ep150",
                   type=pathlib.Path)
    p.add_argument("--alpha-csv",
                   default="outputs/analysis/archetype_pair_level_volat_ep150_coldonly/k4_pair_alpha_with_metrics.csv",
                   type=pathlib.Path)
    p.add_argument("--out",
                   default="outputs/analysis/multi_horizon_stacking_volat_ep150/fig_diamond_pair_hexbin.pdf",
                   type=pathlib.Path)
    p.add_argument("--gridsize", type=int, default=10)
    p.add_argument("--mincnt", type=int, default=20,
                   help="hex 内最小サンプル数 (それ以下は非表示)")
    p.add_argument("--vlim", type=float, default=0.15)
    args = p.parse_args()

    # 1) pair_predictions の全 horizon を pool
    long_df = load_pair_long(args.pred_root)
    print(f"long records: {len(long_df)} from 4 horizons")

    # 2) pair α と merge
    alpha = pd.read_csv(args.alpha_csv)
    long_df = long_df.merge(alpha[["developer", "directory", "A1", "A2", "A3", "A4"]],
                            on=["developer", "directory"], how="inner")
    long_df["x"] = long_df["A2"] - long_df["A4"]
    long_df["y"] = long_df["A3"] - long_df["A1"]
    print(f"after merge: {len(long_df)} samples, "
          f"{long_df.groupby(['developer','directory']).ngroups} pairs")

    # 3) hex grid 割当
    centers, r_hex = make_hex_grid(args.gridsize)
    cx_arr = np.array([c[0] for c in centers])
    cy_arr = np.array([c[1] for c in centers])
    xs, ys = long_df["x"].to_numpy(), long_df["y"].to_numpy()
    dist2 = (xs[:, None] - cx_arr[None, :]) ** 2 + (ys[:, None] - cy_arr[None, :]) ** 2
    long_df["hex"] = dist2.argmin(axis=1)

    # 4) hex 別 pooled AUC
    results = []
    for hidx, grp in long_df.groupby("hex"):
        n = len(grp)
        if n < args.mincnt:
            continue
        y = grp["label"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        try:
            a_irl = roc_auc_score(y, grp["irl_dir_prob_calibrated"])
            a_rf = roc_auc_score(y, grp["rf_dir_prob"])
        except Exception:
            continue
        results.append({
            "hex": hidx, "cx": cx_arr[hidx], "cy": cy_arr[hidx],
            "n": n, "n_pos": int(y.sum()),
            "auc_irl": a_irl, "auc_rf": a_rf,
            "diff": a_irl - a_rf,
        })
    res = pd.DataFrame(results)
    print(f"valid hex: {len(res)} / {len(centers)} "
          f"(mincnt={args.mincnt})")

    # 5) 描画 (3 panel: IRL, RF, Diff)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    for ax, (col, title, cmap, vmn, vmx) in zip(axes, [
        ("auc_irl", "IRL_cold pooled AUC", "RdYlGn", 0.5, 0.9),
        ("auc_rf", "RF_dir pooled AUC", "RdYlGn", 0.5, 0.9),
        ("diff", "Δ = AUC$_{IRL}$ − AUC$_{RF}$", "RdBu_r",
         -args.vlim, args.vlim),
    ]):
        draw_corners(ax)
        patches = []
        vals = []
        for _, row in res.iterrows():
            hexp = RegularPolygon((row["cx"], row["cy"]),
                                  numVertices=6, radius=r_hex,
                                  orientation=np.pi / 6)
            patches.append(hexp)
            vals.append(row[col])
        coll = PatchCollection(patches, cmap=cmap)
        coll.set_array(np.array(vals))
        coll.set_clim(vmn, vmx)
        coll.set_edgecolor("white")
        coll.set_linewidth(0.4)
        ax.add_collection(coll)
        # n 注釈 (薄く)
        for _, row in res.iterrows():
            ax.text(row["cx"], row["cy"], f"{row['n']}",
                    ha="center", va="center", fontsize=5.5,
                    color="black", alpha=0.55, zorder=6)
        ax.set_title(f"{title}\n(mean over hexes = "
                     f"{res[col].mean():.3f})", fontsize=11)
        cbar = fig.colorbar(coll, ax=ax, shrink=0.75, pad=0.03)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        f"Pair-level diamond hex-bin: pooled per-pair AUC "
        f"(n={len(long_df)} samples / {res['n'].sum()} in valid hexes, "
        f"gridsize={args.gridsize}, mincnt={args.mincnt})",
        fontsize=12, y=1.02,
    )
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.out}")
    res.to_csv(args.out.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
