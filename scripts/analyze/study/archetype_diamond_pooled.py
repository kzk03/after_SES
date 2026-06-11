# -*- coding: utf-8 -*-
"""k=4 archetype Diamond: 各 hex 領域内で pooled AUC を計算 (label 一貫 dev も含む)。

母集団:
  全 active dev × 4 eval window = 289 × 4 = 1,156 samples
  (per-dev AUC では n=103 に絞られるが、pooled では全員含む)

各 hex に入る (dev, window) サンプルを pool して AUC_IRL, AUC_RF を計算、
diff = AUC_IRL − AUC_RF を色で表示。
"""
from __future__ import annotations
import argparse
import pathlib
import glob

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

DEV_LABELS = {
    "A1": "A1: Sporadic Newcomer",
    "A2": "A2: Strict Reviewer-Only",
    "A3": "A3: Approving Veteran",
    "A4": "A4: Saturated Active",
}


def aggregate_dev_window(pred_dir: pathlib.Path, diagonal: bool = False,
                         train_h_fixed: str = "0-3m") -> pd.DataFrame:
    """4 eval window 分の pair_predictions を読み、(dev, window) ごとに max-prob 集約。

    diagonal=True なら train_h を eval_h と一致させる (stacking と同じ対角 4 セル)。
    diagonal=False (デフォルト) なら train_h_fixed を全 horizon に固定 (旧挙動)。
    """
    rows = []
    for h in ["0-3m", "3-6m", "6-9m", "9-12m"]:
        train_h = h if diagonal else train_h_fixed
        p = pred_dir / f"lstm_baseline/train_{train_h}/eval_{h}/pair_predictions.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        agg = d.groupby("developer").agg(
            label=("label", "max"),
            irl=("irl_dir_prob_calibrated", "max"),
            rf=("rf_dir_prob", "max"),
        ).reset_index()
        agg["window"] = h
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def make_hex_grid(gridsize: int):
    """diamond [-1,1]×[-1,1] を hex grid に分割。各 hex の中心 (cx, cy) と半径を返す。"""
    extent = 1.1
    r = extent / gridsize  # hex 半径
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=pathlib.Path, required=True)
    p.add_argument("--pred-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_volat_ep150"))
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--gridsize", type=int, default=14)
    p.add_argument("--vmin", type=float, default=-0.15)
    p.add_argument("--vmax", type=float, default=0.15)
    p.add_argument("--mincnt", type=int, default=10)
    p.add_argument("--diagonal", action="store_true",
                   help="train_h を eval_h と一致させる (stacking と同じ対角 4 セル)")
    p.add_argument("--train-h", default="0-3m",
                   help="diagonal=False 時の固定 train_h (旧挙動)")
    args = p.parse_args()

    # 1) alpha
    alpha = pd.read_csv(args.alpha_csv, index_col=0)
    alpha.index.name = "developer"
    alpha = alpha.reset_index()

    # 2) (dev, window) サンプル
    dw = aggregate_dev_window(args.pred_dir, diagonal=args.diagonal,
                              train_h_fixed=args.train_h)
    mode = "diagonal" if args.diagonal else f"fixed train_{args.train_h}"
    print(f"(dev, window) samples: {len(dw)}  [{mode}]")

    # 3) alpha と merge
    m = dw.merge(alpha[["developer", "A1", "A2", "A3", "A4"]], on="developer", how="inner")
    print(f"merged: {len(m)} samples ({m.developer.nunique()} dev × {m.window.nunique()} window)")

    m["x"] = m.A2 - m.A4
    m["y"] = m.A1 - m.A3

    # 4) hex grid 分割
    centers, r_hex = make_hex_grid(args.gridsize)
    # 各 sample をどの hex に割り当てるか (最寄り center)
    cx_arr = np.array([c[0] for c in centers])
    cy_arr = np.array([c[1] for c in centers])
    xs, ys = m["x"].to_numpy(), m["y"].to_numpy()
    dist2 = (xs[:, None] - cx_arr[None, :]) ** 2 + (ys[:, None] - cy_arr[None, :]) ** 2
    hex_idx = dist2.argmin(axis=1)
    m["hex"] = hex_idx

    # 5) hex ごとに pooled AUC
    results = []
    for hidx, grp in m.groupby("hex"):
        n = len(grp)
        if n < args.mincnt:
            continue
        y = grp.label.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        try:
            auc_irl = roc_auc_score(y, grp.irl.to_numpy())
            auc_rf = roc_auc_score(y, grp.rf.to_numpy())
            diff = auc_irl - auc_rf
            results.append({
                "hex": hidx, "cx": cx_arr[hidx], "cy": cy_arr[hidx],
                "n": n, "auc_irl": auc_irl, "auc_rf": auc_rf, "diff": diff,
            })
        except ValueError:
            continue
    rdf = pd.DataFrame(results)
    print(f"valid hex: {len(rdf)} / {len(centers)}, n_samples covered: {rdf.n.sum()}")

    # 6) 描画
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.plot([0, 1, 0, -1, 0], [1, 0, -1, 0, 1], "k-", linewidth=1.2)
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=args.vmin, vmax=args.vmax)
    patches = []
    for _, row in rdf.iterrows():
        hex_patch = RegularPolygon(
            (row.cx, row.cy), numVertices=6, radius=r_hex * 0.95,
            orientation=np.pi / 6,
            facecolor=cmap(norm(row["diff"])),
            edgecolor="white", linewidth=0.4,
        )
        patches.append(hex_patch)
        ax.add_patch(hex_patch)

    # 頂点
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
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("AUC(Cold IRL) − AUC(RF_Dir)  [pooled per hex]\n+ → IRL 強   − → RF 強", fontsize=9)
    ax.set_title(f"k=4 Archetype Diamond: pooled AUC per hex (Cold IRL vs RF)\n"
                 f"n={len(m)} (dev, window) samples, "
                 f"valid hex={len(rdf)}, mincnt={args.mincnt}",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"saved: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
