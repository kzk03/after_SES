# -*- coding: utf-8 -*-
"""k=4 archetype Diamond: IRL/RF が "予測できた" 母集団の違いを可視化。

各 (dev, window) を 4 カテゴリに分類:
  - both:    IRL も RF も正解 (両者カバー)
  - irl_only: IRL のみ正解 (IRL 専属)
  - rf_only:  RF のみ正解 (RF 専属)
  - neither:  両者外れ (誰も予測できない)

threshold は 0.5 で pred=1/0、label と一致なら "correct"。

母集団: 全 289 active dev × 4 window = 1,156 (dev, window) sample。
"""
from __future__ import annotations
import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import RegularPolygon

DEV_LABELS = {
    "A1": "A1: Sporadic Newcomer",
    "A2": "A2: Strict Reviewer-Only",
    "A3": "A3: Approving Veteran",
    "A4": "A4: Saturated Active",
}
CAT_COLORS = {
    "both":     "#7f7f7f",  # gray
    "irl_only": "#d62728",  # red
    "rf_only":  "#1f77b4",  # blue
    "neither":  "#000000",  # black
}
CAT_LABELS_JA = {
    "both": "両者正解",
    "irl_only": "IRL のみ正解",
    "rf_only": "RF のみ正解",
    "neither": "両者外れ",
}


def aggregate_dev_window(pred_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for h in ["0-3m", "3-6m", "6-9m", "9-12m"]:
        p = pred_dir / f"lstm_baseline/train_0-3m/eval_{h}/pair_predictions.csv"
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv"))
    p.add_argument("--pred-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_volat_ep150"))
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    alpha = pd.read_csv(args.alpha_csv, index_col=0)
    alpha.index.name = "developer"
    alpha = alpha.reset_index()

    dw = aggregate_dev_window(args.pred_dir)
    m = dw.merge(alpha[["developer", "A1", "A2", "A3", "A4", "dominant"]],
                 on="developer", how="inner")

    m["irl_correct"] = ((m.irl >= args.threshold).astype(int) == m.label).astype(int)
    m["rf_correct"]  = ((m.rf  >= args.threshold).astype(int) == m.label).astype(int)

    def categorize(row):
        if row.irl_correct and row.rf_correct: return "both"
        if row.irl_correct: return "irl_only"
        if row.rf_correct:  return "rf_only"
        return "neither"
    m["category"] = m.apply(categorize, axis=1)
    m["x"] = m.A2 - m.A4
    m["y"] = m.A1 - m.A3

    # ━━━ 集計 ━━━
    print("=== Overall category counts ===")
    cnt = m.category.value_counts()
    for c in ["both", "irl_only", "rf_only", "neither"]:
        n = cnt.get(c, 0)
        print(f"  {c:<10} n={n:>4} ({n/len(m)*100:>5.1f}%)")
    print()
    print("=== By dominant archetype (%) ===")
    pivot = (m.groupby(["dominant", "category"]).size()
             .unstack(fill_value=0)
             .reindex(columns=["both", "irl_only", "rf_only", "neither"], fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    print(pivot_pct.round(1))

    # ━━━ 描画: 2 panel (scatter + archetype bar) ━━━
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1.4, 1]})

    # Panel A: diamond scatter
    ax = axes[0]
    ax.plot([0, 1, 0, -1, 0], [1, 0, -1, 0, 1], "k-", linewidth=1.2)
    # both は薄く背景に
    for c in ["both", "neither", "irl_only", "rf_only"]:
        sub = m[m.category == c]
        size = 12 if c == "both" else 28
        alpha_p = 0.25 if c == "both" else 0.75
        ax.scatter(sub.x, sub.y, c=CAT_COLORS[c], s=size, alpha=alpha_p,
                   edgecolor="none", label=f"{CAT_LABELS_JA[c]} (n={len(sub)})")
    for arch, (vx, vy) in zip(["A1","A2","A3","A4"], [(0,1),(1,0),(0,-1),(-1,0)]):
        ax.plot(vx, vy, "o", markersize=14, markerfacecolor="white",
                markeredgecolor="black", markeredgewidth=1.5, zorder=5)
        ox, oy, ha, va = {(0,1):(0,0.08,"center","bottom"),
                         (1,0):(0.08,0,"left","center"),
                         (0,-1):(0,-0.08,"center","top"),
                         (-1,0):(-0.08,0,"right","center")}[(vx,vy)]
        ax.text(vx+ox, vy+oy, DEV_LABELS[arch], ha=ha, va=va,
                fontsize=10, fontweight="bold")
    ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title(f"各 (dev, window) sample の予測カテゴリ\n"
                 f"n={len(m)} (= {m.developer.nunique()} dev × {m.window.nunique()} window)",
                 fontsize=11)

    # Panel B: archetype 別 stacked bar
    ax = axes[1]
    bottom = np.zeros(len(pivot_pct))
    arch_order = list(pivot_pct.index)
    width = 0.7
    for c in ["both", "irl_only", "rf_only", "neither"]:
        vals = pivot_pct[c].values
        ax.barh(arch_order, vals, left=bottom, color=CAT_COLORS[c],
                label=CAT_LABELS_JA[c], edgecolor="white", linewidth=1)
        # ラベル
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 5:
                ax.text(b + v/2, i, f"{v:.0f}%", ha="center", va="center",
                        color="white" if c in ["irl_only","rf_only","neither"] else "black",
                        fontsize=9, fontweight="bold")
        bottom += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel("各カテゴリの割合 (%)")
    ax.set_title("Dominant archetype 別カテゴリ分布", fontsize=11)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    ax.invert_yaxis()

    fig.suptitle("IRL と RF が正解できた母集団は archetype 軸で異なる",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"\nsaved: {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
