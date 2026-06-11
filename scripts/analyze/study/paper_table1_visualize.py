# -*- coding: utf-8 -*-
"""論文 Table 1 (Archetype × Horizon の IRL/RF/Uniform AUC) の可視化.

2 panel:
  (A) Δ (IRL − RF) heatmap — どこで IRL/RF が逆転するか一目で
  (B) archetype 別 line plot — IRL / RF / Uniform AUC の 4 horizon 軌跡

入力: outputs/analysis/multi_horizon_stacking_volat_ep150/motivation_archetype_diagonal.csv
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHS = ["A1", "A2", "A3", "A4"]
ARCH_FULL = {
    "A1": "A1: Sporadic Newcomer",
    "A2": "A2: Strict Reviewer-Only",
    "A3": "A3: Approving Veteran",
    "A4": "A4: Saturated Active",
}
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
ARCH_COLORS = {
    "A1": "#E64B35", "A2": "#4DBBD5",
    "A3": "#00A087", "A4": "#F39B7F",
}


def main() -> None:
    src = pathlib.Path(
        "outputs/analysis/multi_horizon_stacking_volat_ep150/motivation_archetype_diagonal.csv"
    )
    df = pd.read_csv(src)
    df = df[df["archetype"].isin(ARCHS)].copy()
    df["diff"] = df["auc_irl"] - df["auc_rf"]

    def mat(col):
        m = np.zeros((len(ARCHS), len(HORIZONS)))
        for i, a in enumerate(ARCHS):
            for j, h in enumerate(HORIZONS):
                r = df[(df["archetype"] == a) & (df["horizon"] == h)]
                m[i, j] = r[col].iloc[0] if len(r) else np.nan
        return m

    fig = plt.figure(figsize=(13.5, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5], wspace=0.28)

    # ---- Panel A: Δ heatmap ----
    ax = fig.add_subplot(gs[0, 0])
    dmat = mat("diff")
    irl = mat("auc_irl")
    rf = mat("auc_rf")
    im = ax.imshow(dmat, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
    for i in range(len(ARCHS)):
        for j in range(len(HORIZONS)):
            v = dmat[i, j]
            color = "white" if abs(v) > 0.09 else "black"
            ax.text(j, i - 0.18, f"{v:+.03f}",
                    ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold")
            ax.text(j, i + 0.22,
                    f"IRL {irl[i,j]:.02f} / RF {rf[i,j]:.02f}",
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS, fontsize=10)
    ax.set_yticks(range(len(ARCHS)))
    ax.set_yticklabels([ARCH_FULL[a] for a in ARCHS], fontsize=9)
    ax.set_title("(A) Δ = AUC$_{IRL}$ − AUC$_{RF}$\n"
                 "青=RF 優位／赤=IRL 優位",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02,
                        label="Δ (IRL − RF)")
    cbar.ax.tick_params(labelsize=8)

    # アノテーション: 補完性の方向
    ax.annotate("A1: 全 horizon で\nRF 一貫優位",
                xy=(3, 0), xytext=(4.6, 0),
                fontsize=8, color="#1f77b4", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1),
                annotation_clip=False, va="center")
    ax.annotate("A2: 全 horizon で\nIRL 一貫優位",
                xy=(3, 1), xytext=(4.6, 1),
                fontsize=8, color="#d62728", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1),
                annotation_clip=False, va="center")

    # ---- Panel B: line plot per archetype ----
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(HORIZONS))
    for i, a in enumerate(ARCHS):
        sub = df[df["archetype"] == a].set_index("horizon").reindex(HORIZONS)
        c = ARCH_COLORS[a]
        ax2.plot(x, sub["auc_irl"], "-o", color=c, lw=1.6, ms=6,
                 label=f"{a} IRL")
        ax2.plot(x, sub["auc_rf"], "--s", color=c, lw=1.2, ms=5,
                 markerfacecolor="white", label=f"{a} RF")
        # Δ shading: 正なら IRL marker を強調、負なら RF marker
        for j, h in enumerate(HORIZONS):
            v_irl = sub.loc[h, "auc_irl"]
            v_rf = sub.loc[h, "auc_rf"]
            ax2.plot([j, j], [v_irl, v_rf], "-", color=c, lw=0.5, alpha=0.4)

    # Uniform を背景に薄く
    uni = df.groupby("horizon")["auc_avg"].mean().reindex(HORIZONS)
    ax2.plot(x, uni.values, ":", color="gray", lw=1.4, label="Uniform 0.5 (mean over arch)")

    ax2.set_xticks(x)
    ax2.set_xticklabels(HORIZONS, fontsize=10)
    ax2.set_ylabel("AUC (per-pair pooled, diagonal cells)", fontsize=10)
    ax2.set_xlabel("Evaluation horizon", fontsize=10)
    ax2.set_ylim(0.50, 0.90)
    ax2.set_title("(B) Archetype × Horizon の IRL / RF AUC 軌跡\n"
                  "実線 = IRL, 点線 = RF, 縦バー = |Δ|",
                  fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_axisbelow(True)
    # 2 列の凡例 (arch+method)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc="lower right", ncol=2,
               fontsize=7.5, frameon=True, framealpha=0.9)

    fig.suptitle(
        "Table 1 可視化: archetype × horizon の IRL/RF 補完性",
        fontsize=12, y=1.02,
    )
    out = pathlib.Path(
        "outputs/analysis/multi_horizon_stacking_volat_ep150/fig_paper_table1.pdf"
    )
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
