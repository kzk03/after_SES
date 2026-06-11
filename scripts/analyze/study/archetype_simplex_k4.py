# -*- coding: utf-8 -*-
"""k=4 archetype α 行列の可視化 (PCA + pairwise simplex)。

k=3 は三角プロットで全情報表示できるが、k=4 は 4-simplex (tetrahedron) で
2D 表示しきれない。3 通りの可視化を出力:
  1. PCA projection of α (n×4 → n×2), dominant archetype で色分け
  2. Pairwise 2D scatter: (A1,A2), (A3,A4) など 6 通り
  3. Stacked bar: dev を dominant archetype で sort して α 内訳を見せる

色 (heatmap) は cold IRL accuracy − RF accuracy で IRL/RF 優勢領域を表示。

入力:
  outputs/analysis/archetype_irl_rf_deep/k4_dev_alpha_with_metrics.csv

出力:
  outputs/analysis/archetype_irl_rf_deep/fig_k4_pca_simplex.pdf
  outputs/analysis/archetype_irl_rf_deep/fig_k4_pairwise.pdf
  outputs/analysis/archetype_irl_rf_deep/fig_k4_stacked_bar.pdf

使い方:
  uv run python scripts/analyze/study/archetype_simplex_k4.py
"""

from __future__ import annotations

import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


ARCH_COLORS = {"A1": "#1f77b4", "A2": "#d62728", "A3": "#2ca02c", "A4": "#ff7f0e"}
ARCH_NAMES = {
    "A1": "Sporadic Newcomer",
    "A2": "Strict Reviewer-Only",
    "A3": "Approving Veteran",
    "A4": "Saturated Active",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_irl_rf_deep/k4_dev_alpha_with_metrics.csv"))
    p.add_argument("--out-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_irl_rf_deep"))
    args = p.parse_args()

    df = pd.read_csv(args.alpha_csv, index_col=0)
    arch_cols = ["A1", "A2", "A3", "A4"]
    A = df[arch_cols].to_numpy()
    n = len(df)

    diff = (df["acc_cold_irl"] - df["acc_rf_dir"]).to_numpy()  # IRL - RF 勝敗
    auc_diff = (df["auc_cold_irl"] - df["auc_rf_dir"]).to_numpy()
    dom = df["dominant"].to_numpy()

    # ── 1. PCA projection of α ──
    pca = PCA(n_components=2, random_state=777)
    Y = pca.fit_transform(A)
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 1a. PCA, color = dominant archetype
    ax = axes[0]
    for arch in arch_cols:
        mask = (dom == arch)
        ax.scatter(Y[mask, 0], Y[mask, 1], s=24, alpha=0.75,
                   color=ARCH_COLORS[arch], label=f"{arch}: {ARCH_NAMES[arch]} (n={int(mask.sum())})",
                   edgecolor="k", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(f"α 行列 PCA (k=4 archetype, n={n})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # 1b. PCA, color = AUC(IRL) - AUC(RF) — IRL 優勢 vs RF 優勢の地形図
    ax = axes[1]
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=auc_diff, cmap="RdBu_r",
                     vmin=-0.5, vmax=0.5, s=30, edgecolor="k", linewidth=0.3)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title("AUC(Cold IRL) − AUC(RF_Dir) 地形図\n+ → IRL 強、− → RF 強、NaN → 灰色")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("AUC diff")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_k4_pca_simplex.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.out_dir/'fig_k4_pca_simplex.pdf'}")

    # ── 2. Pairwise 2D scatter (6 通り) ──
    pairs = [("A1", "A2"), ("A1", "A3"), ("A1", "A4"),
              ("A2", "A3"), ("A2", "A4"), ("A3", "A4")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (a, b) in zip(axes.flat, pairs):
        for arch in arch_cols:
            mask = (dom == arch)
            ax.scatter(df.loc[mask, a], df.loc[mask, b], s=14, alpha=0.6,
                        color=ARCH_COLORS[arch], edgecolor="k", linewidth=0.2,
                        label=arch if (a, b) == pairs[0] else None)
        ax.plot([0, 1], [1, 0], "k--", linewidth=0.5, alpha=0.4)  # α_a + α_b ≤ 1 の境界
        ax.set_xlabel(f"α_{a}")
        ax.set_ylabel(f"α_{b}")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("k=4 archetype α 散布図 (全 6 ペア、点色 = dominant archetype)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_k4_pairwise.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.out_dir/'fig_k4_pairwise.pdf'}")

    # ── 3. Stacked bar (dev sorted by dominant) ──
    # dominant 順 → 各 dominant 内では archetype 重みでソート
    sort_key = []
    for i in range(n):
        d = dom[i]
        sort_key.append((arch_cols.index(d), -A[i, arch_cols.index(d)]))
    order = sorted(range(n), key=lambda i: sort_key[i])
    A_sorted = A[order]

    fig, ax = plt.subplots(figsize=(12, 4))
    bottom = np.zeros(n)
    x = np.arange(n)
    for j, arch in enumerate(arch_cols):
        ax.bar(x, A_sorted[:, j], bottom=bottom, width=1.0,
               color=ARCH_COLORS[arch], label=f"{arch}: {ARCH_NAMES[arch]}")
        bottom += A_sorted[:, j]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f"開発者 (n={n}, dominant archetype でソート)")
    ax.set_ylabel("α (archetype membership)")
    ax.set_title("k=4 archetype: 開発者ごとの archetype 帰属")
    ax.legend(loc="upper right", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_k4_stacked_bar.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.out_dir/'fig_k4_stacked_bar.pdf'}")


if __name__ == "__main__":
    main()
