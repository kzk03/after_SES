# -*- coding: utf-8 -*-
"""全 reliable ディレクトリ (286 件) の予測精度を網羅的に可視化.

メモリ「ディレクトリ別分析は小規模 dir も対象に含める」に従い、
MIN_N で機械的に切り捨てず全 286 dir を集計対象とする。
表示は規模で層別 + 散布図で全体感を出す。

出力 (複数図):
  fig_dirs_n_vs_auc.pdf            散布図: n_pair vs Ens AUC (286 dir)
  fig_dirs_delta_hist.pdf          Δ=IRL-RF のヒストグラム + 累積分布
  fig_dirs_ranked_top72.pdf        Ens AUC 上位 72 dir (Q1)
  fig_dirs_ranked_middle.pdf       中位 142 dir (Q2-Q3)
  fig_dirs_ranked_bottom72.pdf     下位 72 dir (Q4)
  fig_dirs_by_project_<proj>.pdf   project 別 sub-dir (拡張版、>=2 sub-dir)

データ:
  ../thesis/data/directory_pooled_metrics.csv (全 390 dir、286 reliable)
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
THESIS = ROOT / "thesis" / "data"
OUT = ROOT / "results"
PRED_CSV = ROOT / "data" / "pair_trajectories.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def plot_ranked_segment(rel_segment: pd.DataFrame, title: str,
                         out_path: pathlib.Path) -> None:
    """セグメントの dir を AUC 順に横棒で描画."""
    n = len(rel_segment)
    fig, ax = plt.subplots(figsize=(11, max(8, n * 0.22)))
    y = np.arange(n)
    width = 0.27
    ax.barh(y - width, rel_segment["auc_irl"], width, color="#1f77b4",
            alpha=0.85, label="IRL")
    ax.barh(y, rel_segment["auc_rf"], width, color="#d62728",
            alpha=0.85, label="RF")
    ax.barh(y + width, rel_segment["auc_ens"], width, color="#2ca02c",
            alpha=0.85, label="アンサンブル")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r['directory']} (n={int(r['n_pair'])})"
         for _, r in rel_segment.iterrows()],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC (4 区間プール)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.axvline(0.5, color="gray", lw=0.5, ls=":")
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    d = pd.read_csv(THESIS / "directory_pooled_metrics.csv")
    # AUC が算出可能 (両クラス有) な全 ディレクトリを対象 (n=286)
    # auc_reliable=True (46) は厳しすぎるため使わない
    rel = d[d["auc_irl"].notna() & d["auc_rf"].notna() & d["auc_ens"].notna()].copy()
    rel["delta"] = rel["auc_irl"] - rel["auc_rf"]
    rel = rel.sort_values("auc_ens", ascending=False).reset_index(drop=True)
    print(f"全 AUC 算出可能ディレクトリ: {len(rel)} (auc_reliable のみ: {(d['auc_reliable']==True).sum()})")
    print(f"n_pair 中央値: {rel['n_pair'].median():.0f}, 最小: {rel['n_pair'].min()}, 最大: {rel['n_pair'].max()}")
    print(f"AUC レンジ: {rel['auc_ens'].min():.3f} - {rel['auc_ens'].max():.3f}")

    # ── 図1: 散布図 n_pair vs Ens AUC (全 286 dir 一望) ─────
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sc = ax.scatter(
        rel["n_pair"], rel["auc_ens"],
        c=rel["delta"], cmap="RdBu", vmin=-0.2, vmax=0.2,
        s=30, alpha=0.7, edgecolors="black", linewidths=0.3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("ペア数 (ディレクトリの規模, 対数)", fontsize=11)
    ax.set_ylabel("アンサンブル AUC (4 区間プール)", fontsize=11)
    ax.set_title(
        f"全 {len(rel)} ディレクトリの規模 × 予測精度 (色 = IRL−RF)",
        fontsize=12)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    plt.colorbar(sc, ax=ax, label="IRL − RF")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dirs_n_vs_auc.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_dirs_n_vs_auc.pdf'}")

    # ── 図2: Δ ヒストグラム + 累積 ─────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(rel["delta"], bins=40, color="#888", edgecolor="black",
                  alpha=0.85)
    axes[0].axvline(0, color="black", lw=1)
    axes[0].axvspan(-0.03, 0.03, color="gray", alpha=0.2, label="拮抗 (|Δ|<0.03)")
    axes[0].set_xlabel("Δ = IRL − RF", fontsize=11)
    axes[0].set_ylabel("ディレクトリ数", fontsize=11)
    axes[0].set_title(f"Δ 分布 (n={len(rel)} ディレクトリ)", fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    # 累積分布
    sorted_d = np.sort(rel["delta"].dropna().values)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    axes[1].plot(sorted_d, cdf, color="#1f77b4", lw=2)
    axes[1].axvline(0, color="black", lw=1)
    axes[1].axhline(0.5, color="gray", lw=0.5, ls=":")
    axes[1].set_xlabel("Δ = IRL − RF", fontsize=11)
    axes[1].set_ylabel("累積比率", fontsize=11)
    axes[1].set_title("Δ 累積分布", fontsize=12)
    axes[1].grid(alpha=0.3)
    # 重要点を注記
    for q, label in [(0.5, "中央値"), (0.25, "Q1"), (0.75, "Q3")]:
        v = np.quantile(sorted_d, q)
        axes[1].text(v, q, f"  {label}={v:+.3f}", fontsize=9,
                      va="center")
        axes[1].plot(v, q, "ro", markersize=6)

    fig.tight_layout()
    fig.savefig(OUT / "fig_dirs_delta_hist.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_dirs_delta_hist.pdf'}")

    # ── 図3: 全 286 dir を 3 セグメントに分けて ranking ─────
    n_total = len(rel)
    q1_end = n_total // 4  # 上位 1/4
    q3_start = n_total - n_total // 4  # 下位 1/4

    top_seg = rel.iloc[:q1_end]
    mid_seg = rel.iloc[q1_end:q3_start]
    bot_seg = rel.iloc[q3_start:]

    plot_ranked_segment(
        top_seg,
        f"アンサンブル AUC 上位 {len(top_seg)} ディレクトリ (全 {n_total} 中)",
        OUT / "fig_dirs_ranked_top.pdf",
    )
    plot_ranked_segment(
        mid_seg,
        f"アンサンブル AUC 中位 {len(mid_seg)} ディレクトリ (全 {n_total} 中)",
        OUT / "fig_dirs_ranked_middle.pdf",
    )
    plot_ranked_segment(
        bot_seg,
        f"アンサンブル AUC 下位 {len(bot_seg)} ディレクトリ (全 {n_total} 中)",
        OUT / "fig_dirs_ranked_bottom.pdf",
    )

    # ── 図4: project 別 sub-dir breakdown (>=2 sub-dir) ─────
    proj_counts = rel.groupby("top_dir").size()
    big_projs = proj_counts[proj_counts >= 2].index.tolist()
    print(f"\nプロジェクト別 (>=2 sub-dir): {len(big_projs)}")

    for proj in big_projs:
        sub = rel[rel["top_dir"] == proj].sort_values("auc_ens",
                                                       ascending=False)
        n_sub = len(sub)
        fig, ax = plt.subplots(figsize=(9, max(3, n_sub * 0.32)))
        y = np.arange(n_sub)
        width = 0.27
        ax.barh(y - width, sub["auc_irl"], width, color="#1f77b4",
                alpha=0.85, label="IRL")
        ax.barh(y, sub["auc_rf"], width, color="#d62728",
                alpha=0.85, label="RF")
        ax.barh(y + width, sub["auc_ens"], width, color="#2ca02c",
                alpha=0.85, label="アンサンブル")
        ax.set_yticks(y)
        ax.set_yticklabels(
            [f"{r['directory']} (n={int(r['n_pair'])}, Δ={r['delta']:+.2f})"
             for _, r in sub.iterrows()],
            fontsize=9,
        )
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("AUC (4 区間プール)", fontsize=11)
        ax.set_title(f"{proj}: サブディレクトリ別予測精度 ({n_sub} dir)",
                     fontsize=12)
        ax.axvline(0.5, color="gray", lw=0.5, ls=":")
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_dirs_by_project_{proj}.pdf", format="pdf",
                    bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {OUT/f'fig_dirs_by_project_{proj}.pdf'}")

    # ── 数値サマリ ───────────────────────────
    print("\n=== AUC 分布 (n=286) ===")
    for col, name in [("auc_irl", "IRL"), ("auc_rf", "RF"), ("auc_ens", "アンサンブル")]:
        v = rel[col]
        print(f"  {name}: 平均={v.mean():.3f}, 中央={v.median():.3f}, "
              f"最小={v.min():.3f}, 最大={v.max():.3f}")
    print("\n=== Δ = IRL − RF (n=286) ===")
    print(f"  IRL 優位 (Δ>0): {(rel['delta']>0).sum()}/{len(rel)}")
    print(f"  RF 優位 (Δ<0):  {(rel['delta']<0).sum()}/{len(rel)}")
    print(f"  拮抗 (|Δ|<0.03): {(rel['delta'].abs()<0.03).sum()}/{len(rel)}")
    print(f"  IRL 大勝 (Δ>+0.05): {(rel['delta']>0.05).sum()}/{len(rel)}")
    print(f"  RF 大勝 (Δ<-0.05): {(rel['delta']<-0.05).sum()}/{len(rel)}")

    # 規模別 (小規模 dir の挙動が見たい)
    print("\n=== 規模別 (n_pair) AUC 平均 ===")
    rel["size_bin"] = pd.cut(
        rel["n_pair"],
        bins=[0, 5, 10, 30, 100, np.inf],
        labels=["~5", "6-10", "11-30", "31-100", "100+"],
    )
    print(rel.groupby("size_bin", observed=True).agg(
        n_dirs=("directory", "count"),
        irl=("auc_irl", "mean"),
        rf=("auc_rf", "mean"),
        ens=("auc_ens", "mean"),
        delta=("delta", "mean"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
