# -*- coding: utf-8 -*-
"""予測しやすい/しにくい dir の archetype 分布比較.

論文 (Hashimoto_SES.tex) §5.2 / §6 補完性の主張を強化するため、
dir レベルの予測性 (Ens AUC) と (a) 開発者archetype分布, (b) ペアarchetype分布
を関連付けて分析する。

入力:
  data/pair_trajectories.csv         (developer, directory, archetype=A1-A4, predictions)
  data/pair_archetype.csv             (developer, directory, pair archetype dominant=A1-A4)

出力:
  results/dir_archetype_distribution.csv      dir × archetype カウント
  results/fig_dir_archetype_stacked.pdf       top/bottom dir の archetype構成比stacked bar
  results/fig_dir_dev_vs_pair_archetype.pdf   開発者archetype vs ペアarchetype比較
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
MIN_DIR_SIZE = 30  # dir 安定性のしきい値

# 名称マッピング
DEV_ARCH_NAMES = {
    "A1": "A1 低活動の新人",
    "A2": "A2 レビュー専念型",
    "A3": "A3 高承諾のベテラン",
    "A4": "A4 常時活動型",
}
PAIR_ARCH_NAMES = {
    "A1": "P1 ディレクトリ未経験",
    "A2": "P2 経験あり・低承諾",
    "A3": "P3 経験あり・高承諾",
    "A4": "P4 中程度関与",
}
ARCH_COLORS = {
    "A1": "#E64B35",
    "A2": "#4DBBD5",
    "A3": "#00A087",
    "A4": "#F39B7F",
}


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pair_arch = pd.read_csv(DATA / "pair_archetype.csv")
    pair_arch = pair_arch.rename(columns={"dominant": "pair_archetype"})[
        ["developer", "directory", "pair_archetype"]
    ]

    # join pair_archetype to pred
    df = pred.merge(pair_arch, on=["developer", "directory"], how="left")
    print(f"pred: {len(pred)}, with pair_arch: {df['pair_archetype'].notna().sum()}")

    # ── per-dir Ens AUC を計算 (4ホライズン pool) ───────────
    dir_rows = []
    for d in df["directory"].unique():
        sub = df[df["directory"] == d]
        if len(sub) < MIN_DIR_SIZE:
            continue
        # pool 4 horizons
        y_all, ens_all = [], []
        for h in HORIZONS:
            mask = ~sub[f"label_{h}"].isna()
            y_all.extend(sub.loc[mask, f"label_{h}"].astype(int).tolist())
            ens_all.extend(sub.loc[mask, f"avg_{h}"].tolist())
        if len(np.unique(y_all)) < 2:
            continue
        ens_auc = safe_auc(y_all, ens_all)
        # archetype 分布
        dev_dist = sub["archetype"].value_counts(normalize=True).to_dict()
        pair_dist = sub["pair_archetype"].value_counts(normalize=True).to_dict()
        # n_dev (unique developers in this dir)
        n_dev = sub["developer"].nunique()
        dir_rows.append({
            "directory": d,
            "top_dir": d.split("/")[0],
            "n_pair": int(len(sub)),
            "n_dev": int(n_dev),
            "ens_auc": ens_auc,
            "dev_A1": dev_dist.get("A1", 0),
            "dev_A2": dev_dist.get("A2", 0),
            "dev_A3": dev_dist.get("A3", 0),
            "dev_A4": dev_dist.get("A4", 0),
            "dev_unknown": 1 - sum(dev_dist.values()),
            "pair_A1": pair_dist.get("A1", 0),
            "pair_A2": pair_dist.get("A2", 0),
            "pair_A3": pair_dist.get("A3", 0),
            "pair_A4": pair_dist.get("A4", 0),
            "pair_unknown": 1 - sum(pair_dist.values()),
        })

    dist_df = pd.DataFrame(dir_rows).sort_values("ens_auc", ascending=False)
    dist_df.to_csv(OUT / "dir_archetype_distribution.csv", index=False)
    print(f"saved: {OUT/'dir_archetype_distribution.csv'} ({len(dist_df)} dirs)")

    # ── 図1: top/bottom 10 dir × 開発者archetype 分布 (stacked) ───
    top_n = 10
    top_dirs = dist_df.head(top_n).copy()
    bot_dirs = dist_df.tail(top_n).copy()[::-1]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    for col_idx, (arch_type, name_map) in enumerate([
        ("dev", DEV_ARCH_NAMES),
        ("pair", PAIR_ARCH_NAMES),
    ]):
        # 列ごとの handles を蓄積
        col_handles, col_labels = [], []
        for row_idx, (label, dirs_sub) in enumerate([
            (f"予測しやすい dir TOP{top_n}", top_dirs),
            (f"予測しにくい dir BOTTOM{top_n}", bot_dirs),
        ]):
            ax = axes[row_idx, col_idx]
            cols = [f"{arch_type}_{a}" for a in ["A1", "A2", "A3", "A4"]]
            data = dirs_sub[cols].to_numpy()
            y = np.arange(len(dirs_sub))
            left = np.zeros(len(dirs_sub))
            for j, (a, col) in enumerate(zip(["A1", "A2", "A3", "A4"], cols)):
                vals = data[:, j]
                bar = ax.barh(y, vals, left=left, color=ARCH_COLORS[a], alpha=0.85,
                        label=name_map[a], edgecolor="white", lw=0.5)
                left += vals
                if row_idx == 0:
                    col_handles.append(bar)
                    col_labels.append(name_map[a])
            # AUC を右側に注記
            for i, (_, r) in enumerate(dirs_sub.iterrows()):
                ax.text(1.02, i, f"AUC={r['ens_auc']:.2f}, n={int(r['n_pair'])}",
                        va="center", fontsize=8)
            ax.set_yticks(y)
            ax.set_yticklabels(dirs_sub["directory"], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("構成比", fontsize=10)
            kind_ja = "開発者archetype" if arch_type == "dev" else "ペアarchetype"
            ax.set_title(f"{label}: {kind_ja}分布", fontsize=11)
            ax.grid(axis="x", alpha=0.3)
            ax.set_axisbelow(True)

        # 列ごとに凡例を BOTTOM 列の下に表示
        axes[1, col_idx].legend(
            col_handles, col_labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
            fontsize=9, frameon=False,
        )

    fig.suptitle("予測性 × archetype 分布 (dir レベル)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dir_archetype_stacked.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_dir_archetype_stacked.pdf'}")

    # ── 図2: AUC vs archetype 比率 散布図 (4 panel for each archetype) ───
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for col_idx, (arch_type, name_map) in enumerate([
        ("dev", DEV_ARCH_NAMES),
        ("pair", PAIR_ARCH_NAMES),
    ]):
        kind_ja = "開発者" if arch_type == "dev" else "ペア"
        for j, a in enumerate(["A1", "A2", "A3", "A4"]):
            ax = axes[col_idx, j]
            x = dist_df[f"{arch_type}_{a}"].to_numpy()
            y = dist_df["ens_auc"].to_numpy()
            ax.scatter(x, y, s=dist_df["n_pair"] / 2, alpha=0.5,
                       color=ARCH_COLORS[a], edgecolors="black", linewidths=0.4)
            # 相関
            valid = ~(np.isnan(x) | np.isnan(y))
            if valid.sum() > 3:
                from scipy.stats import spearmanr
                rho, p = spearmanr(x[valid], y[valid])
                ax.text(0.05, 0.05, f"ρ={rho:+.2f}\np={p:.3f}",
                        transform=ax.transAxes, fontsize=10, va="bottom",
                        bbox=dict(facecolor="white", alpha=0.7))
            ax.set_xlabel(f"{kind_ja}{name_map[a]} 比率", fontsize=9)
            ax.set_ylabel("Ens AUC", fontsize=9)
            ax.set_xlim(-0.05, 1.05)
            ax.set_ylim(0.5, 1.0)
            ax.grid(alpha=0.3)
    fig.suptitle("Dir 別 Ens AUC vs archetype 比率 (各点=1 dir, n_pair≥30)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dir_auc_vs_archetype.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_dir_auc_vs_archetype.pdf'}")

    # ── 数値サマリ ───────────────────────────────
    print("\n=== TOP 5 予測しやすい dir の archetype 分布 ===")
    cols_show = ["directory", "n_pair", "ens_auc",
                 "dev_A1", "dev_A2", "dev_A3", "dev_A4",
                 "pair_A1", "pair_A2", "pair_A3", "pair_A4"]
    print(dist_df.head(5)[cols_show].to_string(index=False))
    print("\n=== BOTTOM 5 予測しにくい dir の archetype 分布 ===")
    print(dist_df.tail(5)[cols_show].to_string(index=False))

    # 平均比較
    high = dist_df.head(10)
    low = dist_df.tail(10)
    print("\n=== TOP10 vs BOTTOM10 平均 archetype 比率 ===")
    for arch_type in ["dev", "pair"]:
        print(f"\n[{arch_type}_archetype]")
        for a in ["A1", "A2", "A3", "A4"]:
            h = high[f"{arch_type}_{a}"].mean()
            lo = low[f"{arch_type}_{a}"].mean()
            print(f"  {a}: TOP10={h:.3f}, BOTTOM10={lo:.3f}, diff={h - lo:+.3f}")


if __name__ == "__main__":
    main()
