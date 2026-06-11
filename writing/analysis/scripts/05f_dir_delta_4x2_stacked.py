# -*- coding: utf-8 -*-
"""Δ=IRL−RF 大の dir で開発者 archetype と ペア archetype を **独立軸** で見る.

§6 冒頭 `05_dir_archetype_distribution.py` の形式 (4 archetype × 2 種類) を、
§6.7 の Δ ベース dir 群 (IRL 優位 Top5 / RF 優位 Top5) に適用する。
16 セル (4×4) 版より粒度が荒い代わりに、dev / pair それぞれの軸での
構成比が読みやすい。

レイアウト:
  2 row × 2 col panel
    row 0: IRL 優位 Top 5 dir
    row 1: RF 優位 Top 5 dir
    col 0: 開発者archetype (A1-A4) の構成比 stacked bar
    col 1: ペアarchetype (P1-P4) の構成比 stacked bar
  各 dir の右に IRL/RF AUC, Δ, n, base rate を注記

出力:
  results/fig_dir_delta_archetype4x2_stacked.pdf
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
ROBUST_N = 20
N_PICK = 5

CODES = ["A1", "A2", "A3", "A4"]
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
    "A1": "#E64B35",  # 赤
    "A2": "#4DBBD5",  # 水色
    "A3": "#00A087",  # 緑
    "A4": "#F39B7F",  # 橙
}


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def pool_aucs_and_baserate(sub: pd.DataFrame):
    y, irl, rf = [], [], []
    for h in HORIZONS:
        mask = ~sub[f"label_{h}"].isna()
        y.extend(sub.loc[mask, f"label_{h}"].astype(int).tolist())
        irl.extend(sub.loc[mask, f"irl_{h}"].tolist())
        rf.extend(sub.loc[mask, f"rf_{h}"].tolist())
    if not y:
        return None, None, np.nan
    return safe_auc(y, irl), safe_auc(y, rf), float(np.mean(y))


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)

    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv")
    pa = pa.rename(columns={"dominant": "pair_archetype"})[
        ["developer", "directory", "pair_archetype"]
    ]
    full = pred.merge(pa, on=["developer", "directory"], how="left")
    df_arch = full.dropna(subset=["archetype", "pair_archetype"]).copy()

    # dir 別 AUC (全ペアで計算) + 各 archetype 構成比
    rows = []
    for d in full["directory"].unique():
        s_all = full[full["directory"] == d]
        if len(s_all) < ROBUST_N:
            continue
        irl, rf, br = pool_aucs_and_baserate(s_all)
        if irl is None or rf is None:
            continue
        s_arch = df_arch[df_arch["directory"] == d]
        n_arch = len(s_arch)
        dev_dist = (s_arch["archetype"].value_counts(normalize=True).to_dict()
                    if n_arch else {})
        pair_dist = (s_arch["pair_archetype"].value_counts(normalize=True).to_dict()
                     if n_arch else {})
        row = {
            "directory": d, "n_pair": len(s_all), "n_pair_arch": n_arch,
            "base_rate": br, "irl_auc": irl, "rf_auc": rf,
            "delta": irl - rf,
        }
        for a in CODES:
            row[f"dev_{a}"] = dev_dist.get(a, 0)
            row[f"pair_{a}"] = pair_dist.get(a, 0)
        rows.append(row)
    dir_df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print(f"対象 dir (n_pair>={ROBUST_N}): {len(dir_df)}")

    picks_irl = dir_df.head(N_PICK).reset_index(drop=True)
    picks_rf = dir_df.tail(N_PICK).iloc[::-1].reset_index(drop=True)

    print("\n=== IRL 優位 Top 5 (Δ>0) ===")
    print(picks_irl[["directory", "n_pair", "n_pair_arch", "base_rate",
                     "irl_auc", "rf_auc", "delta",
                     "dev_A1", "dev_A2", "dev_A3", "dev_A4",
                     "pair_A1", "pair_A2", "pair_A3", "pair_A4"]]
          .to_string(index=False))
    print("\n=== RF 優位 Top 5 (Δ<0) ===")
    print(picks_rf[["directory", "n_pair", "n_pair_arch", "base_rate",
                    "irl_auc", "rf_auc", "delta",
                    "dev_A1", "dev_A2", "dev_A3", "dev_A4",
                    "pair_A1", "pair_A2", "pair_A3", "pair_A4"]]
          .to_string(index=False))

    # ── 図描画 (2x2 panel) ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))

    for col_idx, (arch_type, name_map) in enumerate([
        ("dev", DEV_ARCH_NAMES),
        ("pair", PAIR_ARCH_NAMES),
    ]):
        col_handles, col_labels = [], []
        for row_idx, (label, dirs_sub) in enumerate([
            (f"IRL 優位 (Δ>0) Top {N_PICK}", picks_irl),
            (f"RF 優位 (Δ<0) Top {N_PICK}", picks_rf),
        ]):
            ax = axes[row_idx, col_idx]
            cols = [f"{arch_type}_{a}" for a in CODES]
            data = dirs_sub[cols].to_numpy()
            y = np.arange(len(dirs_sub))
            left = np.zeros(len(dirs_sub))
            for j, a in enumerate(CODES):
                vals = data[:, j]
                bar = ax.barh(y, vals, left=left,
                              color=ARCH_COLORS[a], alpha=0.88,
                              label=name_map[a],
                              edgecolor="white", lw=0.5)
                left += vals
                if row_idx == 0:
                    col_handles.append(bar)
                    col_labels.append(name_map[a])
            # 右側メタ情報
            for i, (_, r) in enumerate(dirs_sub.iterrows()):
                ax.text(
                    1.02, i,
                    f"IRL={r['irl_auc']:.2f} RF={r['rf_auc']:.2f} "
                    f"Δ={r['delta']:+.2f} n={int(r['n_pair_arch'])}/"
                    f"{int(r['n_pair'])} br={r['base_rate']:.2f}",
                    va="center", fontsize=8, family="monospace",
                )
            ax.set_yticks(y)
            ax.set_yticklabels(dirs_sub["directory"], fontsize=9)
            ax.invert_yaxis()
            ax.set_xlim(0, 1.0)
            ax.set_xlabel("構成比", fontsize=10)
            kind_ja = "開発者archetype" if arch_type == "dev" else "ペアarchetype"
            ax.set_title(f"{label}: {kind_ja}分布", fontsize=11)
            ax.grid(axis="x", alpha=0.3)
            ax.set_axisbelow(True)

        # 列ごとに凡例を下段の下に
        axes[1, col_idx].legend(
            col_handles, col_labels,
            loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
            fontsize=9, frameon=False,
        )

    fig.suptitle(
        f"Δ=IRL−RF 大の dir × archetype 分布 "
        f"(n_pair>={ROBUST_N}, 開発者 4 + ペア 4 を独立軸で表示)",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_dir_delta_archetype4x2_stacked.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_dir_delta_archetype4x2_stacked.pdf'}")


if __name__ == "__main__":
    main()
