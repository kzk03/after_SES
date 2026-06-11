# -*- coding: utf-8 -*-
"""IRL / RF それぞれの dir AUC で sort した TOP/BOTTOM dir の 16 セル占有率比較.

§6.5 では Ens AUC で TOP10/BOTTOM10 dir に分けて 16 セル占有率を見た。
ここでは順位付け基準を **IRL AUC** と **RF AUC** に変えて、
「IRL が当てやすい/にくい dir 群はどんな archetype 構成か」
「RF が当てやすい/にくい dir 群はどうか」を heatmap で比較する。

入力:
  data/pair_trajectories.csv
  data/pair_archetype.csv

出力:
  results/archetype16_dir_aucs.csv             dir × (IRL/RF/Ens AUC, n_pair)
  results/fig_archetype16_irl_top_bot.pdf       IRL AUC TOP10/BOTTOM10/差 の 3 連 heatmap
  results/fig_archetype16_rf_top_bot.pdf        RF AUC TOP10/BOTTOM10/差 の 3 連 heatmap
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
MIN_DIR_SIZE = 30

DEV_LABELS = ["A1\n低活動新人", "A2\nレビュー専念", "A3\n高承諾ベテラン", "A4\n常時活動"]
PAIR_LABELS = ["P1\n未経験", "P2\n経験低承諾", "P3\n経験高承諾", "P4\n中程度関与"]
CODES = ["A1", "A2", "A3", "A4"]


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def pool_auc(sub: pd.DataFrame, score_prefix: str) -> float | None:
    y, s = [], []
    for h in HORIZONS:
        mask = ~sub[f"label_{h}"].isna()
        y.extend(sub.loc[mask, f"label_{h}"].astype(int).tolist())
        s.extend(sub.loc[mask, f"{score_prefix}_{h}"].tolist())
    return safe_auc(y, s)


def cell_share(sub: pd.DataFrame) -> np.ndarray:
    """16 セル占有率 (合計 1)."""
    mat = np.zeros((4, 4), dtype=float)
    for i, a in enumerate(CODES):
        for j, p in enumerate(CODES):
            mat[i, j] = ((sub["archetype"] == a) &
                         (sub["pair_archetype"] == p)).sum()
    return mat / mat.sum() if mat.sum() > 0 else mat


def annotate(ax, mat: np.ndarray, fmt: str = "{:.2f}") -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            text = "—" if (v == 0 and "+" not in fmt) else fmt.format(v)
            ax.text(j, i, text, ha="center", va="center", fontsize=10,
                    color="black")


def plot_three(top_share: np.ndarray, bot_share: np.ndarray,
               diff_share: np.ndarray, model_label: str,
               n_top_dir: int, n_bot_dir: int,
               n_top_pair: int, n_bot_pair: int,
               out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cfgs = [
        (axes[0], top_share,
         f"上位群 ({model_label} 当てやすい)\n{n_top_dir} dir / {n_top_pair} ペア",
         "YlOrBr", (0, 0.30), "{:.2f}"),
        (axes[1], bot_share,
         f"下位群 ({model_label} 当てにくい)\n{n_bot_dir} dir / {n_bot_pair} ペア",
         "YlOrBr", (0, 0.30), "{:.2f}"),
        (axes[2], diff_share, "上位 − 下位 (差)", "RdBu_r", None, "{:+.2f}"),
    ]
    for ax, mat, title, cmap, vlim, fmt in cfgs:
        if vlim is None:
            v = max(0.05, np.nanmax(np.abs(mat)))
            im = ax.imshow(mat, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        else:
            im = ax.imshow(mat, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                           aspect="auto")
        annotate(ax, mat, fmt=fmt)
        ax.set_xticks(range(4))
        ax.set_xticklabels(PAIR_LABELS, fontsize=8)
        ax.set_yticks(range(4))
        ax.set_yticklabels(DEV_LABELS, fontsize=8)
        ax.set_xlabel("ペア archetype", fontsize=10)
        ax.set_ylabel("開発者 archetype", fontsize=10)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"16 セル占有率: {model_label} AUC で中央値分割 (n_pair >= {MIN_DIR_SIZE} の dir)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)

    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv")
    pa = pa.rename(columns={"dominant": "pair_archetype"})[
        ["developer", "directory", "pair_archetype"]
    ]
    df = pred.merge(pa, on=["developer", "directory"], how="left")
    df = df.dropna(subset=["archetype", "pair_archetype"]).copy()
    print(f"全ペア: {len(df)}")

    # n_pair>=30 dir
    dir_sizes = df.groupby("directory").size()
    keep_dirs = dir_sizes[dir_sizes >= MIN_DIR_SIZE].index
    sub = df[df["directory"].isin(keep_dirs)].copy()
    print(f"対象 dir: {len(keep_dirs)}, 対象ペア: {len(sub)}")

    # ── 各 dir で IRL / RF / Ens AUC ──────────────
    rows = []
    for d in sub["directory"].unique():
        s = sub[sub["directory"] == d]
        rows.append({
            "directory": d,
            "n_pair": len(s),
            "irl_auc": pool_auc(s, "irl"),
            "rf_auc": pool_auc(s, "rf"),
            "ens_auc": pool_auc(s, "avg"),
        })
    dir_df = pd.DataFrame(rows)
    dir_df = dir_df.dropna(subset=["irl_auc", "rf_auc", "ens_auc"])
    dir_df.to_csv(OUT / "archetype16_dir_aucs.csv", index=False)
    print(f"saved: {OUT/'archetype16_dir_aucs.csv'}")
    print(dir_df.sort_values("ens_auc", ascending=False).to_string(index=False))

    def split_by_median(score_col: str) -> tuple[set, set]:
        med = dir_df[score_col].median()
        top = set(dir_df[dir_df[score_col] > med]["directory"])
        bot = set(dir_df[dir_df[score_col] < med]["directory"])
        return top, bot

    # ── IRL AUC 中央値分割 ──────────────────────────
    top_dirs_irl, bot_dirs_irl = split_by_median("irl_auc")
    sub_top_irl = sub[sub["directory"].isin(top_dirs_irl)]
    sub_bot_irl = sub[sub["directory"].isin(bot_dirs_irl)]
    top_share_irl = cell_share(sub_top_irl)
    bot_share_irl = cell_share(sub_bot_irl)
    diff_irl = top_share_irl - bot_share_irl
    print(f"\nIRL AUC 上位群 ({len(top_dirs_irl)} dir): {sorted(top_dirs_irl)}")
    print(f"IRL AUC 下位群 ({len(bot_dirs_irl)} dir): {sorted(bot_dirs_irl)}")
    plot_three(top_share_irl, bot_share_irl, diff_irl, "IRL",
               len(top_dirs_irl), len(bot_dirs_irl),
               len(sub_top_irl), len(sub_bot_irl),
               OUT / "fig_archetype16_irl_top_bot.pdf")

    # ── RF AUC 中央値分割 ───────────────────────────
    top_dirs_rf, bot_dirs_rf = split_by_median("rf_auc")
    sub_top_rf = sub[sub["directory"].isin(top_dirs_rf)]
    sub_bot_rf = sub[sub["directory"].isin(bot_dirs_rf)]
    top_share_rf = cell_share(sub_top_rf)
    bot_share_rf = cell_share(sub_bot_rf)
    diff_rf = top_share_rf - bot_share_rf
    print(f"\nRF AUC 上位群 ({len(top_dirs_rf)} dir): {sorted(top_dirs_rf)}")
    print(f"RF AUC 下位群 ({len(bot_dirs_rf)} dir): {sorted(bot_dirs_rf)}")
    plot_three(top_share_rf, bot_share_rf, diff_rf, "RF",
               len(top_dirs_rf), len(bot_dirs_rf),
               len(sub_top_rf), len(sub_bot_rf),
               OUT / "fig_archetype16_rf_top_bot.pdf")

    # ── 数値サマリ ───────────────────────────────
    print("\n=== IRL AUC 上位 - 下位 占有率差 ===")
    print(pd.DataFrame(diff_irl, index=CODES, columns=CODES).round(3))
    print("\n=== RF AUC 上位 - 下位 占有率差 ===")
    print(pd.DataFrame(diff_rf, index=CODES, columns=CODES).round(3))

    print(f"\n上位群 共通 (IRL ∩ RF): {sorted(top_dirs_irl & top_dirs_rf)}")
    print(f"上位群 IRL のみ: {sorted(top_dirs_irl - top_dirs_rf)}")
    print(f"上位群 RF のみ: {sorted(top_dirs_rf - top_dirs_irl)}")
    print(f"下位群 共通: {sorted(bot_dirs_irl & bot_dirs_rf)}")
    print(f"下位群 IRL のみ: {sorted(bot_dirs_irl - bot_dirs_rf)}")
    print(f"下位群 RF のみ: {sorted(bot_dirs_rf - bot_dirs_irl)}")


if __name__ == "__main__":
    main()
