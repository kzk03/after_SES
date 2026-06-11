# -*- coding: utf-8 -*-
"""Δ=IRL−RF が大きい dir の 16 セル分布を stacked bar で可視化.

§6 冒頭の `fig_dir_archetype_stacked.pdf` の形式 (dir × archetype の構成比 stacked bar)
を 16 セル (dev arch 4 × pair arch 4) 版に拡張する。

設計:
  - 横軸: 構成比 (0-1)
  - 縦軸: dir 名
  - 1 本のバーの中で 16 セル (A1×P1 ... A4×P4) を積み上げる
  - 色は dev archetype (A1-A4) で 4 色相、pair archetype (P1-P4) で明度を変える
  - 上段: IRL 優位 Top 5 dir (Δ>0), 下段: RF 優位 Top 5 dir (Δ<0)
  - n_pair>=20 でフィルタ (§6.7 と同じ信頼性ライン)

出力:
  results/fig_dir_delta_archetype16_stacked.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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
# dev archetype ごとの基本色 (色相)
DEV_BASE = {
    "A1": "#D62728",  # 赤  A1 低活動新人
    "A2": "#1F77B4",  # 青  A2 レビュー専念
    "A3": "#2CA02C",  # 緑  A3 高承諾ベテラン
    "A4": "#FF7F0E",  # 橙  A4 常時活動
}
# pair archetype ごとの明度 (濃 → 薄)
PAIR_LIGHTNESS = {"A1": 0.55, "A2": 0.75, "A3": 0.90, "A4": 1.05}
DEV_NAMES = {"A1": "A1 低活動新人", "A2": "A2 レビュー専念",
             "A3": "A3 高承諾ベテラン", "A4": "A4 常時活動"}
PAIR_NAMES = {"A1": "P1 未経験", "A2": "P2 経験低承諾",
              "A3": "P3 経験高承諾", "A4": "P4 中程度関与"}


def cell_color(dev_a: str, pair_a: str) -> str:
    base = DEV_BASE[dev_a]
    r, g, b = mcolors.to_rgb(base)
    factor = PAIR_LIGHTNESS[pair_a]
    # factor>1 は白寄せ、<1 は黒寄せ
    if factor >= 1:
        # mix with white
        t = factor - 1
        rgb = (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)
    else:
        rgb = (r * factor, g * factor, b * factor)
    rgb = tuple(np.clip(v, 0, 1) for v in rgb)
    return mcolors.to_hex(rgb)


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def pool_aucs(sub: pd.DataFrame) -> tuple[float | None, float | None]:
    y, irl, rf = [], [], []
    for h in HORIZONS:
        mask = ~sub[f"label_{h}"].isna()
        y.extend(sub.loc[mask, f"label_{h}"].astype(int).tolist())
        irl.extend(sub.loc[mask, f"irl_{h}"].tolist())
        rf.extend(sub.loc[mask, f"rf_{h}"].tolist())
    return safe_auc(y, irl), safe_auc(y, rf)


def dir_cell_shares(df_arch: pd.DataFrame,
                    directories: list[str]) -> pd.DataFrame:
    """16 セル + Unknown の構成比 (行=dir, 列=セル名)."""
    rows = []
    for d in directories:
        s = df_arch[df_arch["directory"] == d]
        n_total = len(s)
        if n_total == 0:
            continue
        row = {"directory": d}
        for a in CODES:
            for p in CODES:
                n_cell = ((s["archetype"] == a)
                          & (s["pair_archetype"] == p)).sum()
                row[f"{a}x{p}"] = n_cell / n_total
        rows.append(row)
    return pd.DataFrame(rows)


def plot_stacked(picks_irl: pd.DataFrame, picks_rf: pd.DataFrame,
                 shares_irl: pd.DataFrame, shares_rf: pd.DataFrame,
                 out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5),
                             gridspec_kw={"hspace": 0.45})

    for ax, label, picks, shares in [
        (axes[0], "IRL 優位 (Δ=IRL−RF > 0) Top 5", picks_irl, shares_irl),
        (axes[1], "RF 優位 (Δ=IRL−RF < 0) Top 5", picks_rf, shares_rf),
    ]:
        dirs = picks["directory"].tolist()
        # 16 セル順序 (A1×P1, A1×P2, ..., A4×P4)
        cell_order = [f"{a}x{p}" for a in CODES for p in CODES]
        y_pos = np.arange(len(dirs))
        left = np.zeros(len(dirs))

        for cell in cell_order:
            a, p = cell.split("x")
            vals = shares.set_index("directory").loc[dirs, cell].to_numpy()
            ax.barh(y_pos, vals, left=left,
                    color=cell_color(a, p), edgecolor="white", lw=0.4,
                    label=f"{a}×{p}" if ax is axes[0] else None)
            left += vals

        # 右側にメタ情報を注記 (IRL/RF AUC, Δ, n, base_rate)
        for i, (_, r) in enumerate(picks.iterrows()):
            text = (f"IRL={r['irl_auc']:.2f}  RF={r['rf_auc']:.2f}  "
                    f"Δ={r['delta']:+.2f}  n={int(r['n_pair_arch'])}/"
                    f"{int(r['n_pair'])}  br={r['base_rate']:.2f}")
            ax.text(1.02, i, text, va="center", fontsize=8.5,
                    family="monospace")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(dirs, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("16 セル占有率", fontsize=10)
        ax.set_title(label, fontsize=11, loc="left")
        ax.grid(axis="x", alpha=0.3)
        ax.set_axisbelow(True)

    # 凡例 (16 セル) を 4×4 グリッドで下に配置
    legend_handles = []
    legend_labels = []
    for a in CODES:
        for p in CODES:
            legend_handles.append(
                mpatches.Patch(color=cell_color(a, p),
                               label=f"{a}×{p}")
            )
            legend_labels.append(f"{a}×{p}")

    fig.legend(legend_handles, legend_labels,
               loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=8, fontsize=9, frameon=False,
               title="dev_arch × pair_arch (色相=dev, 明度=pair, A1 P1 が最も濃い)")

    # 補足凡例: dev archetype 名と pair archetype 名
    fig.text(0.03, -0.10,
             "dev: " + " / ".join(DEV_NAMES.values()),
             fontsize=8, ha="left")
    fig.text(0.03, -0.12,
             "pair: " + " / ".join(PAIR_NAMES.values()),
             fontsize=8, ha="left")

    fig.suptitle(f"Δ=IRL−RF 大の dir 16 セル分布 (n_pair≥{ROBUST_N} で絞り込み)",
                 fontsize=12, y=0.99)
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
    full = pred.merge(pa, on=["developer", "directory"], how="left")
    df_arch = full.dropna(subset=["archetype", "pair_archetype"]).copy()

    # dir 別 AUC (全ペアで計算)
    rows = []
    for d in full["directory"].unique():
        s_all = full[full["directory"] == d]
        s_arch = df_arch[df_arch["directory"] == d]
        if len(s_all) < ROBUST_N:
            continue
        irl, rf = pool_aucs(s_all)
        if irl is None or rf is None:
            continue
        y = []
        for h in HORIZONS:
            mask = ~s_all[f"label_{h}"].isna()
            y.extend(s_all.loc[mask, f"label_{h}"].astype(int).tolist())
        rows.append({
            "directory": d,
            "n_pair": len(s_all),
            "n_pair_arch": len(s_arch),
            "base_rate": float(np.mean(y)) if y else np.nan,
            "irl_auc": irl,
            "rf_auc": rf,
            "delta": irl - rf,
        })
    dir_df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    print(f"対象 dir (n_pair>={ROBUST_N}): {len(dir_df)}")

    picks_irl = dir_df.head(N_PICK)
    picks_rf = dir_df.tail(N_PICK).iloc[::-1]
    print("\n=== IRL 優位 Top {} ===".format(N_PICK))
    print(picks_irl.to_string(index=False))
    print("\n=== RF 優位 Top {} ===".format(N_PICK))
    print(picks_rf.to_string(index=False))

    shares_irl = dir_cell_shares(df_arch, picks_irl["directory"].tolist())
    shares_rf = dir_cell_shares(df_arch, picks_rf["directory"].tolist())

    plot_stacked(picks_irl, picks_rf, shares_irl, shares_rf,
                 OUT / "fig_dir_delta_archetype16_stacked.pdf")


if __name__ == "__main__":
    main()
