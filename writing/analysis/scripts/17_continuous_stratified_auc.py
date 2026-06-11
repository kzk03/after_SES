# -*- coding: utf-8 -*-
"""A-1: 開発者属性の連続軸ストラタ別 per-horizon AUC.

アーキタイプ離散 (A1-A4 / P1-P4) に依存せず、以下 3 軸の連続値で開発者を 5 階層に分け、
IRL/RF/Ens の AUC 差を可視化する:
  - historical_tenure_days     : Gerrit 全体での在籍日数
  - historical_review_count_all: Gerrit 全体での累積レビュー数 (log1p 正規化前)
  - historical_n_unique_repos  : 触れた unique repo 数

仮説:
  累積レビュー数 / 在籍日数の少ない層 (新参〜中堅) では IRL が RF を逆転、
  上位層では RF が圧勝する。連続軸で「どの量的特性が IRL/RF を分けるか」を直接見る。

モデル (RULES.md 準拠):
  cold IRL (irl_<h>), RF (rf_<h>), アンサンブル ((irl+rf)/2)

入力:
  data/pair_trajectories.csv (developer, directory, label_<h>, irl_<h>, rf_<h>)
  /Users/kazuki-h/Master_research/data/dev_history_features.csv (per-horizon snapshot)
  outputs/analysis/bot_detection/bot_candidates_main32.csv

出力:
  results/continuous_stratified_auc.csv
  results/fig_continuous_stratified_auc.pdf
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
REPO_ROOT = ROOT.parents[1]
DEV_HIST_CSV = REPO_ROOT / "data/dev_history_features.csv"
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
HORIZON_TO_FE = {
    "0-3m": "2022-01-01",
    "3-6m": "2022-04-01",
    "6-9m": "2022-07-01",
    "9-12m": "2022-10-01",
}
AXES = {
    "tenure_days": "historical_tenure_days",
    "review_count": "historical_review_count_all",
    "n_unique_repos": "historical_n_unique_repos",
}
AXIS_LABEL = {
    "tenure_days": "在籍日数",
    "review_count": "累積レビュー数",
    "n_unique_repos": "活動 repo 数",
}
N_STRATA = 5


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def add_ens(pair: pd.DataFrame) -> pd.DataFrame:
    pair = pair.copy()
    for h in HORIZONS:
        pair[f"ens_{h}"] = (pair[f"irl_{h}"] + pair[f"rf_{h}"]) / 2
    return pair


def stratify(values: pd.Series, n: int = N_STRATA) -> pd.Series:
    ranked = values.rank(method="first")
    return pd.qcut(ranked, n, labels=[f"Q{i+1}" for i in range(n)])


def compute(pair: pd.DataFrame, dev_hist: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    print(f"\n========== {label} (n_pair={len(pair):,}) ==========")
    for h in HORIZONS:
        fe = HORIZON_TO_FE[h]
        hist_h = dev_hist[dev_hist["feature_end"] == fe].set_index("developer")
        for axis_key, col in AXES.items():
            vals = hist_h[col]
            strat = stratify(vals)  # dev -> Q1..Q5
            pair_h = pair.copy()
            pair_h["stratum"] = pair_h["developer"].map(strat)
            pair_h = pair_h.dropna(subset=["stratum"])

            for q in [f"Q{i+1}" for i in range(N_STRATA)]:
                sub = pair_h[pair_h["stratum"] == q]
                if len(sub) == 0:
                    continue
                y = sub[f"label_{h}"].astype(int).values
                if len(np.unique(y)) < 2:
                    continue
                irl = safe_auc(y, sub[f"irl_{h}"].values)
                rf = safe_auc(y, sub[f"rf_{h}"].values)
                ens = safe_auc(y, sub[f"ens_{h}"].values)
                # stratum 内中央値 (列の元値)
                med = float(vals.reindex(sub["developer"].unique()).median())
                rows.append({
                    "bot_filter": label,
                    "axis": axis_key,
                    "axis_label": AXIS_LABEL[axis_key],
                    "stratum": q,
                    "horizon": h,
                    "n_pair": len(sub),
                    "n_dev": sub["developer"].nunique(),
                    "n_pos": int(y.sum()),
                    "base_rate": float(y.mean()),
                    "stratum_median": med,
                    "irl_auc": irl,
                    "rf_auc": rf,
                    "ens_auc": ens,
                    "irl_minus_rf": irl - rf,
                    "ens_minus_rf": ens - rf,
                })
    df = pd.DataFrame(rows)
    # 表示
    for axis_key in AXES:
        sub = df[df["axis"] == axis_key]
        if sub.empty:
            continue
        print(f"\n[{AXIS_LABEL[axis_key]}] irl − rf (horizon × Q)")
        pv = sub.pivot_table(index="stratum", columns="horizon",
                             values="irl_minus_rf", aggfunc="first")
        print(pv.round(4).to_string())
    return df


def plot_figure(df: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    def draw(ax, sub, axis_key, title):
        s = sub[sub["axis"] == axis_key].copy()
        pv = s.pivot_table(index="stratum", columns="horizon",
                           values="irl_minus_rf", aggfunc="first")
        pv = pv.reindex([f"Q{i+1}" for i in range(N_STRATA)])
        pv = pv[HORIZONS]
        x = np.arange(len(pv.index))
        w = 0.2
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
        for i, h in enumerate(HORIZONS):
            ax.bar(x + (i - 1.5) * w, pv[h].values, w,
                   color=colors[i], label=h, alpha=0.88)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{q}\n(少→多)" if i == 0 else q
                            for i, q in enumerate(pv.index)], fontsize=9)
        ax.set_ylim(-0.18, 0.18)
        ax.set_ylabel("IRL AUC − RF AUC")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=8, ncols=2)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    df_w = df[df["bot_filter"] == "with_bots"]
    df_wo = df[df["bot_filter"] == "without_bots"]

    for j, axis_key in enumerate(AXES.keys()):
        draw(axes[0, j], df_w, axis_key,
             f"({chr(97 + j)}) {AXIS_LABEL[axis_key]} 階層, bot 込み")
        draw(axes[1, j], df_wo, axis_key,
             f"({chr(100 + j)}) {AXIS_LABEL[axis_key]} 階層, bot 除外")

    fig.suptitle("開発者属性 3 軸 × 5 階層 × 4 horizon: IRL AUC − RF AUC",
                 fontsize=12.5, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    print("loading data...")
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    pair = add_ens(pair)
    dev_hist = pd.read_csv(DEV_HIST_CSV)
    bot_df = pd.read_csv(BOT_CSV)
    bots = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712

    df_w = compute(pair, dev_hist, "with_bots")
    pair_nb = pair[~pair["developer"].isin(bots)].copy()
    df_wo = compute(pair_nb, dev_hist, "without_bots")
    df = pd.concat([df_w, df_wo], ignore_index=True)

    OUT.mkdir(exist_ok=True)
    csv = OUT / "continuous_stratified_auc.csv"
    df.to_csv(csv, index=False)
    print(f"\nsaved: {csv}")

    # bot 影響
    print("\n========== bot 除外影響 (|delta irl_minus_rf|) ==========")
    m = df_w.merge(df_wo, on=["axis", "stratum", "horizon"],
                   suffixes=("_w", "_wo"))
    m["d"] = (m["irl_minus_rf_w"] - m["irl_minus_rf_wo"]).abs()
    print(f"|delta| max: {m['d'].max():.4f}, mean: {m['d'].mean():.4f}")

    plot_figure(df, OUT / "fig_continuous_stratified_auc.pdf")


if __name__ == "__main__":
    main()
