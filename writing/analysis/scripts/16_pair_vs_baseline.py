# -*- coding: utf-8 -*-
"""B-1: dev-only / dir-only / pair baseline の AUC 比較.

目的:
  (dev, dir) ペア予測の純粋な利得を定量化。
  以下 3 種の予測を同じサンプル (= 全ペア) で評価する:
    - pair    : 既存の (dev, dir) ペア予測 (irl_<h>, rf_<h>, ens)
    - dev-only: 当該 dev の全ペア平均 prob を予測値とする (dir 情報を平均化)
    - dir-only: 当該 dir の全ペア平均 prob を予測値とする (dev 情報を平均化)
  ラベルは label_<h> (= 当該 (dev, dir) の正例フラグ) で同じ。

仮説:
  pair > dev-only かつ pair > dir-only であれば「ペア化することで dev 平均・dir 平均では
  得られない解像度が生まれている」= ペア予測の純粋な利得。

モデル:
  - cold IRL (irl_<h>)
  - RF (rf_<h>)
  - アンサンブル ((irl + rf) / 2)
  ※ キャリブレーション後・global head・warm-start は一切使わない (RULES.md 準拠)

入力:
  data/pair_trajectories.csv
  outputs/analysis/bot_detection/bot_candidates_main32.csv

出力:
  results/baseline_comparison.csv
  results/fig_baseline_comparison.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
REPO_ROOT = ROOT.parents[1]
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
MODELS = ["irl", "rf", "ens"]
MODEL_LABELS = {"irl": "cold IRL", "rf": "RF", "ens": "アンサンブル"}


def safe_metric(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def add_ens(pair: pd.DataFrame) -> pd.DataFrame:
    pair = pair.copy()
    for h in HORIZONS:
        pair[f"ens_{h}"] = (pair[f"irl_{h}"] + pair[f"rf_{h}"]) / 2
    return pair


def aggregate_predictions(pair: pd.DataFrame, h: str, model: str,
                          group_col: str) -> np.ndarray:
    """group_col (developer or directory) の平均 prob をペア毎に展開."""
    col = f"{model}_{h}"
    return pair.groupby(group_col)[col].transform("mean").values


def compute_auc_table(pair: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    n_dev = pair["developer"].nunique()
    n_dir = pair["directory"].nunique()
    n_pair = len(pair)
    print(f"\n========== {label} (n_dev={n_dev:,}, n_dir={n_dir:,}, "
          f"n_pair={n_pair:,}) ==========")
    for h in HORIZONS:
        y = pair[f"label_{h}"].values.astype(int)
        n_pos = int(y.sum())
        base = n_pos / len(y)
        for m in MODELS:
            pair_pred = pair[f"{m}_{h}"].values
            dev_pred = aggregate_predictions(pair, h, m, "developer")
            dir_pred = aggregate_predictions(pair, h, m, "directory")

            auc_p, ap_p = safe_metric(y, pair_pred)
            auc_d, ap_d = safe_metric(y, dev_pred)
            auc_r, ap_r = safe_metric(y, dir_pred)

            rows.append({
                "bot_filter": label,
                "horizon": h,
                "model": m,
                "model_label": MODEL_LABELS[m],
                "n_pair": n_pair,
                "n_pos": n_pos,
                "base_rate": base,
                "auc_pair": auc_p,
                "auc_dev_only": auc_d,
                "auc_dir_only": auc_r,
                "ap_pair": ap_p,
                "ap_dev_only": ap_d,
                "ap_dir_only": ap_r,
                "lift_pair_vs_dev": auc_p - auc_d,
                "lift_pair_vs_dir": auc_p - auc_r,
            })
        # 表示
        df_h = pd.DataFrame([r for r in rows if r["horizon"] == h
                             and r["bot_filter"] == label])
        print(f"\n[{h}] base_rate={base:.3f}")
        print(df_h[["model_label", "auc_pair", "auc_dev_only", "auc_dir_only",
                    "lift_pair_vs_dev", "lift_pair_vs_dir"]]
              .round(4).to_string(index=False))
    return pd.DataFrame(rows)


def plot_figure(df_with: pd.DataFrame, df_without: pd.DataFrame,
                out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    def draw(ax, df, model, title):
        sub = df[df["model"] == model].set_index("horizon")
        sub = sub.reindex(HORIZONS)
        x = np.arange(len(HORIZONS))
        width = 0.27
        ax.bar(x - width, sub["auc_dev_only"], width, label="dev-only",
               color="#888888", alpha=0.85)
        ax.bar(x, sub["auc_dir_only"], width, label="dir-only",
               color="#aaccee", alpha=0.85)
        ax.bar(x + width, sub["auc_pair"], width, label="pair",
               color="#d62728", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(HORIZONS, fontsize=10)
        ax.set_ylim(0.45, 0.92)
        ax.set_ylabel("AUC-ROC")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        # 各バーに値を表示
        for i, h in enumerate(HORIZONS):
            for j, col in enumerate(["auc_dev_only", "auc_dir_only", "auc_pair"]):
                v = sub[col].iloc[i]
                ax.text(x[i] + (j - 1) * width, v + 0.005,
                        f"{v:.3f}", ha="center", fontsize=7.5)

    for j, m in enumerate(MODELS):
        draw(axes[0, j], df_with, m,
             f"({chr(97 + j)}) {MODEL_LABELS[m]}, bot 込み")
        draw(axes[1, j], df_without, m,
             f"({chr(100 + j)}) {MODEL_LABELS[m]}, bot 除外")

    fig.suptitle("dev-only / dir-only / pair baseline AUC 比較 "
                 "(ペア化の純粋な利得)", fontsize=12, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {out_path}")


def main() -> None:
    print("loading data...")
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    pair = add_ens(pair)

    bot_df = pd.read_csv(BOT_CSV)
    bots = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712

    df_with = compute_auc_table(pair, "with_bots")
    pair_nb = pair[~pair["developer"].isin(bots)].copy()
    df_without = compute_auc_table(pair_nb, "without_bots")

    summary = pd.concat([df_with, df_without], ignore_index=True)
    OUT.mkdir(exist_ok=True)
    out_csv = OUT / "baseline_comparison.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")

    # bot 影響
    print("\n========== bot 除外影響 (|delta| of lift_pair_vs_dev) ==========")
    merged = df_with.merge(df_without, on=["horizon", "model"],
                           suffixes=("_w", "_wo"))
    merged["delta"] = (merged["lift_pair_vs_dev_w"]
                       - merged["lift_pair_vs_dev_wo"]).abs()
    print(f"|delta| max: {merged['delta'].max():.4f}, "
          f"mean: {merged['delta'].mean():.4f}")

    plot_figure(df_with, df_without, OUT / "fig_baseline_comparison.pdf")


if __name__ == "__main__":
    main()
