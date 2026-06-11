# -*- coding: utf-8 -*-
"""時系列の豊かさ (= dev の活動量・期間分布) と IRL の RF 比較優位の関係を検証。

仮説 (B-29 の延長):
    IRL は時系列を学習するため、月次状態系列がリッチな dev で RF より有利。
    逆に時系列がスパース (低活動 / 短期間) な dev では RF が同等 or 有利。

検証:
    各 dev について 2 つの richness 指標を計算:
      - total_reviews_window: feature window (180日) 内のレビュー件数
      - active_months_ratio: 活動月数 / 全月数 (既存特徴量)
    IRL−RF AUC 差 (per-dev) を richness に対して scatter + 回帰 + 区間平均。
    正相関なら仮説支持。

入力:
    outputs/analysis/archetype_irl_rf_deep/k4_dev_alpha_with_metrics.csv
    data/combined_raw_main32.csv

出力:
    outputs/analysis/richness_vs_irl_advantage/
        per_dev_richness.csv
        binned_summary.csv
        fig_richness_scatter.pdf
        fig_richness_binned.pdf

使い方:
    uv run python scripts/analyze/study/timeseries_richness_vs_irl_advantage.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
from review_predictor.IRL.features.common_features import (  # noqa: E402
    extract_common_features,
    normalize_features,
)


def compute_richness(df_raw: pd.DataFrame, devs: list, feat_start: pd.Timestamp,
                      feat_end: pd.Timestamp) -> pd.DataFrame:
    """dev × richness 指標。"""
    rows = []
    for email in devs:
        rev_mask = (
            (df_raw["email"] == email)
            & (df_raw["timestamp"] >= feat_start)
            & (df_raw["timestamp"] < feat_end)
        )
        dev_data = df_raw[rev_mask]
        total_reviews_window = int(len(dev_data))
        # active_months_ratio
        if total_reviews_window > 0:
            months = dev_data["timestamp"].dt.to_period("M").nunique()
            total_months = max(int((feat_end - feat_start).days / 30), 1)
            active_months_ratio = min(months / total_months, 1.0)
        else:
            active_months_ratio = 0.0
        rows.append({
            "developer": email,
            "total_reviews_window": total_reviews_window,
            "active_months_ratio": active_months_ratio,
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_irl_rf_deep/k4_dev_alpha_with_metrics.csv"))
    p.add_argument("--raw-csv", type=pathlib.Path,
                   default=pathlib.Path("data/combined_raw_main32.csv"))
    p.add_argument("--out-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/richness_vs_irl_advantage"))
    p.add_argument("--train-end", default="2022-01-01")
    p.add_argument("--feature-window-days", type=int, default=180)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feat_end = pd.Timestamp(args.train_end)
    feat_start = feat_end - pd.Timedelta(days=args.feature_window_days)

    print(f"[load] {args.alpha_csv}")
    alpha = pd.read_csv(args.alpha_csv, index_col=0)
    print(f"  devs: {len(alpha)}")
    print(f"[load] raw csv {args.raw_csv}")
    df_raw = pd.read_csv(args.raw_csv, parse_dates=["timestamp"])

    print(f"[richness] computing for {len(alpha)} devs ...")
    richness = compute_richness(df_raw, list(alpha.index), feat_start, feat_end)
    merged = alpha.reset_index().rename(columns={"index": "developer"}).merge(
        richness, on="developer", how="inner"
    )

    # IRL−RF AUC diff (per-dev, NaN if any AUC undefined)
    merged["auc_diff_cold"] = merged["auc_cold_irl"] - merged["auc_rf_dir"]
    merged["auc_diff_warm"] = merged["auc_warm_irl"] - merged["auc_rf_dir"]
    merged["acc_diff_cold"] = merged["acc_cold_irl"] - merged["acc_rf_dir"]
    merged["acc_diff_warm"] = merged["acc_warm_irl"] - merged["acc_rf_dir"]

    merged.to_csv(args.out_dir / "per_dev_richness.csv", index=False)
    print(f"\n[head] per_dev_richness.csv:")
    print(merged[["developer", "dominant", "total_reviews_window", "active_months_ratio",
                    "auc_cold_irl", "auc_rf_dir", "auc_diff_cold"]].head(10).round(3).to_string(index=False))

    # ── 相関 ──
    print("\n[corr] richness vs IRL-RF AUC diff (AUC 定義可 dev only)")
    sub_auc = merged.dropna(subset=["auc_diff_cold"])
    print(f"  n_with_auc = {len(sub_auc)}")
    for rich_col in ["total_reviews_window", "active_months_ratio"]:
        for diff_col, label in [("auc_diff_cold", "AUC diff (cold IRL − RF)"),
                                  ("acc_diff_cold", "acc diff (cold IRL − RF)")]:
            d = sub_auc if "auc" in diff_col else merged
            r_p, p_p = pearsonr(d[rich_col], d[diff_col])
            r_s, p_s = spearmanr(d[rich_col], d[diff_col])
            print(f"  {rich_col:25s} vs {label}: "
                   f"Pearson r={r_p:+.3f} (p={p_p:.3g}), "
                   f"Spearman ρ={r_s:+.3f} (p={p_s:.3g}), n={len(d)}")

    # ── binned 平均 ──
    # total_reviews_window でビン分け
    bins = [0, 5, 15, 50, 200, 10000]
    labels = ["0-5", "6-15", "16-50", "51-200", "201+"]
    merged["reviews_bin"] = pd.cut(merged["total_reviews_window"], bins=bins, labels=labels,
                                     include_lowest=True)
    binned_rows = []
    for b in labels:
        sub = merged[merged["reviews_bin"] == b]
        if len(sub) == 0:
            continue
        for metric in ["auc_diff_cold", "auc_diff_warm", "acc_diff_cold", "acc_diff_warm",
                        "auc_cold_irl", "auc_rf_dir", "auc_warm_irl"]:
            vals = sub[metric].dropna()
            binned_rows.append({
                "reviews_bin": b, "n": len(sub), "metric": metric,
                "n_with_metric": len(vals),
                "mean": float(vals.mean()) if len(vals) else np.nan,
                "median": float(vals.median()) if len(vals) else np.nan,
            })
    binned = pd.DataFrame(binned_rows)
    binned.to_csv(args.out_dir / "binned_summary.csv", index=False)

    # 整形して表示
    print("\n[binned] mean diff by review-count bin")
    pivot = binned.pivot(index="reviews_bin", columns="metric", values="mean").reindex(labels)
    print(pivot.round(3).to_string())

    # ── 可視化 ──
    # Scatter: richness vs AUC diff
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, rich_col, xlim in [(axes[0], "total_reviews_window", (0, 200)),
                                 (axes[1], "active_months_ratio", (0, 1.05))]:
        sub = merged.dropna(subset=["auc_diff_cold"])
        ax.scatter(sub[rich_col], sub["auc_diff_cold"], s=24, alpha=0.6,
                    edgecolor="k", linewidth=0.3,
                    c=sub["dominant"].map({"A1": "#1f77b4", "A2": "#d62728",
                                              "A3": "#2ca02c", "A4": "#ff7f0e"}))
        # regression
        if rich_col == "total_reviews_window":
            x = np.clip(sub[rich_col].to_numpy(), 0, 200)
        else:
            x = sub[rich_col].to_numpy()
        y = sub["auc_diff_cold"].to_numpy()
        if len(x) >= 5:
            coef = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, np.polyval(coef, xs), "k-", linewidth=1, alpha=0.7,
                    label=f"y = {coef[0]:+.5f}x + {coef[1]:+.3f}")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
        ax.set_xlim(*xlim)
        ax.set_xlabel(rich_col)
        ax.set_ylabel("AUC diff (Cold IRL − RF_Dir)")
        ax.set_title(f"richness × IRL 優位 (n={len(sub)} dev with AUC)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_richness_scatter.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Binned bar
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.35
    auc_means = pivot["auc_diff_cold"].values
    acc_means = pivot["acc_diff_cold"].values
    ax.bar(x - width/2, auc_means, width, label="AUC diff (Cold IRL − RF)",
            color="#1f77b4", alpha=0.85)
    ax.bar(x + width/2, acc_means, width, label="Acc diff (Cold IRL − RF)",
            color="#ff7f0e", alpha=0.85)
    counts = [int(binned[binned["reviews_bin"] == b]["n"].iloc[0]) if len(binned[binned["reviews_bin"] == b]) > 0 else 0
              for b in labels]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={c})" for b, c in zip(labels, counts)])
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("レビュー件数ビン (180日窓内)")
    ax.set_ylabel("Cold IRL − RF_Dir 差")
    ax.set_title("時系列豊かさ別の IRL 優位 (区間平均)\n+ → IRL 強、− → RF 強")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_richness_binned.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
