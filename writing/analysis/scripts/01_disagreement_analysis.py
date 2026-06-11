# -*- coding: utf-8 -*-
"""IRL正解∧RF誤り / RF正解∧IRL誤り のペアを抽出し state特徴量プロファイルを比較.

入力:
  data/pair_trajectories.csv: developer, directory, label_<h>, irl_<h>, rf_<h>, archetype
  data/feature_values.csv: 25 state features × (__email, __snapshot)

出力:
  results/disagreement_summary.csv: ホライズン×グループ×サンプル数
  results/disagreement_feature_means.csv: グループ別特徴量平均
  results/fig_disagreement_features.pdf: 上位差分特徴量の bar chart
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
SNAPSHOT_T0 = "2022-01-01"  # 主分析のスナップショット時刻

# 評価対象の state 特徴量 (25次元)
STATE_FEATURES = [
    "window_tenure_days", "total_changes", "total_reviews",
    "recent_activity_frequency", "avg_activity_gap", "activity_trend",
    "unique_collaborator_count", "overall_acceptance_rate",
    "recent_acceptance_rate", "recent_load_ratio_30d_all",
    "days_since_last_activity", "acceptance_trend", "reciprocity_score",
    "recent_load_ratio_7d_30d", "core_reviewer_ratio",
    "recent_rejection_streak", "acceptance_rate_last10",
    "active_months_ratio", "response_time_trend", "complex_pr_bias",
    "avg_action_intensity", "avg_change_lines", "avg_response_time",
    "avg_review_size", "repeat_collaboration_rate",
]


def classify(label: int, pred: float, thresh: float = 0.5) -> str:
    """label と pred 確率から TP/FP/TN/FN."""
    pred_cls = 1 if pred >= thresh else 0
    if label == 1 and pred_cls == 1:
        return "TP"
    if label == 0 and pred_cls == 1:
        return "FP"
    if label == 1 and pred_cls == 0:
        return "FN"
    return "TN"


def disagreement_group(label: int, irl: float, rf: float,
                       thresh: float = 0.5) -> str:
    """IRL/RF のどちらが当てたかでグループ分類."""
    irl_correct = (irl >= thresh) == (label == 1)
    rf_correct = (rf >= thresh) == (label == 1)
    if irl_correct and not rf_correct:
        return "IRL_only"
    if rf_correct and not irl_correct:
        return "RF_only"
    if irl_correct and rf_correct:
        return "both_correct"
    return "both_wrong"


def main() -> None:
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    feat = pd.read_csv(DATA / "feature_values.csv")
    feat_t0 = feat[feat["__snapshot"] == SNAPSHOT_T0].copy()
    print(f"feature_values rows at {SNAPSHOT_T0}: {len(feat_t0)} "
          f"(unique devs: {feat_t0['__email'].nunique()})")

    # email で join (pair_trajectories の developer = email)
    merged = pred.merge(feat_t0, left_on="developer", right_on="__email",
                        how="inner")
    print(f"merged shape: {merged.shape} (pairs with features: {len(merged)})")

    summary_rows = []
    means_rows = []

    for h in HORIZONS:
        sub = merged[~merged[f"label_{h}"].isna()].copy()
        if len(sub) == 0:
            continue
        sub["group"] = sub.apply(
            lambda r: disagreement_group(
                int(r[f"label_{h}"]), r[f"irl_{h}"], r[f"rf_{h}"]
            ),
            axis=1,
        )
        grp_counts = sub["group"].value_counts().to_dict()
        for g, n in grp_counts.items():
            summary_rows.append({"horizon": h, "group": g, "n": n})

        # グループ別特徴量平均
        means = sub.groupby("group")[STATE_FEATURES].mean()
        for grp_name in means.index:
            row = {"horizon": h, "group": grp_name}
            for f_ in STATE_FEATURES:
                row[f_] = means.loc[grp_name, f_]
            means_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    means_df = pd.DataFrame(means_rows)
    summary_df.to_csv(OUT / "disagreement_summary.csv", index=False)
    means_df.to_csv(OUT / "disagreement_feature_means.csv", index=False)
    print(f"saved: {OUT/'disagreement_summary.csv'}")
    print(f"saved: {OUT/'disagreement_feature_means.csv'}")

    # 全ホライズン平均で IRL_only vs RF_only の差分上位
    pooled = means_df.groupby("group")[STATE_FEATURES].mean()
    if "IRL_only" in pooled.index and "RF_only" in pooled.index:
        diff = pooled.loc["IRL_only"] - pooled.loc["RF_only"]
        diff_sorted = diff.reindex(diff.abs().sort_values(ascending=False).index)
        top = diff_sorted.head(15)

        fig, ax = plt.subplots(figsize=(10, 7))
        colors = ["#1f77b4" if v > 0 else "#d62728" for v in top.values]
        ax.barh(range(len(top)), top.values, color=colors)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=12)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("IRL_only − RF_only (feature mean diff)", fontsize=12)
        ax.set_title("IRL正解∧RF誤り vs RF正解∧IRL誤り の特徴量差 (4ホライズン平均)",
                     fontsize=13)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "fig_disagreement_features.pdf", format="pdf",
                    bbox_inches="tight")
        plt.close(fig)
        print(f"saved: {OUT/'fig_disagreement_features.pdf'}")

    # サマリ表示
    print("\n=== disagreement summary ===")
    pivot = summary_df.pivot_table(index="group", columns="horizon",
                                   values="n", fill_value=0)
    print(pivot)


if __name__ == "__main__":
    main()
