# -*- coding: utf-8 -*-
"""4予測区間 (0-3, 3-6, 6-9, 9-12か月) ごとに RF を学習し特徴量重要度を比較.

短期と長期で「効く特徴量」がどう変わるかを観察。IRL は事前計算済み importance を
持たないため，本スクリプトは RF gini importance に絞る (IRL gradient importance は
今後別途)。

入力:
  data/pair_trajectories.csv (predictions)
  data/feature_values.csv (state features at t0)

出力:
  results/horizon_importance.csv: horizon × feature × gini
  results/fig_horizon_importance.pdf: ホライズン別 top-N 重要度ヒートマップ
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
SNAPSHOT_T0 = "2022-01-01"

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

# 日本語ラベル
FEATURE_JA = {
    "window_tenure_days": "在籍日数",
    "total_changes": "総changes数",
    "total_reviews": "総レビュー数",
    "recent_activity_frequency": "直近活動頻度",
    "avg_activity_gap": "平均活動間隔",
    "activity_trend": "活動トレンド",
    "unique_collaborator_count": "協働者数",
    "overall_acceptance_rate": "全体承諾率",
    "recent_acceptance_rate": "直近承諾率",
    "recent_load_ratio_30d_all": "30日負荷比",
    "days_since_last_activity": "最終活動からの日数",
    "acceptance_trend": "承諾トレンド",
    "reciprocity_score": "相互依頼性",
    "recent_load_ratio_7d_30d": "7日負荷比",
    "core_reviewer_ratio": "コアレビュー率",
    "recent_rejection_streak": "直近拒否連続",
    "acceptance_rate_last10": "直近10件承諾率",
    "active_months_ratio": "活動月比",
    "response_time_trend": "応答時間トレンド",
    "complex_pr_bias": "複雑PR傾向",
    "avg_action_intensity": "平均行動強度",
    "avg_change_lines": "平均変更行数",
    "avg_response_time": "平均応答時間",
    "avg_review_size": "平均レビュー規模",
    "repeat_collaboration_rate": "リピート協働率",
}


def main() -> None:
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    feat = pd.read_csv(DATA / "feature_values.csv")
    feat_t0 = feat[feat["__snapshot"] == SNAPSHOT_T0].copy()

    merged = pred.merge(feat_t0, left_on="developer", right_on="__email",
                        how="inner")
    print(f"merged shape: {merged.shape}")

    importance_rows = []
    for h in HORIZONS:
        sub = merged[~merged[f"label_{h}"].isna()].copy()
        X = sub[STATE_FEATURES].fillna(0.0).to_numpy()
        y = sub[f"label_{h}"].astype(int).to_numpy()
        n_pos, n_neg = (y == 1).sum(), (y == 0).sum()
        print(f"horizon {h}: n={len(y)} (pos={n_pos}, neg={n_neg})")
        if n_pos < 10 or n_neg < 10:
            print("  skipped (class imbalance)")
            continue
        # 論文のRF設定 (paper §3.3): n_estimators=500, max_depth=12
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=12, random_state=777, n_jobs=-1
        )
        rf.fit(X, y)
        for f_, imp in zip(STATE_FEATURES, rf.feature_importances_):
            importance_rows.append({
                "horizon": h,
                "feature": f_,
                "feature_ja": FEATURE_JA.get(f_, f_),
                "gini_importance": imp,
            })

    imp_df = pd.DataFrame(importance_rows)
    imp_df.to_csv(OUT / "horizon_importance.csv", index=False)
    print(f"saved: {OUT/'horizon_importance.csv'}")

    # ヒートマップ: 上位 10 特徴量 (全ホライズン平均 importance で選択)
    pivot = imp_df.pivot(index="feature_ja", columns="horizon",
                        values="gini_importance")
    pivot["mean"] = pivot.mean(axis=1)
    top = pivot.nlargest(12, "mean").drop(columns="mean")
    top = top[HORIZONS]  # 順序固定

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(top.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS, fontsize=12)
    ax.set_yticks(range(len(top.index)))
    ax.set_yticklabels(top.index, fontsize=12)
    for i, feat in enumerate(top.index):
        for j, h in enumerate(HORIZONS):
            v = top.loc[feat, h]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if v > top.values.max() * 0.5 else "black")
    ax.set_xlabel("予測区間", fontsize=12)
    ax.set_title("予測区間別 RF特徴量重要度 (Gini, 上位12)", fontsize=13)
    fig.colorbar(im, ax=ax, label="importance")
    fig.tight_layout()
    fig.savefig(OUT / "fig_horizon_importance.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_horizon_importance.pdf'}")

    # 短期 vs 長期の差分
    pivot_diff = pivot[HORIZONS].copy()
    pivot_diff["short_long_diff"] = (
        pivot_diff["0-3m"] - pivot_diff["9-12m"]
    )
    top_short = pivot_diff.nlargest(5, "short_long_diff")[
        ["0-3m", "9-12m", "short_long_diff"]
    ]
    top_long = pivot_diff.nsmallest(5, "short_long_diff")[
        ["0-3m", "9-12m", "short_long_diff"]
    ]
    print("\n=== 短期 (0-3m) で特に効く特徴量 (vs 9-12m) ===")
    print(top_short)
    print("\n=== 長期 (9-12m) で特に効く特徴量 (vs 0-3m) ===")
    print(top_long)


if __name__ == "__main__":
    main()
