# -*- coding: utf-8 -*-
"""volat150 (main32_mce_volat_ep150/lstm_baseline) の IRL と RF の
特徴量重要度を比較し、ユーザー仮説 (IRL は分散・活動間隔系、RF は累積・path 系)
を直接検証する.

入力:
  outputs/main32_mce_volat_ep150/lstm_baseline/train_<H>/eval_<H>/
      irl_feature_importance.json        (IRL gradient-based, |grad|^2 平均)
      irl_feature_importance_signed.json (IRL signed)
      rf_dir_feature_importance.json     (RF Gini)
  → 対角 4 セル (0-3m, 3-6m, 6-9m, 9-12m) を平均

出力:
  writing/analysis/results/feature_importance_irl_vs_rf.csv
  writing/analysis/results/fig_feature_importance_irl_vs_rf.pdf
  writing/analysis/results/fig_feature_importance_group_share.pdf
"""
from __future__ import annotations

import json
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
ANALYSIS_ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "main32_mce_volat_ep150" / "lstm_baseline"
OUT = ANALYSIS_ROOT / "results"
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]

# 特徴量の意味グループ分け (ユーザー仮説検証用)
GROUPS = {
    "path 親和度": [
        "path_review_count", "path_recency", "path_acceptance_rate",
        "path_owner_overlap", "path_hub_score", "path_neighbor_coverage",
    ],
    "累積カウント": [
        "total_changes", "total_reviews", "window_tenure_days",
        "unique_collaborator_count", "core_reviewer_ratio",
        "reviewer_owner_ratio", "n_projects", "cross_project_review_share",
    ],
    "承諾率": [
        "overall_acceptance_rate", "recent_acceptance_rate",
        "acceptance_rate_last10", "accept_rate_drift_30d",
    ],
    "活動間隔・時間": [
        "avg_activity_gap", "days_since_last_activity",
        "time_since_last_accept", "time_since_last_reject",
        "active_months_ratio",
    ],
    "変動性・トレンド": [
        "recent_switch_rate", "activity_trend",
        "response_time_trend", "recent_activity_frequency",
    ],
    "負荷": [
        "recent_load_ratio_30d_all", "recent_load_ratio_7d_30d",
        "concurrent_open_reviews_7d",
    ],
    "関係性": [
        "reciprocity_score", "same_domain_share", "repeat_collaboration_rate",
    ],
    "action": [
        "avg_action_intensity", "avg_change_lines", "avg_response_time",
        "avg_review_size",
    ],
}


def load_importance(model: str, fname: str) -> pd.DataFrame:
    """4 horizon の JSON を読み込み、列を horizon にしたデータフレームを返す."""
    rows: dict[str, dict[str, float]] = {}
    for h in HORIZONS:
        p = BASE / f"train_{h}" / f"eval_{h}" / fname
        if not p.exists():
            print(f"  欠損: {p}")
            continue
        with open(p) as f:
            d = json.load(f)
        for k, v in d.items():
            rows.setdefault(k, {})[h] = float(v)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).T
    df.columns = [f"{model}_{c}" for c in df.columns]
    df[f"{model}_mean"] = df[[c for c in df.columns if c.startswith(f"{model}_")]].mean(axis=1)
    return df


def normalize(s: pd.Series) -> pd.Series:
    """各モデル内で合計 1 に正規化."""
    total = s.sum()
    return s / total if total > 0 else s


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    print(f"BASE: {BASE}")

    irl = load_importance("irl", "irl_feature_importance.json")
    rf = load_importance("rf", "rf_dir_feature_importance.json")

    if irl.empty or rf.empty:
        print("=== どちらかの importance JSON がまだない ===")
        print(f"IRL rows: {len(irl)}, RF rows: {len(rf)}")
        return

    # IRL と RF の平均をマージ
    merged = pd.concat(
        [irl["irl_mean"], rf["rf_mean"]], axis=1
    ).rename(columns={"irl_mean": "irl_raw", "rf_mean": "rf_raw"})
    merged["irl_norm"] = normalize(merged["irl_raw"])
    merged["rf_norm"] = normalize(merged["rf_raw"])
    merged["irl_rank"] = merged["irl_norm"].rank(ascending=False).astype("Int64")
    merged["rf_rank"] = merged["rf_norm"].rank(ascending=False).astype("Int64")
    merged["rank_diff"] = merged["rf_rank"] - merged["irl_rank"]  # 正 = IRL が上位寄せ
    merged["share_diff"] = merged["irl_norm"] - merged["rf_norm"]  # 正 = IRL 寄り

    merged = merged.sort_values("share_diff", ascending=False)
    merged.to_csv(OUT / "feature_importance_irl_vs_rf.csv")
    print(f"saved: {OUT/'feature_importance_irl_vs_rf.csv'}")

    print("\n=== IRL が RF より重視する特徴量 上位 10 (share_diff 降順) ===")
    print(merged.head(10)[["irl_norm", "rf_norm", "irl_rank", "rf_rank", "share_diff"]]
          .to_string())
    print("\n=== RF が IRL より重視する特徴量 上位 10 (share_diff 昇順) ===")
    print(merged.tail(10).iloc[::-1][["irl_norm", "rf_norm", "irl_rank", "rf_rank", "share_diff"]]
          .to_string())

    # ── (1) 特徴量ごとの IRL / RF 重要度を 2 列棒グラフで並列表示 ──
    fig, ax = plt.subplots(figsize=(8.5, 12))
    y = np.arange(len(merged))
    ax.barh(y - 0.2, merged["rf_norm"], height=0.4, label="RF (Gini)",
            color="#D62728", alpha=0.85)
    ax.barh(y + 0.2, merged["irl_norm"], height=0.4, label="IRL (|grad|² 平均)",
            color="#1F77B4", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(merged.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("正規化重要度 (各モデルで合計 1)", fontsize=10)
    ax.set_title(
        "volat150 lstm_baseline: 特徴量重要度 IRL vs RF\n"
        "(対角 4 セル平均、share_diff 降順 = IRL 寄り上位)",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "fig_feature_importance_irl_vs_rf.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_feature_importance_irl_vs_rf.pdf'}")

    # ── (2) グループ別の集計 (ユーザー仮説の直接検証) ──
    grp_rows = []
    feature_to_group = {f: g for g, fs in GROUPS.items() for f in fs}
    for feat in merged.index:
        g = feature_to_group.get(feat, "その他")
        grp_rows.append({
            "feature": feat,
            "group": g,
            "irl_norm": merged.at[feat, "irl_norm"],
            "rf_norm": merged.at[feat, "rf_norm"],
        })
    grp_df = pd.DataFrame(grp_rows)
    grp_agg = grp_df.groupby("group").agg({"irl_norm": "sum", "rf_norm": "sum"})
    grp_agg["share_diff"] = grp_agg["irl_norm"] - grp_agg["rf_norm"]
    grp_agg = grp_agg.sort_values("share_diff", ascending=False)
    grp_agg.to_csv(OUT / "feature_importance_group_share.csv")
    print(f"\nsaved: {OUT/'feature_importance_group_share.csv'}")
    print("\n=== グループ別重要度合計 (share_diff 降順 = IRL 寄り) ===")
    print(grp_agg.round(4).to_string())

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(grp_agg))
    ax.bar(x - 0.2, grp_agg["rf_norm"], width=0.4, label="RF (Gini)",
           color="#D62728", alpha=0.85)
    ax.bar(x + 0.2, grp_agg["irl_norm"], width=0.4, label="IRL (|grad|²)",
           color="#1F77B4", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(grp_agg.index, fontsize=9, rotation=18, ha="right")
    ax.set_ylabel("グループ合計重要度", fontsize=10)
    ax.set_title(
        "特徴量グループ別 重要度シェア (volat150 lstm_baseline, 対角 4 セル平均)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # share_diff を上に注記
    for i, (_, r) in enumerate(grp_agg.iterrows()):
        ax.text(i, max(r["irl_norm"], r["rf_norm"]) + 0.005,
                f"Δ={r['share_diff']:+.3f}",
                ha="center", fontsize=8.5,
                color="green" if r["share_diff"] > 0 else "darkred")

    fig.tight_layout()
    fig.savefig(OUT / "fig_feature_importance_group_share.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_feature_importance_group_share.pdf'}")


if __name__ == "__main__":
    main()
