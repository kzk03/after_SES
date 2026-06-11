"""
§10: IRL 残差の説明

教授の問い (2)「周期的な揺らぎ反証 → では IRL は何を捉えているのか」と (4)「IRL 固有に当てる
レビュアー像」に答えるため、IRL − RF の予測差 (= IRL 固有の信号) を state 特徴量で回帰し
「どの特徴量が IRL の上乗せを説明するか」を特定する。

被説明変数:
  resid_h = irl_dir_prob − rf_dir_prob (各予測区間)
  resid_mean = 4 予測区間の平均残差 (ペアごと、より安定)

説明変数: 25 state 特徴量 (t0=2022-01-01 時点)

手法:
  - Lasso (L1 正則化) で疎な係数を得る
  - RandomForestRegressor で Gini importance を得る (非線形補完)
  - 重要な上位 5 特徴量について partial 依存を描く

出力:
  - results/irl_residual_coef.csv: Lasso 係数 + RF importance
  - results/fig_irl_residual_explain.pdf: 係数の棒グラフ + 上位特徴量の partial 依存
"""

from __future__ import annotations

from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
SNAPSHOT_T0 = "2022-01-01"
SEED = 777

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

LABEL_MAP = {
    "window_tenure_days": "在籍日数",
    "total_changes": "総PR数",
    "total_reviews": "総レビュー数",
    "recent_activity_frequency": "直近活動頻度",
    "avg_activity_gap": "平均活動間隔",
    "activity_trend": "活動トレンド",
    "unique_collaborator_count": "独立協働者数",
    "overall_acceptance_rate": "全期承諾率",
    "recent_acceptance_rate": "直近承諾率",
    "recent_load_ratio_30d_all": "直近負荷比 30d",
    "days_since_last_activity": "最終活動から日数",
    "acceptance_trend": "承諾トレンド",
    "reciprocity_score": "相互依頼性",
    "recent_load_ratio_7d_30d": "直近負荷比 7/30d",
    "core_reviewer_ratio": "コアレビュー率",
    "recent_rejection_streak": "直近拒否連続",
    "acceptance_rate_last10": "直近10件承諾率",
    "active_months_ratio": "活動月比",
    "response_time_trend": "応答時間トレンド",
    "complex_pr_bias": "複雑PRバイアス",
    "avg_action_intensity": "平均行動強度",
    "avg_change_lines": "平均変更行数",
    "avg_response_time": "平均応答時間",
    "avg_review_size": "平均レビュー規模",
    "repeat_collaboration_rate": "反復協働率",
}


def load_data() -> pd.DataFrame:
    pt = pd.read_csv(DATA / "pair_trajectories.csv")
    fv = pd.read_csv(DATA / "feature_values.csv")
    fv = fv[fv["__snapshot"] == SNAPSHOT_T0].copy()
    fv = fv.drop_duplicates(subset=["__email"], keep="last")
    fv = fv.rename(columns={"__email": "developer"})
    merged = pt.merge(fv[["developer"] + STATE_FEATURES], on="developer", how="inner")
    print(f"pair_trajectories n={len(pt)}, merged with features n={len(merged)}")
    return merged


def main() -> None:
    df = load_data()

    # compute residual = irl - rf per horizon, then mean
    for h in HORIZONS:
        df[f"resid_{h}"] = df[f"irl_{h}"] - df[f"rf_{h}"]
    df["resid_mean"] = df[[f"resid_{h}" for h in HORIZONS]].mean(axis=1)

    print(f"resid_mean: mean={df['resid_mean'].mean():+.4f}  std={df['resid_mean'].std():.4f}")
    print(f"resid_mean: 25th={df['resid_mean'].quantile(0.25):+.4f}  75th={df['resid_mean'].quantile(0.75):+.4f}")

    X = df[STATE_FEATURES].fillna(0).values
    y = df["resid_mean"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---- Lasso ----
    lasso = LassoCV(cv=5, random_state=SEED, max_iter=5000, n_alphas=50).fit(Xs, y)
    print(f"Lasso: alpha={lasso.alpha_:.5f}  R²={lasso.score(Xs, y):.4f}")

    # ---- RandomForest ----
    rf = RandomForestRegressor(n_estimators=500, max_depth=8, random_state=SEED, n_jobs=-1).fit(X, y)
    rf_r2 = rf.score(X, y)
    print(f"RF: R²={rf_r2:.4f} (in-sample)")

    coef_df = pd.DataFrame({
        "feature": STATE_FEATURES,
        "label_jp": [LABEL_MAP.get(c, c) for c in STATE_FEATURES],
        "lasso_coef": lasso.coef_,
        "rf_importance": rf.feature_importances_,
    })
    coef_df["abs_lasso"] = coef_df["lasso_coef"].abs()
    coef_df = coef_df.sort_values("abs_lasso", ascending=False).reset_index(drop=True)
    coef_df.to_csv(RESULTS / "irl_residual_coef.csv", index=False)
    print(f"-> {RESULTS / 'irl_residual_coef.csv'}")
    print("\nTop 10 by |Lasso coef|:")
    print(coef_df.head(10).to_string(index=False))

    # ---- per-horizon Lasso (to see if signal differs by horizon) ----
    horizon_rows = []
    for h in HORIZONS:
        y_h = df[f"resid_{h}"].values
        m_h = LassoCV(cv=5, random_state=SEED, max_iter=5000, n_alphas=50).fit(Xs, y_h)
        for f, c in zip(STATE_FEATURES, m_h.coef_):
            horizon_rows.append({"horizon": h, "feature": f, "label_jp": LABEL_MAP.get(f, f),
                                 "lasso_coef": c, "alpha": m_h.alpha_, "r2": m_h.score(Xs, y_h)})
    pd.DataFrame(horizon_rows).to_csv(RESULTS / "irl_residual_coef_per_horizon.csv", index=False)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # left: top 12 |lasso coef|
    ax = axes[0]
    top = coef_df.head(12).iloc[::-1]
    colors = ["#1f77b4" if c > 0 else "#d62728" for c in top["lasso_coef"]]
    ax.barh(np.arange(len(top)), top["lasso_coef"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["label_jp"])
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Lasso 係数 (標準化後)")
    ax.set_title(f"IRL − RF 残差を説明する特徴量 Top 12 (Lasso R²={lasso.score(Xs, y):.3f})")
    ax.grid(True, axis="x", alpha=0.3)
    for yi, (val, p) in enumerate(zip(top["lasso_coef"], top["rf_importance"])):
        ax.text(val + (0.0008 * (1 if val >= 0 else -1)), yi,
                f"RF imp={p:.3f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8, color="gray")

    # right: per-horizon heatmap of lasso coefs for top features (top 10 by |mean coef|)
    ax = axes[1]
    ph = pd.read_csv(RESULTS / "irl_residual_coef_per_horizon.csv")
    pivot = ph.pivot(index="feature", columns="horizon", values="lasso_coef")
    pivot["abs_mean"] = pivot.abs().mean(axis=1)
    pivot_sorted = pivot.sort_values("abs_mean", ascending=False).head(10).drop(columns=["abs_mean"])
    pivot_sorted = pivot_sorted[HORIZONS]
    pivot_sorted.index = [LABEL_MAP.get(c, c) for c in pivot_sorted.index]

    vmax = float(np.abs(pivot_sorted.values).max())
    im = ax.imshow(pivot_sorted.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(HORIZONS)))
    ax.set_xticklabels([h.replace("m", "か月") for h in HORIZONS])
    ax.set_yticks(np.arange(len(pivot_sorted)))
    ax.set_yticklabels(pivot_sorted.index)
    ax.set_title("予測区間別 Lasso 係数 (top 10 特徴量)")
    for i in range(pivot_sorted.shape[0]):
        for j in range(pivot_sorted.shape[1]):
            v = pivot_sorted.values[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    color="white" if abs(v) > vmax * 0.5 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label("係数 (+ = IRL の上乗せを増やす)")

    fig.tight_layout()
    out_pdf = RESULTS / "fig_irl_residual_explain.pdf"
    out_png = RESULTS / "fig_irl_residual_explain.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"-> {out_pdf}")
    print(f"-> {out_png}")


if __name__ == "__main__":
    main()
