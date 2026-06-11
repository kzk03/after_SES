# -*- coding: utf-8 -*-
"""A-3: IRL gain を dev/dir 共変量で多変量回帰.

09_irl_residual_explain.py は state 25 次元のみで残差を説明していた。本スクリプトは
それを拡張し、以下 3 層の共変量を統合してクロスレベル回帰を行う:

  (i) dev 状態 (feature_values.csv @ 2022-01-01, 25 個)
  (ii) dev 累積履歴 (dev_history_features.csv @ 2022-01-01, 3 個)
  (iii) dir 特性 (pair_trajectories.csv から集計, 3 個)

被説明変数:
  irl_gain_mean = mean over 4 horizons of (irl_<h> - rf_<h>)
  per-horizon 版もファイルとして保存

モデル (RULES.md 準拠):
  cold IRL (irl_<h>), RF (rf_<h>) の差分。Ens は使わない (gain 定義の意味が変わるため)

入力:
  data/pair_trajectories.csv, data/feature_values.csv
  /Users/kazuki-h/Master_research/data/dev_history_features.csv
  outputs/analysis/bot_detection/bot_candidates_main32.csv

出力:
  results/irl_gain_multivar_coef.csv
  results/irl_gain_multivar_coef_per_horizon.csv
  results/fig_irl_gain_multivar.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
REPO_ROOT = ROOT.parents[1]
DEV_HIST_CSV = REPO_ROOT / "data/dev_history_features.csv"
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
SNAPSHOT = "2022-01-01"
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

HIST_FEATURES = [
    "historical_tenure_days", "historical_review_count_all",
    "historical_n_unique_repos",
]

DIR_FEATURES = ["dir_log_n_pair", "dir_accept_rate", "dir_dev_diversity"]

ALL_FEATURES = STATE_FEATURES + HIST_FEATURES + DIR_FEATURES

LABEL_MAP = {
    "window_tenure_days": "在籍日数 (期間内)",
    "total_changes": "総PR数 (期間内)",
    "total_reviews": "総レビュー数 (期間内)",
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
    "historical_tenure_days": "累積在籍日数",
    "historical_review_count_all": "累積レビュー数",
    "historical_n_unique_repos": "累積活動 repo 数",
    "dir_log_n_pair": "[dir] log(ペア数)",
    "dir_accept_rate": "[dir] 承諾率",
    "dir_dev_diversity": "[dir] 関与dev数",
}

GROUP_OF = {**{f: "dev_state" for f in STATE_FEATURES},
            **{f: "dev_history" for f in HIST_FEATURES},
            **{f: "dir" for f in DIR_FEATURES}}


def build_dir_features(pair: pd.DataFrame) -> pd.DataFrame:
    g = pair.groupby("directory")
    n_pair = g.size()
    accept_rate = pair.assign(
        any_label=pair[[f"label_{h}" for h in HORIZONS]].max(axis=1)
    ).groupby("directory")["any_label"].mean()
    n_dev = g["developer"].nunique()
    return pd.DataFrame({
        "dir_log_n_pair": np.log1p(n_pair),
        "dir_accept_rate": accept_rate,
        "dir_dev_diversity": np.log1p(n_dev),
    })


def load_data(label_filter: str, bots: set[str]) -> pd.DataFrame:
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    if label_filter == "without_bots":
        pair = pair[~pair["developer"].isin(bots)].copy()

    fv = pd.read_csv(DATA / "feature_values.csv")
    fv = fv[fv["__snapshot"] == SNAPSHOT].copy()
    fv = fv.drop_duplicates(subset=["__email"], keep="last").rename(
        columns={"__email": "developer"})
    fv = fv[["developer"] + STATE_FEATURES]

    hist = pd.read_csv(DEV_HIST_CSV)
    hist = hist[hist["feature_end"] == SNAPSHOT][["developer"] + HIST_FEATURES]

    dir_feat = build_dir_features(pair).reset_index()

    df = pair.merge(fv, on="developer", how="inner")
    df = df.merge(hist, on="developer", how="inner")
    df = df.merge(dir_feat, on="directory", how="inner")
    print(f"[{label_filter}] pair n={len(pair):,} -> merged n={len(df):,}")
    return df


def compute_y(df: pd.DataFrame) -> pd.DataFrame:
    for h in HORIZONS:
        df[f"resid_{h}"] = df[f"irl_{h}"] - df[f"rf_{h}"]
    df["resid_mean"] = df[[f"resid_{h}" for h in HORIZONS]].mean(axis=1)
    return df


def fit_models(df: pd.DataFrame):
    X = df[ALL_FEATURES].fillna(0).values
    y = df["resid_mean"].values
    Xs = StandardScaler().fit_transform(X)

    lasso = LassoCV(cv=5, random_state=SEED, max_iter=8000, n_alphas=60).fit(Xs, y)
    print(f"  Lasso alpha={lasso.alpha_:.5f}  R²={lasso.score(Xs, y):.4f}")

    rf = RandomForestRegressor(
        n_estimators=600, max_depth=10, random_state=SEED, n_jobs=-1
    ).fit(X, y)
    print(f"  RF R²={rf.score(X, y):.4f} (in-sample)")
    return lasso, rf, Xs


def coef_table(lasso, rf) -> pd.DataFrame:
    df = pd.DataFrame({
        "feature": ALL_FEATURES,
        "label_jp": [LABEL_MAP.get(c, c) for c in ALL_FEATURES],
        "group": [GROUP_OF[c] for c in ALL_FEATURES],
        "lasso_coef": lasso.coef_,
        "rf_importance": rf.feature_importances_,
    })
    df["abs_lasso"] = df["lasso_coef"].abs()
    return df.sort_values("abs_lasso", ascending=False).reset_index(drop=True)


def fit_per_horizon(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    X = df[ALL_FEATURES].fillna(0).values
    Xs = StandardScaler().fit_transform(X)
    for h in HORIZONS:
        y = df[f"resid_{h}"].values
        m = LassoCV(cv=5, random_state=SEED, max_iter=8000, n_alphas=60).fit(Xs, y)
        for f, c in zip(ALL_FEATURES, m.coef_):
            rows.append({
                "bot_filter": label, "horizon": h, "feature": f,
                "label_jp": LABEL_MAP.get(f, f), "group": GROUP_OF[f],
                "lasso_coef": c, "alpha": m.alpha_,
                "r2": m.score(Xs, y),
            })
    return pd.DataFrame(rows)


def plot_figure(coef_w: pd.DataFrame, coef_wo: pd.DataFrame,
                ph_w: pd.DataFrame, lasso_w_r2: float,
                rf_w_r2: float, out_path: pathlib.Path) -> None:
    fig = plt.figure(figsize=(15.5, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.15, 1])

    # (a) Lasso coef top 14 (bot 込み)
    ax = fig.add_subplot(gs[0, 0])
    top = coef_w.head(14).iloc[::-1]
    color_map = {"dev_state": "#1f77b4", "dev_history": "#2ca02c", "dir": "#d62728"}
    colors = [color_map[g] for g in top["group"]]
    ax.barh(np.arange(len(top)), top["lasso_coef"], color=colors,
            edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top["label_jp"], fontsize=9)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Lasso 係数 (標準化後)")
    ax.set_title(f"(a) IRL gain を説明する特徴量 Top 14 "
                 f"(Lasso R²={lasso_w_r2:.3f}, RF R²={rf_w_r2:.3f})",
                 fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    for yi, p in enumerate(top["rf_importance"]):
        sign = 1 if top["lasso_coef"].iloc[yi] >= 0 else -1
        ax.text(top["lasso_coef"].iloc[yi] + sign * 0.0008, yi,
                f"RF imp={p:.3f}", va="center",
                ha="left" if sign > 0 else "right", fontsize=7.5, color="gray")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax.legend(handles, ["dev 状態", "dev 累積履歴", "dir 特性"],
              loc="lower right", fontsize=8)

    # (b) Group share (group ごとの |coef| 比率)
    ax = fig.add_subplot(gs[0, 1])
    share = (coef_w.assign(abs_=coef_w["lasso_coef"].abs())
             .groupby("group")["abs_"].sum())
    share = share / share.sum()
    rf_share = coef_w.groupby("group")["rf_importance"].sum()
    rf_share = rf_share / rf_share.sum()
    groups = ["dev_state", "dev_history", "dir"]
    x = np.arange(len(groups))
    w = 0.38
    ax.bar(x - w / 2, [share[g] for g in groups], w, label="Lasso |係数| 比率",
           color="#888888", alpha=0.85)
    ax.bar(x + w / 2, [rf_share[g] for g in groups], w, label="RF importance 比率",
           color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["dev 状態", "dev 累積履歴", "dir 特性"])
    ax.set_ylabel("寄与シェア")
    ax.set_title("(b) 3 層別の説明寄与シェア", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for xi, g in enumerate(groups):
        ax.text(xi - w / 2, share[g] + 0.01, f"{share[g]:.2%}",
                ha="center", fontsize=8)
        ax.text(xi + w / 2, rf_share[g] + 0.01, f"{rf_share[g]:.2%}",
                ha="center", fontsize=8)

    # (c) per-horizon ヒートマップ (top 10 特徴量)
    ax = fig.add_subplot(gs[1, 0])
    pv = ph_w.pivot_table(index="feature", columns="horizon",
                          values="lasso_coef", aggfunc="first")
    pv["abs_mean"] = pv.abs().mean(axis=1)
    pv = pv.sort_values("abs_mean", ascending=False).head(10)
    pv = pv[HORIZONS]
    pv.index = [LABEL_MAP.get(c, c) for c in pv.index]
    vmax = float(np.abs(pv.values).max())
    im = ax.imshow(pv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([h.replace("m", "か月") for h in HORIZONS])
    ax.set_yticks(range(len(pv)))
    ax.set_yticklabels(pv.index, fontsize=9)
    ax.set_title("(c) 予測区間別 Lasso 係数 (top 10)", fontsize=11)
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    color="white" if abs(v) > vmax * 0.55 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.045,
                 label="係数 (+ = IRL gain を増やす)")

    # (d) bot 込み vs 除外の係数散布図
    ax = fig.add_subplot(gs[1, 1])
    m = coef_w.merge(coef_wo, on="feature", suffixes=("_w", "_wo"))
    ax.scatter(m["lasso_coef_w"], m["lasso_coef_wo"],
               c=[color_map[g] for g in m["group_w"]],
               s=40, alpha=0.7, edgecolor="black", linewidth=0.4)
    lim = max(m["lasso_coef_w"].abs().max(), m["lasso_coef_wo"].abs().max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.7)
    ax.axhline(0, color="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.4)
    ax.set_xlabel("Lasso 係数 (bot 込み)")
    ax.set_ylabel("Lasso 係数 (bot 除外)")
    ax.set_title("(d) bot 除外感度 (各点 = 1 特徴量)", fontsize=11)
    ax.grid(alpha=0.3)
    # 主要特徴量にラベル
    for _, r in m.nlargest(4, columns="abs_lasso_w").iterrows():
        ax.annotate(LABEL_MAP.get(r["feature"], r["feature"]),
                    (r["lasso_coef_w"], r["lasso_coef_wo"]),
                    fontsize=7.5, alpha=0.8,
                    xytext=(4, 4), textcoords="offset points")

    fig.suptitle("IRL gain (= irl_dir_prob − rf_dir_prob) のクロスレベル多変量回帰",
                 fontsize=12.5, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    print("loading data...")
    bot_df = pd.read_csv(BOT_CSV)
    bots = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712

    print("\n========== bot 込み ==========")
    df_w = compute_y(load_data("with_bots", bots))
    lasso_w, rf_w, _Xs_w = fit_models(df_w)
    coef_w = coef_table(lasso_w, rf_w)
    coef_w["bot_filter"] = "with_bots"
    print("\nTop 10 by |Lasso coef| (bot 込み):")
    print(coef_w.head(10)[["label_jp", "group", "lasso_coef",
                           "rf_importance"]].to_string(index=False))

    print("\n========== bot 除外 ==========")
    df_wo = compute_y(load_data("without_bots", bots))
    lasso_wo, rf_wo, _Xs_wo = fit_models(df_wo)
    coef_wo = coef_table(lasso_wo, rf_wo)
    coef_wo["bot_filter"] = "without_bots"

    coef = pd.concat([coef_w, coef_wo], ignore_index=True)
    OUT.mkdir(exist_ok=True)
    coef.to_csv(OUT / "irl_gain_multivar_coef.csv", index=False)
    print(f"\nsaved: {OUT / 'irl_gain_multivar_coef.csv'}")

    ph_w = fit_per_horizon(df_w, "with_bots")
    ph_wo = fit_per_horizon(df_wo, "without_bots")
    pd.concat([ph_w, ph_wo], ignore_index=True).to_csv(
        OUT / "irl_gain_multivar_coef_per_horizon.csv", index=False)

    # bot 影響
    m = coef_w.merge(coef_wo, on="feature", suffixes=("_w", "_wo"))
    m["d"] = (m["lasso_coef_w"] - m["lasso_coef_wo"]).abs()
    print(f"\n[bot 影響] |coef delta| max={m['d'].max():.4f}, "
          f"mean={m['d'].mean():.4f}")

    plot_figure(coef_w, coef_wo, ph_w,
                lasso_w.score(StandardScaler().fit_transform(
                    df_w[ALL_FEATURES].fillna(0).values), df_w["resid_mean"]),
                rf_w.score(df_w[ALL_FEATURES].fillna(0).values,
                           df_w["resid_mean"]),
                OUT / "fig_irl_gain_multivar.pdf")


if __name__ == "__main__":
    main()
