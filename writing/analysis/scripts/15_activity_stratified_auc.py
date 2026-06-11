# -*- coding: utf-8 -*-
"""主張 6 v2: 活動量階層 × per-horizon AUC で IRL vs RF を分析.

仮説:
  RF は累積活動量が多い dev を予測しやすい (= 累積指標が支配的)。
  IRL は中程度の活動量の dev で「直近活動の角度」「応答時間の変化」「ステップ間の意図」を
  系列学習から学んで予測精度を上げているはず。

評価方法:
  1. 各 dev の活動量を combined_raw_main32.csv の total_reviews_received で定義
     (各 dev が受けたレビュー依頼の総数、label とは独立)
  2. dev を活動量で 5 階層 (Q1=最少 〜 Q5=最多) に分け、各 (dev, dir) ペアに dev の階層を付与
  3. 各 horizon × 階層で per-horizon AUC を IRL/RF/Ens それぞれ計算
     (per-horizon AUC は同じモデルの出力内で比較するので 4 モデル bias は無関係)
  4. IRL − RF を階層別に並べ、「中程度の階層で IRL が RF を上回る」仮説を検証
  5. 直近活動の角度 (activity_trend) も同様に階層化して別軸で評価

bot 除外あり / なしを両方計算。差が小さければ bot こみを採用 (ユーザー方針)。

入力:
  data/pair_trajectories.csv
  data/feature_values.csv
  data/combined_raw_main32.csv
  outputs/analysis/bot_detection/bot_candidates_main32.csv

出力:
  results/activity_stratified_auc.csv
  results/fig_activity_stratified_auc.pdf
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
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
N_STRATA = 5
STRATUM_LABELS = ["Q1 (最少)", "Q2", "Q3 (中程度)", "Q4", "Q5 (最多)"]


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan, len(y), int(y.sum())
    return float(roc_auc_score(y, s)), len(y), int(y.sum())


def dev_activity_level(combined: pd.DataFrame) -> pd.Series:
    """dev ごとの total review requests received (活動量)."""
    return combined.groupby("email").size().rename("total_requests")


def dev_recent_trend(features: pd.DataFrame) -> pd.Series:
    """dev ごとの activity_trend の平均 (近時活動の角度)."""
    return features.groupby("__email")["activity_trend"].mean().rename("mean_activity_trend")


def stratify(series: pd.Series, n: int = N_STRATA) -> pd.Series:
    """値を n 階層に分ける (qcut, 重複は drop してラベル数を bin 数に合わせる)."""
    # rank で同値タイを散らしてから qcut
    ranked = series.rank(method="first")
    return pd.qcut(ranked, n, labels=[f"Q{i+1}" for i in range(n)])


def compute_per_horizon_auc(
    pair: pd.DataFrame, dev_stratum: pd.Series, stratum_name: str
) -> pd.DataFrame:
    """各 (horizon, 階層) で per-horizon AUC を計算."""
    pair = pair.copy()
    pair["stratum"] = pair["developer"].map(dev_stratum)
    pair = pair.dropna(subset=["stratum"])
    rows = []
    for h in HORIZONS:
        for q in sorted(pair["stratum"].unique()):
            sub = pair[pair["stratum"] == q]
            y = sub[f"label_{h}"].astype(int).values
            if len(np.unique(y)) < 2:
                continue
            irl_auc, n, n_pos = safe_auc(y, sub[f"irl_{h}"].values)
            rf_auc, _, _ = safe_auc(y, sub[f"rf_{h}"].values)
            ens_auc, _, _ = safe_auc(
                y, (sub[f"irl_{h}"].values + sub[f"rf_{h}"].values) / 2
            )
            rows.append({
                "stratum_axis": stratum_name,
                "stratum": str(q),
                "horizon": h,
                "n": n,
                "n_pos": n_pos,
                "base_rate": n_pos / n,
                "irl_auc": irl_auc,
                "rf_auc": rf_auc,
                "ens_auc": ens_auc,
                "irl_minus_rf": irl_auc - rf_auc,
            })
    return pd.DataFrame(rows)


def run_analysis(pair: pd.DataFrame, dev_activity: pd.Series, dev_trend: pd.Series,
                 label: str) -> pd.DataFrame:
    print(f"\n========== {label} ==========")
    print(f"unique devs in pair_trajectories: {pair['developer'].nunique():,}")
    print(f"pair rows: {len(pair):,}")

    # ── 活動量階層 ────────────────────────────────────
    matched_activity = dev_activity.reindex(pair["developer"].unique())
    matched_activity = matched_activity.dropna()
    print(f"\n[活動量] dev with combined_raw match: {len(matched_activity):,}")
    print(f"[活動量] dist (total_requests): "
          f"min={matched_activity.min():.0f}, "
          f"Q25={matched_activity.quantile(0.25):.0f}, "
          f"med={matched_activity.median():.0f}, "
          f"Q75={matched_activity.quantile(0.75):.0f}, "
          f"max={matched_activity.max():.0f}")
    strat = stratify(matched_activity)
    df1 = compute_per_horizon_auc(pair, strat, "activity_level")

    # ── 直近活動角度階層 ───────────────────────────────
    matched_trend = dev_trend.reindex(pair["developer"].unique())
    matched_trend = matched_trend.dropna()
    print(f"\n[活動角度] dev with feature_values match: {len(matched_trend):,}")
    print(f"[活動角度] dist (mean_activity_trend): "
          f"min={matched_trend.min():.3f}, "
          f"Q25={matched_trend.quantile(0.25):.3f}, "
          f"med={matched_trend.median():.3f}, "
          f"Q75={matched_trend.quantile(0.75):.3f}, "
          f"max={matched_trend.max():.3f}")
    strat2 = stratify(matched_trend)
    df2 = compute_per_horizon_auc(pair, strat2, "recent_trend")

    out = pd.concat([df1, df2], ignore_index=True)
    out["bot_filter"] = label

    print("\n[活動量階層 × per-horizon AUC]")
    print(df1.pivot_table(
        index="stratum", columns="horizon",
        values="irl_minus_rf", aggfunc="first"
    ).round(3).to_string())

    print("\n[活動角度階層 × per-horizon AUC]")
    print(df2.pivot_table(
        index="stratum", columns="horizon",
        values="irl_minus_rf", aggfunc="first"
    ).round(3).to_string())

    return out


def main() -> None:
    print("loading data...")
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    combined = pd.read_csv(DATA / "combined_raw_main32.csv", usecols=["email"])
    features = pd.read_csv(DATA / "feature_values.csv",
                           usecols=["__email", "activity_trend"])
    bot_df = pd.read_csv(BOT_CSV)
    bots = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712

    dev_activity = dev_activity_level(combined)
    dev_trend = dev_recent_trend(features)

    # ── bot 除外 vs 込み ───────────────────────────────
    df_with = run_analysis(pair, dev_activity, dev_trend, "with_bots")
    pair_nobot = pair[~pair["developer"].isin(bots)].copy()
    df_without = run_analysis(pair_nobot, dev_activity, dev_trend, "without_bots")

    summary = pd.concat([df_with, df_without], ignore_index=True)
    out_csv = OUT / "activity_stratified_auc.csv"
    summary.to_csv(out_csv, index=False)
    print(f"\nsaved: {out_csv}")

    # ── bot あり/なしの差 ───────────────────────────────
    print("\n========== bot 除外影響 (with - without の irl_minus_rf 差) ==========")
    merged = df_with.merge(
        df_without,
        on=["stratum_axis", "stratum", "horizon"],
        suffixes=("_with", "_without"),
    )
    merged["delta_irl_rf"] = merged["irl_minus_rf_with"] - merged["irl_minus_rf_without"]
    print(f"|delta| max: {merged['delta_irl_rf'].abs().max():.4f}")
    print(f"|delta| mean: {merged['delta_irl_rf'].abs().mean():.4f}")

    # ── 図 ─────────────────────────────────────────────
    plot_figure(df_with, df_without, OUT / "fig_activity_stratified_auc.pdf")


def plot_figure(df_with: pd.DataFrame, df_without: pd.DataFrame,
                out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def draw(ax, df, axis_name, title, ylim):
        sub = df[df["stratum_axis"] == axis_name].copy()
        pivot = sub.pivot_table(
            index="stratum", columns="horizon",
            values="irl_minus_rf", aggfunc="first"
        )
        pivot = pivot.reindex(sorted(pivot.index))
        pivot = pivot[HORIZONS]
        x = np.arange(len(pivot.index))
        width = 0.2
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
        for i, h in enumerate(HORIZONS):
            ax.bar(x + (i - 1.5) * width, pivot[h].values, width,
                   color=colors[i], label=h, alpha=0.88)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, fontsize=10)
        ax.set_ylim(*ylim)
        ax.set_ylabel("IRL AUC − RF AUC")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="lower right", fontsize=8, ncols=2)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        # n を補助テキスト
        n_per_stratum = sub.groupby("stratum")["n"].first().reindex(pivot.index)
        for j, n in enumerate(n_per_stratum):
            ax.text(j, ylim[0] + (ylim[1] - ylim[0]) * 0.04,
                    f"n={int(n):,}", ha="center", fontsize=7.5, color="#666")

    # 活動量
    ylim = (-0.1, 0.1)
    draw(axes[0, 0], df_with, "activity_level",
         "(a) 活動量階層 (Q1=少→Q5=多) × IRL−RF, bot 込み", ylim)
    draw(axes[0, 1], df_without, "activity_level",
         "(b) 活動量階層 × IRL−RF, bot 除外", ylim)
    # 直近活動角度
    draw(axes[1, 0], df_with, "recent_trend",
         "(c) 直近活動角度階層 (Q1=減→Q5=増) × IRL−RF, bot 込み", ylim)
    draw(axes[1, 1], df_without, "recent_trend",
         "(d) 直近活動角度階層 × IRL−RF, bot 除外", ylim)

    fig.suptitle(
        "活動量 / 直近活動角度の階層別 per-horizon AUC 差 (IRL − RF)",
        fontsize=12, y=1.00
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
