# -*- coding: utf-8 -*-
"""区間予測 (4 予測区間分割) の意義に統計的裏付けを付ける [W3 rigor].

論文 §5 / §6 の「区間予測で何が示せたか」は §2 (機序シフト) と §13 (脱落 timing) に
基づくが、いずれも記述統計で CI / 検定がなかった。本スクリプトは 2 つの主張に rigor を付ける:

  Part A [機序シフトの有意性]:
    「短期予測=直近性, 長期予測=累積量」を、直近性特徴量の importance share が
    0-3m で 9-12m より有意に高いかで検定。開発者クラスタ bootstrap (B=200) で
    各予測区間の recency_share と、shift=share(0-3m)−share(9-12m) の 95% CI を出す。

  Part B [脱落 timing の構造的検定 — 4 区間分割の正当化]:
    「同一ディレクトリでも脱落のタイミングが遷移ごとに変動する」を、
    各 dir の 3 遷移 attrition rate の range(max−min) が、一定脱落率の帰無の下での
    サンプリングノイズを超えるかで検定 (Monte Carlo Binomial + χ² 独立性検定)。
    有意な dir 数が多ければ「単一年間予測では失う構造」を統計的に主張できる。

入力:
  data/pair_trajectories.csv, data/feature_values.csv (Part A)
  results/horizon_dir_attrition.csv (Part B)

出力:
  results/interval_mechanism_shift.csv
  results/interval_attrition_timing_test.csv
  results/interval_rigor_summary.json
  results/fig_interval_rigor.pdf
"""
from __future__ import annotations

import json
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
HORIZON_JP = {"0-3m": "0-3か月", "3-6m": "3-6か月",
              "6-9m": "6-9か月", "9-12m": "9-12か月"}
T0 = "2022-01-01"
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
    "avg_action_intensity", "avg_change_lines", "avg_review_size",
    "repeat_collaboration_rate", "avg_response_time",
]

# 直近性 (recency) 系と 累積/規模 (volume) 系の明確なコア集合のみで share を測る
# (trend / 曖昧な特徴量は両群から除外しコントラストを明瞭化)
RECENCY_SET = [
    "window_tenure_days", "days_since_last_activity", "recent_activity_frequency",
    "active_months_ratio", "recent_rejection_streak", "recent_acceptance_rate",
    "recent_load_ratio_30d_all", "recent_load_ratio_7d_30d", "acceptance_rate_last10",
]
VOLUME_SET = [
    "total_changes", "total_reviews", "unique_collaborator_count",
    "core_reviewer_ratio", "avg_action_intensity", "avg_change_lines",
    "avg_review_size", "repeat_collaboration_rate",
]

N_BOOT_A = 200
RF_TREES_A = 200
N_MC_B = 5000


# ---------------- Part A ----------------
def fit_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    rf = RandomForestClassifier(n_estimators=RF_TREES_A, max_depth=12,
                                random_state=SEED, n_jobs=-1)
    rf.fit(X, y)
    return rf.feature_importances_


def recency_share(imp: np.ndarray) -> float:
    idx = {f: i for i, f in enumerate(STATE_FEATURES)}
    rec = sum(imp[idx[f]] for f in RECENCY_SET)
    vol = sum(imp[idx[f]] for f in VOLUME_SET)
    return rec / (rec + vol) if (rec + vol) > 0 else np.nan


def part_a(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    # horizon ごとの X, y, developer を事前構築
    H_data = {}
    for h in HORIZONS:
        sub = merged[~merged[f"label_{h}"].isna()].copy().reset_index(drop=True)
        H_data[h] = {
            "X": sub[STATE_FEATURES].fillna(0.0).to_numpy(),
            "y": sub[f"label_{h}"].astype(int).to_numpy(),
            "dev": sub["developer"].to_numpy(),
        }

    # 観測 share
    obs_share = {}
    for h in HORIZONS:
        obs_share[h] = recency_share(fit_importance(H_data[h]["X"], H_data[h]["y"]))

    # bootstrap (開発者クラスタ)
    boot_share = {h: [] for h in HORIZONS}
    boot_shift = []  # share(0-3m) - share(9-12m)
    for b in range(N_BOOT_A):
        shares_b = {}
        for h in HORIZONS:
            dev = H_data[h]["dev"]
            uniq = np.unique(dev)
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            # 選ばれた開発者の行を集める
            mask_rows = np.concatenate([np.where(dev == d)[0] for d in pick])
            Xb, yb = H_data[h]["X"][mask_rows], H_data[h]["y"][mask_rows]
            if len(np.unique(yb)) < 2:
                shares_b[h] = np.nan
                continue
            shares_b[h] = recency_share(fit_importance(Xb, yb))
            boot_share[h].append(shares_b[h])
        if np.isfinite(shares_b.get("0-3m", np.nan)) and \
                np.isfinite(shares_b.get("9-12m", np.nan)):
            boot_shift.append(shares_b["0-3m"] - shares_b["9-12m"])
        if (b + 1) % 50 == 0:
            print(f"  [A] bootstrap {b+1}/{N_BOOT_A}")

    rows = []
    for h in HORIZONS:
        arr = np.array(boot_share[h])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append({"horizon": h, "recency_share": round(obs_share[h], 4),
                     "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
    shift = np.array(boot_shift)
    shift_obs = obs_share["0-3m"] - obs_share["9-12m"]
    lo, hi = np.percentile(shift, [2.5, 97.5])
    p = float(min(1.0, 2 * min(np.mean(shift <= 0), np.mean(shift >= 0))))
    summary = {
        "recency_share_obs": {h: round(obs_share[h], 4) for h in HORIZONS},
        "shift_0_3m_minus_9_12m": round(float(shift_obs), 4),
        "shift_ci": [round(float(lo), 4), round(float(hi), 4)],
        "shift_p": round(p, 4),
        "sig": bool(lo > 0 or hi < 0),
        "n_boot": N_BOOT_A,
    }
    return pd.DataFrame(rows), summary


# ---------------- Part B ----------------
def part_b(att: pd.DataFrame, min_n: int = 10) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED + 1)
    att = att.dropna(subset=["n_active_prev", "n_attrited"]).copy()
    att = att[att["n_active_prev"] >= min_n]
    rows = []
    for d, g in att.groupby("directory"):
        if g["h_prev"].nunique() < 3:  # 3 遷移揃ったものだけ
            continue
        ns = g["n_active_prev"].to_numpy().astype(int)
        ks = g["n_attrited"].to_numpy().astype(int)
        rates = ks / ns
        rng_obs = float(rates.max() - rates.min())
        p_hat = ks.sum() / ns.sum()
        # Monte Carlo: 一定脱落率 p_hat の下での range 分布
        sim = rng.binomial(ns[None, :], p_hat, size=(N_MC_B, len(ns))) / ns[None, :]
        sim_range = sim.max(axis=1) - sim.min(axis=1)
        p_mc = float((1 + np.sum(sim_range >= rng_obs)) / (1 + N_MC_B))
        # χ² 独立性 (遷移 × {attrited, retained})
        table = np.array([ks, ns - ks])
        try:
            chi2, p_chi2, _, _ = chi2_contingency(table)
        except ValueError:
            chi2, p_chi2 = np.nan, np.nan
        peak = g.loc[g["attrition_rate"].idxmax()]
        rows.append({
            "directory": d, "n_trans": len(ns),
            "attr_range": round(rng_obs, 3), "pooled_rate": round(float(p_hat), 3),
            "p_montecarlo": round(p_mc, 4), "p_chi2": round(float(p_chi2), 4),
            "peak_transition": f"{peak['h_prev']}→{peak['h_curr']}",
            "sig_mc": bool(p_mc < 0.05), "sig_chi2": bool(p_chi2 < 0.05),
        })
    res = pd.DataFrame(rows).sort_values("attr_range", ascending=False)
    n_total = len(res)
    summary = {
        "n_dirs_3trans": n_total,
        "n_sig_montecarlo": int(res["sig_mc"].sum()),
        "n_sig_chi2": int(res["sig_chi2"].sum()),
        "frac_sig_mc": round(res["sig_mc"].mean(), 3) if n_total else None,
        "median_attr_range": round(float(res["attr_range"].median()), 3) if n_total else None,
        "n_range_ge_0.3": int((res["attr_range"] >= 0.3).sum()),
        "min_n_active_prev": min_n, "n_montecarlo": N_MC_B,
    }
    return res, summary


def plot_rigor(shift_df: pd.DataFrame, shift_sum: dict,
               attr_df: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左: recency share across horizons with CI
    ax = axes[0]
    x = np.arange(len(HORIZONS))
    y = shift_df["recency_share"].to_numpy()
    lo = shift_df["ci_lo"].to_numpy()
    hi = shift_df["ci_hi"].to_numpy()
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", color="#1f77b4",
                capsize=5, lw=2, markersize=9)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([HORIZON_JP[h] for h in HORIZONS], fontsize=10)
    ax.set_ylabel("直近性特徴量の importance share", fontsize=10)
    ax.set_title(f"機序シフト: 直近性 share は短期>長期\n"
                 f"shift={shift_sum['shift_0_3m_minus_9_12m']:+.3f} "
                 f"[{shift_sum['shift_ci'][0]:+.3f},{shift_sum['shift_ci'][1]:+.3f}] "
                 f"{'(有意)' if shift_sum['sig'] else '(非有意)'}", fontsize=11)
    ax.grid(alpha=0.25)

    # 右: per-dir attrition range, 有意/非有意で色分け
    ax = axes[1]
    d = attr_df.sort_values("attr_range")
    colors = ["#d62728" if s else "#cccccc" for s in d["sig_mc"]]
    ax.barh(range(len(d)), d["attr_range"], color=colors)
    ax.axvline(0.3, color="black", ls="--", lw=1, label="range=0.3")
    ax.set_yticks([])
    ax.set_xlabel("dir 内の脱落率 range (max−min, 3 遷移)", fontsize=10)
    n_sig = int(d["sig_mc"].sum())
    ax.set_title(f"脱落 timing の dir 別変動\n"
                 f"{n_sig}/{len(d)} dir で有意 (Monte Carlo p<0.05, 赤)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("区間予測 (4 予測区間分割) の意義の統計的裏付け", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    # ---- Part A ----
    print("=== Part A: 機序シフト (recency share) の bootstrap ===")
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    feat = pd.read_csv(DATA / "feature_values.csv")
    feat_t0 = feat[feat["__snapshot"] == T0].copy()
    merged = pred.merge(feat_t0, left_on="developer", right_on="__email",
                        how="inner")
    print(f"merged shape: {merged.shape}")
    shift_df, shift_sum = part_a(merged)
    shift_df.to_csv(OUT / "interval_mechanism_shift.csv", index=False)
    print("recency share:", shift_sum["recency_share_obs"])
    print(f"shift(0-3m − 9-12m) = {shift_sum['shift_0_3m_minus_9_12m']:+.3f} "
          f"CI{shift_sum['shift_ci']} p={shift_sum['shift_p']} "
          f"{'★有意' if shift_sum['sig'] else '非有意'}")

    # ---- Part B ----
    print("\n=== Part B: 脱落 timing の構造的検定 ===")
    att = pd.read_csv(OUT / "horizon_dir_attrition.csv")
    attr_df, attr_sum = part_b(att)
    attr_df.to_csv(OUT / "interval_attrition_timing_test.csv", index=False)
    print(json.dumps(attr_sum, ensure_ascii=False, indent=2))
    print("\n上位 10 (range 降順):")
    print(attr_df.head(10)[["directory", "attr_range", "pooled_rate",
                            "p_montecarlo", "p_chi2", "peak_transition",
                            "sig_mc"]].to_string(index=False))

    with open(OUT / "interval_rigor_summary.json", "w") as f:
        json.dump({"part_a_mechanism_shift": shift_sum,
                   "part_b_attrition_timing": attr_sum}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nsaved {OUT / 'interval_rigor_summary.json'}")

    plot_rigor(shift_df, shift_sum, attr_df, OUT / "fig_interval_rigor.pdf")


if __name__ == "__main__":
    main()
