"""ルールベース 4 グループ分類 (活動量 × 承諾率) で archetype 別 AUC を出す。

軸:
- activity   : total_reviews_window (累積件数)、median split
- recency    : days_since_last_activity ≤ 30 日 (補助軸、後で 8 ビン拡張用)
- accept_rate: overall_acceptance_rate ≥ 0.5

2 軸 4 グループ:
  G1: Active Accepter    (activity=High, accept=High)
  G2: Active Rejecter    (activity=High, accept=Low)
  G3: Casual Accepter    (activity=Low,  accept=High)
  G4: Casual Rejecter    (activity=Low,  accept=Low)

3 軸 8 グループ版も併記。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from review_predictor.IRL.features.common_features import extract_common_features
from sklearn.metrics import average_precision_score, roc_auc_score

RNG = np.random.default_rng(777)


def auc_pair(y, s):
    if pd.Series(y).nunique() < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, s), average_precision_score(y, s)


def bootstrap_ci(y, s, n_boot=1000, alpha=0.05):
    y, s = np.asarray(y), np.asarray(s)
    n = len(y)
    aucs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if len(set(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(aucs, [2.5, 97.5]) if aucs else (np.nan, np.nan)


def paired_p(y, s_a, s_b, n_boot=1000):
    y, s_a, s_b = np.asarray(y), np.asarray(s_a), np.asarray(s_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        if len(set(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    diffs = np.array(diffs)
    return 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if len(diffs) else np.nan


def compute_axes(df_raw: pd.DataFrame, devs, feat_start, feat_end) -> pd.DataFrame:
    """各 dev について 3 軸の raw 値を計算 (normalize 前)。"""
    rows = []
    feat_start_ts = pd.Timestamp(feat_start)
    feat_end_ts = pd.Timestamp(feat_end)
    for email in devs:
        rev_mask = (
            (df_raw["email"] == email)
            & (df_raw["timestamp"] >= feat_start_ts)
            & (df_raw["timestamp"] < feat_end_ts)
        )
        if not rev_mask.any():
            continue
        feats = extract_common_features(
            df_raw, email,
            feat_start_ts.to_pydatetime(), feat_end_ts.to_pydatetime(),
            normalize=False,  # raw 値で軸を切る
        )
        rows.append({
            "developer": email,
            "total_reviews": feats.get("total_reviews", 0),
            "days_since_last_activity": feats.get("days_since_last_activity", 999),
            "overall_acceptance_rate": feats.get("overall_acceptance_rate", 0.0),
            "active_months_ratio": feats.get("active_months_ratio", 0.0),
        })
    return pd.DataFrame(rows)


def assign_groups(axes: pd.DataFrame) -> pd.DataFrame:
    """活動量 (median split on total_reviews) × 承諾率 (>=0.5) で 4 グループ。
    direct activity volume を使う (active_months_ratio はほぼ全員 1.0 で分離しないため)。
    """
    med_activity = axes["total_reviews"].median()
    axes = axes.copy()
    axes["high_activity"] = axes["total_reviews"] >= med_activity
    axes["high_accept"] = axes["overall_acceptance_rate"] >= 0.5
    axes["recent_active"] = axes["days_since_last_activity"] <= 30

    def g4(r):
        if r.high_activity and r.high_accept: return "G1_ActiveAccepter"
        if r.high_activity and not r.high_accept: return "G2_ActiveRejecter"
        if not r.high_activity and r.high_accept: return "G3_CasualAccepter"
        return "G4_CasualRejecter"
    axes["group4"] = axes.apply(g4, axis=1)

    def g8(r):
        a = "A+" if r.high_activity else "A-"
        c = "C+" if r.high_accept else "C-"
        r_ = "R+" if r.recent_active else "R-"
        return f"{a}{c}{r_}"
    axes["group8"] = axes.apply(g8, axis=1)
    return axes, med_activity


def analyze(df: pd.DataFrame, group_col: str):
    rows = []
    for g, sub in df.groupby(group_col):
        y = sub["label"].astype(int).to_numpy()
        irl = sub["irl_dir_prob_calibrated"].to_numpy()
        rf = sub["rf_dir_prob"].to_numpy()
        avg = (irl + rf) / 2
        a_irl, p_irl = auc_pair(y, irl)
        a_rf, p_rf = auc_pair(y, rf)
        a_avg, p_avg = auc_pair(y, avg)
        irl_lo, irl_hi = bootstrap_ci(y, irl)
        rf_lo, rf_hi = bootstrap_ci(y, rf)
        avg_lo, avg_hi = bootstrap_ci(y, avg)
        p_ir = paired_p(y, irl, rf)
        best_single = irl if a_irl >= a_rf else rf
        best_name = "IRL" if a_irl >= a_rf else "RF"
        p_ab = paired_p(y, avg, best_single)

        rows.append({
            "group": g, "n_devs": sub["developer"].nunique(),
            "n_pairs": len(sub), "n_pos": int(y.sum()),
            "auc_irl": a_irl, "irl_ci": f"[{irl_lo:.3f},{irl_hi:.3f}]",
            "auc_rf":  a_rf,  "rf_ci":  f"[{rf_lo:.3f},{rf_hi:.3f}]",
            "auc_avg": a_avg, "avg_ci": f"[{avg_lo:.3f},{avg_hi:.3f}]",
            "diff_ir": a_irl - a_rf, "p_ir": p_ir,
            "best": best_name,
            "diff_avg_best": a_avg - max(a_irl, a_rf), "p_avg_best": p_ab,
        })
    # ALL
    y = df["label"].astype(int).to_numpy()
    irl = df["irl_dir_prob_calibrated"].to_numpy()
    rf = df["rf_dir_prob"].to_numpy()
    avg = (irl + rf) / 2
    a_irl, _ = auc_pair(y, irl); a_rf, _ = auc_pair(y, rf); a_avg, _ = auc_pair(y, avg)
    rows.append({
        "group": "ALL", "n_devs": df["developer"].nunique(),
        "n_pairs": len(df), "n_pos": int(y.sum()),
        "auc_irl": a_irl, "auc_rf": a_rf, "auc_avg": a_avg,
        "diff_ir": a_irl - a_rf, "p_ir": paired_p(y, irl, rf),
        "best": "IRL" if a_irl >= a_rf else "RF",
        "diff_avg_best": a_avg - max(a_irl, a_rf),
        "p_avg_best": paired_p(y, avg, irl if a_irl >= a_rf else rf),
    })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path,
                   default=Path("outputs/main32_mce_volat/lstm_baseline/train_0-3m/eval_0-3m/pair_predictions.csv"))
    p.add_argument("--raw", type=Path, default=Path("data/combined_raw_main32.csv"))
    p.add_argument("--feature-end", default="2022-01-01")
    p.add_argument("--feature-window-days", type=int, default=180)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/analysis/rule_based_groups_volat_t0e0"))
    args = p.parse_args()

    feat_end = pd.Timestamp(args.feature_end)
    feat_start = feat_end - pd.Timedelta(days=args.feature_window_days)

    pred = pd.read_csv(args.pred).dropna(subset=["label", "irl_dir_prob_calibrated", "rf_dir_prob"])
    print(f"[pred] {args.pred}: rows={len(pred)}")

    print(f"[raw] loading {args.raw}")
    raw = pd.read_csv(args.raw, parse_dates=["timestamp"])

    print(f"[axes] computing 3-axis raw values for {pred['developer'].nunique()} devs ...")
    axes = compute_axes(raw, pred["developer"].unique(), feat_start, feat_end)
    axes, med_act = assign_groups(axes)
    print(f"[thresholds] activity median = {med_act:.1f} reviews,  accept>=0.5,  recent<=30d")
    print(f"[axes] dev → group:\n{axes['group4'].value_counts()}")

    merged = pred.merge(axes[["developer", "group4", "group8"]], on="developer", how="inner")
    print(f"[merged] {len(merged)} pairs ({merged['developer'].nunique()} devs after rule filter)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    axes.to_csv(args.out_dir / "dev_groups.csv", index=False)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)

    print("\n=== 4 groups (activity × accept) ===")
    r4 = analyze(merged, "group4")
    print(r4.to_string(index=False))
    r4.to_csv(args.out_dir / "result_g4.csv", index=False)

    print("\n=== 8 groups (activity × accept × recent) ===")
    r8 = analyze(merged, "group8")
    print(r8.to_string(index=False))
    r8.to_csv(args.out_dir / "result_g8.csv", index=False)

    print(f"\nwrote: {args.out_dir}/")


if __name__ == "__main__":
    main()
