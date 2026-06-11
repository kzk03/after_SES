#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""40_archetype_deep_analysis.py
アーキタイプ別 IRL vs RF 性能差の構造的要因を定量分析。

分析項目:
  1. P1-P4 / A1-A4 の pooled AUC (IRL vs RF) と Δ
  2. 各 archetype の t0 時点特徴量の分布統計
     - 特に active_months_ratio, total_reviews, avg_activity_gap, overall_acceptance_rate
  3. P1 で RF 予測値のばらつきが小さい (base_rate 近傍に崩壊) ことの検証
  4. A4 (常時活動型) で IRL の系列長が長く LSTM に有利であることの検証
  5. A2 (レビュー専念型) の活動月比による IRL/RF 寄与の違い
"""
import json
import os

import numpy as np
import pandas as pd

BASE = "/Users/kazuki-h/Master_research/writing/analysis"
DATA = os.path.join(BASE, "data")
RESULTS = os.path.join(BASE, "results")
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
PMAP = {"A1": "P1", "A2": "P2", "A3": "P3", "A4": "P4"}
T0 = "2022-01-01"


def _auc_roc(y_true, y_score):
    """Mann-Whitney U statistic AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    # vectorized: sum of (pos_i > neg_j) + 0.5*(pos_i == neg_j)
    count = 0.0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return count / (len(pos) * len(neg))


def _avg_precision(y_true, y_score):
    """Average Precision (simple implementation)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.sum() == 0:
        return np.nan
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    prec = tp / np.arange(1, len(y_sorted) + 1)
    return float(np.sum(prec * y_sorted) / y_true.sum())



def load_data():
    traj = pd.read_csv(os.path.join(DATA, "pair_trajectories.csv"))
    arch = pd.read_csv(os.path.join(DATA, "pair_archetype.csv"))
    m = traj.merge(
        arch[["developer", "directory", "dominant"]],
        on=["developer", "directory"], how="left"
    )
    m["pair_arch"] = m["dominant"].map(PMAP)
    m["dev_arch"] = m["archetype"]
    return m


def to_long(df):
    rows = []
    for h in HORIZONS:
        sub = df[["developer", "directory", "pair_arch", "dev_arch"]].copy()
        sub["horizon"] = h
        sub["label"] = df[f"label_{h}"].values
        sub["irl"] = df[f"irl_{h}"].values
        sub["rf"] = df[f"rf_{h}"].values
        sub["avg"] = df[f"avg_{h}"].values
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


# ====================================================================
# 1. Archetype 別 pooled AUC
# ====================================================================
def analysis_1_auc(long):
    results = []
    for arch_type, col in [("pair_arch (P)", "pair_arch"), ("dev_arch (A)", "dev_arch")]:
        for g in sorted(long[col].dropna().unique()):
            sub = long[long[col] == g].dropna(subset=["label", "irl", "rf"])
            y = sub["label"].values.astype(int)
            results.append({
                "type": arch_type, "archetype": g,
                "n_obs": len(sub), "n_pos": int(y.sum()),
                "base_rate": round(y.mean(), 4),
                "irl_auc": round(_auc_roc(y, sub["irl"]), 4),
                "rf_auc": round(_auc_roc(y, sub["rf"]), 4),
                "delta_auc": round(_auc_roc(y, sub["irl"]) - _auc_roc(y, sub["rf"]), 4),
                "irl_ap": round(_avg_precision(y, sub["irl"]), 4),
                "rf_ap": round(_avg_precision(y, sub["rf"]), 4),
                "delta_ap": round(_avg_precision(y, sub["irl"]) - _avg_precision(y, sub["rf"]), 4),
            })
    return pd.DataFrame(results)


# ====================================================================
# 2. Archetype 別 per-horizon AUC (Δの推移を見る)
# ====================================================================
def analysis_2_horizon(long):
    results = []
    for col in ["pair_arch", "dev_arch"]:
        for g in sorted(long[col].dropna().unique()):
            for h in HORIZONS:
                sub = long[(long[col] == g) & (long["horizon"] == h)].dropna(subset=["label", "irl", "rf"])
                y = sub["label"].values.astype(int)
                irl_a = _auc_roc(y, sub["irl"])
                rf_a = _auc_roc(y, sub["rf"])
                results.append({
                    "arch_col": col, "archetype": g, "horizon": h,
                    "n_obs": len(sub), "n_pos": int(y.sum()),
                    "base_rate": round(y.mean(), 4),
                    "irl_auc": round(irl_a, 4) if not np.isnan(irl_a) else np.nan,
                    "rf_auc": round(rf_a, 4) if not np.isnan(rf_a) else np.nan,
                    "delta_auc": round(irl_a - rf_a, 4) if not (np.isnan(irl_a) or np.isnan(rf_a)) else np.nan,
                })
    return pd.DataFrame(results)


# ====================================================================
# 3. RF 予測値のばらつき (P1 で base_rate 近傍に崩壊するか)
# ====================================================================
def analysis_3_rf_collapse(long):
    results = []
    for g in ["P1", "P2", "P3", "P4"]:
        sub = long[long["pair_arch"] == g].dropna(subset=["label", "rf", "irl"])
        rf = sub["rf"].values
        irl = sub["irl"].values
        br = sub["label"].mean()
        results.append({
            "archetype": g,
            "n_obs": len(sub),
            "base_rate": round(br, 4),
            "rf_mean": round(rf.mean(), 4),
            "rf_std": round(rf.std(), 4),
            "rf_iqr": round(np.percentile(rf, 75) - np.percentile(rf, 25), 4),
            "rf_frac_near_base": round(float(np.mean(np.abs(rf - br) < 0.05)), 4),
            "irl_mean": round(irl.mean(), 4),
            "irl_std": round(irl.std(), 4),
            "irl_iqr": round(np.percentile(irl, 75) - np.percentile(irl, 25), 4),
        })
    return pd.DataFrame(results)


# ====================================================================
# 4. 特徴量分布 (t0 時点) を archetype 別に集計
# ====================================================================
def analysis_4_features(df):
    fv = pd.read_csv(os.path.join(DATA, "feature_values.csv"))
    fv = fv[fv["__snapshot"] == T0].rename(columns={"__email": "developer"})
    fv = fv.drop_duplicates("developer")

    # developer archetype
    dev_arch = df[["developer", "dev_arch"]].drop_duplicates("developer")
    fv_merged = fv.merge(dev_arch, on="developer", how="inner")

    cols = ["active_months_ratio", "total_reviews", "total_changes",
            "avg_activity_gap", "overall_acceptance_rate",
            "recent_activity_frequency", "days_since_last_activity",
            "unique_collaborator_count"]

    results = []
    for g in sorted(fv_merged["dev_arch"].dropna().unique()):
        sub = fv_merged[fv_merged["dev_arch"] == g]
        row = {"archetype": g, "n_devs": len(sub)}
        for c in cols:
            if c in sub.columns:
                vals = sub[c].dropna()
                row[f"{c}_mean"] = round(vals.mean(), 4)
                row[f"{c}_median"] = round(vals.median(), 4)
                row[f"{c}_std"] = round(vals.std(), 4)
        results.append(row)
    return pd.DataFrame(results)


# ====================================================================
# 5. 系列長の効果: dev_arch 別の active_months_ratio と IRL AUC
# ====================================================================
def analysis_5_sequence_length(df, long):
    fv = pd.read_csv(os.path.join(DATA, "feature_values.csv"))
    fv = fv[fv["__snapshot"] == T0][["__email", "active_months_ratio", "total_reviews"]].rename(
        columns={"__email": "developer"})
    fv = fv.drop_duplicates("developer")

    long_f = long.merge(fv, on="developer", how="inner")
    long_f = long_f.dropna(subset=["label", "irl", "rf", "active_months_ratio"])

    results = []
    for g in sorted(long_f["dev_arch"].dropna().unique()):
        sub = long_f[long_f["dev_arch"] == g]
        # 3分位
        q = sub["active_months_ratio"].quantile([1/3, 2/3]).values
        for bucket_name, lo, hi in [("低(Q1)", -1, q[0]), ("中(Q2)", q[0], q[1]), ("高(Q3)", q[1], 999)]:
            b = sub[(sub["active_months_ratio"] > lo) & (sub["active_months_ratio"] <= hi)]
            y = b["label"].values.astype(int)
            results.append({
                "dev_arch": g, "bucket": bucket_name,
                "n_obs": len(b), "n_pos": int(y.sum()),
                "amr_mean": round(b["active_months_ratio"].mean(), 2),
                "irl_auc": round(_auc_roc(y, b["irl"]), 4),
                "rf_auc": round(_auc_roc(y, b["rf"]), 4),
                "delta_auc": round(_auc_roc(y, b["irl"]) - _auc_roc(y, b["rf"]), 4)
                if not np.isnan(_auc_roc(y, b["irl"])) else np.nan,
            })
    return pd.DataFrame(results)


def main():
    print("Loading data...")
    df = load_data()
    long = to_long(df)
    print(f"Pairs: {len(df)}, Long obs: {len(long)}")

    print("\n" + "="*70)
    print("1. Archetype 別 pooled AUC (IRL vs RF)")
    print("="*70)
    a1 = analysis_1_auc(long)
    print(a1.to_string(index=False))
    a1.to_csv(os.path.join(RESULTS, "archetype_deep_auc.csv"), index=False)

    print("\n" + "="*70)
    print("2. Archetype × Horizon 別 delta_auc")
    print("="*70)
    a2 = analysis_2_horizon(long)
    # pivot for pair_arch
    piv = a2[a2["arch_col"] == "pair_arch"].pivot(index="archetype", columns="horizon", values="delta_auc")
    print("[Pair archetype (P)]")
    print(piv.round(4).to_string())
    piv2 = a2[a2["arch_col"] == "dev_arch"].pivot(index="archetype", columns="horizon", values="delta_auc")
    print("\n[Dev archetype (A)]")
    print(piv2.round(4).to_string())
    a2.to_csv(os.path.join(RESULTS, "archetype_deep_horizon.csv"), index=False)

    print("\n" + "="*70)
    print("3. RF 予測値のばらつき (P1 崩壊仮説)")
    print("="*70)
    a3 = analysis_3_rf_collapse(long)
    print(a3.to_string(index=False))
    a3.to_csv(os.path.join(RESULTS, "archetype_deep_rf_collapse.csv"), index=False)

    print("\n" + "="*70)
    print("4. 特徴量分布 (t0 時点, 開発者 archetype 別)")
    print("="*70)
    a4 = analysis_4_features(df)
    print(a4.to_string(index=False))
    a4.to_csv(os.path.join(RESULTS, "archetype_deep_features.csv"), index=False)

    print("\n" + "="*70)
    print("5. 系列長 (active_months_ratio) の効果")
    print("="*70)
    a5 = analysis_5_sequence_length(df, long)
    print(a5.to_string(index=False))
    a5.to_csv(os.path.join(RESULTS, "archetype_deep_sequence.csv"), index=False)

    # Summary JSON
    summary = {
        "P1_delta_auc_pooled": float(a1[(a1["type"]=="pair_arch (P)") & (a1["archetype"]=="P1")]["delta_auc"].iloc[0]),
        "P2_delta_auc_pooled": float(a1[(a1["type"]=="pair_arch (P)") & (a1["archetype"]=="P2")]["delta_auc"].iloc[0]),
        "P3_delta_auc_pooled": float(a1[(a1["type"]=="pair_arch (P)") & (a1["archetype"]=="P3")]["delta_auc"].iloc[0]),
        "P4_delta_auc_pooled": float(a1[(a1["type"]=="pair_arch (P)") & (a1["archetype"]=="P4")]["delta_auc"].iloc[0]),
        "P1_rf_std": float(a3[a3["archetype"]=="P1"]["rf_std"].iloc[0]),
        "P3_rf_std": float(a3[a3["archetype"]=="P3"]["rf_std"].iloc[0]),
        "P1_rf_frac_near_base": float(a3[a3["archetype"]=="P1"]["rf_frac_near_base"].iloc[0]),
        "P3_rf_frac_near_base": float(a3[a3["archetype"]=="P3"]["rf_frac_near_base"].iloc[0]),
    }

    for arch in ["A1", "A2", "A3", "A4"]:
        row = a1[(a1["type"]=="dev_arch (A)") & (a1["archetype"]==arch)]
        if len(row):
            summary[f"{arch}_delta_auc_pooled"] = float(row["delta_auc"].iloc[0])

    with open(os.path.join(RESULTS, "archetype_deep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nDone. Results saved to", RESULTS)


if __name__ == "__main__":
    main()
