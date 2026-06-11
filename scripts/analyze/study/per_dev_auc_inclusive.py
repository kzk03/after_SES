# -*- coding: utf-8 -*-
"""per-dev AUC を「(dev, dir, horizon) を dev 単位で pool」で計算 (inclusive 版).

旧 multi_horizon_stacking_dev_level.py は (dev, horizon) 単位 (= 各 horizon で
dir を max-pool) で per-dev AUC を計算していたため、4 horizon にまたがる label
switch がない dev は除外され n=103 に留まった。

inclusive 版: 各 dev の **全 (dir, horizon) 行を 1 バケツに pool** して AUC 計算。
1 つの horizon にしか出現しない dev でも、dir 間で label 変動があれば AUC 可能。

archetype α (k4_dev_alpha_with_metrics.csv) と merge し、archetype 別の IRL vs RF
mean / median / Wilcoxon p を出力。
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_long(pred_root: pathlib.Path, variant: str = "lstm_baseline",
              diagonal: bool = True) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        train_h = h if diagonal else "0-3m"
        p = pred_root / variant / f"train_{train_h}" / f"eval_{h}" / "pair_predictions.csv"
        if not p.exists():
            print(f"missing: {p}")
            continue
        d = pd.read_csv(p)
        d["horizon"] = h
        rows.append(d[["developer", "directory", "horizon", "label",
                       "irl_dir_prob_calibrated", "rf_dir_prob"]])
    return pd.concat(rows, ignore_index=True)


def per_dev_aucs(long_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for dev, g in long_df.groupby("developer"):
        y = g["label"].to_numpy().astype(int)
        if len(y) < 2 or len(set(y)) < 2:
            out.append({"developer": dev, "n_records": len(g),
                        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                        "auc_irl": np.nan, "auc_rf": np.nan, "diff": np.nan})
            continue
        try:
            a_irl = roc_auc_score(y, g["irl_dir_prob_calibrated"].to_numpy())
            a_rf = roc_auc_score(y, g["rf_dir_prob"].to_numpy())
            out.append({"developer": dev, "n_records": len(g),
                        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                        "auc_irl": a_irl, "auc_rf": a_rf, "diff": a_irl - a_rf})
        except Exception:
            continue
    return pd.DataFrame(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred-root", type=pathlib.Path,
                   default="outputs/main32_mce_volat_ep150")
    p.add_argument("--alpha-csv", type=pathlib.Path,
                   default="outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv")
    p.add_argument("--out-dir", type=pathlib.Path,
                   default="outputs/analysis/multi_horizon_stacking_volat_ep150")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    long_df = load_long(args.pred_root)
    print(f"long records: {len(long_df)}, devs: {long_df['developer'].nunique()}")

    pdf = per_dev_aucs(long_df)
    print(f"\nper-dev AUC computed for {pdf['auc_irl'].notna().sum()} devs "
          f"(of {len(pdf)} total)")

    alpha = pd.read_csv(args.alpha_csv).rename(columns={"Unnamed: 0": "developer"})
    if "developer" not in alpha.columns:
        alpha = pd.read_csv(args.alpha_csv, index_col=0).reset_index().rename(
            columns={"index": "developer"})
    merged = pdf.merge(alpha[["developer", "A1", "A2", "A3", "A4", "dominant"]],
                       on="developer", how="left")
    merged.to_csv(args.out_dir / "per_dev_auc_inclusive.csv", index=False)
    print(f"saved: {args.out_dir/'per_dev_auc_inclusive.csv'}")

    valid = merged.dropna(subset=["auc_irl", "auc_rf", "dominant"])
    print(f"\n=== archetype 別 IRL vs RF (n_valid={len(valid)}) ===")
    summary_rows = []
    for a in ["A1", "A2", "A3", "A4"]:
        sub = valid[valid["dominant"] == a]
        n = len(sub)
        if n < 5:
            continue
        m_irl, m_rf = sub["auc_irl"].mean(), sub["auc_rf"].mean()
        med_irl, med_rf = sub["auc_irl"].median(), sub["auc_rf"].median()
        diff = sub["auc_irl"] - sub["auc_rf"]
        try:
            _, pval = wilcoxon(sub["auc_irl"], sub["auc_rf"])
        except Exception:
            pval = np.nan
        win_irl = (diff > 0).sum()
        win_rf = (diff < 0).sum()
        tie = (diff == 0).sum()
        summary_rows.append({
            "archetype": a, "n": n,
            "mean_irl": m_irl, "mean_rf": m_rf, "mean_diff": m_irl - m_rf,
            "median_irl": med_irl, "median_rf": med_rf,
            "win_irl": win_irl, "win_rf": win_rf, "tie": tie,
            "wilcoxon_p": pval,
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))
    summary.to_csv(args.out_dir / "per_dev_auc_by_archetype.csv", index=False)

    # overall
    diff_all = valid["auc_irl"] - valid["auc_rf"]
    try:
        _, p_all = wilcoxon(valid["auc_irl"], valid["auc_rf"])
    except Exception:
        p_all = np.nan
    print(f"\nOVERALL: n={len(valid)}, mean IRL={valid['auc_irl'].mean():.3f}, "
          f"mean RF={valid['auc_rf'].mean():.3f}, "
          f"mean diff={diff_all.mean():+.4f}, Wilcoxon p={p_all:.4g}, "
          f"win_irl={(diff_all > 0).sum()}, win_rf={(diff_all < 0).sum()}")


if __name__ == "__main__":
    main()
