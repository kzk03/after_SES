"""Multi-horizon stacking の dev-level 評価。

pair-level で訓練した stacking model の予測を dev 単位に集約 (max-prob)、
per-dev label trajectory に対する AUC で評価。B-30 と整合した評価。

per-dev AUC: 各 dev の 4 horizon prediction を sample として AUC 計算 (label が両クラス含む dev のみ)。
overall dev-level AUC: 全 (dev, horizon) サンプル pool。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_diagonal(base: Path, variant: str = "lstm_baseline") -> pd.DataFrame:
    dfs = []
    for h in HORIZONS:
        p = base / variant / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        d = pd.read_csv(p).rename(columns={
            "label": f"label_{h}",
            "irl_dir_prob_calibrated": f"irl_{h}",
            "rf_dir_prob": f"rf_{h}",
        })[["developer", "directory", f"label_{h}", f"irl_{h}", f"rf_{h}"]]
        dfs.append(d)
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["developer", "directory"], how="outer")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path("outputs/main32_mce_volat_ep150"))
    ap.add_argument("--meta", choices=["lr", "gb"], default="gb")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/analysis/multi_horizon_stacking_volat_ep150"))
    args = ap.parse_args()

    print(f"[load] {args.base}")
    w = load_diagonal(args.base)

    feature_cols = [f"irl_{h}" for h in HORIZONS] + [f"rf_{h}" for h in HORIZONS]
    label_cols = [f"label_{h}" for h in HORIZONS]

    # === Step 1: Train stacking model per horizon (CV) and get OOF predictions ===
    print(f"[train] meta={args.meta}, OOF stacking predictions per horizon...")
    for h in HORIZONS:
        label_col = f"label_{h}"
        sub_idx = w.dropna(subset=feature_cols + [label_col]).index
        X = w.loc[sub_idx, feature_cols].to_numpy()
        y = w.loc[sub_idx, label_col].astype(int).to_numpy()
        oof_pred = np.full(len(w), np.nan)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
        for tr, te in skf.split(X, y):
            if args.meta == "gb":
                clf = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=777)
            else:
                clf = LogisticRegression(C=1.0, max_iter=500, random_state=777)
            clf.fit(X[tr], y[tr])
            oof_pred[sub_idx[te]] = clf.predict_proba(X[te])[:, 1]
        w[f"stack_{h}"] = oof_pred

    # === Step 2: Aggregate to dev-level (max-prob per dev per horizon) ===
    print("[aggregate] dev-level max-prob aggregation ...")
    dev_rows = []
    for dev, sub in w.groupby("developer"):
        row = {"developer": dev}
        for h in HORIZONS:
            valid = sub.dropna(subset=[f"label_{h}", f"irl_{h}", f"rf_{h}"])
            if valid.empty:
                continue
            row[f"label_{h}"] = int(valid[f"label_{h}"].max())
            row[f"irl_{h}"] = valid[f"irl_{h}"].max()
            row[f"rf_{h}"] = valid[f"rf_{h}"].max()
            row[f"avg_{h}"] = ((valid[f"irl_{h}"] + valid[f"rf_{h}"]) / 2).max()
            stack_v = valid[f"stack_{h}"].dropna()
            if not stack_v.empty:
                row[f"stack_{h}"] = stack_v.max()
        dev_rows.append(row)
    dev = pd.DataFrame(dev_rows)
    print(f"  devs: {len(dev)}")

    # === Step 3: per-dev AUC (across 4 horizons, devs with both labels) ===
    print("\n=== per-dev AUC (4-horizon ranking, devs with mixed labels) ===")
    per_dev_aucs = []
    for _, row in dev.iterrows():
        labels = [row.get(f"label_{h}", np.nan) for h in HORIZONS]
        labels_arr = np.array(labels, dtype=float)
        valid = ~np.isnan(labels_arr)
        if valid.sum() < 2: continue
        if len(set(labels_arr[valid].astype(int))) < 2: continue
        try:
            auc_irl = roc_auc_score(labels_arr[valid].astype(int),
                                    [row[f"irl_{h}"] for h in HORIZONS if not np.isnan(row.get(f"label_{h}", np.nan))])
            auc_rf  = roc_auc_score(labels_arr[valid].astype(int),
                                    [row[f"rf_{h}"]  for h in HORIZONS if not np.isnan(row.get(f"label_{h}", np.nan))])
            auc_avg = roc_auc_score(labels_arr[valid].astype(int),
                                    [row[f"avg_{h}"] for h in HORIZONS if not np.isnan(row.get(f"label_{h}", np.nan))])
            stack_vals = [row.get(f"stack_{h}", np.nan) for h in HORIZONS]
            stack_arr = np.array(stack_vals, dtype=float)
            v2 = valid & ~np.isnan(stack_arr)
            if v2.sum() >= 2 and len(set(labels_arr[v2].astype(int))) >= 2:
                auc_stack = roc_auc_score(labels_arr[v2].astype(int), stack_arr[v2])
            else:
                auc_stack = np.nan
            per_dev_aucs.append({
                "developer": row["developer"],
                "auc_irl": auc_irl, "auc_rf": auc_rf, "auc_avg": auc_avg, "auc_stack": auc_stack,
            })
        except Exception:
            pass
    pdf = pd.DataFrame(per_dev_aucs)
    print(f"  devs with per-dev AUC: {len(pdf)}")
    print(f"  median: IRL={pdf['auc_irl'].median():.3f}  RF={pdf['auc_rf'].median():.3f}  "
          f"Avg={pdf['auc_avg'].median():.3f}  Stack={pdf['auc_stack'].median():.3f}")
    print(f"  mean:   IRL={pdf['auc_irl'].mean():.3f}  RF={pdf['auc_rf'].mean():.3f}  "
          f"Avg={pdf['auc_avg'].mean():.3f}  Stack={pdf['auc_stack'].mean():.3f}")

    # Wilcoxon paired tests
    print("\n=== per-dev AUC: paired Wilcoxon test ===")
    for a, b in [("auc_stack", "auc_irl"), ("auc_stack", "auc_rf"), ("auc_stack", "auc_avg"),
                 ("auc_avg",   "auc_irl"), ("auc_avg",   "auc_rf"), ("auc_irl",   "auc_rf")]:
        valid = pdf[[a, b]].dropna()
        if len(valid) < 5: continue
        diff = valid[a] - valid[b]
        if (diff == 0).all(): continue
        try:
            stat, p = wilcoxon(valid[a], valid[b])
        except Exception:
            continue
        print(f"  {a} vs {b}: n={len(valid)}, median_diff={diff.median():+.3f}, "
              f"mean_diff={diff.mean():+.3f}, p={p:.4f}")

    # === Step 4: overall dev-level pooled AUC ===
    print("\n=== pooled dev-level AUC (all (dev, horizon) samples) ===")
    long_rows = []
    for _, row in dev.iterrows():
        for h in HORIZONS:
            if pd.notna(row.get(f"label_{h}", np.nan)):
                long_rows.append({
                    "developer": row["developer"], "horizon": h,
                    "label": int(row[f"label_{h}"]),
                    "irl": row[f"irl_{h}"], "rf": row[f"rf_{h}"],
                    "avg": row[f"avg_{h}"], "stack": row.get(f"stack_{h}", np.nan),
                })
    long_df = pd.DataFrame(long_rows)
    print(f"  n_samples: {len(long_df)} ({long_df['developer'].nunique()} devs × ~4 horizons)")
    for name in ["irl", "rf", "avg", "stack"]:
        v = long_df.dropna(subset=[name, "label"])
        if v["label"].nunique() < 2: continue
        auc = roc_auc_score(v["label"], v[name])
        print(f"  {name:>5}: AUC = {auc:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(args.out_dir / "dev_level_per_dev_aucs.csv", index=False)
    long_df.to_csv(args.out_dir / "dev_level_long.csv", index=False)
    dev.to_csv(args.out_dir / "dev_level_aggregated.csv", index=False)
    print(f"\nwrote: {args.out_dir}/dev_level_*")


if __name__ == "__main__":
    main()
