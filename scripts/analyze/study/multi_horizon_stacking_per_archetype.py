"""Multi-horizon stacking を archetype 別に評価する。

「A1 (Newcomer) は IRL/RF 単独でも単純 ensemble でも救えないが、
stacking なら救えるか？」を検証。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
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


def cv_auc(X, y, irl_solo, rf_solo, meta="gb", n_splits=5):
    if pd.Series(y).nunique() < 2 or len(y) < 10:
        return None
    skf = StratifiedKFold(n_splits=min(n_splits, sum(y == 1), sum(y == 0)),
                          shuffle=True, random_state=777)
    avg = (irl_solo + rf_solo) / 2
    aucs_irl, aucs_rf, aucs_avg, aucs_stack = [], [], [], []
    for tr, te in skf.split(X, y):
        if len(set(y[tr])) < 2: continue
        if meta == "gb":
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                             learning_rate=0.1, random_state=777)
        else:
            clf = LogisticRegression(C=1.0, max_iter=500, random_state=777)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs_stack.append(roc_auc_score(y[te], p))
        aucs_irl.append(roc_auc_score(y[te], irl_solo[te]))
        aucs_rf.append(roc_auc_score(y[te], rf_solo[te]))
        aucs_avg.append(roc_auc_score(y[te], avg[te]))
    return {
        "auc_irl": np.mean(aucs_irl), "auc_rf": np.mean(aucs_rf),
        "auc_avg": np.mean(aucs_avg), "auc_stack": np.mean(aucs_stack),
        "stack_std": np.std(aucs_stack),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path("outputs/main32_mce_volat_ep150"))
    ap.add_argument("--alpha", type=Path,
                    default=Path("outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv"))
    ap.add_argument("--meta", choices=["gb", "lr"], default="gb")
    ap.add_argument("--out", type=Path,
                    default=Path("outputs/analysis/multi_horizon_stacking_volat_ep150/per_archetype.csv"))
    args = ap.parse_args()

    w = load_diagonal(args.base)
    alpha = pd.read_csv(args.alpha, index_col=0)
    alpha.index.name = "developer"
    alpha = alpha.reset_index()[["developer", "dominant"]].rename(
        columns={"dominant": "archetype"})
    w = w.merge(alpha, on="developer", how="left")

    feature_cols = [f"irl_{h}" for h in HORIZONS] + [f"rf_{h}" for h in HORIZONS]

    rows = []
    for arch in sorted(w["archetype"].dropna().unique()):
        sub_arch = w[w["archetype"] == arch]
        for target_h in HORIZONS:
            label_col = f"label_{target_h}"
            sub = sub_arch.dropna(subset=feature_cols + [label_col])
            if len(sub) < 30: continue
            X = sub[feature_cols].to_numpy()
            y = sub[label_col].astype(int).to_numpy()
            irl_s = sub[f"irl_{target_h}"].to_numpy()
            rf_s = sub[f"rf_{target_h}"].to_numpy()
            r = cv_auc(X, y, irl_s, rf_s, meta=args.meta)
            if r is None: continue
            rows.append({
                "archetype": arch, "target": target_h,
                "n": len(sub), "n_pos": int(y.sum()),
                **r,
            })

    # ALL archetypes
    for target_h in HORIZONS:
        label_col = f"label_{target_h}"
        sub = w.dropna(subset=feature_cols + [label_col])
        X = sub[feature_cols].to_numpy()
        y = sub[label_col].astype(int).to_numpy()
        irl_s = sub[f"irl_{target_h}"].to_numpy()
        rf_s = sub[f"rf_{target_h}"].to_numpy()
        r = cv_auc(X, y, irl_s, rf_s, meta=args.meta)
        rows.append({
            "archetype": "ALL", "target": target_h,
            "n": len(sub), "n_pos": int(y.sum()),
            **r,
        })

    out = pd.DataFrame(rows)
    out["gain_vs_avg"] = out["auc_stack"] - out["auc_avg"]
    out["gain_vs_best"] = out["auc_stack"] - out[["auc_irl", "auc_rf"]].max(axis=1)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 200)
    print("\n=== Multi-horizon stacking by archetype × target horizon ===")
    print(out.to_string(index=False))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote: {args.out}")

    # Highlight A1
    print("\n=== A1 (Newcomer) per-horizon gain ===")
    a1 = out[out["archetype"] == "A1"]
    print(a1[["target", "n", "auc_irl", "auc_rf", "auc_avg", "auc_stack", "gain_vs_avg", "gain_vs_best"]].to_string(index=False))


if __name__ == "__main__":
    main()
