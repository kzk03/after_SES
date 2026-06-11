"""Multi-horizon stacking: 4 horizon の base model predictions を特徴量にして
target horizon の label を予測する meta-learner。

入力:
  - train_W/eval_W (W in 0-3, 3-6, 6-9, 9-12) の pair_predictions.csv
  - 各 (dev, dir) について 8 次元特徴量:
    [irl_0-3, irl_3-6, irl_6-9, irl_9-12, rf_0-3, rf_3-6, rf_6-9, rf_9-12]

target: 4 個の label_W (各 horizon の正解)
学習: target horizon ごとに別 logistic regression (5-fold CV)

ベースライン:
  - IRL 単独 (train_h モデルの予測)
  - RF 単独
  - Ensemble (uniform 0.5)
  - Multi-horizon stacking (本提案)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
RNG = np.random.default_rng(777)


def load_diagonal(base: Path, variant: str = "lstm_baseline") -> pd.DataFrame:
    dfs = []
    for h in HORIZONS:
        p = base / variant / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        if not p.exists():
            print(f"  missing: {p}")
            continue
        d = pd.read_csv(p)
        d = d.rename(columns={
            "label": f"label_{h}",
            "irl_dir_prob_calibrated": f"irl_{h}",
            "rf_dir_prob": f"rf_{h}",
        })[["developer", "directory", f"label_{h}", f"irl_{h}", f"rf_{h}"]]
        dfs.append(d)
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["developer", "directory"], how="outer")
    return merged


def evaluate_horizon(
    df: pd.DataFrame, target_h: str, feature_cols: list[str],
    n_splits: int = 5, meta: str = "lr",
):
    """target horizon の label を、feature_cols で予測。CV で AUC を出す。

    また同時に baseline (uniform 0.5 / IRL 単独 / RF 単独) も同 fold で計算。
    """
    label_col = f"label_{target_h}"
    sub = df.dropna(subset=feature_cols + [label_col]).copy()
    X = sub[feature_cols].to_numpy()
    y = sub[label_col].astype(int).to_numpy()
    irl_solo = sub[f"irl_{target_h}"].to_numpy()
    rf_solo = sub[f"rf_{target_h}"].to_numpy()
    avg_uniform = (irl_solo + rf_solo) / 2

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=777)
    aucs_stack, aucs_irl, aucs_rf, aucs_avg = [], [], [], []
    prs_stack, prs_avg = [], []
    preds_stack = np.zeros(len(y))

    for tr, te in skf.split(X, y):
        if meta == "lr":
            clf = LogisticRegression(C=1.0, max_iter=500, random_state=777)
        elif meta == "gb":
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=777,
            )
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        preds_stack[te] = p
        aucs_stack.append(roc_auc_score(y[te], p))
        prs_stack.append(average_precision_score(y[te], p))
        aucs_irl.append(roc_auc_score(y[te], irl_solo[te]))
        aucs_rf.append(roc_auc_score(y[te], rf_solo[te]))
        aucs_avg.append(roc_auc_score(y[te], avg_uniform[te]))
        prs_avg.append(average_precision_score(y[te], avg_uniform[te]))

    # 全データで再学習して coef を見る (interpretability)
    if meta == "lr":
        clf_full = LogisticRegression(C=1.0, max_iter=500, random_state=777)
        clf_full.fit(X, y)
        coefs = dict(zip(feature_cols, clf_full.coef_.flatten()))
        intercept = float(clf_full.intercept_[0])
    else:
        clf_full = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=777)
        clf_full.fit(X, y)
        coefs = dict(zip(feature_cols, clf_full.feature_importances_))
        intercept = None

    return {
        "target_horizon": target_h,
        "n": len(y), "n_pos": int(y.sum()),
        "auc_irl_solo": np.mean(aucs_irl),
        "auc_rf_solo": np.mean(aucs_rf),
        "auc_avg_uniform": np.mean(aucs_avg),
        "auc_stack_cv_mean": np.mean(aucs_stack),
        "auc_stack_cv_std": np.std(aucs_stack),
        "pr_stack_cv_mean": np.mean(prs_stack),
        "pr_avg_uniform": np.mean(prs_avg),
        "gain_vs_avg": np.mean(aucs_stack) - np.mean(aucs_avg),
        "gain_vs_best_single": np.mean(aucs_stack) - max(np.mean(aucs_irl), np.mean(aucs_rf)),
        "coefs": coefs,
        "intercept": intercept,
        "preds_stack": preds_stack,
        "labels": y,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path("outputs/main32_mce_volat_ep150"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/analysis/multi_horizon_stacking_volat_ep150"))
    ap.add_argument("--meta", choices=["lr", "gb"], default="lr",
                    help="meta-learner: lr=logistic, gb=gradient boosting")
    args = ap.parse_args()

    print(f"[load] diagonal from {args.base}")
    w = load_diagonal(args.base)
    print(f"  pairs: {len(w)}")

    feature_cols = [f"irl_{h}" for h in HORIZONS] + [f"rf_{h}" for h in HORIZONS]
    print(f"[features] {feature_cols}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in HORIZONS:
        if f"label_{h}" not in w.columns:
            print(f"  skip {h}: missing label column")
            continue
        r = evaluate_horizon(w, h, feature_cols, meta=args.meta)
        rows.append(r)

    # 表示
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 200)
    table = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("coefs", "preds_stack", "labels", "intercept")}
        for r in rows
    ])
    print("\n=== Multi-horizon stacking (5-fold CV) ===")
    print(table.to_string(index=False))

    # coefs
    print("\n=== meta-learner coefficients (full fit) ===")
    for r in rows:
        print(f"\n  target = {r['target_horizon']}  (intercept={r['intercept']})" if r['intercept'] is not None else f"\n  target = {r['target_horizon']}  (feature importances)")
        for f, c in sorted(r["coefs"].items(), key=lambda x: -abs(x[1])):
            print(f"    {f:>12}  {c:+.4f}")

    # save table
    table.to_csv(args.out_dir / "stacking_results.csv", index=False)

    # per-horizon predictions
    for r in rows:
        h = r["target_horizon"]
        df_h = pd.DataFrame({"label": r["labels"], "pred_stack": r["preds_stack"]})
        df_h.to_csv(args.out_dir / f"preds_target_{h}.csv", index=False)

    print(f"\nwrote: {args.out_dir}")


if __name__ == "__main__":
    main()
