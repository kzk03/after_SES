"""train_0-3m × eval_0-3m 単一セルで archetype 別 AUC を bootstrap CI 付きで出す。

「真の価値」を測る論文 Section 4 用の表生成。
- 各 archetype × {Cold IRL, RF, Ensemble (uniform 0.5)} の AUC / PR
- bootstrap 1,000 で 95% CI
- IRL vs RF, AVG vs best-single の paired bootstrap p-value
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

RNG = np.random.default_rng(777)


def auc_pair(y, s):
    if pd.Series(y).nunique() < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, s), average_precision_score(y, s)


def bootstrap_ci(y, s, n_boot=1000, alpha=0.05):
    y = np.asarray(y); s = np.asarray(s)
    n = len(y)
    aucs, prs = [], []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yb = y[idx]; sb = s[idx]
        if len(set(yb)) < 2:
            continue
        aucs.append(roc_auc_score(yb, sb))
        prs.append(average_precision_score(yb, sb))
    aucs = np.array(aucs); prs = np.array(prs)
    auc_lo, auc_hi = np.percentile(aucs, [100*alpha/2, 100*(1-alpha/2)])
    pr_lo, pr_hi = np.percentile(prs, [100*alpha/2, 100*(1-alpha/2)])
    return (auc_lo, auc_hi), (pr_lo, pr_hi)


def paired_bootstrap_pvalue(y, s_a, s_b, n_boot=1000):
    """H0: AUC(a) = AUC(b). 片側 p (a > b の証拠)。"""
    y = np.asarray(y); s_a = np.asarray(s_a); s_b = np.asarray(s_b)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yb = y[idx]
        if len(set(yb)) < 2:
            continue
        diffs.append(roc_auc_score(yb, s_a[idx]) - roc_auc_score(yb, s_b[idx]))
    diffs = np.array(diffs)
    # 両側 p
    return 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), diffs.mean(), np.percentile(diffs, [2.5, 97.5])


def analyze_archetype(sub: pd.DataFrame, arch: str):
    y = sub["label"].astype(int).to_numpy()
    irl = sub["irl_dir_prob_calibrated"].to_numpy()
    rf = sub["rf_dir_prob"].to_numpy()
    avg = (irl + rf) / 2

    a_irl, p_irl = auc_pair(y, irl)
    a_rf, p_rf = auc_pair(y, rf)
    a_avg, p_avg = auc_pair(y, avg)

    (irl_auc_ci, irl_pr_ci) = bootstrap_ci(y, irl)
    (rf_auc_ci, rf_pr_ci) = bootstrap_ci(y, rf)
    (avg_auc_ci, avg_pr_ci) = bootstrap_ci(y, avg)

    p_irl_vs_rf, mean_diff_ir, ci_ir = paired_bootstrap_pvalue(y, irl, rf)
    best_single = irl if a_irl >= a_rf else rf
    best_name = "IRL" if a_irl >= a_rf else "RF"
    p_avg_vs_best, mean_diff_ab, ci_ab = paired_bootstrap_pvalue(y, avg, best_single)

    return {
        "archetype": arch, "n": len(sub), "n_pos": int(y.sum()),
        "auc_irl": a_irl, "auc_irl_lo": irl_auc_ci[0], "auc_irl_hi": irl_auc_ci[1],
        "auc_rf":  a_rf,  "auc_rf_lo":  rf_auc_ci[0],  "auc_rf_hi":  rf_auc_ci[1],
        "auc_avg": a_avg, "auc_avg_lo": avg_auc_ci[0], "auc_avg_hi": avg_auc_ci[1],
        "pr_irl": p_irl, "pr_rf": p_rf, "pr_avg": p_avg,
        "diff_irl_minus_rf": a_irl - a_rf,
        "p_irl_vs_rf": p_irl_vs_rf,
        "best_single": best_name,
        "diff_avg_minus_best": a_avg - max(a_irl, a_rf),
        "p_avg_vs_best": p_avg_vs_best,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", type=Path,
                   default=Path("outputs/main32_mce_volat/lstm_baseline/train_0-3m/eval_0-3m/pair_predictions.csv"))
    p.add_argument("--alpha", type=Path,
                   default=Path("outputs/analysis/archetype_irl_rf_volat_coldonly/k4_dev_alpha_with_metrics.csv"))
    p.add_argument("--output", type=Path,
                   default=Path("outputs/analysis/archetype_irl_rf_volat_coldonly/single_cell_train0-3_eval0-3.csv"))
    args = p.parse_args()

    df = pd.read_csv(args.pred).dropna(subset=["label", "irl_dir_prob_calibrated", "rf_dir_prob"])
    alpha = pd.read_csv(args.alpha, index_col=0)
    alpha.index.name = "developer"
    alpha = alpha.reset_index()
    alpha["hard_archetype"] = alpha["dominant"]
    df = df.merge(alpha[["developer", "hard_archetype"]], on="developer", how="inner")

    print(f"[data] cell={args.pred.parent.name}/{args.pred.parent.parent.name}, n_pairs={len(df)}, n_pos={df['label'].sum()}")

    rows = []
    for arch in sorted(df["hard_archetype"].unique()):
        sub = df[df["hard_archetype"] == arch]
        rows.append(analyze_archetype(sub, arch))
    # ALL
    rows.append(analyze_archetype(df.assign(hard_archetype="ALL"), "ALL"))

    out = pd.DataFrame(rows)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)

    print("\n=== AUC (bootstrap 95% CI) ===")
    disp = out[["archetype", "n", "n_pos",
                "auc_irl", "auc_irl_lo", "auc_irl_hi",
                "auc_rf",  "auc_rf_lo",  "auc_rf_hi",
                "auc_avg", "auc_avg_lo", "auc_avg_hi"]]
    print(disp.to_string(index=False))

    print("\n=== PR AUC ===")
    print(out[["archetype", "pr_irl", "pr_rf", "pr_avg"]].to_string(index=False))

    print("\n=== paired bootstrap tests ===")
    print(out[["archetype", "diff_irl_minus_rf", "p_irl_vs_rf",
               "best_single", "diff_avg_minus_best", "p_avg_vs_best"]].to_string(index=False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
