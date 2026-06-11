"""ルールベース 4 グループ × データドリブン A1-A4 の AUC 比較図 (PDF)。

train_0-3m × eval_0-3m 単一セル、95% bootstrap CI 付きの棒グラフ。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 10


def parse_ci(s):
    s = s.strip("[]")
    lo, hi = s.split(",")
    return float(lo), float(hi)


def make_bar_panel(ax, df, group_col, group_labels, title):
    n = len(df)
    x = np.arange(n)
    w = 0.27

    irl = df["auc_irl"].to_numpy()
    rf  = df["auc_rf"].to_numpy()
    avg = df["auc_avg"].to_numpy()

    # CIs may be in "irl_ci" string format
    def errs(col):
        if col not in df.columns:
            return None
        lo, hi = [], []
        for s in df[col]:
            try:
                a, b = parse_ci(str(s))
                lo.append(a); hi.append(b)
            except Exception:
                lo.append(np.nan); hi.append(np.nan)
        return np.array(lo), np.array(hi)

    def yerr(vals, ci):
        lo, hi = ci
        return np.vstack([vals - lo, hi - vals])

    irl_ci = errs("irl_ci")
    rf_ci  = errs("rf_ci")
    avg_ci = errs("avg_ci")

    b_irl = ax.bar(x - w, irl, w, yerr=(yerr(irl, irl_ci) if irl_ci else None),
                   label="Cold IRL", color="#4A90D9", capsize=3)
    b_rf  = ax.bar(x,     rf,  w, yerr=(yerr(rf,  rf_ci)  if rf_ci  else None),
                   label="RF_Dir",   color="#E07B39", capsize=3)
    b_avg = ax.bar(x + w, avg, w, yerr=(yerr(avg, avg_ci) if avg_ci else None),
                   label="Ensemble (AVG)", color="#5BA85B", capsize=3)

    # significance markers (p<0.05) for IRL vs RF and AVG vs best
    if "p_ir" in df.columns:
        for i, p in enumerate(df["p_ir"]):
            if pd.notna(p) and p < 0.05:
                y = max(irl[i], rf[i]) + 0.04
                ax.plot([x[i]-w, x[i]], [y, y], "k-", lw=0.8)
                star = "★★" if p < 0.01 else "★"
                ax.text(x[i]-w/2, y + 0.005, star, ha="center", fontsize=8)
    if "p_avg_best" in df.columns:
        for i, p in enumerate(df["p_avg_best"]):
            if pd.notna(p) and p < 0.05:
                y = max(irl[i], rf[i], avg[i]) + 0.09
                ax.plot([x[i], x[i]+w], [y, y], "k-", lw=0.8)
                star = "★★" if p < 0.01 else "★"
                ax.text(x[i]+w/2, y + 0.005, star, ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.45, 0.98)
    ax.set_title(title)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", lw=0.3, alpha=0.5)


def annotate_n(ax, df, group_col):
    """各バー群の下に n_pairs / n_devs を小さく表示。"""
    n = len(df)
    for i in range(n):
        np_ = int(df["n_pairs"].iloc[i])
        nd_raw = df["n_devs"].iloc[i] if "n_devs" in df.columns else None
        try:
            nd_ = int(nd_raw) if nd_raw not in (None, "", np.nan) else None
        except (ValueError, TypeError):
            nd_ = None
        txt = f"n={np_}" if nd_ is None else f"n={np_}\n({nd_} devs)"
        ax.text(i, 0.46, txt, ha="center", va="bottom", fontsize=7, color="gray")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rule-csv", type=Path,
                   default=Path("outputs/analysis/rule_based_groups_volat_t0e0/result_g4.csv"))
    p.add_argument("--data-csv", type=Path,
                   default=Path("outputs/analysis/archetype_irl_rf_volat_coldonly/single_cell_train0-3_eval0-3.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("outputs/analysis/rule_based_groups_volat_t0e0/fig_rule_vs_archetype.pdf"))
    args = p.parse_args()

    rule = pd.read_csv(args.rule_csv)
    rule = rule[rule["group"] != "ALL"].reset_index(drop=True)

    data = pd.read_csv(args.data_csv)
    data = data[data["archetype"] != "ALL"].reset_index(drop=True)
    # build ci strings from archetype_single_cell.py columns (numeric -> string)
    if "irl_ci" not in data.columns and "auc_irl_lo" in data.columns:
        for k, lo, hi in [("irl_ci","auc_irl_lo","auc_irl_hi"),
                          ("rf_ci", "auc_rf_lo", "auc_rf_hi"),
                          ("avg_ci","auc_avg_lo","auc_avg_hi")]:
            data[k] = data.apply(lambda r: f"[{r[lo]:.3f},{r[hi]:.3f}]", axis=1)
    # rename for unified plot
    data = data.rename(columns={
        "diff_irl_minus_rf": "diff_ir",
        "p_irl_vs_rf": "p_ir",
        "diff_avg_minus_best": "diff_avg_best",
        "p_avg_vs_best": "p_avg_best",
        "n": "n_pairs",
    })
    data["n_devs"] = ""  # archetype side: per-dev count is in alpha file, skip

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    rule_labels = [
        "G1 Active\nAccepter", "G2 Active\nRejecter",
        "G3 Casual\nAccepter", "G4 Casual\nRejecter",
    ]
    # ensure G1..G4 order
    rule["sort_key"] = rule["group"].str.extract(r"G(\d)").astype(int)
    rule = rule.sort_values("sort_key").reset_index(drop=True)

    arch_labels = [
        "A1 Sporadic\nNewcomer", "A2 Strict\nReviewer-Only",
        "A3 Approving\nVeteran", "A4 Saturated\nActive",
    ]
    data = data.sort_values("archetype").reset_index(drop=True)

    make_bar_panel(axes[0], rule, "group", rule_labels,
                   "(a) Rule-based 4 groups (activity × accept_rate)")
    annotate_n(axes[0], rule, "group")

    make_bar_panel(axes[1], data, "archetype", arch_labels,
                   "(b) Data-driven Archetypal Analysis (K=4)")
    annotate_n(axes[1], data, "archetype")

    fig.suptitle("AUC by reviewer type — train_0-3m × eval_0-3m (volat, n_pair=3,677 / 3,747)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
