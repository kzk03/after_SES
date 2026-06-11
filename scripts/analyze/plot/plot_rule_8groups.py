"""8 グループ (活動量 × 承諾率 × 直近活動) の AUC 比較図 (PDF)。

train_0-3m × eval_0-3m 単一セル。
特に A-C+R+ (低活動・承諾派・直近あり) で IRL の +15pt 優位を強調。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 10


def parse_ci(s):
    s = str(s).strip("[]")
    lo, hi = s.split(",")
    return float(lo), float(hi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path,
                   default=Path("outputs/analysis/rule_based_groups_volat_t0e0/result_g8.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("outputs/analysis/rule_based_groups_volat_t0e0/fig_8groups.pdf"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["group"] != "ALL"].reset_index(drop=True)
    # 順序: 高活動から → A+ で始まる順、R+ を前に
    order = ["A+C+R+", "A+C+R-", "A+C-R+", "A+C-R-",
             "A-C+R+", "A-C+R-", "A-C-R+", "A-C-R-"]
    df["sort_key"] = df["group"].map({g: i for i, g in enumerate(order)})
    df = df.sort_values("sort_key").reset_index(drop=True)

    labels = [
        "High-act\nAccepter\nRecent",
        "High-act\nAccepter\nDormant",
        "High-act\nRejecter\nRecent",
        "High-act\nRejecter\nDormant",
        "Low-act\nAccepter\nRecent",
        "Low-act\nAccepter\nDormant",
        "Low-act\nRejecter\nRecent",
        "Low-act\nRejecter\nDormant",
    ]
    n = len(df)
    x = np.arange(n)
    w = 0.27

    def errs(col):
        lo, hi = [], []
        for s in df[col]:
            try:
                a, b = parse_ci(s); lo.append(a); hi.append(b)
            except Exception:
                lo.append(np.nan); hi.append(np.nan)
        return np.array(lo), np.array(hi)

    def yerr(vals, ci):
        lo, hi = ci
        return np.vstack([np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)])

    irl = df["auc_irl"].to_numpy(); rf = df["auc_rf"].to_numpy(); avg = df["auc_avg"].to_numpy()
    irl_ci, rf_ci, avg_ci = errs("irl_ci"), errs("rf_ci"), errs("avg_ci")

    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.bar(x - w, irl, w, yerr=yerr(irl, irl_ci), label="Cold IRL", color="#4A90D9", capsize=3)
    ax.bar(x,     rf,  w, yerr=yerr(rf,  rf_ci),  label="RF_Dir",   color="#E07B39", capsize=3)
    ax.bar(x + w, avg, w, yerr=yerr(avg, avg_ci), label="Ensemble (AVG)", color="#5BA85B", capsize=3)

    # significance markers
    for i in range(n):
        p_ir = df["p_ir"].iloc[i]
        if pd.notna(p_ir) and p_ir < 0.05:
            y = max(irl[i], rf[i]) + 0.04
            ax.plot([x[i]-w, x[i]], [y, y], "k-", lw=0.8)
            star = "★★" if p_ir < 0.01 else "★"
            ax.text(x[i]-w/2, y + 0.005, star, ha="center", fontsize=9)

    # n labels and group separator
    for i in range(n):
        np_ = int(df["n_pairs"].iloc[i])
        nd_ = int(df["n_devs"].iloc[i])
        ax.text(i, 0.38, f"n={np_}\n({nd_} devs)", ha="center", va="bottom", fontsize=7, color="gray")

    # divider between High-activity and Low-activity halves
    ax.axvline(3.5, color="black", lw=0.6, ls="--", alpha=0.5)
    ax.text(1.5, 0.97, "High activity (≥14.5 reviews)", ha="center", fontsize=10, fontweight="bold")
    ax.text(5.5, 0.97, "Low activity (<14.5 reviews)", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.35, 1.02)
    ax.set_title("AUC by 8-group split (activity × accept_rate × recency) — train_0-3m × eval_0-3m (volat)")
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", lw=0.3, alpha=0.5)

    # 注目ボックス: A-C+R+
    idx = df[df["group"] == "A-C+R+"].index[0]
    ax.annotate(f"IRL +{(df['diff_ir'].iloc[idx]*100):.1f}pt\n(p=0.000)",
                xy=(idx, irl[idx]), xytext=(idx-0.5, 0.95),
                fontsize=9, color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

    idx2 = df[df["group"] == "A-C+R-"].index[0]
    ax.annotate(f"RF +{(-df['diff_ir'].iloc[idx2]*100):.1f}pt\n(p=0.030)",
                xy=(idx2, rf[idx2]), xytext=(idx2+0.3, 0.90),
                fontsize=9, color="darkblue",
                arrowprops=dict(arrowstyle="->", color="darkblue", lw=0.8))

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
