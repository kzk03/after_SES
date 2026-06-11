"""dev 散布図 (活動量 × 承諾率) と 4 グループ重ね合わせ (PDF)。

threshold (median activity, accept 0.5) を点線で示し、ルール分類の妥当性を視覚化。
直近活動 (recent_active) の有無を marker shape で。
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path,
                   default=Path("outputs/analysis/rule_based_groups_volat_t0e0/dev_groups.csv"))
    p.add_argument("--out", type=Path,
                   default=Path("outputs/analysis/rule_based_groups_volat_t0e0/fig_dev_scatter.pdf"))
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    print(f"loaded {len(df)} devs")

    med_act = df["total_reviews"].median()

    colors = {
        "G1_ActiveAccepter": "#2E7D32",
        "G2_ActiveRejecter": "#C62828",
        "G3_CasualAccepter": "#66BB6A",
        "G4_CasualRejecter": "#EF9A9A",
    }
    labels = {
        "G1_ActiveAccepter": "G1 Active Accepter",
        "G2_ActiveRejecter": "G2 Active Rejecter",
        "G3_CasualAccepter": "G3 Casual Accepter",
        "G4_CasualRejecter": "G4 Casual Rejecter",
    }

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # log-scale total_reviews + jitter accept_rate に
    np.random.seed(7)
    jitter = np.random.uniform(-0.015, 0.015, size=len(df))

    for g, sub in df.groupby("group4"):
        for recent_flag, marker, lbl_suffix in [(True, "o", " (recent)"), (False, "x", " (dormant)")]:
            ss = sub[sub["recent_active"] == recent_flag]
            if ss.empty:
                continue
            ax.scatter(
                np.maximum(ss["total_reviews"], 0.5),  # avoid 0 for log
                ss["overall_acceptance_rate"] + jitter[ss.index],
                s=35, c=colors[g], marker=marker, alpha=0.65,
                edgecolors="white", linewidths=0.4,
                label=f"{labels[g]}{lbl_suffix}",
            )

    # threshold lines
    ax.axvline(med_act, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.7)
    ax.text(med_act, 1.05, f"activity median = {med_act:.0f}",
            ha="center", fontsize=8, color="black")
    ax.text(0.5, 0.51, "accept = 0.5", ha="left", fontsize=8, color="black")

    # group counts in each quadrant
    counts = df["group4"].value_counts()
    quadrant_pos = {
        "G1_ActiveAccepter": (max(df["total_reviews"])*0.6, 0.92),
        "G2_ActiveRejecter": (max(df["total_reviews"])*0.6, 0.08),
        "G3_CasualAccepter": (1.0, 0.92),
        "G4_CasualRejecter": (1.0, 0.08),
    }
    for g, (x, y) in quadrant_pos.items():
        ax.text(x, y, f"n={counts.get(g, 0)}", fontsize=11, fontweight="bold",
                color=colors[g], ha="center",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[g], alpha=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("total_reviews in 180-day window (log scale)")
    ax.set_ylabel("overall_acceptance_rate")
    ax.set_title(f"Developer distribution by activity × accept_rate (n={len(df)} active devs)\n"
                 f"feature_end=2022-01-01, train_0-3m")
    ax.set_ylim(-0.05, 1.10)
    ax.grid(lw=0.3, alpha=0.4)
    ax.legend(loc="center right", fontsize=8, bbox_to_anchor=(1.45, 0.5))

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
