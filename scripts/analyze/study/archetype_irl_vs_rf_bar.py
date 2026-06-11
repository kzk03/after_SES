# -*- coding: utf-8 -*-
"""Archetype 別 IRL vs RF の per-dev / per-pair AUC 比較 bar chart.

各 archetype で IRL と RF の mean AUC を並べ、Wilcoxon paired test の p 値を注釈。
diamond 散布図と組み合わせて読む motivation 図。
"""
from __future__ import annotations

import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ARCH_LABELS = {
    "A1": "A1\nSporadic\nNewcomer",
    "A2": "A2\nStrict\nReviewer-Only",
    "A3": "A3\nApproving\nVeteran",
    "A4": "A4\nSaturated\nActive",
}


def stars(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return f"p={p:.3f}**"
    if p < 0.05:
        return f"p={p:.3f}*"
    return f"p={p:.2f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=pathlib.Path)
    p.add_argument("--out", required=True, type=pathlib.Path)
    p.add_argument("--unit", choices=["dev", "pair"], default="dev")
    p.add_argument("--irl-col", default="auc_irl")
    p.add_argument("--rf-col", default="auc_rf")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna(subset=[args.irl_col, args.rf_col, "dominant"])

    archetypes = ["A1", "A2", "A3", "A4"]
    means_irl, means_rf, sems_irl, sems_rf, pvals, ns = [], [], [], [], [], []
    for a in archetypes:
        sub = df[df["dominant"] == a]
        ns.append(len(sub))
        if len(sub) < 5:
            means_irl.append(np.nan); means_rf.append(np.nan)
            sems_irl.append(0); sems_rf.append(0); pvals.append(np.nan)
            continue
        means_irl.append(sub[args.irl_col].mean())
        means_rf.append(sub[args.rf_col].mean())
        sems_irl.append(sub[args.irl_col].sem())
        sems_rf.append(sub[args.rf_col].sem())
        try:
            _, pval = wilcoxon(sub[args.irl_col], sub[args.rf_col])
        except Exception:
            pval = np.nan
        pvals.append(pval)

    # overall
    overall_irl = df[args.irl_col].mean()
    overall_rf = df[args.rf_col].mean()
    try:
        _, p_all = wilcoxon(df[args.irl_col], df[args.rf_col])
    except Exception:
        p_all = np.nan

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(archetypes) + 1)  # +1 for ALL
    w = 0.38
    all_irl = means_irl + [overall_irl]
    all_rf = means_rf + [overall_rf]
    all_sem_irl = sems_irl + [df[args.irl_col].sem()]
    all_sem_rf = sems_rf + [df[args.rf_col].sem()]
    all_pvals = pvals + [p_all]
    all_ns = ns + [len(df)]
    labels = [ARCH_LABELS[a] for a in archetypes] + [f"ALL\n{args.unit}s"]

    b1 = ax.bar(x - w / 2, all_irl, w, yerr=all_sem_irl, capsize=3,
                color="#E64B35", alpha=0.85, label="IRL_cold")
    b2 = ax.bar(x + w / 2, all_rf, w, yerr=all_sem_rf, capsize=3,
                color="#4DBBD5", alpha=0.85, label="RF_dir")

    for i, (m_irl, m_rf, pv, n) in enumerate(zip(all_irl, all_rf, all_pvals, all_ns)):
        if np.isnan(m_irl):
            continue
        top = max(m_irl, m_rf) + 0.03
        sig = stars(pv)
        diff = m_irl - m_rf
        ax.text(i, top, f"{sig}\nΔ={diff:+.3f}", ha="center", va="bottom",
                fontsize=8, color="black" if pv >= 0.05 else "red")
        ax.text(i, 0.02, f"n={n}", ha="center", va="bottom",
                fontsize=7, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(f"Per-{args.unit} AUC (mean ± SE)", fontsize=10)
    ax.set_title(f"Per-{args.unit} AUC by archetype: IRL_cold vs RF_dir "
                 f"(paired Wilcoxon, inclusive sampling)", fontsize=11)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.6, zorder=0)
    ax.text(len(x) - 0.5, 0.5, "chance", color="gray", fontsize=7,
            va="bottom", ha="right")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(args.out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
