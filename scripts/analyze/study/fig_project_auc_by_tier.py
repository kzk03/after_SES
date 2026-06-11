"""32 project の AUC を tier 別に 4 panel grouped bar で表示。

各 project ごとに IRL / RF / Ensemble の 3 bar を並べる。
AUC 算出不能な project (n_pos=0 等) は "N/A" として bar なしで表示。

出力:
  outputs/analysis/per_repo/fig_project_auc_by_tier.pdf
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "sans-serif"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-csv",
                    default="outputs/analysis/per_repo/top_dir_pooled_metrics.csv")
    ap.add_argument("--main-repos-csv", default="data/service_teams_main_repos.csv")
    ap.add_argument("--output", default="outputs/analysis/per_repo/fig_project_auc_by_tier.pdf")
    args = ap.parse_args()

    td = pd.read_csv(args.project_csv)
    mr = pd.read_csv(args.main_repos_csv).dropna(subset=["main_repo"])
    mr["repo"] = mr["main_repo"].astype(str).str.split("/").str[-1]
    main_repos = set(mr["repo"])
    tier_map = mr.set_index("repo")["tier"].to_dict()
    n_review_map = {}
    # n_review を project_stats.csv から merge
    proj_stats = pd.read_csv("outputs/analysis/per_repo_basic_stats/project_stats.csv")
    n_review_map = proj_stats.set_index("repo")["n_review"].to_dict()

    # project-specific のみ
    df = td[td["top_dir"].isin(main_repos)].copy()
    # main_repos のうち td に出現しない project は AUC=NaN で補完
    missing = main_repos - set(df["top_dir"])
    for p in missing:
        df = pd.concat([df, pd.DataFrame([{
            "top_dir": p, "category": "project-specific",
            "n_pair": 0, "n_pos": 0, "n_neg": 0,
            "auc_irl": np.nan, "auc_rf": np.nan, "auc_ens": np.nan,
        }])], ignore_index=True)
    df["tier"] = df["top_dir"].map(tier_map).fillna("?")
    df["n_review"] = df["top_dir"].map(n_review_map).fillna(0).astype(int)

    # tier 別に並べ替え (各 tier 内は n_review 降順)
    tier_order = ["大", "中", "小", "極小"]
    tier_colors_label = {"大": "#2E7D32", "中": "#558B2F", "小": "#E65100", "極小": "#B71C1C"}
    model_colors = {"IRL": "#4C72B0", "RF": "#DD8452", "Ens": "#55A868"}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, tier in enumerate(tier_order):
        ax = axes[idx]
        sub = df[df["tier"] == tier].sort_values("n_review", ascending=False).reset_index(drop=True)
        n_proj = len(sub)
        n_auc_ok = sub["auc_ens"].notna().sum()
        x = np.arange(n_proj)
        width = 0.27
        for i, (key, label) in enumerate([("auc_irl", "IRL"), ("auc_rf", "RF"), ("auc_ens", "Ens")]):
            vals = sub[key].to_numpy()
            offsets = (i - 1) * width
            ax.bar(x + offsets, np.where(np.isnan(vals), 0, vals),
                   width, label=label, color=model_colors[label], alpha=0.85,
                   edgecolor="white", linewidth=0.5)
        # AUC 計算不能の project に "N/A" マーク
        for i, row in sub.iterrows():
            if pd.isna(row["auc_ens"]):
                ax.text(i, 0.05, "N/A\n(n_pos=0)", ha="center", va="bottom",
                        fontsize=8, color="#888", fontweight="bold")
        ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.6)
        ax.set_xticks(x)
        labels = [f"{r}\n(rev={n})" for r, n in zip(sub["top_dir"], sub["n_review"])]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Pooled AUC")
        ax.set_ylim(0, 1.02)
        ax.set_title(
            f"tier {tier} ({n_proj} project, AUC 算出可 {n_auc_ok})",
            color=tier_colors_label[tier], fontsize=11, fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=8, ncol=3)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Project 単位 AUC (tier 別、IRL / RF / Ensemble 並列)", y=1.00, fontsize=13)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
