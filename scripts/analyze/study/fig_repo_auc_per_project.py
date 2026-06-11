"""全 32 project の repo (depth=2 path) 別 AUC を bar で表示。

各 panel = 1 project。
panel 内: 当該 project に属する repo × IRL/RF/Ens の 3 bar grouped。
AUC 計算不能な repo は N/A マーク。

出力:
  outputs/analysis/per_repo/fig_repo_auc_per_project.pdf
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
    ap.add_argument("--repo-csv",
                    default="outputs/analysis/per_dir_all/directory_pooled_metrics.csv")
    ap.add_argument("--main-repos-csv", default="data/service_teams_main_repos.csv")
    ap.add_argument("--output", default="outputs/analysis/per_repo/fig_repo_auc_per_project.pdf")
    args = ap.parse_args()

    repo_df = pd.read_csv(args.repo_csv)
    mr = pd.read_csv(args.main_repos_csv).dropna(subset=["main_repo"])
    mr["project"] = mr["main_repo"].astype(str).str.split("/").str[-1]
    tier_map = mr.set_index("project")["tier"].to_dict()

    # 全 32 main project (tier 別 → tier 内 n_review 降順)
    main_projects = list(mr["project"])
    proj_stats = pd.read_csv("outputs/analysis/per_repo_basic_stats/project_stats.csv")
    n_review_map = proj_stats.set_index("repo")["n_review"].to_dict()
    tier_order = {"大": 0, "中": 1, "小": 2, "極小": 3}
    main_projects.sort(key=lambda p: (
        tier_order.get(tier_map.get(p, "?"), 9),
        -n_review_map.get(p, 0),
    ))

    # repo を top_dir で group
    repo_df["top_dir"] = repo_df["directory"].astype(str).str.split("/").str[0]
    by_project = {p: repo_df[repo_df["top_dir"] == p].copy() for p in main_projects}

    tier_colors = {"大": "#2E7D32", "中": "#558B2F", "小": "#E65100", "極小": "#B71C1C", "?": "#666"}
    model_colors = {"IRL": "#4C72B0", "RF": "#DD8452", "Ens": "#55A868"}

    n_proj = len(main_projects)  # 32
    ncol = 4
    nrow = 8
    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 32))
    axes = axes.flatten()

    for idx, project in enumerate(main_projects):
        ax = axes[idx]
        sub = by_project.get(project, pd.DataFrame()).copy()
        tier = tier_map.get(project, "?")
        if len(sub) == 0:
            ax.text(0.5, 0.5, "pair データなし",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888", fontweight="bold")
            ax.set_title(f"{project} (tier {tier}) — pair 無し",
                         color=tier_colors[tier], fontsize=10, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        sub = sub.sort_values("n_pair", ascending=False).reset_index(drop=True)
        n_repo = len(sub)
        n_auc_ok = sub["auc_ens"].notna().sum()
        x = np.arange(n_repo)
        width = 0.27

        for i, (key, label) in enumerate([("auc_irl", "IRL"), ("auc_rf", "RF"), ("auc_ens", "Ens")]):
            vals = sub[key].to_numpy()
            offsets = (i - 1) * width
            ax.bar(x + offsets, np.where(np.isnan(vals), 0, vals),
                   width, label=label, color=model_colors[label], alpha=0.85,
                   edgecolor="white", linewidth=0.4)
        # N/A マーク
        for i, row in sub.iterrows():
            if pd.isna(row["auc_ens"]):
                ax.text(i, 0.05, "N/A",
                        ha="center", va="bottom", fontsize=7,
                        color="#888", fontweight="bold")
        ax.axhline(0.5, color="gray", lw=0.4, ls=":", alpha=0.6)
        ax.set_xticks(x)
        # 短縮ラベル: directory から project 名を省いて sub-dir のみ表示
        short_labels = []
        for d in sub["directory"]:
            d = str(d)
            if d.startswith(f"{project}/"):
                short = d[len(project) + 1:]  # nova/compute → compute
            else:
                short = d
            short_labels.append(f"{short}\n(n={int(sub.loc[short_labels.__len__(), 'n_pair'])})")
        # 上の参照が壊れるので別ループで安全に
        short_labels = []
        for i, row in sub.iterrows():
            d = str(row["directory"])
            short = d[len(project) + 1:] if d.startswith(f"{project}/") else d
            short_labels.append(f"{short}\n(n={int(row['n_pair'])})")
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC", fontsize=8)
        ax.set_title(
            f"{project} (tier {tier}, {n_repo} repo, AUC可 {n_auc_ok})",
            color=tier_colors[tier], fontsize=9, fontweight="bold",
        )
        ax.tick_params(axis="y", labelsize=7)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=7, ncol=3)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "全 32 project の repo (depth=2) 別 AUC — IRL / RF / Ensemble",
        y=1.00, fontsize=13,
    )
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
