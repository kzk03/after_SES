"""project 単位 (32) と repo 単位 (390) の AUC 分布を比較する図を生成。

- 左 panel: project 単位 (32 全件、うち AUC 計算可は 26、不能 6 は注記)
- 右 panel: repo 単位 (390 全件、うち AUC 計算可は 286、不能 104 は注記)
- 各 panel: IRL / RF / Ensemble の 3 violin を並列
- AUC reliable (n_pos>=30 & n_neg>=30) の境界線も併記

出力:
  outputs/analysis/per_repo/fig_project_vs_repo_auc.pdf
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
    ap.add_argument("--repo-csv",
                    default="outputs/analysis/per_dir_all/directory_pooled_metrics.csv")
    ap.add_argument("--main-repos-csv", default="data/service_teams_main_repos.csv")
    ap.add_argument("--output", default="outputs/analysis/per_repo/fig_project_vs_repo_auc.pdf")
    args = ap.parse_args()

    # ── project (32 main_repos)
    td = pd.read_csv(args.project_csv)
    mr = pd.read_csv(args.main_repos_csv).dropna(subset=["main_repo"])
    main_repos = set(mr["main_repo"].astype(str).str.split("/").str[-1])
    # main_repos に一致するもののみ抽出 (cross-cutting を除外)
    proj = td[td["top_dir"].isin(main_repos)].copy()
    # main_repos のうち td に出現しない project も AUC=NaN として補完 (32 全件にする)
    missing_proj = main_repos - set(proj["top_dir"])
    for p in missing_proj:
        proj = pd.concat([proj, pd.DataFrame([{
            "top_dir": p, "category": "project-specific",
            "n_pair": 0, "n_pos": 0, "n_neg": 0,
            "auc_irl": np.nan, "auc_rf": np.nan, "auc_ens": np.nan,
        }])], ignore_index=True)
    n_proj_total = len(proj)
    n_proj_auc_ok = proj["auc_ens"].notna().sum()
    n_proj_auc_na = n_proj_total - n_proj_auc_ok

    # ── repo (390 全 directory)
    repo = pd.read_csv(args.repo_csv)
    n_repo_total = len(repo)
    n_repo_auc_ok = repo["auc_ens"].notna().sum()
    n_repo_auc_na = n_repo_total - n_repo_auc_ok

    print(f"project: 全 {n_proj_total} / AUC 算出可 {n_proj_auc_ok} / 計算不能 {n_proj_auc_na}")
    print(f"repo:    全 {n_repo_total} / AUC 算出可 {n_repo_auc_ok} / 計算不能 {n_repo_auc_na}")

    model_cols = ["auc_irl", "auc_rf", "auc_ens"]
    model_labels = ["IRL", "RF", "Ensemble"]
    model_colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    def plot_panel(ax, df, n_total, n_ok, n_na, title):
        positions = [1, 2, 3]
        data_list = [df[c].dropna().to_numpy() for c in model_cols]
        parts = ax.violinplot(data_list, positions=positions,
                              showmedians=True, widths=0.7)
        for body, color in zip(parts["bodies"], model_colors):
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.6)
        for k in ["cmedians", "cbars", "cmaxes", "cmins"]:
            if k in parts:
                parts[k].set_color("black")
                parts[k].set_linewidth(0.8)
        # 個別点を jitter で重ねる
        for i, (arr, color) in enumerate(zip(data_list, model_colors)):
            jitter = (np.random.default_rng(seed=i).random(len(arr)) - 0.5) * 0.2
            ax.scatter(np.full_like(arr, positions[i]) + jitter, arr,
                       s=12, color="black", alpha=0.35, zorder=3)
            # median 値を表示
            if len(arr) > 0:
                m = float(np.median(arr))
                ax.text(positions[i], m, f" {m:.3f}",
                        fontsize=8, va="center", ha="left", color="darkred")
        ax.axhline(0.5, color="gray", lw=0.5, ls=":", alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [f"{lab}\n(n={len(d)})" for lab, d in zip(model_labels, data_list)],
            fontsize=10,
        )
        ax.set_title(
            f"{title}\n全 {n_total} 中 AUC 算出可 {n_ok}, 計算不能 {n_na}",
            fontsize=11,
        )
        ax.set_ylim(0.0, 1.05)

    plot_panel(axes[0], proj, n_proj_total, n_proj_auc_ok, n_proj_auc_na,
               "Project 単位 (32 main_repos)")
    plot_panel(axes[1], repo, n_repo_total, n_repo_auc_ok, n_repo_auc_na,
               "Repo 単位 (depth=2 path)")
    axes[0].set_ylabel("Pooled AUC")
    fig.suptitle("Project 単位 vs Repo 単位 の AUC 分布比較", y=1.02, fontsize=12)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
