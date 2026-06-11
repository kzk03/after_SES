"""Repo 特性 (basic stats) と予測精度 (AUC) の相関・回帰分析。

集計対象: project_stats.csv (32 repo) + top_dir_pooled_metrics.csv の AUC を repo key で merge。

出力 (outputs/analysis/per_repo_correlation/):
  - merged_repo_metrics.csv     : 各 repo の特性 + AUC のフラットテーブル
  - correlation_matrix.csv      : 数値変数間の Spearman 相関マトリクス
  - regression_summary.csv      : log10(n_review), accept_rate, n_dev で AUC を回帰
  - fig_corr_heatmap.pdf        : 相関マトリクスのヒートマップ
  - fig_accept_vs_auc.pdf       : 承諾率 vs Ens AUC scatter
  - fig_n_review_vs_auc.pdf     : n_review vs IRL/RF/Ens の 3 panel scatter
  - fig_repo_2d_cluster.pdf     : 2D (log n_review × accept_rate) で AUC を色付け
  - README.md                   : 発見サマリ
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "sans-serif"]


def merge_repo_metrics(
    project_stats: Path, top_dir_pooled: Path, main_repos_csv: Path,
) -> pd.DataFrame:
    """basic stats と AUC を repo key で結合。"""
    proj = pd.read_csv(project_stats)
    auc = pd.read_csv(top_dir_pooled)
    auc = auc.rename(columns={"top_dir": "repo"})
    mr = pd.read_csv(main_repos_csv).dropna(subset=["main_repo"])
    mr["repo"] = mr["main_repo"].astype(str).str.split("/").str[-1]
    tier_map = mr.set_index("repo")["tier"].to_dict()

    merged = proj.merge(auc[["repo", "n_pair", "n_pos", "n_neg",
                              "auc_irl", "auc_rf", "auc_ens", "gain_vs_best",
                              "auc_reliable"]],
                          on="repo", how="inner", suffixes=("_raw", "_pair"))
    merged["tier"] = merged["repo"].map(tier_map).fillna("unknown")
    return merged


def correlation_matrix(merged: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Spearman 相関マトリクス。"""
    rho = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pvals = pd.DataFrame(index=cols, columns=cols, dtype=float)
    sub = merged.dropna(subset=cols).copy()
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            r, p = spearmanr(sub[a], sub[b])
            rho.loc[a, b] = r
            pvals.loc[a, b] = p
    return rho.astype(float), pvals.astype(float)


def regression_analysis(merged: pd.DataFrame) -> pd.DataFrame:
    """log10(n_review), accept_rate, log10(n_dev) で IRL/RF/Ens を線形回帰。"""
    rel = merged.dropna(subset=["auc_ens"]).copy()
    rel["log_n_review"] = np.log10(rel["n_review"] + 1)
    rel["log_n_dev"] = np.log10(rel["n_unique_dev"] + 1)
    rows = []
    feats = ["log_n_review", "accept_rate", "log_n_dev"]
    X = rel[feats].to_numpy()
    for target in ["auc_irl", "auc_rf", "auc_ens", "gain_vs_best"]:
        y = rel[target].to_numpy()
        lr = LinearRegression().fit(X, y)
        pred = lr.predict(X)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        row = {"target": target, "r2": r2, "intercept": float(lr.intercept_)}
        for f, c in zip(feats, lr.coef_):
            row[f"coef_{f}"] = float(c)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# 可視化
# ----------------------------------------------------------------------

def fig_corr_heatmap(rho: pd.DataFrame, pvals: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(rho.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rho.columns)))
    ax.set_xticklabels(rho.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rho.index)))
    ax.set_yticklabels(rho.index, fontsize=8)
    for i in range(rho.shape[0]):
        for j in range(rho.shape[1]):
            v = rho.values[i, j]
            p = pvals.values[i, j]
            star = "*" if p < 0.05 else ""
            ax.text(j, i, f"{v:.2f}{star}", ha="center", va="center",
                    fontsize=7, color="black" if abs(v) < 0.6 else "white")
    plt.colorbar(im, ax=ax, label="Spearman ρ")
    ax.set_title("Repo 特性 × AUC の Spearman 相関 (* = p<0.05)")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


def _fig_accept_vs_auc_single(
    merged: pd.DataFrame, out: Path, model_col: str, model_label: str,
) -> None:
    rel = merged.dropna(subset=[model_col]).copy()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    tier_colors = {
        "大": "#2E7D32", "中": "#558B2F", "小": "#E65100", "極小": "#B71C1C",
        "unknown": "#888",
    }
    for tier, c in tier_colors.items():
        sub = rel[rel["tier"] == tier]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["accept_rate"], sub[model_col],
            s=np.log10(sub["n_review"] + 1) * 30 + 20,
            c=c, alpha=0.7, edgecolors="white", linewidths=0.6,
            label=f"tier {tier} (n={len(sub)})",
        )
    for _, r in rel.iterrows():
        ax.annotate(
            r["repo"], (r["accept_rate"], r[model_col]),
            fontsize=7, xytext=(3, 3), textcoords="offset points", alpha=0.85,
        )
    rho, p = spearmanr(rel["accept_rate"], rel[model_col])
    ax.text(
        0.02, 0.98,
        f"Spearman ρ = {rho:.3f}, p = {p:.3g}\nn = {len(rel)} repos",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
    )
    ax.set_xlabel("Project 承諾率 (raw review-level)")
    ax.set_ylabel(f"Pooled {model_label} AUC")
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"承諾率 vs {model_label} AUC (バブル = log10 n_review)")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


def fig_accept_vs_auc(merged: pd.DataFrame, out: Path) -> None:
    """Ens 版 (既存名互換)。"""
    _fig_accept_vs_auc_single(merged, out, "auc_ens", "Ensemble")


def fig_accept_vs_auc_irl(merged: pd.DataFrame, out: Path) -> None:
    _fig_accept_vs_auc_single(merged, out, "auc_irl", "IRL")


def fig_accept_vs_auc_rf(merged: pd.DataFrame, out: Path) -> None:
    _fig_accept_vs_auc_single(merged, out, "auc_rf", "RF")


def fig_accept_vs_auc_combined(merged: pd.DataFrame, out: Path) -> None:
    """IRL/RF/Ens を横並びの 3 panel で比較。"""
    rel = merged.dropna(subset=["auc_irl", "auc_rf", "auc_ens"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    tier_colors = {
        "大": "#2E7D32", "中": "#558B2F", "小": "#E65100", "極小": "#B71C1C",
        "unknown": "#888",
    }
    for ax, model_col, model_label in [
        (axes[0], "auc_irl", "IRL"),
        (axes[1], "auc_rf", "RF"),
        (axes[2], "auc_ens", "Ensemble"),
    ]:
        for tier, c in tier_colors.items():
            sub = rel[rel["tier"] == tier]
            if len(sub) == 0:
                continue
            ax.scatter(
                sub["accept_rate"], sub[model_col],
                s=np.log10(sub["n_review"] + 1) * 30 + 20,
                c=c, alpha=0.7, edgecolors="white", linewidths=0.6,
                label=f"tier {tier} (n={len(sub)})" if ax is axes[0] else None,
            )
        for _, r in rel.iterrows():
            ax.annotate(
                r["repo"], (r["accept_rate"], r[model_col]),
                fontsize=6, xytext=(3, 3), textcoords="offset points", alpha=0.75,
            )
        rho, p = spearmanr(rel["accept_rate"], rel[model_col])
        ax.text(
            0.02, 0.98,
            f"ρ = {rho:.3f}\np = {p:.3g}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
        )
        ax.set_xlabel("Project 承諾率 (raw)")
        ax.set_title(f"承諾率 vs {model_label} AUC")
        ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    axes[0].set_ylabel("Pooled AUC")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"承諾率 vs IRL / RF / Ensemble AUC (n = {len(rel)} repos)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_n_review_vs_auc(merged: pd.DataFrame, out: Path) -> None:
    rel = merged.dropna(subset=["auc_ens"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, target, color in [
        (axes[0], "auc_irl", "#4C72B0"),
        (axes[1], "auc_rf", "#DD8452"),
        (axes[2], "auc_ens", "#55A868"),
    ]:
        sub = rel.dropna(subset=[target])
        ax.scatter(np.log10(sub["n_review"]), sub[target],
                   s=50, c=color, alpha=0.7, edgecolors="white")
        for _, r in sub.iterrows():
            ax.annotate(
                r["repo"], (np.log10(r["n_review"]), r[target]),
                fontsize=6, xytext=(3, 3), textcoords="offset points", alpha=0.7,
            )
        rho, p = spearmanr(np.log10(sub["n_review"]), sub[target])
        ax.text(
            0.02, 0.98,
            f"ρ={rho:.3f}\np={p:.3g}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.85),
        )
        ax.set_xlabel("log₁₀(n_review)")
        ax.set_title(target.upper())
        ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    axes[0].set_ylabel("Pooled AUC")
    fig.suptitle("n_review (project 規模) vs AUC: IRL / RF / Ensemble", y=1.02)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_repo_2d_cluster(merged: pd.DataFrame, out: Path) -> None:
    """x = log n_review, y = accept_rate, 色 = Ens AUC で 2D map。"""
    rel = merged.dropna(subset=["auc_ens"]).copy()
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        np.log10(rel["n_review"]), rel["accept_rate"],
        c=rel["auc_ens"], s=np.log10(rel["n_unique_dev"] + 1) * 40 + 20,
        cmap="RdYlGn", vmin=0.4, vmax=0.95, alpha=0.85, edgecolors="black", linewidths=0.5,
    )
    for _, r in rel.iterrows():
        ax.annotate(
            r["repo"], (np.log10(r["n_review"]), r["accept_rate"]),
            fontsize=7, xytext=(3, 3), textcoords="offset points", alpha=0.85,
        )
    plt.colorbar(sc, ax=ax, label="Pooled Ens AUC")
    ax.set_xlabel("log₁₀(n_review)")
    ax.set_ylabel("Project 承諾率")
    ax.set_title("Repo 二次元マップ: n_review × 承諾率, 色 = Ens AUC, バブル = log n_dev")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# README
# ----------------------------------------------------------------------

def write_readme(
    merged: pd.DataFrame, rho: pd.DataFrame, pvals: pd.DataFrame,
    reg: pd.DataFrame, out_dir: Path,
) -> None:
    md = []
    md.append("# Repo 特性 × AUC の相関・回帰分析")
    md.append("")
    md.append("`per_repo_basic_stats/project_stats.csv` の repo 特性と")
    md.append("`per_repo/top_dir_pooled_metrics.csv` の AUC を merge し、")
    md.append("どの特性が予測精度を説明するかを定量化する。")
    md.append("")

    rel = merged.dropna(subset=["auc_ens"]).copy()
    md.append(f"## 対象: {len(rel)} repo (基本統計と AUC の両方が揃うもの)")
    md.append("")

    # 主要相関
    md.append("## 主要相関 (Spearman)")
    md.append("")
    md.append("AUC を予測する特徴量との相関 (* = p<0.05):")
    md.append("")
    md.append("| 特徴量 | vs IRL AUC | vs RF AUC | vs Ens AUC | vs gain |")
    md.append("|---|---:|---:|---:|---:|")
    for feat in ["n_review", "accept_rate", "n_unique_dev", "reviews_per_dev",
                 "median_monthly_unique_acc_dev"]:
        if feat not in rho.index:
            continue
        row = f"| {feat} |"
        for tgt in ["auc_irl", "auc_rf", "auc_ens", "gain_vs_best"]:
            r = rho.loc[feat, tgt]
            p = pvals.loc[feat, tgt]
            star = "*" if p < 0.05 else ""
            row += f" **{r:+.3f}**{star} |" if abs(r) > 0.4 and p < 0.05 else f" {r:+.3f}{star} |"
        md.append(row)
    md.append("")

    # 解釈
    md.append("### 解釈")
    interp = []
    if "accept_rate" in rho.index:
        r_ar = rho.loc["accept_rate", "auc_ens"]
        p_ar = pvals.loc["accept_rate", "auc_ens"]
        if p_ar < 0.05 and r_ar > 0.4:
            interp.append(f"- **承諾率 → Ens AUC**: ρ={r_ar:+.3f} (p={p_ar:.3g}) → "
                          f"承諾率が高い repo ほど予測しやすい")
        elif p_ar < 0.05:
            interp.append(f"- **承諾率 → Ens AUC**: ρ={r_ar:+.3f} (p={p_ar:.3g}) → "
                          f"弱い相関")
        else:
            interp.append(f"- **承諾率 → Ens AUC**: ρ={r_ar:+.3f} (p={p_ar:.3g}) → "
                          f"有意な関連なし")
    if "n_review" in rho.index:
        r_nr = rho.loc["n_review", "auc_ens"]
        p_nr = pvals.loc["n_review", "auc_ens"]
        if p_nr < 0.05 and r_nr > 0.4:
            interp.append(f"- **n_review → Ens AUC**: ρ={r_nr:+.3f} (p={p_nr:.3g}) → "
                          f"規模 (レビュー件数) が大きいほど予測しやすい")
        else:
            interp.append(f"- **n_review → Ens AUC**: ρ={r_nr:+.3f} (p={p_nr:.3g})")
    md.extend(interp)
    md.append("")

    # 回帰
    md.append("## 線形回帰: 3 特徴量で AUC を予測")
    md.append("")
    md.append("特徴量: log10(n_review+1), accept_rate, log10(n_unique_dev+1)")
    md.append("")
    md.append("| target | R² | intercept | β log_n_review | β accept_rate | β log_n_dev |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in reg.iterrows():
        md.append(
            f"| {r['target']} | {r['r2']:.3f} | {r['intercept']:.3f} | "
            f"{r['coef_log_n_review']:+.3f} | {r['coef_accept_rate']:+.3f} | "
            f"{r['coef_log_n_dev']:+.3f} |"
        )
    md.append("")
    best_r2 = reg.loc[reg["target"] == "auc_ens", "r2"].values[0]
    md.append(f"- **Ens AUC の説明力 R² = {best_r2:.3f}** "
              f"({'高' if best_r2 > 0.5 else '中' if best_r2 > 0.3 else '低'})")
    md.append("")

    # 異常 repo (residual 大)
    md.append("## 回帰モデルから外れる repo (期待値より大きく逸脱)")
    md.append("")
    rel_r = rel.dropna(subset=["auc_ens"]).copy()
    rel_r["log_n_review"] = np.log10(rel_r["n_review"] + 1)
    rel_r["log_n_dev"] = np.log10(rel_r["n_unique_dev"] + 1)
    X = rel_r[["log_n_review", "accept_rate", "log_n_dev"]].to_numpy()
    y = rel_r["auc_ens"].to_numpy()
    lr = LinearRegression().fit(X, y)
    rel_r["pred_auc_ens"] = lr.predict(X)
    rel_r["residual"] = rel_r["auc_ens"] - rel_r["pred_auc_ens"]
    md.append("### 期待値より高い AUC (positive surprise top 3)")
    md.append("")
    md.append("| repo | n_review | accept_rate | actual AUC | predicted | residual |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in rel_r.nlargest(3, "residual").iterrows():
        md.append(
            f"| {r['repo']} | {r['n_review']} | {r['accept_rate']:.3f} | "
            f"**{r['auc_ens']:.3f}** | {r['pred_auc_ens']:.3f} | "
            f"{r['residual']:+.3f} |"
        )
    md.append("")
    md.append("### 期待値より低い AUC (negative surprise top 3)")
    md.append("")
    md.append("| repo | n_review | accept_rate | actual AUC | predicted | residual |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in rel_r.nsmallest(3, "residual").iterrows():
        md.append(
            f"| {r['repo']} | {r['n_review']} | {r['accept_rate']:.3f} | "
            f"**{r['auc_ens']:.3f}** | {r['pred_auc_ens']:.3f} | "
            f"{r['residual']:+.3f} |"
        )
    md.append("")

    md.append("## 出力ファイル")
    md.append("- `merged_repo_metrics.csv` : basic stats + AUC 結合テーブル")
    md.append("- `correlation_matrix.csv` : Spearman ρ マトリクス")
    md.append("- `regression_summary.csv` : 線形回帰係数・R²")
    md.append("- `fig_corr_heatmap.pdf` : 相関ヒートマップ")
    md.append("- `fig_accept_vs_auc.pdf` : 承諾率 vs Ens AUC scatter")
    md.append("- `fig_n_review_vs_auc.pdf` : n_review vs IRL/RF/Ens の 3 panel")
    md.append("- `fig_repo_2d_cluster.pdf` : 2D map (n_review × accept_rate, 色 = AUC)")

    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-stats",
                    default="outputs/analysis/per_repo_basic_stats/project_stats.csv")
    ap.add_argument("--top-dir-pooled",
                    default="outputs/analysis/per_repo/top_dir_pooled_metrics.csv")
    ap.add_argument("--main-repos-csv", default="data/service_teams_main_repos.csv")
    ap.add_argument("--output-dir", default="outputs/analysis/per_repo_correlation")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Merging repo metrics...")
    merged = merge_repo_metrics(
        Path(args.project_stats), Path(args.top_dir_pooled), Path(args.main_repos_csv),
    )
    merged.to_csv(out_dir / "merged_repo_metrics.csv", index=False)
    print(f"  -> merged_repo_metrics.csv ({len(merged)} repos)")

    # 相関
    cols = ["n_review", "accept_rate", "n_unique_dev", "reviews_per_dev",
            "median_monthly_unique_acc_dev",
            "auc_irl", "auc_rf", "auc_ens", "gain_vs_best"]
    print("Computing Spearman correlation matrix...")
    rho, pvals = correlation_matrix(merged, cols)
    rho.to_csv(out_dir / "correlation_matrix.csv")
    pvals.to_csv(out_dir / "correlation_pvalues.csv")
    print(f"  -> correlation_matrix.csv, correlation_pvalues.csv")

    # 回帰
    print("Running regression analysis...")
    reg = regression_analysis(merged)
    reg.to_csv(out_dir / "regression_summary.csv", index=False)
    print(f"  -> regression_summary.csv")

    # 可視化
    print("Generating figures...")
    fig_corr_heatmap(rho, pvals, out_dir / "fig_corr_heatmap.pdf")
    print("  -> fig_corr_heatmap.pdf")
    fig_accept_vs_auc(merged, out_dir / "fig_accept_vs_auc.pdf")
    print("  -> fig_accept_vs_auc.pdf (Ensemble)")
    fig_accept_vs_auc_irl(merged, out_dir / "fig_accept_vs_auc_irl.pdf")
    print("  -> fig_accept_vs_auc_irl.pdf")
    fig_accept_vs_auc_rf(merged, out_dir / "fig_accept_vs_auc_rf.pdf")
    print("  -> fig_accept_vs_auc_rf.pdf")
    fig_accept_vs_auc_combined(merged, out_dir / "fig_accept_vs_auc_combined.pdf")
    print("  -> fig_accept_vs_auc_combined.pdf (IRL/RF/Ens 3 panel)")
    fig_n_review_vs_auc(merged, out_dir / "fig_n_review_vs_auc.pdf")
    print("  -> fig_n_review_vs_auc.pdf")
    fig_repo_2d_cluster(merged, out_dir / "fig_repo_2d_cluster.pdf")
    print("  -> fig_repo_2d_cluster.pdf")

    # README
    write_readme(merged, rho, pvals, reg, out_dir)
    print("  -> README.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
