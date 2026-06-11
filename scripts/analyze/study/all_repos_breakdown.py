"""全 main repos の sub-directory × window 深掘り分析 + 集計単位の対比。

cinder に絞らず、全 main repos について以下を分析:

1. 各 repo の sub-dir × window AUC heatmap (IRL / RF / Δ の 3 panel)
2. 全 repo の Δ 分布 summary (どの repo で IRL/RF どちらが優位か)
3. project (raw review-level) と top_dir (pair-level) の集計比較
4. IRL が苦戦するパターン / RF が苦戦するパターンの cross-repo 集約

出力 (outputs/analysis/all_repos_breakdown/):
  - per_repo_subdir_breakdown.csv  : 全 main repo × sub-dir × window 詳細
  - repo_delta_summary.csv         : 各 repo の Δ の分布統計
  - project_vs_repo_compare.csv    : project (raw) vs top_dir (pair) 並列比較
  - irl_struggle_pattern.csv       : IRL が大幅に劣る (Δ<-0.15) cell 集約
  - rf_struggle_pattern.csv        : RF が大幅に劣る (Δ>+0.15) cell 集約
  - fig_all_repos_delta_heatmap.pdf : 全 repo × window の Δ summary heatmap
  - fig_all_repos_subdir_grid.pdf  : 主要 repo の sub-dir × window heatmap grid
  - fig_project_vs_repo.pdf        : 集計単位の対比 scatter
  - README.md
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "sans-serif"]

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_diagonal(base: Path, variant: str) -> pd.DataFrame:
    parts = []
    for h in HORIZONS:
        p = base / variant / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d.dropna(subset=["irl_dir_prob_calibrated", "rf_dir_prob", "label"]).copy()
        d["ensemble"] = (d["irl_dir_prob_calibrated"] + d["rf_dir_prob"]) / 2.0
        d["window"] = h
        d["top_dir"] = d["directory"].astype(str).str.split("/").str[0]
        parts.append(d)
    return pd.concat(parts, axis=0, ignore_index=True)


def safe_auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


# ----------------------------------------------------------------------
# 1. 各 repo の sub-dir × window breakdown
# ----------------------------------------------------------------------

def per_repo_subdir_breakdown(big: pd.DataFrame, main_repos: set[str]) -> pd.DataFrame:
    rows = []
    for repo in sorted(set(big["top_dir"]) & main_repos):
        sub = big[big["top_dir"] == repo]
        for (dir_name, h), g in sub.groupby(["directory", "window"]):
            y = g["label"].astype(int).to_numpy()
            n_pos = int(y.sum())
            n_neg = int((y == 0).sum())
            if n_pos == 0 and n_neg == 0:
                continue
            row = {
                "repo": repo, "directory": dir_name, "window": h,
                "n_pair": len(g), "n_pos": n_pos, "n_neg": n_neg,
                "accept_rate": n_pos / len(g),
                "auc_irl": safe_auc(y, g["irl_dir_prob_calibrated"].to_numpy()),
                "auc_rf": safe_auc(y, g["rf_dir_prob"].to_numpy()),
                "auc_ens": safe_auc(y, g["ensemble"].to_numpy()),
            }
            row["delta"] = (row["auc_irl"] - row["auc_rf"]
                            if (pd.notna(row["auc_irl"]) and pd.notna(row["auc_rf"])) else np.nan)
            row["gain_vs_best"] = (
                row["auc_ens"] - max(row["auc_irl"], row["auc_rf"])
                if all(pd.notna(row[k]) for k in ["auc_ens", "auc_irl", "auc_rf"])
                else np.nan
            )
            rows.append(row)
    return pd.DataFrame(rows)


def repo_delta_summary(breakdown: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for repo, g in breakdown.dropna(subset=["delta"]).groupby("repo"):
        delta = g["delta"]
        rows.append({
            "repo": repo,
            "n_cells": len(g),
            "n_irl_win": int((delta > 0.03).sum()),
            "n_tied": int(((delta >= -0.03) & (delta <= 0.03)).sum()),
            "n_rf_win": int((delta < -0.03).sum()),
            "median_delta": float(delta.median()),
            "min_delta": float(delta.min()),
            "max_delta": float(delta.max()),
            "median_irl": float(g["auc_irl"].median()),
            "median_rf": float(g["auc_rf"].median()),
            "median_ens": float(g["auc_ens"].median()),
            "median_gain": float(g["gain_vs_best"].median()),
        })
    return pd.DataFrame(rows).sort_values("median_delta")


# ----------------------------------------------------------------------
# 2. project (raw) vs repo (pair) 対比
# ----------------------------------------------------------------------

def project_vs_repo_compare(
    breakdown: pd.DataFrame, raw_csv: Path, top_dir_pooled: Path,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_csv, parse_dates=["timestamp"])
    raw["repo"] = raw["project"].astype(str).str.split("/").str[-1]
    if raw["timestamp"].dt.tz is not None:
        raw["timestamp"] = raw["timestamp"].dt.tz_localize(None)
    t0 = pd.Timestamp("2022-01-01")
    t_end = pd.Timestamp("2023-01-01")
    raw = raw[(raw["timestamp"] >= t0) & (raw["timestamp"] < t_end)]

    proj = raw.groupby("repo").agg(
        proj_n_review=("label", "size"),
        proj_accept_rate=("label", "mean"),
        proj_n_dev=("email", "nunique"),
    ).reset_index()

    auc = pd.read_csv(top_dir_pooled).rename(columns={"top_dir": "repo"})
    auc_keep = auc[["repo", "n_pair", "n_pos", "n_neg",
                    "auc_irl", "auc_rf", "auc_ens", "gain_vs_best"]].copy()
    auc_keep["pair_accept_rate"] = auc_keep["n_pos"] / auc_keep["n_pair"]
    auc_keep = auc_keep.rename(columns={
        "n_pair": "pair_n", "n_pos": "pair_n_pos", "n_neg": "pair_n_neg",
    })

    bd_agg = breakdown.groupby("repo").agg(
        n_subdir=("directory", "nunique"),
        n_subdir_cells=("directory", "size"),
    ).reset_index()

    df = proj.merge(auc_keep, on="repo", how="inner").merge(bd_agg, on="repo", how="left")
    return df.sort_values("proj_n_review", ascending=False)


# ----------------------------------------------------------------------
# 3. struggle pattern (IRL / RF が大幅劣る cell 集約)
# ----------------------------------------------------------------------

def struggle_patterns(breakdown: pd.DataFrame, threshold: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    rel = breakdown.dropna(subset=["delta"]).copy()
    irl_struggle = rel[rel["delta"] < -threshold].sort_values("delta").copy()
    rf_struggle = rel[rel["delta"] > threshold].sort_values("delta", ascending=False).copy()
    return irl_struggle, rf_struggle


# ----------------------------------------------------------------------
# 可視化
# ----------------------------------------------------------------------

def fig_all_repos_delta_heatmap(breakdown: pd.DataFrame, out: Path) -> None:
    """全 main repo × window で IRL−RF の中央値を heatmap."""
    rel = breakdown.dropna(subset=["delta"])
    pivot = rel.groupby(["repo", "window"])["delta"].median().unstack("window")
    pivot = pivot.reindex(columns=HORIZONS)
    # repo を median_delta で並べる
    order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.loc[order]
    n_cells = rel.groupby(["repo", "window"]).size().unstack("window").reindex(columns=HORIZONS).loc[order]

    fig, ax = plt.subplots(figsize=(7, max(5, len(pivot) * 0.32)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-0.25, vmax=0.25)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS, fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            n = n_cells.values[i, j] if pd.notna(n_cells.values[i, j]) else 0
            if pd.notna(v):
                ax.text(j, i, f"{v:+.2f}\n(n={int(n)})",
                        ha="center", va="center", fontsize=6,
                        color="black" if abs(v) < 0.15 else "white")
    plt.colorbar(im, ax=ax, label="median Δ = IRL − RF\n(青: RF 優位 / 赤: IRL 優位)")
    ax.set_title("全 main repo × window の sub-dir median Δ")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


def fig_repo_subdir_grid(
    breakdown: pd.DataFrame, repos: list[str], out: Path,
) -> None:
    """指定 repo の sub-dir × window Δ heatmap を grid で並べる。"""
    n_repos = len(repos)
    # 全 main repo を一覧化するため列数を増やす (n>=12 で 4 列に)
    if n_repos >= 16:
        ncol = 4
    elif n_repos >= 9:
        ncol = 3
    else:
        ncol = min(3, n_repos)
    nrow = int(np.ceil(n_repos / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.5, nrow * 3.5))
    if n_repos == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes).flatten()

    for idx, repo in enumerate(repos):
        ax = axes[idx]
        sub = breakdown[breakdown["repo"] == repo].copy()
        if sub.empty:
            ax.axis("off")
            continue
        pivot = sub.pivot_table(index="directory", columns="window", values="delta")
        pivot = pivot.reindex(columns=HORIZONS)
        pivot_n = sub.pivot_table(index="directory", columns="window", values="n_pair")
        pivot_n = pivot_n.reindex(columns=HORIZONS)
        # n_pair 合計順 (delta が全 NaN な dir は除外)
        common = pivot.dropna(how="all").index.intersection(pivot_n.index)
        order = pivot_n.loc[common].sum(axis=1).sort_values(ascending=False).index
        pivot = pivot.reindex(order)
        pivot_n = pivot_n.reindex(order)
        if len(order) == 0:
            ax.axis("off")
            continue
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu", vmin=-0.5, vmax=0.5)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels(HORIZONS, fontsize=7)
        ax.set_yticks(range(len(pivot)))
        ax.set_yticklabels([d.split("/", 1)[-1] for d in pivot.index], fontsize=6)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:+.2f}",
                            ha="center", va="center", fontsize=5,
                            color="black" if abs(v) < 0.3 else "white")
        ax.set_title(f"{repo} (n_subdir={len(pivot)})", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.6, label="Δ" if idx == 0 else None)

    for idx in range(len(repos), len(axes)):
        axes[idx].axis("off")
    fig.suptitle("各 main repo の sub-dir × window 補完性 (Δ = IRL − RF)", y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_project_vs_repo(cmp: pd.DataFrame, out: Path) -> None:
    """raw project accept_rate と pair top_dir accept_rate の対比。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rel = cmp.dropna(subset=["proj_accept_rate", "pair_accept_rate", "auc_ens"]).copy()
    sc = axes[0].scatter(
        rel["proj_accept_rate"], rel["pair_accept_rate"],
        c=rel["auc_ens"], s=np.log10(rel["proj_n_review"] + 1) * 25 + 15,
        cmap="RdYlGn", vmin=0.4, vmax=0.95, alpha=0.85, edgecolors="black", linewidths=0.5,
    )
    for _, r in rel.iterrows():
        axes[0].annotate(r["repo"], (r["proj_accept_rate"], r["pair_accept_rate"]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc, ax=axes[0], label="pooled Ens AUC")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3, lw=0.7)
    axes[0].set_xlabel("Project 承諾率 (raw review-level)")
    axes[0].set_ylabel("Pair 承諾率 (window-level)")
    axes[0].set_title("集計単位の承諾率対比")

    # 2 番目: pair_n vs proj_n_review
    sc2 = axes[1].scatter(
        np.log10(rel["proj_n_review"]), np.log10(rel["pair_n"]),
        c=rel["auc_ens"], s=60, cmap="RdYlGn", vmin=0.4, vmax=0.95,
        alpha=0.85, edgecolors="black", linewidths=0.5,
    )
    for _, r in rel.iterrows():
        axes[1].annotate(r["repo"], (np.log10(r["proj_n_review"]), np.log10(r["pair_n"])),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    plt.colorbar(sc2, ax=axes[1], label="pooled Ens AUC")
    axes[1].plot([0, 5], [0, 5], "k--", alpha=0.3, lw=0.7)
    axes[1].set_xlabel("log₁₀(project n_review, raw)")
    axes[1].set_ylabel("log₁₀(pair n)")
    axes[1].set_title("集計単位のサンプル数対比")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# README
# ----------------------------------------------------------------------

def write_readme(
    breakdown: pd.DataFrame, delta_summ: pd.DataFrame, cmp: pd.DataFrame,
    irl_struggle: pd.DataFrame, rf_struggle: pd.DataFrame, out_dir: Path,
) -> None:
    md = []
    md.append("# 全 main repos の sub-dir × window 深掘り + 集計単位対比")
    md.append("")
    md.append("Cinder に絞らず全 main repos について sub-dir × window 詳細を集計し、")
    md.append("project (raw review-level) と repo (pair-level) の対比を行う。")
    md.append("")

    # 1. 全 repo Δ summary
    md.append("## 1. 全 main repo の Δ = IRL − RF 分布")
    md.append("")
    md.append("`median_delta` 昇順 (RF 優位 ← → IRL 優位):")
    md.append("")
    md.append("| repo | n_cells | n_irl_win | n_tied | n_rf_win | median Δ | min Δ | max Δ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in delta_summ.iterrows():
        md.append(
            f"| {r['repo']} | {r['n_cells']} | {r['n_irl_win']} | {r['n_tied']} | "
            f"{r['n_rf_win']} | {r['median_delta']:+.3f} | "
            f"{r['min_delta']:+.3f} | {r['max_delta']:+.3f} |"
        )
    md.append("")
    md.append("### 中央値での 3 群分類")
    rf_dominant = delta_summ[delta_summ["median_delta"] < -0.03]["repo"].tolist()
    tied = delta_summ[(delta_summ["median_delta"] >= -0.03) & (delta_summ["median_delta"] <= 0.03)]["repo"].tolist()
    irl_dominant = delta_summ[delta_summ["median_delta"] > 0.03]["repo"].tolist()
    md.append(f"- **RF 中央値優位** (median Δ < -3pp): {len(rf_dominant)} repo — {', '.join(rf_dominant)}")
    md.append(f"- **拮抗** (|median Δ| ≤ 3pp): {len(tied)} repo — {', '.join(tied)}")
    md.append(f"- **IRL 中央値優位** (median Δ > +3pp): {len(irl_dominant)} repo — {', '.join(irl_dominant)}")
    md.append("")

    # 2. project vs repo compare
    md.append("## 2. Project (raw) vs Repo (pair) 集計単位の対比")
    md.append("")
    md.append("評価期間 (2022-01〜2023-01) の raw review-level と pair (window) ベースを並列表示。")
    md.append("")
    md.append("| repo | proj_n_review | proj_accept | proj_n_dev | pair_n | pair_accept | pair_n_dev (n_subdir) | IRL | RF | Ens |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    cmp_show = cmp.sort_values("proj_n_review", ascending=False).head(20)
    for _, r in cmp_show.iterrows():
        md.append(
            f"| {r['repo']} | {int(r['proj_n_review'])} | {r['proj_accept_rate']:.3f} | "
            f"{int(r['proj_n_dev'])} | {int(r['pair_n'])} | "
            f"{r['pair_accept_rate']:.3f} | "
            f"({int(r['n_subdir']) if pd.notna(r['n_subdir']) else '—'}) | "
            f"{r['auc_irl']:.3f} | {r['auc_rf']:.3f} | **{r['auc_ens']:.3f}** |"
        )
    md.append("")
    md.append("### Project (raw) vs Pair (window) 承諾率の対応")
    rel = cmp.dropna(subset=["proj_accept_rate", "pair_accept_rate"])
    if len(rel) >= 5:
        from scipy.stats import spearmanr
        rho, p = spearmanr(rel["proj_accept_rate"], rel["pair_accept_rate"])
        md.append(f"- Spearman ρ(proj承諾率, pair承諾率) = **{rho:+.3f}** (p={p:.3g})")
        md.append("- proj_accept > pair_accept なのは pair が「window 内に依頼あり&承諾」を要求するため")
    md.append("")

    # 3. IRL struggle patterns
    if not irl_struggle.empty:
        md.append("## 3. IRL が大幅に劣る cell (Δ < -0.15)")
        md.append("")
        md.append(f"- 該当 cell 数: {len(irl_struggle)}")
        repo_counts = irl_struggle["repo"].value_counts()
        md.append("- repo 別出現数: " + ", ".join(f"{r}={n}" for r, n in repo_counts.head(10).items()))
        md.append("")
        md.append("### Top 15 worst (IRL 大幅劣勢)")
        md.append("")
        md.append("| repo | directory | window | n_pair | accept_rate | IRL | RF | Ens | Δ |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in irl_struggle.head(15).iterrows():
            md.append(
                f"| {r['repo']} | {r['directory']} | {r['window']} | {int(r['n_pair'])} | "
                f"{r['accept_rate']:.2f} | "
                f"{r['auc_irl']:.3f} | {r['auc_rf']:.3f} | "
                f"{r['auc_ens']:.3f} | {r['delta']:+.3f} |"
            )
        md.append("")

    # 4. RF struggle patterns
    if not rf_struggle.empty:
        md.append("## 4. RF が大幅に劣る cell (Δ > +0.15)")
        md.append("")
        md.append(f"- 該当 cell 数: {len(rf_struggle)}")
        repo_counts = rf_struggle["repo"].value_counts()
        md.append("- repo 別出現数: " + ", ".join(f"{r}={n}" for r, n in repo_counts.head(10).items()))
        md.append("")
        md.append("### Top 15 best (IRL 大幅優位)")
        md.append("")
        md.append("| repo | directory | window | n_pair | accept_rate | IRL | RF | Ens | Δ |")
        md.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for _, r in rf_struggle.head(15).iterrows():
            md.append(
                f"| {r['repo']} | {r['directory']} | {r['window']} | {int(r['n_pair'])} | "
                f"{r['accept_rate']:.2f} | "
                f"{r['auc_irl']:.3f} | {r['auc_rf']:.3f} | "
                f"{r['auc_ens']:.3f} | {r['delta']:+.3f} |"
            )
        md.append("")

    md.append("## 5. 出力ファイル")
    md.append("- `per_repo_subdir_breakdown.csv` : 全 main repo × sub-dir × window 詳細")
    md.append("- `repo_delta_summary.csv` : 各 repo の Δ 分布統計")
    md.append("- `project_vs_repo_compare.csv` : project (raw) と repo (pair) 並列比較")
    md.append("- `irl_struggle_pattern.csv` : IRL が大幅劣勢の cell")
    md.append("- `rf_struggle_pattern.csv` : RF が大幅劣勢の cell")
    md.append("- `fig_all_repos_delta_heatmap.pdf` : 全 main repo × window の median Δ heatmap")
    md.append("- `fig_all_repos_subdir_grid.pdf` : 主要 repo の sub-dir × window grid")
    md.append("- `fig_project_vs_repo.pdf` : 集計単位の対比 scatter")

    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/main32_mce_volat_ep150")
    ap.add_argument("--variant", default="lstm_hybrid_snapshot")
    ap.add_argument("--main-repos-csv", default="data/service_teams_main_repos.csv")
    ap.add_argument("--raw-csv", default="data/combined_raw_main32.csv")
    ap.add_argument("--top-dir-pooled",
                    default="outputs/analysis/per_repo/top_dir_pooled_metrics.csv")
    ap.add_argument("--output-dir", default="outputs/analysis/all_repos_breakdown")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mr_df = pd.read_csv(args.main_repos_csv).dropna(subset=["main_repo"])
    main_repos = set(mr_df["main_repo"].astype(str).str.split("/").str[-1]) - {"nan"}

    print("Loading predictions...")
    big = load_diagonal(Path(args.base), args.variant)
    print(f"  total rows: {len(big):,}")

    print("Computing per-repo sub-dir breakdown...")
    breakdown = per_repo_subdir_breakdown(big, main_repos)
    breakdown.to_csv(out_dir / "per_repo_subdir_breakdown.csv", index=False)
    print(f"  -> per_repo_subdir_breakdown.csv ({len(breakdown)} cells "
          f"in {breakdown['repo'].nunique()} repos)")

    print("Computing repo delta summary...")
    delta_summ = repo_delta_summary(breakdown)
    delta_summ.to_csv(out_dir / "repo_delta_summary.csv", index=False)
    print(f"  -> repo_delta_summary.csv ({len(delta_summ)} repos)")

    print("Computing project vs repo compare...")
    cmp = project_vs_repo_compare(
        breakdown, Path(args.raw_csv), Path(args.top_dir_pooled),
    )
    cmp.to_csv(out_dir / "project_vs_repo_compare.csv", index=False)
    print(f"  -> project_vs_repo_compare.csv ({len(cmp)} repos)")

    print("Computing struggle patterns...")
    irl_str, rf_str = struggle_patterns(breakdown)
    irl_str.to_csv(out_dir / "irl_struggle_pattern.csv", index=False)
    rf_str.to_csv(out_dir / "rf_struggle_pattern.csv", index=False)
    print(f"  -> irl_struggle ({len(irl_str)}), rf_struggle ({len(rf_str)})")

    print("Generating figures...")
    fig_all_repos_delta_heatmap(breakdown, out_dir / "fig_all_repos_delta_heatmap.pdf")
    # grid: 上位 9 repo (n_cells 多い順)
    all_repos = delta_summ.sort_values("n_cells", ascending=False)["repo"].tolist()
    fig_repo_subdir_grid(breakdown, all_repos, out_dir / "fig_all_repos_subdir_grid.pdf")
    fig_project_vs_repo(cmp, out_dir / "fig_project_vs_repo.pdf")
    print("  -> 3 figures")

    write_readme(breakdown, delta_summ, cmp, irl_str, rf_str, out_dir)
    print("  -> README.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
