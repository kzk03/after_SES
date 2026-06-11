"""Directory / Repository ごとの基本統計を集計する。

予測精度の話の前に「データ特性自体が directory/repo ごとに違うのか」を見る分析。

集計内容:
  - project (repo) 単位 (combined_raw_main32.csv を使用、評価期間 2022-01〜2023-01):
    - 総 review 件数, 承諾率 (mean label), ユニーク reviewer 数,
    - 月次レビュー件数, 月次承諾件数, 月次承諾ユニーク dev 数, 平均承諾 dev 数 / 月
  - directory 単位 (pair_predictions.csv 4 window を使用):
    - n_pair, 承諾率 = mean(label), ユニーク dev 数, dev / pair ratio
  - top_dir (repo) 単位 (同上):
    - 集約版

出力 (outputs/analysis/per_repo_basic_stats/):
  - project_stats.csv         : 32 project の基本統計
  - project_monthly.csv       : project × 月 の活動量
  - directory_stats.csv       : directory 単位の基本統計
  - top_dir_stats.csv         : top_dir (repo) 単位の基本統計
  - fig_accept_rate_dist.pdf  : 承諾率の分布 (project vs directory)
  - fig_repo_monthly.pdf      : 主要 repo の月次活動量推移
  - fig_dev_n_vs_accept.pdf   : dev 数 vs 承諾率 (project 単位)
  - README.md
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "sans-serif"]

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_raw(p: Path, t0: str = "2022-01-01", t_end: str = "2023-01-01") -> pd.DataFrame:
    """combined_raw_main32.csv を読み、評価期間にフィルタ。"""
    df = pd.read_csv(p, parse_dates=["timestamp"])
    df["repo"] = df["project"].astype(str).str.split("/").str[-1]
    t0_dt = pd.Timestamp(t0).tz_localize(None)
    t_end_dt = pd.Timestamp(t_end).tz_localize(None)
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    df = df[(df["timestamp"] >= t0_dt) & (df["timestamp"] < t_end_dt)].copy()
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    return df


def project_stats(raw: pd.DataFrame) -> pd.DataFrame:
    """project (= repo) 単位の集計 (評価期間)。"""
    rows = []
    for repo, g in raw.groupby("repo"):
        n_review = len(g)
        n_acc = int((g["label"] == 1).sum())
        n_rej = int((g["label"] == 0).sum())
        acc_rate = n_acc / n_review if n_review else np.nan
        n_dev = g["email"].nunique()
        n_owners = g["owner_email"].nunique()
        # 月次集計
        monthly = g.groupby("month").agg(
            n_reviews=("label", "size"),
            n_acc=("label", lambda x: int((x == 1).sum())),
            n_unique_dev=("email", "nunique"),
            n_unique_acc_dev=("email", lambda x: g.loc[x.index][g.loc[x.index, "label"] == 1]["email"].nunique()),
        )
        rows.append({
            "repo": repo,
            "n_review": n_review,
            "n_accept": n_acc,
            "n_reject": n_rej,
            "accept_rate": acc_rate,
            "n_unique_dev": n_dev,
            "n_unique_owner": n_owners,
            "reviews_per_dev": n_review / n_dev if n_dev else np.nan,
            "active_months": int(monthly.index.nunique()),
            "median_monthly_reviews": float(monthly["n_reviews"].median()),
            "median_monthly_accepts": float(monthly["n_acc"].median()),
            "median_monthly_unique_acc_dev": float(monthly["n_unique_acc_dev"].median()),
            "mean_monthly_unique_acc_dev": float(monthly["n_unique_acc_dev"].mean()),
            "max_monthly_unique_acc_dev": int(monthly["n_unique_acc_dev"].max()),
        })
    return pd.DataFrame(rows).sort_values("n_review", ascending=False).reset_index(drop=True)


def project_monthly(raw: pd.DataFrame) -> pd.DataFrame:
    """project × month の活動量。"""
    rows = []
    for (repo, month), g in raw.groupby(["repo", "month"]):
        n_reviews = len(g)
        n_acc = int((g["label"] == 1).sum())
        n_dev = g["email"].nunique()
        acc_devs = g[g["label"] == 1]["email"].nunique()
        rows.append({
            "repo": repo, "month": month,
            "n_reviews": n_reviews, "n_accept": n_acc,
            "accept_rate": n_acc / n_reviews if n_reviews else np.nan,
            "n_unique_dev": n_dev,
            "n_unique_acc_dev": acc_devs,
        })
    return pd.DataFrame(rows).sort_values(["repo", "month"])


def load_pair_pred(base: Path, variant: str) -> pd.DataFrame:
    """pair_predictions を 4 window 結合し top_dir 付与。"""
    parts = []
    for h in HORIZONS:
        p = base / variant / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d.dropna(subset=["label"]).copy()
        d["window"] = h
        parts.append(d[["developer", "directory", "label", "window"]])
    big = pd.concat(parts, axis=0, ignore_index=True)
    big["top_dir"] = big["directory"].astype(str).str.split("/").str[0]
    return big


def directory_stats(pp: pd.DataFrame, main_repos: set[str]) -> pd.DataFrame:
    rows = []
    for dir_name, g in pp.groupby("directory"):
        n_pair = len(g)
        n_pos = int((g["label"] == 1).sum())
        n_neg = int((g["label"] == 0).sum())
        top_dir = str(dir_name).split("/")[0]
        cat = "project-specific" if top_dir in main_repos else "cross-cutting"
        rows.append({
            "directory": dir_name,
            "top_dir": top_dir,
            "category": cat,
            "n_pair": n_pair,
            "n_pos": n_pos, "n_neg": n_neg,
            "accept_rate": n_pos / n_pair if n_pair else np.nan,
            "n_unique_dev": g["developer"].nunique(),
            "n_acc_dev": g[g["label"] == 1]["developer"].nunique(),
            "pairs_per_dev": n_pair / g["developer"].nunique() if g["developer"].nunique() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_pair", ascending=False).reset_index(drop=True)


def top_dir_stats(pp: pd.DataFrame, main_repos: set[str]) -> pd.DataFrame:
    rows = []
    for tdir, g in pp.groupby("top_dir"):
        n_pair = len(g)
        n_pos = int((g["label"] == 1).sum())
        n_neg = int((g["label"] == 0).sum())
        cat = "project-specific" if tdir in main_repos else "cross-cutting"
        rows.append({
            "top_dir": tdir,
            "category": cat,
            "n_directory": g["directory"].nunique(),
            "n_pair": n_pair,
            "n_pos": n_pos, "n_neg": n_neg,
            "accept_rate": n_pos / n_pair if n_pair else np.nan,
            "n_unique_dev": g["developer"].nunique(),
            "n_acc_dev": g[g["label"] == 1]["developer"].nunique(),
            "pairs_per_dev": n_pair / g["developer"].nunique() if g["developer"].nunique() else np.nan,
        })
    return pd.DataFrame(rows).sort_values("n_pair", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# 可視化
# ----------------------------------------------------------------------

def fig_accept_rate_dist(
    proj: pd.DataFrame, dir_df: pd.DataFrame, out: Path,
    main_repos_csv: Path | None = None,
) -> None:
    """承諾率分布: 左 = 32 project (tier 色分け)、右 = 全 386 repo (単色 bar)。"""
    # tier merge
    proj = proj.copy()
    if main_repos_csv and main_repos_csv.exists():
        mr = pd.read_csv(main_repos_csv).dropna(subset=["main_repo"])
        mr["repo_name"] = mr["main_repo"].astype(str).str.split("/").str[-1]
        tier_map = mr.set_index("repo_name")["tier"].to_dict()
        proj["tier"] = proj["repo"].map(tier_map).fillna("?")
    else:
        proj["tier"] = "?"
    tier_colors = {
        "大": "#2E7D32", "中": "#558B2F", "小": "#E65100", "極小": "#B71C1C",
        "?": "#888888",
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # project: 全 32 件、tier 色分け
    proj_sorted = proj.sort_values("accept_rate", ascending=False).reset_index(drop=True)
    bar_colors = [tier_colors.get(t, "#888888") for t in proj_sorted["tier"]]
    axes[0].bar(range(len(proj_sorted)), proj_sorted["accept_rate"],
                color=bar_colors, alpha=0.85)
    axes[0].set_xticks(range(len(proj_sorted)))
    axes[0].set_xticklabels(proj_sorted["repo"], rotation=90, fontsize=7)
    axes[0].set_ylabel("Accept rate")
    axes[0].axhline(proj["accept_rate"].median(), color="red", lw=0.8, ls="--",
                    label=f"median={proj['accept_rate'].median():.2f}")
    # tier 凡例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=tier_colors[t],
              label=f"tier {t} (n={(proj_sorted['tier']==t).sum()})")
        for t in ["大", "中", "小", "極小"]
        if (proj_sorted["tier"] == t).sum() > 0
    ]
    legend_elements.append(
        plt.Line2D([0], [0], color="red", lw=0.8, ls="--",
                   label=f"median={proj['accept_rate'].median():.2f}")
    )
    axes[0].legend(handles=legend_elements, loc="upper right", fontsize=8)
    axes[0].set_title(f"Project 別 accept rate (全 {len(proj)} project, tier 色分け)")
    axes[0].set_ylim(0, 1.05)

    # repo (directory): 全件 bar、accept_rate 降順、カテゴリ塗り分けなし
    dir_sorted = dir_df.sort_values("accept_rate", ascending=False).reset_index(drop=True)
    n = len(dir_sorted)
    axes[1].bar(range(n), dir_sorted["accept_rate"], color="#3F5D7D", width=1.0, alpha=0.85)
    med = dir_sorted["accept_rate"].median()
    axes[1].axhline(med, color="red", lw=0.8, ls="--", label=f"median={med:.2f}")
    axes[1].set_ylabel("Accept rate")
    axes[1].set_xlabel(f"全 {n} repo (depth=2 path)、accept_rate 降順")
    axes[1].set_xticks([])
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_title(f"Repo 別 accept rate (全 {n} repo)")
    axes[1].set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


def fig_repo_monthly(monthly: pd.DataFrame, out: Path, top_n_repos: int | None = None) -> None:
    """全 repo (top_n_repos=None) または上位 N repo の月次推移を描画。

    log スケールで全 repo を 1 図に重ねる。色は規模降順で tab20。
    """
    totals = monthly.groupby("repo")["n_reviews"].sum().sort_values(ascending=False)
    if top_n_repos is None:
        target_repos = totals.index.tolist()
        title_label = f"全 {len(target_repos)} repo"
    else:
        target_repos = totals.head(top_n_repos).index.tolist()
        title_label = f"主要 {top_n_repos} repo"
    n = len(target_repos)
    cmap = plt.cm.tab20 if n > 10 else plt.cm.tab10
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for i, repo in enumerate(target_repos):
        sub = monthly[monthly["repo"] == repo].sort_values("month")
        if sub.empty:
            continue
        color = cmap(i % cmap.N)
        lw = 1.5 if i < 10 else 0.8
        ms = 4 if i < 10 else 2
        alpha = 0.95 if i < 10 else 0.5
        axes[0].plot(sub["month"], sub["n_reviews"], marker="o", label=repo,
                     color=color, lw=lw, ms=ms, alpha=alpha)
        axes[1].plot(sub["month"], sub["n_unique_acc_dev"], marker="s", label=repo,
                     color=color, lw=lw, ms=ms, alpha=alpha)
    axes[0].set_ylabel("月次レビュー件数 (log scale)")
    axes[0].set_yscale("symlog")
    axes[0].set_title(f"{title_label} の月次活動量 (2022-01〜2023-01)")
    # 凡例を 2 列で外に出す
    axes[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                   fontsize=7, ncol=2 if n > 16 else 1)
    axes[1].set_ylabel("月次ユニーク承諾 dev 数")
    axes[1].set_xlabel("month")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


def fig_dev_n_vs_accept(
    proj: pd.DataFrame, out: Path, main_repos_csv: Path | None = None,
) -> None:
    """unique dev 数 vs 承諾率 scatter。tier で色分け、点の大きさ = log10 n_review。"""
    proj = proj.copy()
    if main_repos_csv and main_repos_csv.exists():
        mr = pd.read_csv(main_repos_csv).dropna(subset=["main_repo"])
        mr["repo_name"] = mr["main_repo"].astype(str).str.split("/").str[-1]
        tier_map = mr.set_index("repo_name")["tier"].to_dict()
        proj["tier"] = proj["repo"].map(tier_map).fillna("?")
    else:
        proj["tier"] = "?"
    tier_colors = {
        "大": "#2E7D32", "中": "#558B2F", "小": "#E65100", "極小": "#B71C1C",
        "?": "#888888",
    }
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for tier in ["大", "中", "小", "極小"]:
        sub = proj[proj["tier"] == tier]
        if len(sub) == 0:
            continue
        ax.scatter(
            np.log10(sub["n_unique_dev"]), sub["accept_rate"],
            s=np.log10(sub["n_review"] + 1) * 25 + 10,
            c=tier_colors[tier], alpha=0.75, edgecolors="white", linewidths=0.6,
            label=f"tier {tier} (n={len(sub)})",
        )
    for _, r in proj.iterrows():
        ax.annotate(
            r["repo"], (np.log10(r["n_unique_dev"]), r["accept_rate"]),
            fontsize=7, xytext=(3, 3), textcoords="offset points",
        )
    ax.axhline(proj["accept_rate"].median(), color="red", lw=0.6, ls="--", alpha=0.6,
               label=f"median={proj['accept_rate'].median():.2f}")
    ax.set_xlabel("log₁₀(unique dev 数)")
    ax.set_ylabel("accept rate")
    ax.set_title("Project: unique dev 数 vs 承諾率 (点の大きさ = log₁₀ レビュー依頼件数)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# README
# ----------------------------------------------------------------------

def write_readme(
    proj: pd.DataFrame, dir_df: pd.DataFrame, td_df: pd.DataFrame, out_dir: Path,
) -> None:
    md = []
    md.append("# Directory / Repository 別の基本統計")
    md.append("")
    md.append("予測精度の話に入る前に「データ特性自体が directory / repo ごとに違うのか」を見る分析。")
    md.append("")

    # Project stats summary
    md.append("## 1. Project (repo) 単位の基本統計 (評価期間 2022-01〜2023-01)")
    md.append("")
    md.append(f"- 対象 project: {len(proj)} repo")
    md.append(f"- 総 review 件数: {proj['n_review'].sum():,}")
    md.append(f"- 総ユニーク dev 数 (重複含む): {proj['n_unique_dev'].sum():,}")
    md.append(f"- accept_rate 中央値: **{proj['accept_rate'].median():.3f}**, "
              f"min={proj['accept_rate'].min():.3f}, max={proj['accept_rate'].max():.3f}")
    md.append("")
    md.append("### accept_rate 上位/下位 5 project")
    md.append("")
    md.append("| repo | n_review | accept_rate | n_dev | reviews/dev | median月次承諾dev |")
    md.append("|---|---:|---:|---:|---:|---:|")
    top = proj.nlargest(5, "accept_rate")
    bot = proj.nsmallest(5, "accept_rate")
    md.append("**上位 5 (高承諾)**")
    md.append("")
    md.append("| repo | n_review | accept_rate | n_dev | reviews/dev | med月次承諾dev |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in top.iterrows():
        md.append(
            f"| {r['repo']} | {r['n_review']} | **{r['accept_rate']:.3f}** | "
            f"{r['n_unique_dev']} | {r['reviews_per_dev']:.1f} | "
            f"{r['median_monthly_unique_acc_dev']:.0f} |"
        )
    md.append("")
    md.append("**下位 5 (低承諾)**")
    md.append("")
    md.append("| repo | n_review | accept_rate | n_dev | reviews/dev | med月次承諾dev |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in bot.iterrows():
        md.append(
            f"| {r['repo']} | {r['n_review']} | **{r['accept_rate']:.3f}** | "
            f"{r['n_unique_dev']} | {r['reviews_per_dev']:.1f} | "
            f"{r['median_monthly_unique_acc_dev']:.0f} |"
        )
    md.append("")

    # n_review top
    md.append("### レビュー件数上位 5 project (規模)")
    md.append("")
    md.append("| repo | n_review | accept_rate | n_dev | reviews/dev | med月次承諾dev |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in proj.head(5).iterrows():
        md.append(
            f"| {r['repo']} | **{r['n_review']}** | {r['accept_rate']:.3f} | "
            f"{r['n_unique_dev']} | {r['reviews_per_dev']:.1f} | "
            f"{r['median_monthly_unique_acc_dev']:.0f} |"
        )
    md.append("")
    md.append("### レビュー件数下位 5 project (極小規模)")
    md.append("")
    md.append("| repo | n_review | accept_rate | n_dev | reviews/dev | med月次承諾dev |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in proj.tail(5).iterrows():
        md.append(
            f"| {r['repo']} | **{r['n_review']}** | {r['accept_rate']:.3f} | "
            f"{r['n_unique_dev']} | {r['reviews_per_dev']:.1f} | "
            f"{r['median_monthly_unique_acc_dev']:.0f} |"
        )
    md.append("")

    # Directory stats
    md.append("## 2. Directory 単位の基本統計")
    md.append("")
    md.append(f"- 対象 directory: {len(dir_df)} (4 window 跨ぐ unique)")
    md.append(f"- 内訳: project-specific {(dir_df['category']=='project-specific').sum()} / "
              f"cross-cutting {(dir_df['category']=='cross-cutting').sum()}")
    md.append(f"- accept_rate 中央値: **{dir_df['accept_rate'].median():.3f}**, "
              f"std={dir_df['accept_rate'].std():.3f}")
    md.append(f"- accept_rate 範囲: min={dir_df['accept_rate'].min():.3f}, "
              f"max={dir_df['accept_rate'].max():.3f}")
    md.append("")
    md.append("### accept_rate の category 別")
    for cat in ["project-specific", "cross-cutting"]:
        sub = dir_df[dir_df["category"] == cat]
        md.append(
            f"- **{cat}** (n={len(sub)}): "
            f"median accept_rate = {sub['accept_rate'].median():.3f} "
            f"(IQR {sub['accept_rate'].quantile(0.25):.3f}〜{sub['accept_rate'].quantile(0.75):.3f})"
        )
    md.append("")
    md.append("### accept_rate 上位/下位 directory (n_pair ≥ 50 で絞る)")
    md.append("")
    sub_df = dir_df[dir_df["n_pair"] >= 50].copy()
    md.append("**上位 5**")
    md.append("")
    md.append("| directory | category | n_pair | accept_rate | n_dev |")
    md.append("|---|---|---:|---:|---:|")
    for _, r in sub_df.nlargest(5, "accept_rate").iterrows():
        md.append(
            f"| {r['directory']} | {r['category']} | {r['n_pair']} | "
            f"**{r['accept_rate']:.3f}** | {r['n_unique_dev']} |"
        )
    md.append("")
    md.append("**下位 5**")
    md.append("")
    md.append("| directory | category | n_pair | accept_rate | n_dev |")
    md.append("|---|---|---:|---:|---:|")
    for _, r in sub_df.nsmallest(5, "accept_rate").iterrows():
        md.append(
            f"| {r['directory']} | {r['category']} | {r['n_pair']} | "
            f"**{r['accept_rate']:.3f}** | {r['n_unique_dev']} |"
        )
    md.append("")

    # top_dir stats
    md.append("## 3. Top_dir (≈ repo) 単位の集約 (pair 単位)")
    md.append("")
    md.append(f"- 対象 top_dir: {len(td_df)} (4 window 跨ぐ unique)")
    md.append("")
    md.append("### サンプルが多い top_dir top 10")
    md.append("")
    md.append("| top_dir | category | n_dir | n_pair | accept_rate | n_dev | pairs/dev |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in td_df.head(10).iterrows():
        md.append(
            f"| {r['top_dir']} | {r['category']} | {r['n_directory']} | "
            f"{r['n_pair']} | {r['accept_rate']:.3f} | "
            f"{r['n_unique_dev']} | {r['pairs_per_dev']:.1f} |"
        )
    md.append("")

    # Insights
    md.append("## 4. 主要観察")
    md.append("")
    md.append(f"- **project 間の承諾率レンジ**: "
              f"{proj['accept_rate'].min():.3f} ({proj.loc[proj['accept_rate'].idxmin(), 'repo']}) "
              f"〜 {proj['accept_rate'].max():.3f} "
              f"({proj.loc[proj['accept_rate'].idxmax(), 'repo']})")
    if proj["accept_rate"].max() - proj["accept_rate"].min() > 0.3:
        md.append(f"  → project 間で承諾文化に大きな差があり、これが予測精度の repo 依存性に寄与する可能性")
    md.append(f"- **dev 数のレンジ**: "
              f"{proj['n_unique_dev'].min()} 〜 {proj['n_unique_dev'].max()}")
    md.append(f"- **dev 一人あたり review 数**: "
              f"中央値 {proj['reviews_per_dev'].median():.1f}, "
              f"min={proj['reviews_per_dev'].min():.1f}, max={proj['reviews_per_dev'].max():.1f}")
    md.append(f"- **月次活動量の中央値レンジ**: "
              f"{proj['median_monthly_reviews'].min():.0f} 〜 {proj['median_monthly_reviews'].max():.0f}")
    md.append("")

    md.append("## 出力ファイル")
    md.append("- `project_stats.csv` : 32 project の基本統計 (評価期間)")
    md.append("- `project_monthly.csv` : project × 月の活動量")
    md.append("- `directory_stats.csv` : directory 単位の n_pair, accept_rate, n_dev 等")
    md.append("- `top_dir_stats.csv` : top_dir (repo) 単位の集約")
    md.append("- `fig_accept_rate_dist.pdf` : 承諾率分布 (project vs directory)")
    md.append("- `fig_repo_monthly.pdf` : 主要 repo の月次活動量推移 (レビュー件数/承諾 dev 数)")
    md.append("- `fig_dev_n_vs_accept.pdf` : dev 数 vs 承諾率 scatter")

    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/combined_raw_main32.csv")
    ap.add_argument("--base", default="outputs/main32_mce_volat_ep150")
    ap.add_argument("--variant", default="lstm_hybrid_snapshot")
    ap.add_argument("--main-repos-csv", default="data/service_teams_main_repos.csv")
    ap.add_argument("--output-dir", default="outputs/analysis/per_repo_basic_stats")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # main_repos
    mr_df = pd.read_csv(args.main_repos_csv).dropna(subset=["main_repo"])
    main_repos = set(mr_df["main_repo"].astype(str).str.split("/").str[-1]) - {"nan"}

    # Raw → project stats
    print(f"Loading raw data from {args.raw}...")
    raw = load_raw(Path(args.raw))
    print(f"  evaluation period rows: {len(raw):,}")

    print("Computing project stats...")
    proj = project_stats(raw)
    proj.to_csv(out_dir / "project_stats.csv", index=False)
    print(f"  -> project_stats.csv ({len(proj)} projects)")

    print("Computing project monthly stats...")
    monthly = project_monthly(raw)
    monthly.to_csv(out_dir / "project_monthly.csv", index=False)
    print(f"  -> project_monthly.csv ({len(monthly)} rows)")

    # Pair predictions → directory stats
    print(f"Loading pair predictions from {args.base}/{args.variant}...")
    pp = load_pair_pred(Path(args.base), args.variant)
    print(f"  pair_predictions rows (4 windows): {len(pp):,}")

    print("Computing directory stats...")
    dir_df = directory_stats(pp, main_repos)
    dir_df.to_csv(out_dir / "directory_stats.csv", index=False)
    print(f"  -> directory_stats.csv ({len(dir_df)} directories)")

    print("Computing top_dir stats...")
    td_df = top_dir_stats(pp, main_repos)
    td_df.to_csv(out_dir / "top_dir_stats.csv", index=False)
    print(f"  -> top_dir_stats.csv ({len(td_df)} top_dirs)")

    # Figures
    print("Generating figures...")
    fig_accept_rate_dist(proj, dir_df, out_dir / "fig_accept_rate_dist.pdf",
                          main_repos_csv=Path(args.main_repos_csv))
    print("  -> fig_accept_rate_dist.pdf")
    fig_repo_monthly(monthly, out_dir / "fig_repo_monthly.pdf", top_n_repos=None)
    print("  -> fig_repo_monthly.pdf (全 repo)")
    fig_dev_n_vs_accept(proj, out_dir / "fig_dev_n_vs_accept.pdf",
                         main_repos_csv=Path(args.main_repos_csv))
    print("  -> fig_dev_n_vs_accept.pdf")

    # README
    write_readme(proj, dir_df, td_df, out_dir)
    print("  -> README.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
