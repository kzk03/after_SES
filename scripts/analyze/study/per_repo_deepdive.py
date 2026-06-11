"""Repo 別予測精度傾向の深堀り分析。

per_dir_analysis.py の出力 (per_repo/) を受けて、以下を追加で分析する:

1. window 別の repo AUC 推移と時間安定性
2. tier (大/中/小/極小) 別の AUC 分布と検定
3. archetype × repo の構成比 (どんな archetype の dev がいる repo か)
4. cinder 異常性の深掘り (sub-dir 別、window 別)

出力 (outputs/analysis/per_repo_deepdive/):
  - repo_window_evolution.csv     : repo × window の AUC マトリクス
  - tier_summary.csv              : tier 別 AUC / gain 中央値と分散
  - tier_stats.json               : tier 群間検定
  - repo_archetype_mix.csv        : repo × archetype のペア数構成比
  - cinder_breakdown.csv          : cinder の sub-dir × window 詳細
  - fig_repo_window_heatmap.pdf   : repo × window の Ens AUC heatmap
  - fig_tier_boxplot.pdf          : tier 別 violin/box
  - fig_repo_archetype_stack.pdf  : repo 別 archetype 構成比 stacked bar
  - fig_cinder_breakdown.pdf      : cinder の sub-dir × window heatmap
  - README.md                     : 発見サマリ
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "sans-serif"]

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_diagonal(base: Path, variant: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for h in HORIZONS:
        p = base / variant / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d.dropna(subset=["irl_dir_prob_calibrated", "rf_dir_prob", "label"]).copy()
        d["ensemble"] = (d["irl_dir_prob_calibrated"] + d["rf_dir_prob"]) / 2.0
        d["window"] = h
        d["top_dir"] = d["directory"].astype(str).str.split("/").str[0]
        frames[h] = d
    return frames


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


# ----------------------------------------------------------------------
# 1. window 別の repo 推移
# ----------------------------------------------------------------------

def repo_window_evolution(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """各 repo × window で IRL/RF/Ens AUC を計算しワイド表で返す。"""
    rows = []
    for h, d in frames.items():
        for repo, g in d.groupby("top_dir"):
            y = g["label"].astype(int).to_numpy()
            n_pos = int(y.sum())
            n_neg = int((y == 0).sum())
            rows.append({
                "top_dir": repo, "window": h,
                "n_pair": len(g), "n_pos": n_pos, "n_neg": n_neg,
                "auc_irl": safe_auc(y, g["irl_dir_prob_calibrated"].to_numpy()),
                "auc_rf": safe_auc(y, g["rf_dir_prob"].to_numpy()),
                "auc_ens": safe_auc(y, g["ensemble"].to_numpy()),
            })
    long = pd.DataFrame(rows)
    return long


def fig_repo_window_heatmap(long: pd.DataFrame, out: Path, min_total_n: int = 100) -> None:
    """repo × window の IRL / RF / Ens AUC ヒートマップ 3 パネル。"""
    totals = long.groupby("top_dir")["n_pair"].sum()
    keep = totals[totals >= min_total_n].index

    pivots = {}
    for model in ["auc_irl", "auc_rf", "auc_ens"]:
        piv = long.pivot_table(index="top_dir", columns="window", values=model)
        piv = piv.reindex(columns=HORIZONS).loc[piv.index.intersection(keep)]
        pivots[model] = piv

    # Ens の降順で y 軸固定
    order = pivots["auc_ens"].dropna(how="all").sort_values(
        HORIZONS, ascending=False, na_position="last").index
    for k in pivots:
        pivots[k] = pivots[k].reindex(order)

    titles = {"auc_irl": "IRL AUC", "auc_rf": "RF AUC", "auc_ens": "Ensemble AUC"}
    fig, axes = plt.subplots(1, 3, figsize=(14, max(6, len(order) * 0.25)), sharey=True)
    for ax, key in zip(axes, ["auc_irl", "auc_rf", "auc_ens"]):
        piv = pivots[key]
        im = ax.imshow(piv.values, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels(HORIZONS, fontsize=8)
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels(piv.index, fontsize=7)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6, color="black" if 0.55 < v < 0.85 else "white")
        ax.set_title(titles[key])
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle(f"Repo × window AUC heatmap (n_pair total ≥ {min_total_n})", y=1.02)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# 2. tier 別集計
# ----------------------------------------------------------------------

def merge_tier(pooled: pd.DataFrame, main_repos_csv: Path) -> pd.DataFrame:
    mr = pd.read_csv(main_repos_csv).dropna(subset=["main_repo"])
    mr["top_dir"] = mr["main_repo"].astype(str).str.split("/").str[-1]
    tier_map = mr.set_index("top_dir")["tier"].to_dict()
    out = pooled.copy()
    out["tier"] = out["top_dir"].map(tier_map)
    out.loc[out["category"] == "cross-cutting", "tier"] = "cross-cutting"
    out["tier"] = out["tier"].fillna("unknown")
    return out


def tier_summary(pooled_t: pd.DataFrame) -> pd.DataFrame:
    rel = pooled_t.dropna(subset=["auc_ens"]).copy()
    rows = []
    for tier, g in rel.groupby("tier"):
        rows.append({
            "tier": tier,
            "n_repo": len(g),
            "total_n_pair": int(g["n_pair"].sum()),
            "median_auc_irl": float(g["auc_irl"].median()),
            "median_auc_rf": float(g["auc_rf"].median()),
            "median_auc_ens": float(g["auc_ens"].median()),
            "median_gain": float(g["gain_vs_best"].median()),
            "auc_irl_q25": float(g["auc_irl"].quantile(0.25)),
            "auc_irl_q75": float(g["auc_irl"].quantile(0.75)),
            "auc_rf_q25": float(g["auc_rf"].quantile(0.25)),
            "auc_rf_q75": float(g["auc_rf"].quantile(0.75)),
            "auc_ens_q25": float(g["auc_ens"].quantile(0.25)),
            "auc_ens_q75": float(g["auc_ens"].quantile(0.75)),
        })
    return pd.DataFrame(rows).sort_values("median_auc_ens", ascending=False)


def tier_stats_tests(pooled_t: pd.DataFrame) -> dict:
    rel = pooled_t.dropna(subset=["auc_ens"]).copy()
    rel = rel[rel["tier"].isin(["大", "中", "小", "極小"])]
    groups = {t: rel[rel["tier"] == t]["auc_ens"].to_numpy() for t in ["大", "中", "小", "極小"]}
    out = {
        "n_per_tier": {t: int(len(v)) for t, v in groups.items()},
    }
    valid = [v for v in groups.values() if len(v) >= 2]
    if len(valid) >= 2:
        h, p = kruskal(*valid)
        out["kruskal_wallis"] = {"H": float(h), "p": float(p), "n_groups": len(valid)}
    # 大 vs 極小 pairwise
    if len(groups["大"]) >= 2 and len(groups["極小"]) >= 2:
        u, p = mannwhitneyu(groups["大"], groups["極小"], alternative="greater")
        out["mannwhitney_dai_vs_kyokushou"] = {
            "U": float(u), "p_one_sided": float(p),
            "median_dai": float(np.median(groups["大"])),
            "median_kyokushou": float(np.median(groups["極小"])),
        }
    return out


def fig_tier_box(pooled_t: pd.DataFrame, out: Path) -> None:
    rel = pooled_t.dropna(subset=["auc_ens"]).copy()
    tier_order = ["大", "中", "小", "極小", "cross-cutting"]
    data = [rel[rel["tier"] == t]["auc_ens"].to_numpy() for t in tier_order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, tick_labels=[f"{t}\n(n={len(d)})" for t, d in zip(tier_order, data)],
                    patch_artist=True, widths=0.6)
    colors = ["#2E7D32", "#558B2F", "#9E9D24", "#E65100", "#5D4037"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    # 各点も散布
    for i, arr in enumerate(data):
        jitter = (np.random.default_rng(seed=i).random(len(arr)) - 0.5) * 0.18
        ax.scatter(np.full_like(arr, i + 1) + jitter, arr, s=15, color="#222", alpha=0.6)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("Pooled Ens AUC")
    ax.set_title("Tier 別 repo の Ens AUC 分布")
    ax.set_ylim(0.1, 1.0)
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. repo × archetype 構成比
# ----------------------------------------------------------------------

def repo_archetype_mix(
    frames: dict[str, pd.DataFrame], archetype_csv: Path,
    dir_project_map_csv: Path | None = None,
    group_col: str = "directory",
) -> pd.DataFrame:
    """archetype 構成比を group_col 単位で集計。

    group_col="directory" で repo (depth=2 path) 単位、"top_dir" で project 単位。
    dir_project_map_csv があれば directory ごとの主要 project / shared フラグを merge。
    """
    if not archetype_csv.exists():
        return pd.DataFrame()
    arch = pd.read_csv(archetype_csv)
    arch_lookup = arch.set_index(["developer", "window"])["dominant"].to_dict()
    big = pd.concat(frames.values(), axis=0, ignore_index=True)
    big["archetype"] = big.apply(
        lambda r: arch_lookup.get((r["developer"], r["window"]), None), axis=1,
    )
    big = big.dropna(subset=["archetype"])
    pivot = (
        big.groupby([group_col, "archetype"]).size()
        .unstack(fill_value=0)
        .reindex(columns=["A1", "A2", "A3", "A4"], fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot[pivot["total"] >= 50].sort_values("total", ascending=False)
    for a in ["A1", "A2", "A3", "A4"]:
        pivot[f"{a}_pct"] = pivot[a] / pivot["total"] * 100
    df = pivot.reset_index()

    # directory_project_map を merge
    if group_col == "directory" and dir_project_map_csv and dir_project_map_csv.exists():
        dp = pd.read_csv(dir_project_map_csv)
        df = df.merge(
            dp[["directory", "top1_project", "top1_pct", "category", "n_projects"]],
            on="directory", how="left",
        )
    return df


def fig_repo_archetype_stack(mix: pd.DataFrame, out: Path, top_n: int = 20) -> None:
    if mix.empty:
        return
    # mix may use either "directory" or "top_dir" as group column
    group_col = "directory" if "directory" in mix.columns else "top_dir"
    show = mix.head(top_n).set_index(group_col)
    pct = show[["A1_pct", "A2_pct", "A3_pct", "A4_pct"]]
    colors = {"A1_pct": "#D9543E", "A2_pct": "#E89C3F", "A3_pct": "#5BA85D", "A4_pct": "#4474B6"}
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = np.zeros(len(show))
    for col in ["A1_pct", "A2_pct", "A3_pct", "A4_pct"]:
        ax.bar(
            show.index, pct[col], bottom=bottom, label=col.replace("_pct", ""),
            color=colors[col], edgecolor="white", linewidth=0.5,
        )
        bottom += pct[col].to_numpy()
    ax.set_ylabel("Archetype 構成比 (%)")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(show.index, rotation=45, ha="right", fontsize=9)
    ax.legend(loc="upper right", title="Archetype", ncol=4, fontsize=9)
    ax.set_title(f"Repo 別 archetype 構成比 (上位 {len(show)} repo, total_pair ≥ 50)")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. cinder の深掘り
# ----------------------------------------------------------------------

def cinder_breakdown(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for h, d in frames.items():
        sub = d[d["top_dir"] == "cinder"]
        for dir_name, g in sub.groupby("directory"):
            y = g["label"].astype(int).to_numpy()
            n_pos = int(y.sum())
            n_neg = int((y == 0).sum())
            if n_pos == 0 or n_neg == 0:
                continue
            rows.append({
                "directory": dir_name, "window": h,
                "n_pair": len(g), "n_pos": n_pos, "n_neg": n_neg,
                "auc_irl": safe_auc(y, g["irl_dir_prob_calibrated"].to_numpy()),
                "auc_rf": safe_auc(y, g["rf_dir_prob"].to_numpy()),
                "auc_ens": safe_auc(y, g["ensemble"].to_numpy()),
            })
    df = pd.DataFrame(rows)
    df["delta"] = df["auc_irl"] - df["auc_rf"]
    return df.sort_values(["directory", "window"])


def fig_cinder_breakdown(cb: pd.DataFrame, out: Path) -> None:
    if cb.empty:
        return
    pivot_irl = cb.pivot_table(index="directory", columns="window", values="auc_irl").reindex(columns=HORIZONS)
    pivot_rf = cb.pivot_table(index="directory", columns="window", values="auc_rf").reindex(columns=HORIZONS)
    pivot_delta = cb.pivot_table(index="directory", columns="window", values="delta").reindex(columns=HORIZONS)
    pivot_n = cb.pivot_table(index="directory", columns="window", values="n_pair").reindex(columns=HORIZONS)
    order = pivot_n.sum(axis=1).sort_values(ascending=False).index
    pivot_irl = pivot_irl.loc[order]
    pivot_rf = pivot_rf.loc[order]
    pivot_delta = pivot_delta.loc[order]

    fig, axes = plt.subplots(1, 3, figsize=(14, max(4, len(order) * 0.28)))
    for ax, piv, title, vmin, vmax, cmap in [
        (axes[0], pivot_irl, "IRL AUC", 0.4, 1.0, "RdYlGn"),
        (axes[1], pivot_rf, "RF AUC", 0.4, 1.0, "RdYlGn"),
        (axes[2], pivot_delta, "Δ = IRL−RF", -0.2, 0.2, "RdBu"),
    ]:
        im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels(HORIZONS, fontsize=8)
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}" if "Δ" in title else f"{v:.2f}",
                            ha="center", va="center", fontsize=6,
                            color="black" if (cmap == "RdBu" and abs(v) < 0.15) or
                                              (cmap == "RdYlGn" and 0.55 < v < 0.85) else "white")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.6)
    fig.suptitle("Cinder: sub-directory × window の AUC 詳細", y=1.02)
    fig.tight_layout()
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def write_readme(
    long: pd.DataFrame, pooled_t: pd.DataFrame, tier_summ: pd.DataFrame, tier_stats: dict,
    mix: pd.DataFrame, cb: pd.DataFrame, out_dir: Path,
) -> None:
    md = []
    md.append("# Per-Repository Deep Dive Analysis")
    md.append("")
    md.append("`outputs/analysis/per_repo/` の repo 単位集計結果を、")
    md.append("(1) window 別推移, (2) tier 別集計, (3) archetype 構成比, (4) cinder 深掘り")
    md.append("の 4 軸で深堀りした補足分析。")
    md.append("")

    # 1. window 別
    rel = long.dropna(subset=["auc_ens"]).copy()
    md.append("## 1. Window 別の repo 推移")
    md.append("")
    md.append(f"- repo × window cell 数: {len(rel)} (auc 算出可能のみ)")
    pivot_ens = rel.pivot_table(index="top_dir", columns="window", values="auc_ens")
    pivot_irl = rel.pivot_table(index="top_dir", columns="window", values="auc_irl")
    pivot_rf = rel.pivot_table(index="top_dir", columns="window", values="auc_rf")
    pivot_ens = pivot_ens.reindex(columns=HORIZONS)
    pivot_ens["std"] = pivot_ens[HORIZONS].std(axis=1)
    pivot_ens["mean"] = pivot_ens[HORIZONS].mean(axis=1)
    pivot_ens = pivot_ens.dropna(subset=["std"])
    most_volatile = pivot_ens.sort_values("std", ascending=False).head(5)
    most_stable = pivot_ens[pivot_ens["mean"] > 0.6].sort_values("std").head(5)

    def render_row(repo, row, pivot_i, pivot_r):
        parts = [f"| {repo} | {row['mean']:.3f} | {row['std']:.3f} |"]
        for model_name, piv in [("IRL", pivot_i), ("RF", pivot_r), ("Ens", None)]:
            for w in HORIZONS:
                if model_name == "Ens":
                    v = row[w]
                else:
                    v = piv.loc[repo, w] if repo in piv.index and w in piv.columns else np.nan
                parts.append(f" {v:.3f} |" if pd.notna(v) else " — |")
        return "".join(parts)

    md.append("### 時系列変動の大きい repo (Ens AUC std 上位 5、IRL/RF/Ens 全 window 値)")
    md.append("")
    header = "| top_dir | Ens mean | Ens std | " \
             + " | ".join(f"IRL {w}" for w in HORIZONS) + " | " \
             + " | ".join(f"RF {w}" for w in HORIZONS) + " | " \
             + " | ".join(f"Ens {w}" for w in HORIZONS) + " |"
    md.append(header)
    md.append("|---|---:|---:|" + "---:|" * 12)
    for repo, row in most_volatile.iterrows():
        md.append(render_row(repo, row, pivot_irl, pivot_rf))
    md.append("")
    md.append("### 時系列安定な repo (Ens mean > 0.6 で std 最小 5、IRL/RF/Ens 全 window 値)")
    md.append("")
    md.append(header)
    md.append("|---|---:|---:|" + "---:|" * 12)
    for repo, row in most_stable.iterrows():
        md.append(render_row(repo, row, pivot_irl, pivot_rf))
    md.append("")

    # 2. tier
    md.append("## 2. Tier 別の AUC 分布")
    md.append("")
    md.append("| tier | n_repo | total_n_pair | median IRL | median RF | median Ens | median gain |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in tier_summ.iterrows():
        md.append(
            f"| {r['tier']} | {r['n_repo']} | {r['total_n_pair']} | "
            f"{r['median_auc_irl']:.3f} | {r['median_auc_rf']:.3f} | "
            f"**{r['median_auc_ens']:.3f}** | {r['median_gain']:+.3f} |"
        )
    md.append("")
    if "kruskal_wallis" in tier_stats:
        kw = tier_stats["kruskal_wallis"]
        verdict = "有意差あり" if kw["p"] < 0.05 else "有意差なし"
        md.append(
            f"- **Kruskal-Wallis (大/中/小/極小)**: H = {kw['H']:.3f}, "
            f"p = {kw['p']:.3g} → {verdict}"
        )
    if "mannwhitney_dai_vs_kyokushou" in tier_stats:
        mw = tier_stats["mannwhitney_dai_vs_kyokushou"]
        md.append(
            f"- **Mann-Whitney 大 > 極小** (one-sided): U = {mw['U']:.0f}, "
            f"p = {mw['p_one_sided']:.3g} "
            f"(median 大={mw['median_dai']:.3f} vs 極小={mw['median_kyokushou']:.3f})"
        )
    md.append("")

    # 3. archetype
    if not mix.empty:
        md.append("## 3. Repo × Archetype 構成比 (total_pair ≥ 50 の repo)")
        md.append("")
        md.append(f"- 対象 repo 数: {len(mix)}")
        md.append("")
        md.append("### A1 (Sporadic Newcomer) 比率の高い repo top 5")
        a1_top = mix.sort_values("A1_pct", ascending=False).head(5)
        md.append("| top_dir | total | A1% | A2% | A3% | A4% |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for _, r in a1_top.iterrows():
            md.append(
                f"| {r.get('top_dir', r.get('directory', ''))} | {r['total']} | "
                f"**{r['A1_pct']:.1f}** | {r['A2_pct']:.1f} | "
                f"{r['A3_pct']:.1f} | {r['A4_pct']:.1f} |"
            )
        md.append("")
        md.append("### A3 (Approving Veteran) 比率の高い repo top 5")
        a3_top = mix.sort_values("A3_pct", ascending=False).head(5)
        md.append("| top_dir | total | A1% | A2% | A3% | A4% |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for _, r in a3_top.iterrows():
            md.append(
                f"| {r.get('top_dir', r.get('directory', ''))} | {r['total']} | "
                f"{r['A1_pct']:.1f} | {r['A2_pct']:.1f} | "
                f"**{r['A3_pct']:.1f}** | {r['A4_pct']:.1f} |"
            )
        md.append("")
        md.append("### A2 (Strict Reviewer-Only) 比率の高い repo top 5")
        a2_top = mix.sort_values("A2_pct", ascending=False).head(5)
        md.append("| top_dir | total | A1% | A2% | A3% | A4% |")
        md.append("|---|---:|---:|---:|---:|---:|")
        for _, r in a2_top.iterrows():
            md.append(
                f"| {r.get('top_dir', r.get('directory', ''))} | {r['total']} | "
                f"{r['A1_pct']:.1f} | **{r['A2_pct']:.1f}** | "
                f"{r['A3_pct']:.1f} | {r['A4_pct']:.1f} |"
            )
        md.append("")

    # 4. cinder
    if not cb.empty:
        md.append("## 4. Cinder の異常性 (RF 一貫優位) の深掘り")
        md.append("")
        md.append(f"- sub-directory × window cell 数: {len(cb)}")
        md.append(f"- Δ (=IRL−RF) 中央値: {cb['delta'].median():+.3f}")
        md.append(f"- Δ 最大 (IRL 寄り): {cb['delta'].max():+.3f}")
        md.append(f"- Δ 最小 (RF 寄り): {cb['delta'].min():+.3f}")
        md.append("")
        # 最も RF 優位な cell top 5
        md.append("### RF が大幅優位な cinder sub-dir × window (Δ 小さい top 5)")
        md.append("")
        md.append("| directory | window | n_pair | IRL | RF | Δ |")
        md.append("|---|---|---:|---:|---:|---:|")
        for _, r in cb.nsmallest(5, "delta").iterrows():
            md.append(
                f"| {r['directory']} | {r['window']} | {r['n_pair']} | "
                f"{r['auc_irl']:.3f} | {r['auc_rf']:.3f} | {r['delta']:+.3f} |"
            )
        md.append("")

    md.append("## 出力ファイル")
    md.append("- `repo_window_evolution.csv` : repo × window の生メトリクス")
    md.append("- `tier_summary.csv` : tier 別 AUC / gain 中央値")
    md.append("- `tier_stats.json` : tier 群間 Kruskal-Wallis / Mann-Whitney")
    md.append("- `repo_archetype_mix.csv` : repo × archetype のペア数構成比")
    md.append("- `cinder_breakdown.csv` : cinder sub-dir × window 詳細")
    md.append("- `fig_repo_window_heatmap.pdf` : repo × window の Ens AUC heatmap")
    md.append("- `fig_tier_boxplot.pdf` : tier 別 boxplot")
    md.append("- `fig_repo_archetype_stack.pdf` : repo 別 archetype 構成比 stacked bar")
    md.append("- `fig_cinder_breakdown.pdf` : cinder の sub-dir × window heatmap")

    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/main32_mce_volat_ep150")
    ap.add_argument("--variant", default="lstm_hybrid_snapshot")
    ap.add_argument("--main-repos-csv", default="data/service_teams_main_repos.csv")
    ap.add_argument("--pooled-csv", default="outputs/analysis/per_repo/top_dir_pooled_metrics.csv")
    ap.add_argument(
        "--archetype-csv",
        default="outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv",
    )
    ap.add_argument("--output-dir", default="outputs/analysis/per_repo_deepdive")
    args = ap.parse_args()

    base = Path(args.base)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading from {base}/{args.variant}...")
    frames = load_diagonal(base, args.variant)
    print(f"  {len(frames)} windows loaded")

    # 1. window evolution
    print("1. Repo × window evolution...")
    long = repo_window_evolution(frames)
    long.to_csv(out_dir / "repo_window_evolution.csv", index=False)
    fig_repo_window_heatmap(long, out_dir / "fig_repo_window_heatmap.pdf")
    print("   -> repo_window_evolution.csv, fig_repo_window_heatmap.pdf")

    # 2. tier
    print("2. Tier summary...")
    pooled = pd.read_csv(args.pooled_csv)
    pooled_t = merge_tier(pooled, Path(args.main_repos_csv))
    pooled_t.to_csv(out_dir / "pooled_with_tier.csv", index=False)
    tier_summ = tier_summary(pooled_t)
    tier_summ.to_csv(out_dir / "tier_summary.csv", index=False)
    tier_stats = tier_stats_tests(pooled_t)
    (out_dir / "tier_stats.json").write_text(
        json.dumps(tier_stats, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    fig_tier_box(pooled_t, out_dir / "fig_tier_boxplot.pdf")
    print("   -> tier_summary.csv, tier_stats.json, fig_tier_boxplot.pdf")

    # 3. archetype mix
    print("3. Repo × archetype mix...")
    mix = repo_archetype_mix(frames, Path(args.archetype_csv))
    if not mix.empty:
        mix.to_csv(out_dir / "repo_archetype_mix.csv", index=False)
        fig_repo_archetype_stack(mix, out_dir / "fig_repo_archetype_stack.pdf")
        print("   -> repo_archetype_mix.csv, fig_repo_archetype_stack.pdf")
    else:
        print("   (skipped: archetype csv not found)")

    # 4. cinder breakdown
    print("4. Cinder deep dive...")
    cb = cinder_breakdown(frames)
    if not cb.empty:
        cb.to_csv(out_dir / "cinder_breakdown.csv", index=False)
        fig_cinder_breakdown(cb, out_dir / "fig_cinder_breakdown.pdf")
        print("   -> cinder_breakdown.csv, fig_cinder_breakdown.pdf")

    # README
    write_readme(long, pooled_t, tier_summ, tier_stats, mix, cb, out_dir)
    print("   -> README.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
