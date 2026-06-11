"""ディレクトリ (モジュール) 別の予測精度傾向を分析する。

入力:
  - outputs/main32_mce_volat_ep150/lstm_hybrid_snapshot/train_W/eval_W/pair_predictions.csv
    (W in 0-3m, 3-6m, 6-9m, 9-12m の対角 4 セル)
  - data/service_teams_main_repos.csv  (32 main repos のリスト)

出力 (outputs/analysis/per_dir/ 配下):
  - dir_window_metrics.csv    : directory x window x {IRL, RF, Ens} の AUC, n_pair, n_pos, n_neg
  - dir_pooled_metrics.csv    : directory x {IRL, RF, Ens} の 4 window pool AUC + 95% bootstrap CI + gain
  - stats_test.json           : Spearman / Mann-Whitney / paired Wilcoxon の検定結果
  - fig_dir_auc_ranked.pdf    : Top/Bottom directory ranked bar
  - fig_category_violin.pdf   : project-specific vs cross-cutting violin
  - fig_n_vs_auc_scatter.pdf  : log(n_pair) vs Ens AUC scatter
  - README.md                 : 発見サマリ
"""

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["font.family"] = ["Hiragino Sans", "Yu Gothic", "Meiryo", "sans-serif"]

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_diagonal(base: Path, variant: str) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for h in HORIZONS:
        p = base / variant / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        if not p.exists():
            print(f"  missing: {p}")
            continue
        d = pd.read_csv(p)
        d = d.dropna(subset=["irl_dir_prob_calibrated", "rf_dir_prob", "label"]).copy()
        d["ensemble"] = (d["irl_dir_prob_calibrated"] + d["rf_dir_prob"]) / 2.0
        d["window"] = h
        frames[h] = d[
            ["developer", "directory", "label", "window",
             "irl_dir_prob_calibrated", "rf_dir_prob", "ensemble"]
        ]
    return frames


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return np.nan


def bootstrap_auc_ci(
    y: np.ndarray, p_irl: np.ndarray, p_rf: np.ndarray, p_ens: np.ndarray,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 777,
) -> dict[str, float]:
    """directory pooled 用の bootstrap CI を 3 モデル × 1 gain で計算。"""
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    auc_irl = safe_auc(y, p_irl)
    auc_rf = safe_auc(y, p_rf)
    auc_ens = safe_auc(y, p_ens)
    if not (np.isfinite(auc_irl) and np.isfinite(auc_rf) and np.isfinite(auc_ens)):
        return {
            "auc_irl": auc_irl, "auc_rf": auc_rf, "auc_ens": auc_ens,
            "gain_vs_best": np.nan,
            "ci_irl_low": np.nan, "ci_irl_high": np.nan,
            "ci_rf_low": np.nan, "ci_rf_high": np.nan,
            "ci_ens_low": np.nan, "ci_ens_high": np.nan,
            "ci_gain_low": np.nan, "ci_gain_high": np.nan,
        }
    gain = auc_ens - max(auc_irl, auc_rf)

    irls, rfs, enss, gains = [], [], [], []
    for _ in range(n_boot):
        bp = rng.choice(pos_idx, len(pos_idx), replace=True)
        bn = rng.choice(neg_idx, len(neg_idx), replace=True)
        idx = np.concatenate([bp, bn])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        try:
            ai = roc_auc_score(yb, p_irl[idx])
            ar = roc_auc_score(yb, p_rf[idx])
            ae = roc_auc_score(yb, p_ens[idx])
            irls.append(ai); rfs.append(ar); enss.append(ae)
            gains.append(ae - max(ai, ar))
        except Exception:
            continue

    def q(a, lo):
        a = np.asarray(a)
        return float(np.quantile(a, lo)) if len(a) else np.nan

    return {
        "auc_irl": auc_irl, "auc_rf": auc_rf, "auc_ens": auc_ens,
        "gain_vs_best": gain,
        "ci_irl_low": q(irls, alpha / 2), "ci_irl_high": q(irls, 1 - alpha / 2),
        "ci_rf_low": q(rfs, alpha / 2), "ci_rf_high": q(rfs, 1 - alpha / 2),
        "ci_ens_low": q(enss, alpha / 2), "ci_ens_high": q(enss, 1 - alpha / 2),
        "ci_gain_low": q(gains, alpha / 2), "ci_gain_high": q(gains, 1 - alpha / 2),
    }


def compute_dir_window_metrics(
    frames: dict[str, pd.DataFrame], group_col: str = "directory",
) -> pd.DataFrame:
    rows = []
    for h, d in frames.items():
        if group_col == "top_dir" and "top_dir" not in d.columns:
            d = d.copy()
            d["top_dir"] = d["directory"].astype(str).str.split("/").str[0]
        for gkey, g in d.groupby(group_col):
            y = g["label"].astype(int).to_numpy()
            n_pos = int(y.sum())
            n_neg = int((y == 0).sum())
            n = len(y)
            rows.append({
                group_col: gkey, "window": h,
                "n_pair": n, "n_pos": n_pos, "n_neg": n_neg,
                "auc_irl": safe_auc(y, g["irl_dir_prob_calibrated"].to_numpy()),
                "auc_rf": safe_auc(y, g["rf_dir_prob"].to_numpy()),
                "auc_ens": safe_auc(y, g["ensemble"].to_numpy()),
                "auc_reliable": (n_pos >= 30) and (n_neg >= 30),
            })
    return pd.DataFrame(rows)


def categorize_directory(dir_name: str, main_repos: set[str]) -> tuple[str, str]:
    """directory 名から (top_dir, category) を返す。

    category は top_dir が main_repos の 32 repo 名のいずれかと一致すれば
    "project-specific"、そうでなければ "cross-cutting"。
    """
    top = str(dir_name).split("/")[0] if isinstance(dir_name, str) else "unknown"
    cat = "project-specific" if top in main_repos else "cross-cutting"
    return top, cat


def compute_dir_pooled_metrics(
    frames: dict[str, pd.DataFrame], main_repos: set[str],
    group_col: str = "directory", n_boot: int = 1000,
) -> pd.DataFrame:
    big = pd.concat(frames.values(), axis=0, ignore_index=True)
    if group_col == "top_dir":
        big = big.copy()
        big["top_dir"] = big["directory"].astype(str).str.split("/").str[0]
    rows = []
    for gkey, g in big.groupby(group_col):
        y = g["label"].astype(int).to_numpy()
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        n = len(y)
        if group_col == "directory":
            top_dir, cat = categorize_directory(gkey, main_repos)
        else:
            top_dir = gkey
            cat = "project-specific" if gkey in main_repos else "cross-cutting"
        stats = bootstrap_auc_ci(
            y,
            g["irl_dir_prob_calibrated"].to_numpy(),
            g["rf_dir_prob"].to_numpy(),
            g["ensemble"].to_numpy(),
            n_boot=n_boot,
        )
        row = {
            group_col: gkey,
            "category": cat,
            "n_pair": n, "n_pos": n_pos, "n_neg": n_neg,
            "auc_reliable": (n_pos >= 30) and (n_neg >= 30),
            **stats,
        }
        if group_col == "directory":
            row["top_dir"] = top_dir
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("n_pair", ascending=False).reset_index(drop=True)
    return df


def run_stats_tests(pooled: pd.DataFrame, use_reliable_filter: bool = True) -> dict:
    if use_reliable_filter:
        rel = pooled[pooled["auc_reliable"]].copy()
    else:
        # AUC が NaN でない (両クラス >=1) ものに最低限フィルタ
        rel = pooled.dropna(subset=["auc_irl", "auc_rf", "auc_ens"]).copy()
    results = {
        "n_used": int(len(rel)), "n_total": int(len(pooled)),
        "filter": "reliable (n_pos>=30 & n_neg>=30)" if use_reliable_filter
                  else "no reliable filter (auc computable only)",
    }

    # Spearman: log(n_pair) vs auc_ens
    if len(rel) >= 5:
        rho_n_ens, p_n_ens = spearmanr(np.log10(rel["n_pair"]), rel["auc_ens"])
        rho_n_gain, p_n_gain = spearmanr(np.log10(rel["n_pair"]), rel["gain_vs_best"])
        results["spearman_logn_vs_auc_ens"] = {"rho": float(rho_n_ens), "p": float(p_n_ens)}
        results["spearman_logn_vs_gain"] = {"rho": float(rho_n_gain), "p": float(p_n_gain)}

    # Mann-Whitney: category 別 AUC 差
    ps = rel[rel["category"] == "project-specific"]["auc_ens"].to_numpy()
    cc = rel[rel["category"] == "cross-cutting"]["auc_ens"].to_numpy()
    if len(ps) >= 3 and len(cc) >= 3:
        u, p = mannwhitneyu(ps, cc, alternative="two-sided")
        results["mannwhitney_category_auc_ens"] = {
            "U": float(u), "p": float(p),
            "n_project_specific": int(len(ps)),
            "n_cross_cutting": int(len(cc)),
            "median_project_specific": float(np.median(ps)),
            "median_cross_cutting": float(np.median(cc)),
        }
        # gain も比較
        ps_g = rel[rel["category"] == "project-specific"]["gain_vs_best"].to_numpy()
        cc_g = rel[rel["category"] == "cross-cutting"]["gain_vs_best"].to_numpy()
        u_g, p_g = mannwhitneyu(ps_g, cc_g, alternative="two-sided")
        results["mannwhitney_category_gain"] = {
            "U": float(u_g), "p": float(p_g),
            "median_project_specific": float(np.median(ps_g)),
            "median_cross_cutting": float(np.median(cc_g)),
        }

    # paired Wilcoxon across directories: Ens vs IRL, Ens vs RF
    if len(rel) >= 5:
        w_ei = wilcoxon(rel["auc_ens"], rel["auc_irl"], alternative="two-sided")
        w_er = wilcoxon(rel["auc_ens"], rel["auc_rf"], alternative="two-sided")
        # rank-biserial 効果量
        def rank_biserial(x, y):
            d = x - y
            d = d[d != 0]
            if len(d) == 0:
                return np.nan
            return float(np.sum(d > 0) / len(d) * 2 - 1)
        results["wilcoxon_ens_vs_irl"] = {
            "W": float(w_ei.statistic), "p": float(w_ei.pvalue),
            "p_bonferroni": float(min(w_ei.pvalue * 2, 1.0)),
            "r_rank_biserial": rank_biserial(
                rel["auc_ens"].to_numpy(), rel["auc_irl"].to_numpy()
            ),
            "n": int(len(rel)),
        }
        results["wilcoxon_ens_vs_rf"] = {
            "W": float(w_er.statistic), "p": float(w_er.pvalue),
            "p_bonferroni": float(min(w_er.pvalue * 2, 1.0)),
            "r_rank_biserial": rank_biserial(
                rel["auc_ens"].to_numpy(), rel["auc_rf"].to_numpy()
            ),
            "n": int(len(rel)),
        }
    return results


def fig_ranked_bar(
    pooled: pd.DataFrame, out: Path,
    group_col: str = "directory", use_reliable_filter: bool = True,
    top_n: int = 10,
) -> None:
    if use_reliable_filter:
        rel = pooled[pooled["auc_reliable"]].copy()
    else:
        rel = pooled.dropna(subset=["auc_ens"]).copy()
    rel = rel.sort_values("auc_ens", ascending=False)
    if len(rel) <= top_n * 2:
        show = rel.copy().reset_index(drop=True)
    else:
        top = rel.head(top_n)
        bot = rel.tail(top_n)
        show = pd.concat([top, bot]).reset_index(drop=True)
    n = len(show)
    x = np.arange(n)
    width = 0.28
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars_irl = ax.bar(x - width, show["auc_irl"], width, label="IRL", color="#4C72B0")
    bars_rf = ax.bar(x, show["auc_rf"], width, label="RF", color="#DD8452")
    bars_ens = ax.bar(x + width, show["auc_ens"], width, label="Ensemble", color="#55A868")
    # Ens の CI 誤差棒
    yerr_lo = show["auc_ens"] - show["ci_ens_low"]
    yerr_hi = show["ci_ens_high"] - show["auc_ens"]
    ax.errorbar(
        x + width, show["auc_ens"], yerr=[yerr_lo, yerr_hi],
        fmt="none", ecolor="#333", capsize=2, lw=0.8,
    )
    # category で xtick 色分け
    colors = {"project-specific": "#3F5D7D", "cross-cutting": "#A36E3F"}
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{d}\n(n={n_})" for d, n_ in zip(show[group_col], show["n_pair"])],
        rotation=45, ha="right", fontsize=8,
    )
    for tick, cat in zip(ax.get_xticklabels(), show["category"]):
        tick.set_color(colors.get(cat, "#000"))
    if len(rel) > top_n * 2:
        ax.axvline(top_n - 0.5, color="gray", lw=0.6, ls="--")
        ax.text(top_n / 2 - 0.5, 0.98, f"Top {top_n}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color="gray")
        ax.text(top_n + top_n / 2 - 0.5, 0.98, f"Bottom {top_n}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10, color="gray")
    ax.set_ylabel("Pooled AUC (4 windows)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.legend(loc="lower left", fontsize=9, ncol=3)
    level = "Repository (top_dir)" if group_col == "top_dir" else "Directory"
    filter_note = "auc_reliable only" if use_reliable_filter else "no reliability filter"
    ax.set_title(f"{level}-level pooled AUC ({filter_note})")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


def fig_category_violin(
    pooled: pd.DataFrame, stats_results: dict, out: Path,
    use_reliable_filter: bool = True,
) -> None:
    if use_reliable_filter:
        rel = pooled[pooled["auc_reliable"]].copy()
    else:
        rel = pooled.dropna(subset=["auc_ens"]).copy()
    cats = ["project-specific", "cross-cutting"]
    data = [rel[rel["category"] == c]["auc_ens"].to_numpy() for c in cats]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    parts = ax.violinplot(data, showmedians=True, widths=0.7)
    for body, color in zip(parts["bodies"], ["#3F5D7D", "#A36E3F"]):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    # 個別点も散布
    for i, arr in enumerate(data):
        jitter = (np.random.default_rng(seed=i).random(len(arr)) - 0.5) * 0.15
        ax.scatter(np.full_like(arr, i + 1) + jitter, arr, s=14, color="#333", alpha=0.5)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{c}\n(n={len(d)})" for c, d in zip(cats, data)])
    ax.set_ylabel("Pooled Ensemble AUC")
    mw = stats_results.get("mannwhitney_category_auc_ens", {})
    if mw:
        ax.set_title(
            f"Directory category vs Ens AUC\n"
            f"Mann-Whitney U={mw.get('U', float('nan')):.0f}, p={mw.get('p', float('nan')):.3g}"
        )
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


def fig_scatter(
    pooled: pd.DataFrame, stats_results: dict, out: Path,
    group_col: str = "directory", use_reliable_filter: bool = True,
) -> None:
    if use_reliable_filter:
        rel = pooled[pooled["auc_reliable"]].copy()
    else:
        rel = pooled.dropna(subset=["auc_ens"]).copy()
    colors = {"project-specific": "#3F5D7D", "cross-cutting": "#A36E3F"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for cat, c in colors.items():
        sub = rel[rel["category"] == cat]
        ax.scatter(
            np.log10(sub["n_pair"]), sub["auc_ens"],
            s=50, color=c, alpha=0.7, label=f"{cat} (n_dir={len(sub)})",
            edgecolors="white", linewidths=0.5,
        )
    # 上位 / 下位 5 directory にラベル
    top5 = rel.nlargest(5, "auc_ens")
    bot5 = rel.nsmallest(5, "auc_ens")
    for _, row in pd.concat([top5, bot5]).iterrows():
        ax.annotate(
            row[group_col], (np.log10(row["n_pair"]), row["auc_ens"]),
            fontsize=7, xytext=(3, 3), textcoords="offset points",
        )
    sp = stats_results.get("spearman_logn_vs_auc_ens", {})
    if sp:
        ax.text(
            0.05, 0.95,
            f"Spearman ρ = {sp.get('rho', float('nan')):.3f}, p = {sp.get('p', float('nan')):.3g}",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.8),
        )
    ax.set_xlabel("log₁₀(n_pair)")
    ax.set_ylabel("Pooled Ensemble AUC")
    ax.axhline(0.5, color="gray", lw=0.5, ls=":")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Sample size vs Ensemble AUC (directory level)")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)


def write_readme(
    pooled: pd.DataFrame, stats_results: dict, out_dir: Path, base: Path, variant: str,
    group_col: str = "directory", use_reliable_filter: bool = True,
) -> None:
    if use_reliable_filter:
        rel = pooled[pooled["auc_reliable"]].copy()
    else:
        rel = pooled.dropna(subset=["auc_ens"]).copy()
    top = rel.nlargest(5, "auc_ens")
    bot = rel.nsmallest(5, "auc_ens")

    md = []
    md.append("# Directory-level Prediction Trend Analysis")
    md.append("")
    md.append("## 実行")
    md.append("")
    md.append("```bash")
    md.append("uv run python scripts/analyze/study/per_dir_analysis.py")
    md.append("```")
    md.append("")
    md.append("## データ概要")
    md.append(f"- 入力: `{base / variant}/train_W/eval_W/pair_predictions.csv` (W in {HORIZONS})")
    unit = "ディレクトリ (モジュール)" if group_col == "directory" else "リポジトリ (top_dir)"
    md.append(f"- {unit}数: {len(pooled)} 件")
    if use_reliable_filter:
        md.append(f"- 集計対象 (auc_reliable, n_pos >= 30 & n_neg >= 30): {len(rel)} 件")
    else:
        md.append(f"- 集計対象 (両クラス AUC 算出可能): {len(rel)} 件 [no reliability filter]")
    md.append(f"- 内訳: project-specific {(rel['category']=='project-specific').sum()} / "
              f"cross-cutting {(rel['category']=='cross-cutting').sum()}")
    md.append("")
    md.append("## 主要発見")
    md.append("")

    sp = stats_results.get("spearman_logn_vs_auc_ens", {})
    if sp:
        rho = sp.get("rho", float("nan"))
        p = sp.get("p", float("nan"))
        verdict = "顕著な正相関あり" if rho > 0.4 and p < 0.05 else (
            "弱い相関のみ" if abs(rho) > 0.2 else "ほぼ相関なし"
        )
        md.append(f"### H1: サンプル数と AUC の相関")
        md.append(f"- Spearman ρ(log n_pair, Ens AUC) = **{rho:.3f}** (p = {p:.3g}) → {verdict}")
        md.append("")

    mw = stats_results.get("mannwhitney_category_auc_ens", {})
    if mw:
        md.append(f"### H2: project-specific vs cross-cutting の Ens AUC 差")
        md.append(
            f"- Mann-Whitney U = {mw.get('U'):.0f}, p = {mw.get('p'):.3g}"
        )
        md.append(
            f"- 中央値: project-specific {mw.get('median_project_specific'):.3f} vs "
            f"cross-cutting {mw.get('median_cross_cutting'):.3f}"
        )
        verdict = "有意差あり" if mw.get("p", 1.0) < 0.05 else "有意差なし"
        md.append(f"- 判定: {verdict}")
        md.append("")

    we = stats_results.get("wilcoxon_ens_vs_irl", {})
    wr = stats_results.get("wilcoxon_ens_vs_rf", {})
    if we and wr:
        md.append("### H3: Ensemble の単独優位性 (directory 横断)")
        md.append(
            f"- Ens vs IRL: W = {we.get('W'):.0f}, p_bonf = {we.get('p_bonferroni'):.3g}, "
            f"r = {we.get('r_rank_biserial'):.3f}"
        )
        md.append(
            f"- Ens vs RF:  W = {wr.get('W'):.0f}, p_bonf = {wr.get('p_bonferroni'):.3g}, "
            f"r = {wr.get('r_rank_biserial'):.3f}"
        )
        md.append("")

    label = "directories" if group_col == "directory" else "repositories"
    md.append(f"## Top 5 / Bottom 5 {label}")
    md.append("")
    md.append("### Top 5 by Ens AUC")
    md.append("")
    md.append(f"| {group_col} | category | n_pair | IRL | RF | Ens | gain |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in top.iterrows():
        md.append(
            f"| {r[group_col]} | {r['category']} | {r['n_pair']} | "
            f"{r['auc_irl']:.3f} | {r['auc_rf']:.3f} | "
            f"**{r['auc_ens']:.3f}** | {r['gain_vs_best']:+.3f} |"
        )
    md.append("")
    md.append("### Bottom 5 by Ens AUC")
    md.append("")
    md.append(f"| {group_col} | category | n_pair | IRL | RF | Ens | gain |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in bot.iterrows():
        md.append(
            f"| {r[group_col]} | {r['category']} | {r['n_pair']} | "
            f"{r['auc_irl']:.3f} | {r['auc_rf']:.3f} | "
            f"**{r['auc_ens']:.3f}** | {r['gain_vs_best']:+.3f} |"
        )
    md.append("")
    md.append("## 出力ファイル")
    md.append(f"- `{group_col}_window_metrics.csv` : {group_col} × window の生メトリクス")
    md.append(f"- `{group_col}_pooled_metrics.csv` : {group_col} 単位 4 window pool + bootstrap CI")
    md.append("- `stats_test.json` : 検定結果 (Spearman, Mann-Whitney, Wilcoxon)")
    md.append(f"- `fig_{group_col}_auc_ranked.pdf` : Top/Bottom の AUC ranked bar")
    md.append("- `fig_category_violin.pdf` : project-specific vs cross-cutting の violin")
    md.append("- `fig_n_vs_auc_scatter.pdf` : log(n_pair) vs Ens AUC の scatter")
    md.append("")
    md.append("## 解釈方針")
    if use_reliable_filter:
        md.append("- **auc_reliable=False** の単位 (n_pos<30 or n_neg<30) は信頼区間が広く")
        md.append("  個別解釈は避ける。Top/Bottom 集計でも除外している。")
    else:
        md.append("- **信頼可能フィルタなし**: AUC が算出可能な (両クラス >=1) 全単位を含む。")
        md.append("  小サンプル単位は信頼区間が広いことに留意。")
    md.append("- **project-specific category** は main_repos 32 と一致する top_dir 名")
    md.append("  (例: nova, neutron, cinder)。**cross-cutting** は repo を跨いで共有される")
    md.append("  ディレクトリ (例: releasenotes, doc, etc, tools)。")

    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base", default="outputs/main32_mce_volat_ep150",
        help="model 出力のベースディレクトリ",
    )
    ap.add_argument("--variant", default="lstm_hybrid_snapshot")
    ap.add_argument("--main-repos-csv", default="data/service_teams_main_repos.csv")
    ap.add_argument("--output-dir", default="outputs/analysis/per_dir")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument(
        "--group-by", choices=["directory", "top_dir"], default="directory",
        help="集計単位: directory (例 nova/compute) or top_dir (例 nova, repo 相当)",
    )
    ap.add_argument(
        "--no-reliable-filter", action="store_true",
        help="auc_reliable フィルタを外し全単位を解析",
    )
    args = ap.parse_args()
    group_col = args.group_by
    use_reliable_filter = not args.no_reliable_filter

    base = Path(args.base)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # main_repos セット (main_repo 列は "openstack/nova" 形式なので / の右側を取る)
    mr_df = pd.read_csv(args.main_repos_csv)
    mr_df = mr_df.dropna(subset=["main_repo"])
    main_repos = set(mr_df["main_repo"].astype(str).str.split("/").str[-1])
    main_repos.discard("nan")
    print(f"main_repos ({len(main_repos)}): {sorted(main_repos)[:10]}...")

    # データ読み込み
    print(f"Loading from {base}/{args.variant}/...")
    frames = load_diagonal(base, args.variant)
    print(f"  loaded {len(frames)} windows: {list(frames.keys())}")
    total = sum(len(d) for d in frames.values())
    print(f"  total rows: {total}")

    # window 別 metrics
    print(f"Computing per-({group_col}, window) metrics...")
    win_df = compute_dir_window_metrics(frames, group_col=group_col)
    win_csv = out_dir / f"{group_col}_window_metrics.csv"
    win_df.to_csv(win_csv, index=False)
    print(f"  -> {win_csv} ({len(win_df)} rows)")

    # pooled metrics
    print(f"Computing pooled metrics with {args.n_boot}-iter bootstrap...")
    pooled = compute_dir_pooled_metrics(
        frames, main_repos, group_col=group_col, n_boot=args.n_boot,
    )
    pooled_csv = out_dir / f"{group_col}_pooled_metrics.csv"
    pooled.to_csv(pooled_csv, index=False)
    print(f"  -> {pooled_csv} ({len(pooled)} {group_col}s)")

    # 統計検定
    print(f"Running statistical tests (reliable_filter={use_reliable_filter})...")
    stats_results = run_stats_tests(pooled, use_reliable_filter=use_reliable_filter)
    with open(out_dir / "stats_test.json", "w", encoding="utf-8") as f:
        json.dump(stats_results, f, indent=2, ensure_ascii=False)
    print(f"  -> {out_dir / 'stats_test.json'}")

    # 可視化
    print("Generating figures...")
    fig_ranked_bar(
        pooled, out_dir / f"fig_{group_col}_auc_ranked.pdf",
        group_col=group_col, use_reliable_filter=use_reliable_filter,
    )
    print(f"  -> fig_{group_col}_auc_ranked.pdf")
    fig_category_violin(
        pooled, stats_results, out_dir / "fig_category_violin.pdf",
        use_reliable_filter=use_reliable_filter,
    )
    print(f"  -> fig_category_violin.pdf")
    fig_scatter(
        pooled, stats_results, out_dir / "fig_n_vs_auc_scatter.pdf",
        group_col=group_col, use_reliable_filter=use_reliable_filter,
    )
    print(f"  -> fig_n_vs_auc_scatter.pdf")

    # README
    write_readme(
        pooled, stats_results, out_dir, base, args.variant,
        group_col=group_col, use_reliable_filter=use_reliable_filter,
    )
    print(f"  -> README.md")

    print("\nDone.")


if __name__ == "__main__":
    main()
