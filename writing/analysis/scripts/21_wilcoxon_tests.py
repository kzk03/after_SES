# -*- coding: utf-8 -*-
"""C-2: per-dev / per-dir AUC に基づく Wilcoxon signed-rank 検定一式.

論文では pair-pooled AUC の差は示されているが、per-dev / per-dir 粒度での統計的有意性は
未提示。本スクリプトは以下 3 ペアを per-dev (n≈270 after bot exclusion) と
per-dir (n≈40-50) で検定する:
  - IRL vs RF
  - アンサンブル vs IRL
  - アンサンブル vs RF

モデル (RULES.md 準拠): cold IRL (irl_<h>), RF (rf_<h>), アンサンブル ((irl+rf)/2)

評価指標: pooled AUC (per dev / per dir で 4 horizon × all obs を pool)
検定: Wilcoxon signed-rank (two-sided), Cliff's delta (効果量)

入力:
  data/pair_trajectories.csv
  outputs/analysis/bot_detection/bot_candidates_main32.csv

出力:
  results/wilcoxon_tests.csv (全比較の p, effect size, n)
  results/fig_wilcoxon_pairwise.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
REPO_ROOT = ROOT.parents[1]
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
COMPARISONS = [("irl", "rf"), ("ens", "irl"), ("ens", "rf")]
MIN_OBS_DEV = 12
MIN_OBS_DIR = 12


def add_ens(pair: pd.DataFrame) -> pd.DataFrame:
    pair = pair.copy()
    for h in HORIZONS:
        pair[f"ens_{h}"] = (pair[f"irl_{h}"] + pair[f"rf_{h}"]) / 2
    return pair


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def per_group_auc(pair: pd.DataFrame, group_col: str, min_obs: int) -> pd.DataFrame:
    """group_col (developer or directory) ごとに 4 horizon pool AUC を出す."""
    rows = []
    for key, g in pair.groupby(group_col):
        ys, irl_p, rf_p, ens_p = [], [], [], []
        for h in HORIZONS:
            ys.append(g[f"label_{h}"].values)
            irl_p.append(g[f"irl_{h}"].values)
            rf_p.append(g[f"rf_{h}"].values)
            ens_p.append(g[f"ens_{h}"].values)
        y = np.concatenate(ys)
        if len(np.unique(y)) < 2 or len(y) < min_obs:
            continue
        rows.append({
            group_col: key, "n_obs": len(y), "n_pos": int(y.sum()),
            "auc_irl": safe_auc(y, np.concatenate(irl_p)),
            "auc_rf": safe_auc(y, np.concatenate(rf_p)),
            "auc_ens": safe_auc(y, np.concatenate(ens_p)),
        })
    return pd.DataFrame(rows)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta. (a > b の頻度 − a < b の頻度), -1〜+1."""
    a = np.asarray(a); b = np.asarray(b)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff).sum()) / (n * m))


def test_pair(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    a = df[col_a].values
    b = df[col_b].values
    valid = ~(np.isnan(a) | np.isnan(b))
    a, b = a[valid], b[valid]
    diff = a - b
    if (diff == 0).all() or len(diff) < 5:
        return {"n": len(diff), "median_a_minus_b": np.nan,
                "wilcoxon_stat": np.nan, "p_value": np.nan,
                "cliffs_delta": np.nan, "win_a": np.nan, "win_b": np.nan,
                "ties": np.nan}
    try:
        stat, p = wilcoxon(a, b, alternative="two-sided",
                           zero_method="wilcox")
    except ValueError:
        stat, p = np.nan, np.nan
    return {
        "n": int(len(diff)),
        "median_a_minus_b": float(np.median(diff)),
        "mean_a_minus_b": float(np.mean(diff)),
        "wilcoxon_stat": float(stat) if stat == stat else np.nan,
        "p_value": float(p) if p == p else np.nan,
        "cliffs_delta": cliffs_delta(a, b),
        "win_a": int((diff > 0).sum()),
        "win_b": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def run(pair: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.DataFrame,
                                                   pd.DataFrame]:
    print(f"\n========== {label} ==========")
    dev = per_group_auc(pair, "developer", MIN_OBS_DEV)
    dir_ = per_group_auc(pair, "directory", MIN_OBS_DIR)
    print(f"per-dev n={len(dev):,}, per-dir n={len(dir_):,}")

    rows = []
    for level, df in [("per_dev", dev), ("per_dir", dir_)]:
        for a, b in COMPARISONS:
            t = test_pair(df, f"auc_{a}", f"auc_{b}")
            t.update({"bot_filter": label, "level": level,
                      "model_a": a, "model_b": b})
            rows.append(t)
            print(f"  [{level}] {a} vs {b}: "
                  f"n={t['n']}, median Δ={t['median_a_minus_b']:+.4f}, "
                  f"p={t['p_value']:.2e}, "
                  f"Cliff's δ={t['cliffs_delta']:+.3f}, "
                  f"win {a}={t['win_a']}, win {b}={t['win_b']}")
    return pd.DataFrame(rows), dev, dir_


def plot_figure(tests: pd.DataFrame, dev_w: pd.DataFrame, dir_w: pd.DataFrame,
                out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

    pair_names = {("irl", "rf"): "cold IRL vs RF",
                  ("ens", "irl"): "アンサンブル vs cold IRL",
                  ("ens", "rf"): "アンサンブル vs RF"}

    # (上段) per-dev / per-dir の AUC 差ヒストグラム (bot 込み)
    for col_idx, (a, b) in enumerate(COMPARISONS):
        ax = axes[0, col_idx]
        diff_dev = dev_w[f"auc_{a}"] - dev_w[f"auc_{b}"]
        diff_dir = dir_w[f"auc_{a}"] - dir_w[f"auc_{b}"]
        bins = np.linspace(-0.5, 0.5, 41)
        ax.hist(diff_dev, bins=bins, alpha=0.55, color="#1f77b4",
                label=f"per-dev (n={len(diff_dev)})", edgecolor="black",
                linewidth=0.3)
        ax.hist(diff_dir, bins=bins, histtype="step",
                edgecolor="#d62728", linewidth=1.8,
                label=f"per-dir (n={len(diff_dir)})")
        ax.axvline(0, color="black", linewidth=0.7)
        ax.axvline(diff_dev.median(), color="#1f77b4", linestyle="--",
                   linewidth=1.4,
                   label=f"per-dev median={diff_dev.median():+.3f}")
        ax.axvline(diff_dir.median(), color="#d62728", linestyle="--",
                   linewidth=1.4,
                   label=f"per-dir median={diff_dir.median():+.3f}")
        ax.set_xlabel(f"AUC ({a}) − AUC ({b})")
        ax.set_ylabel("頻度")
        ax.set_title(f"(a-{col_idx+1}) {pair_names[(a, b)]}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    # (下段) Wilcoxon の forest plot: per-level × per-comparison の Cliff's δ + 95% CI 風
    # CI は bootstrap で近似
    rng = np.random.default_rng(777)
    forest_rows = []
    for level, df_ in [("per_dev", dev_w), ("per_dir", dir_w)]:
        for a, b in COMPARISONS:
            diff = (df_[f"auc_{a}"] - df_[f"auc_{b}"]).dropna().values
            if len(diff) < 5:
                continue
            # bootstrap median
            boots = [np.median(rng.choice(diff, len(diff), replace=True))
                     for _ in range(1000)]
            forest_rows.append({
                "level": level, "comp": f"{a} vs {b}",
                "median": float(np.median(diff)),
                "ci_lo": float(np.quantile(boots, 0.025)),
                "ci_hi": float(np.quantile(boots, 0.975)),
                "n": len(diff),
            })
    forest = pd.DataFrame(forest_rows)

    ax = axes[1, 0]
    levels = ["per_dev", "per_dir"]
    comps = [f"{a} vs {b}" for a, b in COMPARISONS]
    y_labels = []
    y = 0
    for lv in levels:
        for c in comps:
            r = forest[(forest["level"] == lv) & (forest["comp"] == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            color = "#1f77b4" if lv == "per_dev" else "#d62728"
            ax.errorbar(r["median"], y,
                        xerr=[[r["median"] - r["ci_lo"]],
                              [r["ci_hi"] - r["median"]]],
                        fmt="o", color=color, capsize=4)
            y_labels.append(f"{lv} {c} (n={r['n']})")
            y += 1
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("AUC 差 median (bootstrap 95% CI)")
    ax.set_title("(b-1) per-dev / per-dir 比較のメディアン + 95% CI",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    # (b-2) p 値表
    ax = axes[1, 1]
    ax.axis("off")
    show = tests[tests["bot_filter"] == "with_bots"].copy()
    show["p_value_str"] = show["p_value"].apply(
        lambda p: "—" if (p != p) else (f"<1e-15" if p < 1e-15
                                         else f"{p:.2e}"))
    show["δ"] = show["cliffs_delta"].apply(lambda v: f"{v:+.3f}")
    show["Δ med"] = show["median_a_minus_b"].apply(lambda v: f"{v:+.4f}")
    show["wins"] = show.apply(lambda r: f"{int(r['win_a'])}/{int(r['win_b'])}"
                              + (f"/{int(r['ties'])}" if r['ties'] > 0 else ""),
                              axis=1)
    table_df = show[["level", "model_a", "model_b", "n", "Δ med",
                     "p_value_str", "δ", "wins"]].copy()
    table_df.columns = ["粒度", "A", "B", "n", "Δ median",
                        "p", "Cliff's δ", "win A/B/tie"]
    ax.set_title("(b-2) Wilcoxon signed-rank (bot 込み)",
                 fontsize=10, loc="left")
    tab = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                   loc="center", cellLoc="center", colLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.4)

    # (b-3) bot 影響 (with vs without p-value 散布)
    ax = axes[1, 2]
    tw = tests[tests["bot_filter"] == "with_bots"]
    two = tests[tests["bot_filter"] == "without_bots"]
    merged = tw.merge(two, on=["level", "model_a", "model_b"],
                      suffixes=("_w", "_wo"))
    ax.scatter(-np.log10(merged["p_value_w"].clip(lower=1e-300)),
               -np.log10(merged["p_value_wo"].clip(lower=1e-300)),
               s=80, alpha=0.75,
               c=["#1f77b4" if lv == "per_dev" else "#d62728"
                  for lv in merged["level"]])
    lim = max(-np.log10(merged["p_value_w"].clip(lower=1e-300)).max(),
              -np.log10(merged["p_value_wo"].clip(lower=1e-300)).max()) + 1
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.7)
    ax.set_xlabel("−log10 p (bot 込み)")
    ax.set_ylabel("−log10 p (bot 除外)")
    ax.set_title("(b-3) bot 除外感度 (各点 = 1 比較)", fontsize=10)
    ax.grid(alpha=0.3)
    for _, r in merged.iterrows():
        ax.annotate(f"{r['level'].replace('per_', '')}: {r['model_a']}-{r['model_b']}",
                    (-np.log10(r['p_value_w']),
                     -np.log10(r['p_value_wo'])),
                    fontsize=7.5, alpha=0.85,
                    xytext=(4, 4), textcoords="offset points")

    fig.suptitle("per-dev / per-dir AUC の Wilcoxon signed-rank 検定",
                 fontsize=12.5, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    print("loading data...")
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    pair = add_ens(pair)
    bot_df = pd.read_csv(BOT_CSV)
    bots = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712

    tests_w, dev_w, dir_w = run(pair, "with_bots")
    tests_wo, _, _ = run(pair[~pair["developer"].isin(bots)].copy(),
                          "without_bots")

    tests = pd.concat([tests_w, tests_wo], ignore_index=True)
    OUT.mkdir(exist_ok=True)
    tests.to_csv(OUT / "wilcoxon_tests.csv", index=False)
    print(f"\nsaved: {OUT / 'wilcoxon_tests.csv'}")

    # bot 影響
    m = tests_w.merge(tests_wo, on=["level", "model_a", "model_b"],
                      suffixes=("_w", "_wo"))
    m["dp"] = (np.log10(m["p_value_w"].clip(lower=1e-300))
               - np.log10(m["p_value_wo"].clip(lower=1e-300))).abs()
    print(f"\n[bot 影響] |Δ log10 p| max={m['dp'].max():.4f}, "
          f"mean={m['dp'].mean():.4f}")

    plot_figure(tests, dev_w, dir_w, OUT / "fig_wilcoxon_pairwise.pdf")


if __name__ == "__main__":
    main()
