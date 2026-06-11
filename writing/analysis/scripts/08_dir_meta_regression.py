"""
§9: ディレクトリ AUC のメタ回帰

教授の問い (3)「どんなディレクトリなら当てれるのか」に対し、
記述統計 (上位 / 下位) から「ディレクトリ属性が AUC をどれだけ説明するか」の定量モデルへ進める。

被説明変数: Ens AUC (auc_ens, ディレクトリ単位)
説明変数:
  - log_n_pair       (規模)
  - accept_rate      (= n_pos / n_pair)
  - n_devs           (関与開発者数)
  - dev_A4_ratio     (常時活動型の比率)
  - pair_P1_ratio    (ディレクトリ未経験ペアの比率)
  - is_test          (ディレクトリ名に 'tests' を含むか)
  - is_doc           (ディレクトリ名に 'doc' / 'api-ref' / 'releasenotes' を含むか)
  - is_infra         (ディレクトリ名が 'devstack' / 'tools' / 'etc' / 'zuul.d' で始まるか)

サンプル: thesis/data/directory_pooled_metrics.csv の auc_reliable=True 行 (46 件)
archetype 比率は data/pair_archetype.csv からディレクトリ単位に集計

出力:
  - results/dir_meta_regression_coef.csv  係数 / 標準誤差 / t / p
  - results/dir_meta_regression_summary.json 全体の R^2 / F 検定
  - results/fig_dir_meta_regression.pdf   係数プロット + 規模の partial 効果図
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
THESIS_DATA = ROOT / "thesis" / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True, parents=True)


def load_dir_metrics(reliable_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(THESIS_DATA / "directory_pooled_metrics.csv")
    df = df.dropna(subset=["auc_ens"]).copy()
    if reliable_only:
        df = df[df["auc_reliable"] == True].copy()  # noqa: E712
    df["accept_rate"] = df["n_pos"] / df["n_pair"].clip(lower=1)
    df["log_n_pair"] = np.log10(df["n_pair"].clip(lower=1))
    return df


def compute_archetype_mix(reliable_dirs: list[str]) -> pd.DataFrame:
    """data/pair_archetype.csv から dir 単位に dev / pair archetype の比率を集計."""
    arch = pd.read_csv(DATA_DIR / "pair_archetype.csv")
    rows: list[dict] = []
    for d in reliable_dirs:
        sub = arch[arch["directory"] == d]
        if len(sub) == 0:
            rows.append({"directory": d, "n_devs": 0, "dev_A4_ratio": np.nan, "pair_P1_ratio": np.nan})
            continue
        # dev archetype: 各 dev の最頻 archetype (per dev 集約は A1-A4 weight argmax)
        dev_groups = sub.groupby("developer")[["A1", "A2", "A3", "A4"]].mean()
        dev_dom = dev_groups.idxmax(axis=1)
        a4_ratio = (dev_dom == "A4").mean()

        # pair archetype: dominant (=P1..P4) は acc_cold_irl/auc 等から決まる. ファイルの "dominant" 列は dev側 archetype.
        # P1-P4 は pair の dominant 別集計が必要だが pair_archetype.csv は dev archetype しか持たない構造。
        # → results/dir_archetype_distribution.csv に pair archetype 比率が記録されているのでそちらと結合する。
        rows.append({
            "directory": d,
            "n_devs": sub["developer"].nunique(),
            "dev_A4_ratio": float(a4_ratio),
        })
    return pd.DataFrame(rows)


def attach_pair_arch(df: pd.DataFrame) -> pd.DataFrame:
    """既存の dir_archetype_distribution.csv から pair_P1 比率を取る (23 dir 限定)."""
    src = RESULTS / "dir_archetype_distribution.csv"
    if not src.exists():
        df["pair_P1_ratio"] = np.nan
        return df
    arch_dist = pd.read_csv(src)
    arch_dist = arch_dist[["directory", "pair_A1"]].rename(columns={"pair_A1": "pair_P1_ratio"})
    return df.merge(arch_dist, on="directory", how="left")


DOC_PAT = re.compile(r"(doc|api-?ref|releasenotes)", re.I)
INFRA_PAT = re.compile(r"^(devstack|tools|etc|zuul\.d|playbooks)", re.I)


def make_categorical_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_test"] = df["directory"].str.contains(r"tests?", regex=True, case=False).astype(int)
    df["is_doc"] = df["directory"].str.contains(DOC_PAT).astype(int)
    df["is_infra"] = df["directory"].str.contains(INFRA_PAT).astype(int)
    return df


def fit_regression(df: pd.DataFrame, features: list[str], y_col: str = "auc_ens",
                   weights: np.ndarray | None = None):
    sub = df.dropna(subset=features + [y_col]).copy()
    X = sub[features].astype(float)
    X = sm.add_constant(X)
    y = sub[y_col].astype(float)
    if weights is not None:
        w = weights[sub.index] if isinstance(weights, pd.Series) else weights
        model = sm.WLS(y, X, weights=w).fit()
    else:
        model = sm.OLS(y, X).fit()
    return model, sub


def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu, sd = out[c].mean(), out[c].std(ddof=0)
        out[c + "_z"] = (out[c] - mu) / (sd if sd > 0 else 1)
    return out


def run_spec(metrics: pd.DataFrame, name: str, weighted: bool = False) -> dict:
    """One regression specification. Returns dict of artifacts."""
    metrics = make_categorical_flags(metrics)
    cont_feats = ["log_n_pair", "accept_rate", "n_devs", "dev_A4_ratio"]
    metrics_z = standardize(metrics, cont_feats)
    feats_z = [c + "_z" for c in cont_feats] + ["is_test", "is_doc", "is_infra"]

    weights = np.sqrt(metrics["n_pair"].values) if weighted else None
    result, used = fit_regression(metrics_z, feats_z, "auc_ens", weights=weights)
    print(f"\n=== spec: {name} (n={len(used)}, weighted={weighted}) ===")
    print(result.summary())

    coef_df = pd.DataFrame({
        "variable": result.params.index,
        "coef": result.params.values,
        "std_err": result.bse.values,
        "t": result.tvalues.values,
        "p_value": result.pvalues.values,
        "ci_low": result.conf_int()[0].values,
        "ci_high": result.conf_int()[1].values,
    })
    coef_df["n_obs"] = len(used)
    coef_df["r_squared"] = result.rsquared
    coef_df["adj_r_squared"] = result.rsquared_adj
    coef_df["spec"] = name
    return {"coef": coef_df, "result": result, "used": used, "feats": feats_z}


def main() -> None:
    # ---- spec 1: reliable 46 dirs (unweighted) ----
    rel = load_dir_metrics(reliable_only=True)
    arch_mix_rel = compute_archetype_mix(rel["directory"].tolist())
    rel = rel.merge(arch_mix_rel, on="directory", how="left")
    rel = attach_pair_arch(rel)

    spec_reliable = run_spec(rel, "reliable_n_ge_30", weighted=False)

    # ---- spec 2: all 286 dirs with auc_ens defined, weighted by sqrt(n_pair) ----
    full = load_dir_metrics(reliable_only=False)
    arch_mix_full = compute_archetype_mix(full["directory"].tolist())
    full = full.merge(arch_mix_full, on="directory", how="left")
    full = attach_pair_arch(full)

    spec_all = run_spec(full, "all_dirs_weighted", weighted=True)

    # ---- combine and save ----
    coef_df = pd.concat([spec_reliable["coef"], spec_all["coef"]], ignore_index=True)
    coef_df.to_csv(RESULTS / "dir_meta_regression_coef.csv", index=False)
    print(f"\n-> {RESULTS / 'dir_meta_regression_coef.csv'}")

    # ---- visualization ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    label_map = {
        "log_n_pair_z": "log10(n_pair)",
        "accept_rate_z": "accept rate",
        "n_devs_z": "n_devs",
        "dev_A4_ratio_z": "A4 ratio",
        "is_test": "is_test",
        "is_doc": "is_doc",
        "is_infra": "is_infra",
    }

    def draw_coef(ax, coef_df_local, title):
        plot_df = coef_df_local[coef_df_local["variable"] != "const"].copy()
        plot_df["label"] = plot_df["variable"].map(label_map).fillna(plot_df["variable"])
        plot_df = plot_df.sort_values("coef")
        y_pos = np.arange(len(plot_df))
        err_low = plot_df["coef"] - plot_df["ci_low"]
        err_high = plot_df["ci_high"] - plot_df["coef"]
        colors = ["#1f77b4" if c > 0 else "#d62728" for c in plot_df["coef"]]
        ax.errorbar(plot_df["coef"], y_pos, xerr=[err_low, err_high], fmt="o",
                    ecolor="gray", capsize=3, markersize=8,
                    markerfacecolor="white", markeredgecolor="black")
        for yi, (val, p) in enumerate(zip(plot_df["coef"], plot_df["p_value"])):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.scatter(val, yi, s=80, color=colors[yi], zorder=3, edgecolor="black")
            ax.text(val + (err_high.iloc[yi] + 0.005) * (1 if val >= 0 else -1),
                    yi, sig, va="center", ha="left" if val >= 0 else "right", fontsize=11)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["label"])
        ax.set_xlabel("coefficient on Ens AUC (z-scored)")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.3)

    draw_coef(axes[0], spec_reliable["coef"],
              f"reliable dirs (n={len(spec_reliable['used'])}, R^2={spec_reliable['result'].rsquared:.3f})")
    draw_coef(axes[1], spec_all["coef"],
              f"all dirs WLS (n={len(spec_all['used'])}, R^2={spec_all['result'].rsquared:.3f})")

    # right: partial residual plot for log_n_pair on the WEIGHTED ALL spec (where effect is real)
    ax = axes[2]
    feats_z = spec_all["feats"]
    used_full = spec_all["used"]
    other_feats = [c for c in feats_z if c != "log_n_pair_z"]
    w = np.sqrt(full.loc[used_full.index, "n_pair"].values)
    Xo = sm.add_constant(used_full[other_feats].astype(float))
    res_y = sm.WLS(used_full["auc_ens"], Xo, weights=w).fit().resid
    res_x = sm.WLS(used_full["log_n_pair_z"], Xo, weights=w).fit().resid
    sizes = np.clip(w * 0.6, 8, 80)
    ax.scatter(res_x, res_y, s=sizes, alpha=0.5, edgecolor="black", linewidth=0.3)
    slope = (w * res_x * res_y).sum() / (w * res_x ** 2).sum()
    xs = np.linspace(res_x.min(), res_x.max(), 50)
    ax.plot(xs, slope * xs, color="red", linewidth=1.5, label=f"slope={slope:+.4f}")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("log_n_pair residual (z-score)")
    ax.set_ylabel("Ens AUC residual")
    ax.set_title("partial effect of log_n_pair (all dirs)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_pdf = RESULTS / "fig_dir_meta_regression.pdf"
    out_png = RESULTS / "fig_dir_meta_regression.png"
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"-> {out_pdf}")
    print(f"-> {out_png}")

    # ---- summary stat dump ----
    summary = {
        "spec_reliable": {
            "n_dir": int(len(spec_reliable["used"])),
            "r_squared": float(spec_reliable["result"].rsquared),
            "adj_r_squared": float(spec_reliable["result"].rsquared_adj),
            "f_pvalue": float(spec_reliable["result"].f_pvalue),
        },
        "spec_all_weighted": {
            "n_dir": int(len(spec_all["used"])),
            "r_squared": float(spec_all["result"].rsquared),
            "adj_r_squared": float(spec_all["result"].rsquared_adj),
            "f_pvalue": float(spec_all["result"].f_pvalue),
        },
        "features": feats_z,
    }
    (RESULTS / "dir_meta_regression_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"-> {RESULTS / 'dir_meta_regression_summary.json'}")


if __name__ == "__main__":
    main()
