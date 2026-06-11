"""28d: 網羅的メタ回帰 (交絡分離).

全特性を同時に投入し, dir の
  (a) IRL/RF 優劣  : delta_auc (= irl_auc - rf_auc; 正=IRL優位)
  (b) 予測性       : ens_auc (アンサンブル AUC)
を何が決めるかを多変量で分離する.

説明変数 (すべて標準化):
  log10(n_pair), base_rate, dev_A4_share, pair_P1_share, pair_P2_share,
  domain_hhi, corporate_share, top_corp_share_of_all

2 spec:
  (i)  AUC定義可 286 dir, sqrt(n_pair) 重み WLS
  (ii) reliable 46 dir, OLS

statsmodels 非依存. numpy.linalg で (W)LS を解き, SE/t/p を自前算出.

実行:
  cd /Users/kazuki-h/Master_research/writing/analysis && uv run python scripts/28d_meta_regression.py
"""
from __future__ import annotations

import json
from pathlib import Path

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("/Users/kazuki-h/Master_research/writing/analysis/results")
MASTER = RESULTS / "directory_master.csv"
COEF_CSV = RESULTS / "dir_master_regression_coef.csv"
UNI_CSV = RESULTS / "dir_master_regression_univariate.csv"
FIG = RESULTS / "fig_dir_meta_regression_full.pdf"

# 説明変数 (生列名) と表示名
PREDICTORS = [
    ("log10_n_pair", "log10(n_pair)"),
    ("base_rate", "base_rate"),
    ("dev_A4_share", "dev_A4_share (常時活動)"),
    ("pair_P1_share", "pair_P1_share (未経験)"),
    ("pair_P2_share", "pair_P2_share (経験/低承諾)"),
    ("domain_hhi", "domain_hhi (集中)"),
    ("corporate_share", "corporate_share"),
    ("top_corp_share_of_all", "top_corp_share_of_all"),
]
TARGETS = [
    ("delta_auc", "delta_auc (IRL-RF 優劣)"),
    ("ens_auc", "ens_auc (予測性)"),
]


def wls_fit(X: np.ndarray, y: np.ndarray, w: np.ndarray):
    """重み付き最小二乗. X は切片列を含む. 返り値: beta, SE, t, p, r2, n, k."""
    n, k = X.shape
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    # 正規方程式を最小二乗で
    beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    resid = y - X @ beta
    dof = n - k
    # 重み付き残差分散
    sigma2 = float((w * resid**2).sum() / dof)
    XtWX = Xw.T @ Xw
    XtWX_inv = np.linalg.inv(XtWX)
    cov = sigma2 * XtWX_inv
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = 2.0 * stats.t.sf(np.abs(tvals), dof)
    # 重み付き R^2
    ybar = float((w * y).sum() / w.sum())
    ss_res = float((w * resid**2).sum())
    ss_tot = float((w * (y - ybar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    r2_adj = 1.0 - (1.0 - r2) * (n - 1) / dof
    return beta, se, tvals, pvals, r2, r2_adj, n, k


def standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        out[c + "__z"] = (out[c] - mu) / sd if sd > 0 else 0.0
    return out


def run_spec(df: pd.DataFrame, target: str, use_weight: bool, spec_label: str):
    raw_cols = [p[0] for p in PREDICTORS]
    sub = df.dropna(subset=[target] + raw_cols).copy()
    sub = standardize(sub, raw_cols)
    zcols = [c + "__z" for c in raw_cols]
    X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in zcols])
    y = sub[target].values.astype(float)
    if use_weight:
        w = np.sqrt(sub["n_pair"].values.astype(float))
    else:
        w = np.ones(len(sub))
    beta, se, tvals, pvals, r2, r2_adj, n, k = wls_fit(X, y, w)
    names = ["(Intercept)"] + [p[1] for p in PREDICTORS]
    rows = []
    for i, nm in enumerate(names):
        ci_lo = beta[i] - 1.96 * se[i]
        ci_hi = beta[i] + 1.96 * se[i]
        rows.append(
            dict(
                spec=spec_label,
                target=target,
                term=nm,
                coef=beta[i],
                se=se[i],
                t=tvals[i],
                p=pvals[i],
                ci_lo=ci_lo,
                ci_hi=ci_hi,
                sig=bool(pvals[i] < 0.05),
                n=n,
                r2=r2,
                r2_adj=r2_adj,
            )
        )
    return rows


def univariate(df: pd.DataFrame, target: str, spec_label: str, use_weight: bool):
    """単変量 (規模統制なし) の各説明変数→target の標準化係数."""
    raw_cols = [p[0] for p in PREDICTORS]
    rows = []
    for col, disp in PREDICTORS:
        sub = df.dropna(subset=[target, col]).copy()
        sub = standardize(sub, [col])
        X = np.column_stack([np.ones(len(sub)), sub[col + "__z"].values])
        y = sub[target].values.astype(float)
        w = np.sqrt(sub["n_pair"].values.astype(float)) if use_weight else np.ones(len(sub))
        beta, se, tvals, pvals, r2, _, n, _ = wls_fit(X, y, w)
        rows.append(
            dict(
                spec=spec_label,
                target=target,
                term=disp,
                uni_coef=beta[1],
                uni_se=se[1],
                uni_p=pvals[1],
                uni_sig=bool(pvals[1] < 0.05),
                n=n,
            )
        )
    return rows


def main():
    df = pd.read_csv(MASTER)
    df["log10_n_pair"] = np.log10(df["n_pair"].clip(lower=1))

    df_auc = df[df["auc_defined"]].copy()   # 286
    df_rel = df[df["reliable"]].copy()       # 46

    coef_rows = []
    uni_rows = []
    for target, _ in TARGETS:
        # spec (i): 286 dir WLS
        coef_rows += run_spec(df_auc, target, use_weight=True, spec_label="i_WLS_286")
        uni_rows += univariate(df_auc, target, "i_WLS_286", use_weight=True)
        # spec (ii): 46 dir OLS
        coef_rows += run_spec(df_rel, target, use_weight=False, spec_label="ii_OLS_46")
        uni_rows += univariate(df_rel, target, "ii_OLS_46", use_weight=False)

    coef_df = pd.DataFrame(coef_rows)
    uni_df = pd.DataFrame(uni_rows)

    # 単変量と多変量をマージして「規模統制で消える/残る」表を作る
    merged = coef_df.merge(
        uni_df[["spec", "target", "term", "uni_coef", "uni_se", "uni_p", "uni_sig"]],
        on=["spec", "target", "term"],
        how="left",
    )
    # 切片は単変量対応なし
    merged.to_csv(COEF_CSV, index=False)
    uni_df.to_csv(UNI_CSV, index=False)

    # R2 サマリ
    r2_summary = (
        coef_df.groupby(["spec", "target"])[["r2", "r2_adj", "n"]].first().reset_index()
    )

    # ---- 図: forest plot (2 target x 2 spec = 4 panel) ----
    specs = ["i_WLS_286", "ii_OLS_46"]
    spec_titles = {"i_WLS_286": "(i) 286 dir, sqrt(n_pair) 重み WLS",
                   "ii_OLS_46": "(ii) reliable 46 dir, OLS"}
    target_titles = {t[0]: t[1] for t in TARGETS}
    term_order = [p[1] for p in PREDICTORS]  # 切片除外

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for r, (target, _) in enumerate(TARGETS):
        for c, spec in enumerate(specs):
            ax = axes[r, c]
            sel = coef_df[
                (coef_df["spec"] == spec)
                & (coef_df["target"] == target)
                & (coef_df["term"] != "(Intercept)")
            ].set_index("term").loc[term_order].reset_index()
            ypos = np.arange(len(sel))[::-1]
            for j, row in sel.iterrows():
                yy = ypos[j]
                color = "#c0392b" if row["sig"] else "#999999"
                filled = row["sig"]
                ax.plot([row["ci_lo"], row["ci_hi"]], [yy, yy], color=color, lw=2, zorder=2)
                ax.scatter(
                    [row["coef"]], [yy],
                    s=70, color=color if filled else "white",
                    edgecolor=color, linewidth=1.6, zorder=3,
                )
            ax.axvline(0, color="black", lw=0.8, ls="--", zorder=1)
            ax.set_yticks(ypos[::-1])
            ax.set_yticklabels(term_order[::-1] if c == 0 else [])
            r2v = r2_summary[(r2_summary["spec"] == spec) & (r2_summary["target"] == target)]
            r2t = r2v["r2"].iloc[0]
            nt = int(r2v["n"].iloc[0])
            ax.set_title(f"{target_titles[target]}\n{spec_titles[spec]}  (R²={r2t:.2f}, n={nt})",
                         fontsize=10)
            ax.set_xlabel("標準化偏回帰係数 (±95%CI)")
            ax.grid(axis="x", alpha=0.3)
    # 凡例
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="#c0392b", markerfacecolor="#c0392b",
               markersize=9, lw=2, label="有意 (p<0.05)"),
        Line2D([0], [0], marker="o", color="#999999", markerfacecolor="white",
               markersize=9, lw=2, label="非有意"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("dir 特性のメタ回帰: IRL/RF優劣(delta_auc) と 予測性(ens_auc) の規定因",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(FIG, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- コンソール出力 ----
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("=== R2 summary ===")
    print(r2_summary.to_string(index=False))
    print("\n=== 多変量係数 (有意のみ抜粋, 非切片) ===")
    show = coef_df[(coef_df["term"] != "(Intercept)")][
        ["spec", "target", "term", "coef", "se", "p", "sig"]
    ].copy()
    print(show.round(4).to_string(index=False))

    print("\n=== 単変量→多変量 (規模統制) で消える/残る (spec i_WLS_286) ===")
    tab = merged[(merged["spec"] == "i_WLS_286") & (merged["term"] != "(Intercept)")][
        ["target", "term", "uni_coef", "uni_p", "uni_sig", "coef", "p", "sig"]
    ].copy()
    tab["status"] = np.where(
        tab["uni_sig"] & ~tab["sig"], "消える(交絡)",
        np.where(~tab["uni_sig"] & tab["sig"], "出現",
                 np.where(tab["uni_sig"] & tab["sig"], "残る", "非有意のまま")))
    print(tab.round(4).to_string(index=False))

    # JSON サマリ (キー所見)
    summary = {
        "r2": r2_summary.to_dict(orient="records"),
        "delta_auc_sig_terms_286": show[(show.target == "delta_auc") & (show.sig)
                                         & (show.spec == "i_WLS_286")][["term", "coef", "p"]]
        .to_dict(orient="records"),
        "ens_auc_sig_terms_286": show[(show.target == "ens_auc") & (show.sig)
                                      & (show.spec == "i_WLS_286")][["term", "coef", "p"]]
        .to_dict(orient="records"),
    }
    (RESULTS / "dir_master_regression_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nwrote:", COEF_CSV)
    print("wrote:", UNI_CSV)
    print("wrote:", FIG)


if __name__ == "__main__":
    main()
