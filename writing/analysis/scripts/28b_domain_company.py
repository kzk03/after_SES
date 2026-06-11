"""28b: 企業ドメイン構成 → IRL/RF 予測優劣の分析。

担当: メールドメイン判定による企業集中度がディレクトリ予測 (IRL vs RF, Ensemble) に効くか。
size 交絡 (log10(n_pair)) と base_rate を必ず統制する。
"""
import json

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "/Users/kazuki-h/Master_research/writing/analysis/results"
MASTER = f"{RESULTS}/directory_master.csv"

DOMAIN_VARS = ["domain_hhi", "corporate_share", "top_corp_share_of_all", "n_domains"]
TARGET_VARS = ["delta_auc", "ens_auc", "ens_lift_auc"]


def spearman_safe(x, y):
    """NaN を落とした上で Spearman 相関。"""
    m = (~pd.isna(x)) & (~pd.isna(y))
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    rho, p = stats.spearmanr(x[m], y[m])
    return rho, p, int(m.sum())


def partial_spearman(df, x_col, y_col, ctrl_cols):
    """rank 変換 → 制御変数で線形回帰した残差同士の Pearson 相関 = 偏 Spearman。"""
    cols = [x_col, y_col] + ctrl_cols
    sub = df[cols].dropna()
    if len(sub) < (len(ctrl_cols) + 5):
        return np.nan, np.nan, len(sub)
    # rank 変換
    R = sub.rank()
    Z = np.column_stack([np.ones(len(R))] + [R[c].values for c in ctrl_cols])

    def resid(yvec):
        beta, *_ = np.linalg.lstsq(Z, yvec, rcond=None)
        return yvec - Z @ beta

    rx = resid(R[x_col].values)
    ry = resid(R[y_col].values)
    n = len(sub)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan, np.nan, n
    r = np.corrcoef(rx, ry)[0, 1]
    # 偏相関の t 検定 (制御変数 k 個)
    k = len(ctrl_cols)
    dof = n - 2 - k
    if dof <= 0:
        return r, np.nan, n
    t = r * np.sqrt(dof / max(1e-12, (1 - r * r)))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, n


def main():
    df = pd.read_csv(MASTER)
    df["log_npair"] = np.log10(df["n_pair"])
    reliable = df[df["reliable"] == True].copy()  # noqa: E712  n_pair>=20
    aucdef = df[df["auc_defined"] == True].copy()  # noqa: E712

    out = {}

    # ---- 1. ドメイン集中度の分布 ----
    dist = {}
    for grp_name, grp in [("reliable46", reliable), ("aucdef286", aucdef)]:
        d = {}
        for v in DOMAIN_VARS + ["top_corp_share_of_all", "corp_hhi", "n_corporate_domains"]:
            s = grp[v].dropna()
            d[v] = {
                "n": int(len(s)),
                "mean": float(s.mean()),
                "median": float(s.median()),
                "min": float(s.min()),
                "max": float(s.max()),
                "q25": float(s.quantile(0.25)),
                "q75": float(s.quantile(0.75)),
            }
        dist[grp_name] = d
    out["distribution"] = dist

    # ---- 2 & 3. Spearman + 偏 Spearman (log_npair, base_rate 統制) ----
    ctrl = ["log_npair", "base_rate"]
    corr_rows = []
    for grp_name, grp in [("reliable46", reliable), ("aucdef286", aucdef)]:
        for xv in DOMAIN_VARS:
            for yv in TARGET_VARS:
                rho, p, n = spearman_safe(grp[xv], grp[yv])
                prho, pp, pn = partial_spearman(grp, xv, yv, ctrl)
                corr_rows.append(
                    {
                        "group": grp_name,
                        "x": xv,
                        "y": yv,
                        "n": n,
                        "spearman_rho": round(rho, 4) if not np.isnan(rho) else None,
                        "spearman_p": round(p, 5) if not np.isnan(p) else None,
                        "partial_rho": round(prho, 4) if not np.isnan(prho) else None,
                        "partial_p": round(pp, 5) if not np.isnan(pp) else None,
                        "partial_n": pn,
                        "survives_control": bool(
                            (not np.isnan(pp)) and pp < 0.05
                        ),
                    }
                )
    corr_df = pd.DataFrame(corr_rows)
    corr_csv = f"{RESULTS}/domain_company_correlations.csv"
    corr_df.to_csv(corr_csv, index=False)
    out["correlations"] = corr_rows

    # ---- 4. 仮説検証: 単一企業集中 → RF優位? ----
    # top_corp_share_of_all 高 = 単一企業集中。delta_auc<0 = RF優位。
    # 高集中 vs 低分散の delta_auc, ens_auc 差を見る (reliable)。
    hyp = {}
    for grp_name, grp in [("reliable46", reliable), ("aucdef286", aucdef)]:
        g = grp.dropna(subset=["top_corp_share_of_all"]).copy()
        med = g["top_corp_share_of_all"].median()
        hi = g[g["top_corp_share_of_all"] >= med]  # 単一企業集中
        lo = g[g["top_corp_share_of_all"] < med]  # 多企業分散
        rec = {"median_top_corp_share": float(med), "n_hi": len(hi), "n_lo": len(lo)}
        for yv in ["delta_auc", "ens_auc", "ens_lift_auc", "irl_auc", "rf_auc"]:
            hv = hi[yv].dropna()
            lv = lo[yv].dropna()
            rec[f"{yv}_hi_median"] = float(hv.median()) if len(hv) else None
            rec[f"{yv}_lo_median"] = float(lv.median()) if len(lv) else None
            if len(hv) >= 3 and len(lv) >= 3:
                u, pu = stats.mannwhitneyu(hv, lv, alternative="two-sided")
                rec[f"{yv}_mwu_p"] = round(pu, 5)
            else:
                rec[f"{yv}_mwu_p"] = None
        hyp[grp_name] = rec
    out["hypothesis_split"] = hyp

    # ---- 5. top5 集中 / top5 分散 ----
    rel = reliable.dropna(subset=["top_corp_share_of_all", "delta_auc"]).copy()
    cols_show = [
        "directory",
        "top_dir",
        "n_pair",
        "base_rate",
        "top_corp_share_of_all",
        "domain_hhi",
        "n_corporate_domains",
        "top_corp_domain",
        "irl_auc",
        "rf_auc",
        "ens_auc",
        "delta_auc",
    ]
    top_concentrated = rel.nlargest(5, "top_corp_share_of_all")[cols_show]
    top_dispersed = rel.nsmallest(5, "top_corp_share_of_all")[cols_show]
    out["top5_concentrated"] = top_concentrated.round(4).to_dict("records")
    out["top5_dispersed"] = top_dispersed.round(4).to_dict("records")
    extremes_csv = f"{RESULTS}/domain_company_extremes.csv"
    ex = pd.concat(
        [
            top_concentrated.assign(group="concentrated"),
            top_dispersed.assign(group="dispersed"),
        ]
    )
    ex.round(4).to_csv(extremes_csv, index=False)

    # ---- 6. 図 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    plot_df = aucdef.dropna(subset=["top_corp_share_of_all"]).copy()
    tops = plot_df["top_dir"].value_counts().index.tolist()
    cmap = plt.get_cmap("tab20")
    color_map = {t: cmap(i % 20) for i, t in enumerate(tops)}

    for ax, yv, title in [
        (axes[0], "delta_auc", "domain集中度 vs delta_auc (IRL−RF)"),
        (axes[1], "ens_auc", "domain集中度 vs Ensemble AUC"),
    ]:
        sub = plot_df.dropna(subset=[yv, "top_corp_share_of_all"])
        sizes = 12 + (sub["n_pair"] / sub["n_pair"].max()) * 240
        for t in tops:
            m = sub["top_dir"] == t
            if m.sum() == 0:
                continue
            ax.scatter(
                sub.loc[m, "top_corp_share_of_all"],
                sub.loc[m, yv],
                s=sizes[m],
                color=color_map[t],
                alpha=0.6,
                edgecolors="k",
                linewidths=0.3,
            )
        # reliable を強調枠
        relm = sub.index.isin(reliable.index)
        ax.scatter(
            sub.loc[relm, "top_corp_share_of_all"],
            sub.loc[relm, yv],
            s=sizes[relm] + 30,
            facecolors="none",
            edgecolors="red",
            linewidths=0.8,
            label="reliable (n_pair>=20)",
        )
        if yv == "delta_auc":
            ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("top_corp_share_of_all (単一企業ドメイン比)")
        ax.set_ylabel(yv)
        ax.set_title(title)
        # 注記: reliable の Spearman / partial
        rrow = corr_df[
            (corr_df.group == "reliable46")
            & (corr_df.x == "top_corp_share_of_all")
            & (corr_df.y == yv)
        ].iloc[0]
        note = (
            f"reliable46:\n"
            f"Spearman rho={rrow.spearman_rho} (p={rrow.spearman_p})\n"
            f"partial(|log n_pair,base_rate)\n  rho={rrow.partial_rho} (p={rrow.partial_p})"
        )
        ax.text(
            0.02,
            0.02,
            note,
            transform=ax.transAxes,
            fontsize=8,
            va="bottom",
            ha="left",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
        )

    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("企業ドメイン集中度と予測優劣 (点サイズ=n_pair, 色=project)", fontsize=12)
    fig.tight_layout()
    fig_path = f"{RESULTS}/fig_dir_domain_vs_pred.pdf"
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ---- 出力サマリ ----
    summary_json = f"{RESULTS}/domain_company_summary.json"
    with open(summary_json, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # コンソール出力
    print("=== 分布 (reliable46) ===")
    for v, d in dist["reliable46"].items():
        print(f"  {v}: median={d['median']:.3f} [{d['min']:.3f},{d['max']:.3f}] n={d['n']}")
    print("\n=== 相関 (reliable46) ===")
    print(
        corr_df[corr_df.group == "reliable46"][
            ["x", "y", "n", "spearman_rho", "spearman_p", "partial_rho", "partial_p", "survives_control"]
        ].to_string(index=False)
    )
    print("\n=== 相関 (aucdef286) ===")
    print(
        corr_df[corr_df.group == "aucdef286"][
            ["x", "y", "n", "spearman_rho", "spearman_p", "partial_rho", "partial_p", "survives_control"]
        ].to_string(index=False)
    )
    print("\n=== 仮説 split (reliable46) ===")
    print(json.dumps(hyp["reliable46"], indent=2, ensure_ascii=False))
    print("\n=== top5 集中 ===")
    print(top_concentrated.round(3).to_string(index=False))
    print("\n=== top5 分散 ===")
    print(top_dispersed.round(3).to_string(index=False))

    return corr_csv, extremes_csv, summary_json, fig_path


if __name__ == "__main__":
    main()
