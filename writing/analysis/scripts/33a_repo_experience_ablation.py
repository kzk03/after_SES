# -*- coding: utf-8 -*-
"""
33a: repo経験プロキシで「RF優位の正体 = 経験 vs base_rate」を決着
================================================================

B-46で「P1のIRL優位はpath/経験でなくbase_rate感応度の非対称」と再解釈した。
ここでは新規の repo経験プロキシ (pair_repo_experience.csv) を使い、経験を直接
測って RF優位が経験で生じるのか base_rate で生じるのかを分離する。

被覆 (repo_exp_covered=True, 2284ペア = repo名dirのみ) で:

1. repo経験 (log_prior_n, prior_acc, prior_recency_days) は継続(label)を予測するか:
   各指標の単変量 label AUC (区間別 + pooled, bootstrap CI seed=777)。

2. RFはIRLより repo経験に依存するか:
   IRL prob・RF prob を log_prior_n / prior_acc と Spearman相関。
   RFの方が経験指標と強く連動する(=RFは経験を主信号にする)仮説を検証。
   さらに base_rate(=pair_base, 4区間 label平均) を統制した偏相関でも確認。

3. 核心 (base_rate と経験の分離):
   被覆ペアを (低経験 / 高経験 : log_prior_n 中央分割) × base_rate統制 で層別し、
   各層の delta_auc(IRL-RF)。経験が低い層でIRL優位、高い層でRF優位なら
   「経験がRF優位を生む」を支持。
   - base_rate を揃える: base_rate 三分位 × 経験二分の 2D層別で各セル delta_auc。
   - ペアレベル偏相関: per-pair の「IRL-RF rank advantage」近似 (Δ符号付き)を
     log_prior_n と base_rate で偏相関し、どちらが効くか比較。

4. 図 fig_repo_experience_ablation.pdf:
   (a) 経験指標の単変量 label AUC (区間別)
   (b) IRL/RF prob vs log_prior_n の連動 (Spearman, 区間別 + base_rate統制偏相関)
   (c) 経験二分 × base_rate三分位 の IRL/RF AUC と delta

入力:
  data/pair_trajectories.csv
  data/pair_repo_experience.csv
出力:
  results/repo_exp_univariate_auc.csv
  results/repo_exp_prob_correlation.csv
  results/repo_exp_x_baserate_strata.csv
  results/fig_repo_experience_ablation.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

SEED = 777
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
EXP_VARS = ["log_prior_n", "prior_acc", "prior_recency_days"]
EXP_LABEL = {
    "log_prior_n": "log(事前レビュー数)",
    "prior_acc": "事前承諾率",
    "prior_recency_days": "最終活動からの日数",
}
N_BOOT = 2000


# ───────────────────────── helpers ─────────────────────────
def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def boot_auc_ci(y, s, n_boot=N_BOOT, seed=SEED):
    """AUC の bootstrap 95% CI。両クラス揃う時のみ。"""
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y)
    base = float(roc_auc_score(y, s))
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, s[idx]))
    if not boots:
        return base, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return base, float(lo), float(hi)


def partial_spearman(x, y, z):
    """x と y の Spearman 偏相関 (z を統制)。
    rank 化 → z(rank) で線形回帰した残差同士の Pearson 相関。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[m], y[m], z[m]
    if len(x) < 10:
        return np.nan, np.nan, len(x)
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex = rx - Z @ bx
    ey = ry - Z @ by
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan, np.nan, len(x)
    r = np.corrcoef(ex, ey)[0, 1]
    # t-test p
    n = len(x)
    if abs(r) >= 1:
        p = 0.0
    else:
        from scipy.stats import t as tdist
        tstat = r * np.sqrt((n - 3) / (1 - r * r))
        p = 2 * tdist.sf(abs(tstat), n - 3)
    return float(r), float(p), int(n)


def boot_delta_auc_ci(y, s_irl, s_rf, n_boot=N_BOOT, seed=SEED):
    """delta_auc = AUC(IRL) - AUC(RF) の bootstrap CI。"""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    irl = safe_auc(y, s_irl)
    rf = safe_auc(y, s_rf)
    base = irl - rf
    rng = np.random.default_rng(seed)
    n = len(y)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(
            roc_auc_score(yb, np.asarray(s_irl)[idx])
            - roc_auc_score(yb, np.asarray(s_rf)[idx])
        )
    if not boots:
        return base, np.nan, np.nan
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(base), float(lo), float(hi)


# ───────────────────────── load ─────────────────────────
def load() -> pd.DataFrame:
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    re = pd.read_csv(DATA / "pair_repo_experience.csv")
    m = pair.merge(
        re[["developer", "directory", "top_dir", "prior_n", "prior_acc",
            "prior_recency_days", "log_prior_n", "repo_exp_covered"]],
        on=["developer", "directory"], how="left",
    )
    labs = [f"label_{h}" for h in HORIZONS]
    m["pair_base"] = m[labs].mean(axis=1)  # ペアレベル base_rate プロキシ
    cov = m[m["repo_exp_covered"] == True].copy()  # noqa: E712
    cov = cov.dropna(subset=EXP_VARS)
    return cov


# ─────────────────── 1. 単変量 label AUC ───────────────────
def univariate_auc(cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # 区間別
    for h in HORIZONS:
        y = cov[f"label_{h}"].astype(int).values
        for v in EXP_VARS:
            s = cov[v].values
            # recency は「大きいほど離脱」想定なので符号を反転して AUC を取る
            sgn = -1.0 if v == "prior_recency_days" else 1.0
            auc, lo, hi = boot_auc_ci(y, sgn * s)
            rows.append({
                "scope": h, "var": v, "auc": auc, "ci_lo": lo, "ci_hi": hi,
                "n": len(y), "n_pos": int(y.sum()),
                "base_rate": float(y.mean()),
            })
    # pooled (4区間スタック)
    yp = np.concatenate([cov[f"label_{h}"].astype(int).values for h in HORIZONS])
    for v in EXP_VARS:
        sgn = -1.0 if v == "prior_recency_days" else 1.0
        sp = np.concatenate([sgn * cov[v].values for _ in HORIZONS])
        auc, lo, hi = boot_auc_ci(yp, sp)
        rows.append({
            "scope": "pooled", "var": v, "auc": auc, "ci_lo": lo, "ci_hi": hi,
            "n": len(yp), "n_pos": int(yp.sum()), "base_rate": float(yp.mean()),
        })
    return pd.DataFrame(rows)


# ─────────────── 2. prob vs 経験 連動 (相関) ───────────────
def prob_correlation(cov: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        bvar = f"label_{h}"  # この区間の base_rate を統制に使う
        for expv in ["log_prior_n", "prior_acc"]:
            for model, col in [("IRL", f"irl_{h}"), ("RF", f"rf_{h}")]:
                sub = cov[[col, expv, bvar]].dropna()
                # 素の Spearman
                r, p = spearmanr(sub[col], sub[expv])
                # base_rate (この区間 label) を統制した偏相関
                pr, pp, pn = partial_spearman(
                    sub[expv].values, sub[col].values, sub[bvar].values
                )
                rows.append({
                    "horizon": h, "model": model, "exp_var": expv,
                    "spearman": float(r), "spearman_p": float(p),
                    "partial_spearman_ctrl_base": pr, "partial_p": pp,
                    "n": len(sub),
                })
    return pd.DataFrame(rows)


# ─────── 3. 経験 × base_rate 層別 delta_auc (核心) ───────
def strata_delta(cov: pd.DataFrame) -> pd.DataFrame:
    """経験二分(log_prior_n 中央分割) × base_rate 三分位 で各セル delta_auc。
    base_rate は per-pair の pair_base を使う。delta_auc は区間 pool で算出。"""
    cov = cov.copy()
    med = cov["log_prior_n"].median()
    cov["exp_grp"] = np.where(cov["log_prior_n"] <= med, "低経験", "高経験")
    # base_rate 三分位 (pair_base)
    cov["base_grp"] = pd.qcut(
        cov["pair_base"].rank(method="first"), 3,
        labels=["base低", "base中", "base高"],
    )
    rows = []
    for eg in ["低経験", "高経験"]:
        for bg in ["base低", "base中", "base高"]:
            sub = cov[(cov["exp_grp"] == eg) & (cov["base_grp"] == bg)]
            if len(sub) < 20:
                continue
            # 区間 pool して AUC
            ys, irls, rfs = [], [], []
            for h in HORIZONS:
                ys.append(sub[f"label_{h}"].astype(int).values)
                irls.append(sub[f"irl_{h}"].values)
                rfs.append(sub[f"rf_{h}"].values)
            y = np.concatenate(ys)
            si = np.concatenate(irls)
            sr = np.concatenate(rfs)
            irl_auc = safe_auc(y, si)
            rf_auc = safe_auc(y, sr)
            d, dlo, dhi = boot_delta_auc_ci(y, si, sr)
            rows.append({
                "exp_grp": eg, "base_grp": str(bg),
                "n_pair": len(sub),
                "mean_log_prior_n": float(sub["log_prior_n"].mean()),
                "mean_base": float(sub["pair_base"].mean()),
                "irl_auc": irl_auc, "rf_auc": rf_auc,
                "delta_auc": d, "delta_lo": dlo, "delta_hi": dhi,
            })
    df = pd.DataFrame(rows)

    # 経験のみ二分 (base 統制せず) の marginal も別途
    marg = []
    for eg in ["低経験", "高経験"]:
        sub = cov[cov["exp_grp"] == eg]
        ys, irls, rfs = [], [], []
        for h in HORIZONS:
            ys.append(sub[f"label_{h}"].astype(int).values)
            irls.append(sub[f"irl_{h}"].values)
            rfs.append(sub[f"rf_{h}"].values)
        y, si, sr = np.concatenate(ys), np.concatenate(irls), np.concatenate(rfs)
        d, dlo, dhi = boot_delta_auc_ci(y, si, sr)
        marg.append({
            "exp_grp": eg, "base_grp": "ALL(base統制なし)",
            "n_pair": len(sub),
            "mean_log_prior_n": float(sub["log_prior_n"].mean()),
            "mean_base": float(sub["pair_base"].mean()),
            "irl_auc": safe_auc(y, si), "rf_auc": safe_auc(y, sr),
            "delta_auc": d, "delta_lo": dlo, "delta_hi": dhi,
        })
    return pd.concat([df, pd.DataFrame(marg)], ignore_index=True)


# ─────── 3b. ペアレベル偏相関: Δ符号 vs 経験/base ───────
def pairlevel_partial(cov: pd.DataFrame) -> dict:
    """per-pair の IRL-RF 相対優位 (Δ) を、両モデルの per-horizon rank 差で近似し、
    log_prior_n と base_rate のどちらが Δ を動かすか偏相関で比較。
    Δ_pair = 区間平均 [ rankpct(irl) - rankpct(rf) ] (正例だけが理想だが
    全ペアで rank を取り、label と組み合わせるのは別途 AUC が担うため、
    ここは『IRLとRFがそのペアをどれだけ相対的に高く評価したか』の差を見る)。"""
    cov = cov.copy()
    deltas = np.zeros(len(cov))
    cnt = 0
    for h in HORIZONS:
        ri = rankdata(cov[f"irl_{h}"].values) / len(cov)
        rr = rankdata(cov[f"rf_{h}"].values) / len(cov)
        deltas += (ri - rr)
        cnt += 1
    cov["delta_rankpct"] = deltas / cnt

    out = {}
    # 素の Spearman
    for v in ["log_prior_n", "base"]:
        col = "log_prior_n" if v == "log_prior_n" else "pair_base"
        r, p = spearmanr(cov["delta_rankpct"], cov[col])
        out[f"spearman_delta_vs_{v}"] = (float(r), float(p))
    # 偏相関: log_prior_n 効果 (base 統制) / base 効果 (log_prior_n 統制)
    r1, p1, n1 = partial_spearman(
        cov["log_prior_n"].values, cov["delta_rankpct"].values, cov["pair_base"].values
    )
    r2, p2, n2 = partial_spearman(
        cov["pair_base"].values, cov["delta_rankpct"].values, cov["log_prior_n"].values
    )
    out["partial_delta_vs_logn_ctrl_base"] = (r1, p1, n1)
    out["partial_delta_vs_base_ctrl_logn"] = (r2, p2, n2)
    return out


# ───────────────────────── figure ─────────────────────────
def plot_figure(uni: pd.DataFrame, corr: pd.DataFrame, strata: pd.DataFrame,
                out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) 経験指標の単変量 label AUC (区間別)
    ax = axes[0]
    sub = uni[uni["scope"] != "pooled"]
    x = np.arange(len(HORIZONS))
    width = 0.25
    colors = {"log_prior_n": "#1f77b4", "prior_acc": "#2ca02c",
              "prior_recency_days": "#d62728"}
    for i, v in enumerate(EXP_VARS):
        sv = sub[sub["var"] == v].set_index("scope").reindex(HORIZONS)
        yerr = np.vstack([sv["auc"] - sv["ci_lo"], sv["ci_hi"] - sv["auc"]])
        ax.bar(x + (i - 1) * width, sv["auc"].values, width,
               yerr=yerr, capsize=2.5, color=colors[v],
               label=EXP_LABEL[v], alpha=0.88)
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS)
    ax.set_ylim(0.45, 0.75)
    ax.set_ylabel("単変量 label AUC")
    ax.set_title("(a) repo経験指標の継続予測力 (区間別, 95%CI)", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.4)

    # (b) IRL/RF prob vs log_prior_n 連動 (Spearman + base統制偏相関)
    ax = axes[1]
    sub = corr[corr["exp_var"] == "log_prior_n"]
    x = np.arange(len(HORIZONS))
    width = 0.2
    for j, (model, base_c, part_c) in enumerate([
        ("IRL", "#ff7f0e", "#ffbb78"), ("RF", "#9467bd", "#c5b0d5")
    ]):
        sv = sub[sub["model"] == model].set_index("horizon").reindex(HORIZONS)
        ax.bar(x + (2 * j - 1.5) * width, sv["spearman"].values, width,
               color=base_c, label=f"{model} 素のρ", alpha=0.9)
        ax.bar(x + (2 * j - 0.5) * width, sv["partial_spearman_ctrl_base"].values,
               width, color=part_c, label=f"{model} base統制偏ρ", alpha=0.9)
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS)
    ax.set_ylabel("Spearman ρ (prob vs log_prior_n)")
    ax.set_title("(b) prob と repo経験の連動 — RFの方が経験依存?", fontsize=11)
    ax.legend(fontsize=7.5, ncols=2, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.4)

    # (c) 経験二分 × base_rate三分位 の delta_auc(IRL-RF)
    ax = axes[2]
    cell = strata[strata["base_grp"] != "ALL(base統制なし)"].copy()
    base_order = ["base低", "base中", "base高"]
    x = np.arange(len(base_order))
    width = 0.35
    for j, (eg, c) in enumerate([("低経験", "#1f77b4"), ("高経験", "#d62728")]):
        sv = cell[cell["exp_grp"] == eg].set_index("base_grp").reindex(base_order)
        yerr = np.vstack([
            sv["delta_auc"] - sv["delta_lo"],
            sv["delta_hi"] - sv["delta_auc"],
        ])
        ax.bar(x + (j - 0.5) * width, sv["delta_auc"].values, width,
               yerr=yerr, capsize=3, color=c, label=eg, alpha=0.88)
        for k, (xx, n) in enumerate(zip(x, sv["n_pair"].values)):
            if np.isfinite(n):
                ax.text(xx + (j - 0.5) * width, -0.085,
                        f"n={int(n)}", ha="center", fontsize=7, color="#555")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(base_order)
    ax.set_ylim(-0.1, 0.1)
    ax.set_ylabel("delta_auc = IRL AUC − RF AUC")
    ax.set_title("(c) 経験 × base_rate統制 の delta_auc", fontsize=11)
    ax.legend(fontsize=9, title="log_prior_n 二分")
    ax.grid(axis="y", ls=":", alpha=0.4)

    fig.suptitle(
        "repo経験 ablation: RF優位の正体は経験か base_rate か (被覆2284ペア)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


# ───────────────────────── main ─────────────────────────
def main() -> None:
    cov = load()
    print(f"被覆ペア: {len(cov):,} (devs={cov['developer'].nunique()})")
    print(f"log_prior_n median={cov['log_prior_n'].median():.3f}, "
          f"prior_acc mean={cov['prior_acc'].mean():.3f}")

    # 1
    uni = univariate_auc(cov)
    uni.to_csv(OUT / "repo_exp_univariate_auc.csv", index=False)
    print("\n[1] 単変量 label AUC (pooled)")
    print(uni[uni["scope"] == "pooled"][["var", "auc", "ci_lo", "ci_hi"]]
          .round(3).to_string(index=False))

    # 2
    corr = prob_correlation(cov)
    corr.to_csv(OUT / "repo_exp_prob_correlation.csv", index=False)
    print("\n[2] prob vs log_prior_n Spearman (素 / base統制偏相関)")
    piv = corr[corr["exp_var"] == "log_prior_n"].pivot_table(
        index="horizon", columns="model",
        values=["spearman", "partial_spearman_ctrl_base"],
    )
    print(piv.round(3).to_string())
    # pooled-ish summary across horizons
    for expv in ["log_prior_n", "prior_acc"]:
        s = corr[corr["exp_var"] == expv]
        irl_r = s[s["model"] == "IRL"]["spearman"].mean()
        rf_r = s[s["model"] == "RF"]["spearman"].mean()
        irl_pr = s[s["model"] == "IRL"]["partial_spearman_ctrl_base"].mean()
        rf_pr = s[s["model"] == "RF"]["partial_spearman_ctrl_base"].mean()
        print(f"  [{expv}] mean素ρ IRL={irl_r:.3f} RF={rf_r:.3f} | "
              f"base統制偏ρ IRL={irl_pr:.3f} RF={rf_pr:.3f} "
              f"(|RF|>|IRL|? {abs(rf_pr) > abs(irl_pr)})")

    # 3
    strata = strata_delta(cov)
    strata.to_csv(OUT / "repo_exp_x_baserate_strata.csv", index=False)
    print("\n[3] 経験 × base_rate統制 delta_auc(IRL-RF)")
    print(strata[["exp_grp", "base_grp", "n_pair", "mean_base",
                  "irl_auc", "rf_auc", "delta_auc", "delta_lo", "delta_hi"]]
          .round(3).to_string(index=False))

    # 3b
    pl = pairlevel_partial(cov)
    print("\n[3b] ペアレベル Δ(IRL-RF rank優位) の偏相関比較")
    for k, v in pl.items():
        print(f"  {k}: {tuple(round(x, 4) for x in v)}")

    # figure
    plot_figure(uni, corr, strata, OUT / "fig_repo_experience_ablation.pdf")
    print("\nDONE")


if __name__ == "__main__":
    main()
