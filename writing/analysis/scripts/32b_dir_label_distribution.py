"""32b: ディレクトリごとの正例/負例分布の不均一が予測難易度を生むことの実証.

主張: 「ディレクトリごとに正例・負例の分布が全然違う、それで予測が難しくなる」を定量化。

入力:
  - results/directory_master.csv : directory 粒度の base_rate, ens_auc, auc_defined, n_pair ...
  - data/pair_trajectories.csv   : pair × 4 期間の label/score (per-dir × period の base_rate 用)

出力:
  - results/fig_dir_label_distribution.pdf : (a)base_rate ヒストグラム, (b)base_rate vs ens_auc, (c)主要 dir の per-period base_rate
  - results/dir_label_distribution_brbins.csv     : base_rate 群別の AUC算出可率・ens_auc
  - results/dir_label_distribution_perperiod.csv  : 主要 dir の per-dir × period base_rate
  - results/dir_label_distribution_partial.csv    : size 統制偏相関の結果

規約: 図 PDF のみ。相関主張は log10(n_pair)+base_rate 統制の偏相関で確認。
"""

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = "/Users/kazuki-h/Master_research/writing/analysis/results"
DATA = "/Users/kazuki-h/Master_research/writing/analysis/data"

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]
LABEL_COLS = [f"label_{p}" for p in PERIODS]


def partial_corr(df, x, y, covars):
    """x と y の偏相関(covars を線形除去後の残差の Pearson 相関)を返す。"""
    sub = df[[x, y] + covars].dropna()
    n = len(sub)
    if n < len(covars) + 3:
        return np.nan, np.nan, n
    Z = np.column_stack([np.ones(n)] + [sub[c].values for c in covars])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    rx = resid(sub[x].values.astype(float))
    ry = resid(sub[y].values.astype(float))
    if rx.std() == 0 or ry.std() == 0:
        return np.nan, np.nan, n
    r = np.corrcoef(rx, ry)[0, 1]
    # t 検定 (自由度 = n - 2 - len(covars))
    dof = n - 2 - len(covars)
    from scipy import stats

    t = r * np.sqrt(dof / max(1e-12, 1 - r**2))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, n


def main():
    dm = pd.read_csv(f"{RESULTS}/directory_master.csv")
    pt = pd.read_csv(f"{DATA}/pair_trajectories.csv")

    dm["br_extreme"] = np.minimum(dm["base_rate"], 1 - dm["base_rate"])  # 最も近いクラス純度までの距離
    dm["dist_half"] = np.abs(dm["base_rate"] - 0.5)  # 0.5(均衡)からの距離
    dm["log_npair"] = np.log10(dm["n_pair"])

    # base_rate 群分け (dm 全体に付与してから ad を作る)
    edges = [0.0, 0.1, 0.3, 0.7, 0.9, 1.0001]
    labels = ["近0[0,0.1)", "低[0.1,0.3)", "中庸[0.3,0.7)", "高[0.7,0.9)", "近1[0.9,1]"]
    dm["br_grp"] = pd.cut(dm["base_rate"], bins=edges, labels=labels, right=False, include_lowest=True)

    ad = dm[dm["auc_defined"]].copy()  # AUC 定義可 286 dir

    # ========================================================================
    # 1. base_rate の dir 間ばらつき記述
    # ========================================================================
    print("=" * 70)
    print("[1] base_rate(正例率)の dir 間ばらつき")
    print("=" * 70)
    print(f"全 dir 数: {len(dm)}  (AUC 定義可: {len(ad)}, 定義不能: {(~dm.auc_defined).sum()})")
    print("全 dir の base_rate: min=%.3f / 中央=%.3f / max=%.3f / 平均=%.3f / std=%.3f" % (
        dm["base_rate"].min(), dm["base_rate"].median(), dm["base_rate"].max(),
        dm["base_rate"].mean(), dm["base_rate"].std()))
    print("AUC定義可 dir の base_rate: min=%.3f / 中央=%.3f / max=%.3f / std=%.3f" % (
        ad["base_rate"].min(), ad["base_rate"].median(), ad["base_rate"].max(), ad["base_rate"].std()))

    n_pure0 = int((dm["base_rate"] == 0).sum())
    n_pure1 = int((dm["base_rate"] == 1).sum())
    n_extreme = int((dm["br_extreme"] < 0.1).sum())
    print(f"極端な dir: 全員拒否(base_rate=0)={n_pure0}, 全員承諾(base_rate=1)={n_pure1}, "
          f"超不均衡(br_extreme<0.1)={n_extreme} / {len(dm)} ({100*n_extreme/len(dm):.0f}%)")
    # AUC 不能 dir はすべて単一クラスか?
    undef = dm[~dm["auc_defined"]]
    n_undef_pure = int(undef["base_rate"].isin([0, 1]).sum())
    print(f"AUC 定義不能 {len(undef)} dir のうち単一クラス(base_rate=0 or 1): {n_undef_pure} "
          f"({100*n_undef_pure/len(undef):.0f}%) → 不均衡の極致で AUC が算出不能")

    # ========================================================================
    # 2. base_rate と予測しやすさ(ens_auc)の関係 + AUC算出可率
    # ========================================================================
    print("\n" + "=" * 70)
    print("[2] base_rate 群 × (AUC算出可率, ens_auc)")
    print("=" * 70)
    # 近0 / 中庸 / 近1 の 3 群 + さらに細かい分位を併記 (br_grp は冒頭で付与済み)
    rows = []
    for g in labels:
        sub = dm[dm["br_grp"] == g]
        if len(sub) == 0:
            continue
        sub_ad = sub[sub["auc_defined"]]
        rows.append({
            "br_group": g,
            "n_dir": len(sub),
            "auc_def_rate": round(sub["auc_defined"].mean(), 3),
            "mean_n_pair": round(sub["n_pair"].mean(), 1),
            "mean_base_rate": round(sub["base_rate"].mean(), 3),
            "n_auc_defined": len(sub_ad),
            "mean_ens_auc": round(sub_ad["ens_auc"].mean(), 4) if len(sub_ad) else np.nan,
            "median_ens_auc": round(sub_ad["ens_auc"].median(), 4) if len(sub_ad) else np.nan,
            "std_ens_auc": round(sub_ad["ens_auc"].std(), 4) if len(sub_ad) > 1 else np.nan,
        })
    brbins = pd.DataFrame(rows)
    print(brbins.to_string(index=False))
    brbins.to_csv(f"{RESULTS}/dir_label_distribution_brbins.csv", index=False)

    # br_extreme(不均衡度) でも算出可率を確認
    print("\n不均衡度 br_extreme=min(br,1-br) 別の AUC算出可率:")
    ext_edges = [0.0, 0.001, 0.05, 0.15, 0.5001]
    ext_lab = ["純単一クラス≈0", "極不均衡[.001,.05)", "強不均衡[.05,.15)", "中庸[.15,.5]"]
    dm["ext_grp"] = pd.cut(dm["br_extreme"], bins=ext_edges, labels=ext_lab, right=False, include_lowest=True)
    for g in ext_lab:
        sub = dm[dm["ext_grp"] == g]
        if len(sub):
            print(f"  {g}: n_dir={len(sub):3d}, AUC算出可率={sub['auc_defined'].mean():.3f}, "
                  f"mean_npair={sub['n_pair'].mean():.1f}")

    # 相関 (within auc_defined)
    print("\nAUC定義可 dir 内の相関:")
    print("  base_rate   vs ens_auc : r=%.3f" % ad["base_rate"].corr(ad["ens_auc"]))
    print("  dist_half   vs ens_auc : r=%.3f  (0.5均衡からの距離→負なら均衡ほど高AUC)" %
          ad["dist_half"].corr(ad["ens_auc"]))
    print("  br_extreme  vs ens_auc : r=%.3f  (不均衡度→正なら均衡ほど低?要確認)" %
          ad["br_extreme"].corr(ad["ens_auc"]))
    print("  log_npair   vs ens_auc : r=%.3f" % ad["log_npair"].corr(ad["ens_auc"]))

    # ----- size(log n_pair) 統制の偏相関 -----
    print("\nsize 統制偏相関 (covar=log10 n_pair):")
    prows = []
    for x, desc in [("base_rate", "正例率"), ("dist_half", "0.5からの距離"),
                    ("br_extreme", "不均衡度min(br,1-br)")]:
        r, p, n = partial_corr(ad, x, "ens_auc", ["log_npair"])
        print(f"  {x:12s} vs ens_auc | log_npair: r={r:+.3f}, p={p:.4f}, n={n}  ({desc})")
        prows.append({"x": x, "desc": desc, "y": "ens_auc", "covar": "log_npair",
                      "partial_r": round(r, 4), "p": round(p, 4), "n": n})
    # 線形項とU字項(dist_half)を同時に統制した base_rate の効果 (U字検討)
    # ens_auc ~ base_rate + dist_half + log_npair の係数で線形/二次を見る
    from numpy.linalg import lstsq
    sub = ad[["base_rate", "dist_half", "log_npair", "ens_auc"]].dropna()
    X = np.column_stack([np.ones(len(sub)), sub["base_rate"], sub["dist_half"], sub["log_npair"]])
    beta, *_ = lstsq(X, sub["ens_auc"].values, rcond=None)
    print("\nU字検討 多変量回帰 ens_auc ~ 1 + base_rate + dist_half + log_npair:")
    print("  係数: const=%.4f, base_rate=%.4f, dist_half=%.4f, log_npair=%.4f" % tuple(beta))
    print("  → dist_half(均衡からの距離)の係数が負なら『均衡 dir ほど高AUC=易しい』,")
    print("     base_rate の正係数は『承諾多い dir ほど高AUC』(U字ではなく単調寄り)")
    prows.append({"x": "regr_dist_half_coef", "desc": "ens_auc~br+dist_half+lognpair の dist_half 係数",
                  "y": "ens_auc", "covar": "base_rate+log_npair", "partial_r": round(beta[2], 4),
                  "p": np.nan, "n": len(sub)})
    pd.DataFrame(prows).to_csv(f"{RESULTS}/dir_label_distribution_partial.csv", index=False)

    # AUC の不安定性(ばらつき): 不均衡 dir ほど ens_auc の散らばりが大きいか
    print("\nAUC の不安定性(群内 std of ens_auc):")
    for g in labels:
        sub = ad[ad["br_grp"] == g]
        if len(sub) > 1:
            print(f"  {g}: n={len(sub):3d}, ens_auc std={sub['ens_auc'].std():.4f}, "
                  f"min={sub['ens_auc'].min():.3f}, max={sub['ens_auc'].max():.3f}")

    # ========================================================================
    # 3. per-dir × period の正例率変動
    # ========================================================================
    print("\n" + "=" * 70)
    print("[3] 同一 dir 内でも 4 予測区間で base_rate が動く")
    print("=" * 70)
    perdir_rows = []
    g = pt.groupby("directory")
    for d, sub in g:
        n_pair = len(sub)
        rec = {"directory": d, "top_dir": d.split("/")[0], "n_pair": n_pair}
        brs = []
        for p in PERIODS:
            col = f"label_{p}"
            vals = sub[col].dropna()
            br = vals.mean() if len(vals) else np.nan
            rec[f"br_{p}"] = round(br, 4) if not np.isnan(br) else np.nan
            brs.append(br)
        brs = np.array(brs, dtype=float)
        rec["br_min"] = round(np.nanmin(brs), 4)
        rec["br_max"] = round(np.nanmax(brs), 4)
        rec["br_range"] = round(np.nanmax(brs) - np.nanmin(brs), 4)
        rec["br_std"] = round(np.nanstd(brs), 4)
        perdir_rows.append(rec)
    perdir = pd.DataFrame(perdir_rows).sort_values("n_pair", ascending=False)
    perdir.to_csv(f"{RESULTS}/dir_label_distribution_perperiod.csv", index=False)

    # 主要 dir (n_pair 上位, reliable 相当) の期間変動
    big = perdir[perdir["n_pair"] >= 20]
    print(f"n_pair>=20 の主要 {len(big)} dir における dir 内 period 変動:")
    print("  base_rate の period 間 range: 中央=%.3f, 平均=%.3f, max=%.3f" % (
        big["br_range"].median(), big["br_range"].mean(), big["br_range"].max()))
    print("  base_rate の period 間 std  : 中央=%.3f, 平均=%.3f" % (
        big["br_std"].median(), big["br_std"].mean()))
    # 全 dir vs dir 間 std の比較で「dir 内変動 < dir 間変動」を提示
    across_dir_std = big[[f"br_{p}" for p in PERIODS]].mean(axis=1).std()
    within_dir_std_mean = big["br_std"].mean()
    print(f"  参考: dir 間ばらつき(平均base_rate の std)={across_dir_std:.3f}  >  "
          f"dir 内 period ばらつき平均={within_dir_std_mean:.3f}")
    print("  → dir 間差が主因だが, dir 内でも period で正例率が動く(2次的変動)")
    print("\n  主要 dir の例 (n_pair 上位 12):")
    show = big.head(12)[["directory", "n_pair"] + [f"br_{p}" for p in PERIODS] + ["br_range"]]
    print(show.to_string(index=False))

    # ========================================================================
    # 4. 図 fig_dir_label_distribution.pdf
    # ========================================================================
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.30)

    # --- (a) base_rate ヒストグラム ---
    ax = fig.add_subplot(gs[0, 0])
    # 単一クラス(AUC不能)と両クラス(AUC可)を色分け
    br_def = dm.loc[dm["auc_defined"], "base_rate"]
    br_undef = dm.loc[~dm["auc_defined"], "base_rate"]
    bins = np.linspace(0, 1, 21)
    ax.hist(br_def, bins=bins, color="#4C72B0", alpha=0.85, label=f"AUC算出可 (n={len(br_def)})")
    ax.hist(br_undef, bins=bins, color="#C44E52", alpha=0.85, label=f"AUC算出不能 (n={len(br_undef)})")
    ax.axvline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("ディレクトリの正例率 (base_rate)")
    ax.set_ylabel("ディレクトリ数")
    ax.set_title("(a) 正例率は dir 間で大きくばらつく\n（極端な不均衡 dir で AUC が算出不能）")
    ax.legend(fontsize=9, loc="upper right")

    # --- (b) base_rate vs ens_auc 散布 + 分位別平均 ---
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(ad["base_rate"], ad["ens_auc"], s=8 + 20 * ad["log_npair"],
                    c=ad["log_npair"], cmap="viridis", alpha=0.6, edgecolor="none")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("log10(n_pair)", fontsize=8)
    # 分位別平均(±std)を重ねる
    qb = pd.qcut(ad["base_rate"], 5, duplicates="drop")
    grp = ad.groupby(qb, observed=True)
    mids = grp["base_rate"].mean()
    means = grp["ens_auc"].mean()
    stds = grp["ens_auc"].std()
    ax.errorbar(mids, means, yerr=stds, color="#C44E52", marker="o", lw=2,
                capsize=3, label="正例率5分位の平均±std", zorder=5)
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xlabel("ディレクトリの正例率 (base_rate)")
    ax.set_ylabel("ens_auc (予測しやすさ)")
    rr = ad["base_rate"].corr(ad["ens_auc"])
    rp, pp, _ = partial_corr(ad, "base_rate", "ens_auc", ["log_npair"])
    ax.set_title(f"(b) 不均衡(低 base_rate)ほど AUC が低く不安定\n"
                 f"r={rr:+.2f}, size統制偏相関={rp:+.2f}")
    ax.legend(fontsize=8, loc="lower right")

    # --- (c) 主要 dir の per-period base_rate 推移 ---
    ax = fig.add_subplot(gs[0, 2])
    top_show = big.head(8)
    cmap = plt.get_cmap("tab10")
    x = np.arange(len(PERIODS))
    for i, (_, r) in enumerate(top_show.iterrows()):
        ys = [r[f"br_{p}"] for p in PERIODS]
        ax.plot(x, ys, marker="o", lw=1.6, color=cmap(i % 10),
                label=f"{r['directory']} (n={int(r['n_pair'])})")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(PERIODS)
    ax.set_xlabel("予測区間")
    ax.set_ylabel("正例率 (base_rate)")
    ax.set_title("(c) 同一 dir 内でも区間で正例率が変動\n（dir 内の 2 次的な分布変化）")
    ax.legend(fontsize=6.5, ncol=1, loc="upper right")

    fig.suptitle("ディレクトリごとの正例/負例分布の不均一と予測難易度", fontsize=13, y=1.02)
    out = f"{RESULTS}/fig_dir_label_distribution.pdf"
    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"\n[OK] 図を保存: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
