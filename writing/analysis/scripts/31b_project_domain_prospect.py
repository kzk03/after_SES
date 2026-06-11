"""
31b: project 粒度の企業ドメイン分散と予測しやすさの関係、ドメイン特徴量化の展望。

担当タスク:
1. project 別のドメイン構成 (n_domains, domain_hhi, corporate_share, top_domain) の記述と分散の要約。
2. ドメイン分散指標 vs 予測しやすさ (ens_auc) / IRL-RF 優劣 (delta_auc) の Spearman 相関
   (log10(n_pair)+base_rate 統制の偏相関も)。
3. ドメイン特徴量化の展望評価 (既知 B-43: dir 粒度 within-project 相関ゼロ)。
4. 図 fig_project_domain.pdf。

reliable (n_pair>=30) の 28 project を主対象とする。
"""
import japanize_matplotlib  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

BASE = "/Users/kazuki-h/Master_research/writing/analysis"
PM = f"{BASE}/results/project_master.csv"
OUT = f"{BASE}/results"


def spearman_partial(df, x, y, controls):
    """x と y の Spearman 偏相関 (controls を線形回帰で除去した残差同士の Spearman)。
    順位ベースで残差化するため各変数を rank に変換してから残差をとる。"""
    sub = df[[x, y] + controls].dropna()
    n = len(sub)
    if n < 5:
        return np.nan, np.nan, n
    # rank 変換
    R = sub.rank()
    Xc = R[controls].values
    rx = R[x].values - LinearRegression().fit(Xc, R[x].values).predict(Xc)
    ry = R[y].values - LinearRegression().fit(Xc, R[y].values).predict(Xc)
    rho, p = stats.spearmanr(rx, ry)
    return rho, p, n


def main():
    df = pd.read_csv(PM)
    rel = df[df["reliable"]].copy().reset_index(drop=True)
    rel["log_npair"] = np.log10(rel["n_pair"])
    n = len(rel)
    print(f"reliable projects: {n}")

    # ---- 1. ドメイン構成の記述 ----
    desc_cols = ["top_dir", "n_pair", "base_rate", "ens_auc", "delta_auc",
                 "n_domains", "domain_hhi", "corporate_share", "top_domain"]
    comp = rel[desc_cols].copy().sort_values("domain_hhi", ascending=False)
    # top_domain のシェア = domain_hhi ではない。top_domain シェアは別途近似できないため
    # corporate_share と n_domains, hhi で構成を表現する。
    comp.to_csv(f"{OUT}/project_domain_composition.csv", index=False)

    def summ(col):
        s = rel[col]
        return dict(min=s.min(), q25=s.quantile(.25), median=s.median(),
                    mean=s.mean(), q75=s.quantile(.75), max=s.max())

    print("\n=== ドメイン構成の分散 (reliable 28) ===")
    for c in ["n_domains", "domain_hhi", "corporate_share"]:
        st_ = summ(c)
        print(f"{c}: min={st_['min']:.3f} q25={st_['q25']:.3f} med={st_['median']:.3f} "
              f"mean={st_['mean']:.3f} q75={st_['q75']:.3f} max={st_['max']:.3f}")

    # top_domain の頻度
    print("\ntop_domain 分布:")
    print(rel["top_domain"].value_counts())

    # 類型分け: redhat 支配 / gmail (個人ドメイン) 分散 / その他企業集中
    def archetype(row):
        td = str(row["top_domain"])
        hhi = row["domain_hhi"]
        if td == "gmail.com":
            return "個人(gmail)優位"
        if hhi >= 0.30:
            return f"{td}集中"
        return f"{td}優位(分散)"
    rel["domain_type"] = rel.apply(archetype, axis=1)
    print("\nドメイン類型:")
    print(rel["domain_type"].value_counts())

    # corporate_share の低い/高い極
    print("\ncorporate_share 低い5:")
    print(rel.nsmallest(5, "corporate_share")[["top_dir", "corporate_share",
                                               "top_domain", "domain_hhi"]].to_string(index=False))
    print("\ndomain_hhi 高い5 (集中):")
    print(rel.nlargest(5, "domain_hhi")[["top_dir", "domain_hhi", "n_domains",
                                         "top_domain", "corporate_share"]].to_string(index=False))
    print("\ndomain_hhi 低い5 (分散):")
    print(rel.nsmallest(5, "domain_hhi")[["top_dir", "domain_hhi", "n_domains",
                                          "top_domain", "corporate_share"]].to_string(index=False))

    # ---- 2. 相関分析 ----
    div_vars = ["domain_hhi", "corporate_share", "n_domains"]
    targets = ["ens_auc", "delta_auc"]
    controls = ["log_npair", "base_rate"]

    rows = []
    print("\n=== Spearman 相関 (reliable 28) ===")
    for t in targets:
        for v in div_vars:
            rho, p = stats.spearmanr(rel[v], rel[t])
            prho, pp, nn = spearman_partial(rel, v, t, controls)
            rows.append(dict(target=t, predictor=v, n=n,
                             spearman_rho=rho, spearman_p=p,
                             partial_rho=prho, partial_p=pp))
            print(f"{t:10s} ~ {v:16s}: rho={rho:+.3f} (p={p:.3f}) | "
                  f"partial(log_npair,base_rate) rho={prho:+.3f} (p={pp:.3f})")
    corr = pd.DataFrame(rows)
    corr.to_csv(f"{OUT}/project_domain_corr.csv", index=False)

    # 補助: n_domains は規模(n_pair)と強く交絡しうるので確認
    rho_nd, p_nd = stats.spearmanr(rel["n_domains"], rel["n_pair"])
    rho_hh, p_hh = stats.spearmanr(rel["domain_hhi"], rel["n_pair"])
    rho_cs, p_cs = stats.spearmanr(rel["corporate_share"], rel["n_pair"])
    print(f"\n交絡確認 (vs n_pair): n_domains rho={rho_nd:+.3f}(p={p_nd:.3f}) "
          f"domain_hhi rho={rho_hh:+.3f}(p={p_hh:.3f}) corporate_share rho={rho_cs:+.3f}(p={p_cs:.3f})")

    # ---- 4. 図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8))

    # (a) project 別の top_domain シェア(=corporate_share の内、企業比) + corporate_share 構成
    # corporate_share = 企業ドメイン比、(1-corporate_share)=個人/その他。
    # ens_auc 順に並べる。
    order = rel.sort_values("ens_auc").reset_index(drop=True)
    ypos = np.arange(len(order))
    ax1.barh(ypos, order["corporate_share"], color="#4C72B0", label="企業ドメイン比")
    ax1.barh(ypos, 1 - order["corporate_share"], left=order["corporate_share"],
             color="#DD8452", label="個人/その他")
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([f"{r.top_dir} (AUC={r.ens_auc:.2f}, {r.top_domain})"
                         for r in order.itertuples()], fontsize=7)
    ax1.set_xlabel("ドメイン構成比 (横軸=1.0)")
    ax1.set_xlim(0, 1)
    ax1.set_title("(a) project別 企業/個人ドメイン構成 (ens_auc 昇順)", fontsize=10)
    ax1.legend(loc="lower right", fontsize=8)

    # (b) domain_hhi vs ens_auc 散布図
    sizes = (rel["n_pair"] / rel["n_pair"].max() * 400) + 20
    sc = ax2.scatter(rel["domain_hhi"], rel["ens_auc"], s=sizes,
                     c=rel["corporate_share"], cmap="viridis",
                     alpha=0.75, edgecolors="k", linewidths=0.5)
    for r in rel.itertuples():
        ax2.annotate(r.top_dir, (r.domain_hhi, r.ens_auc), fontsize=6,
                     xytext=(2, 2), textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax2)
    cb.set_label("corporate_share (企業比)", fontsize=8)
    rho_main, p_main = stats.spearmanr(rel["domain_hhi"], rel["ens_auc"])
    prho_main, ppm, _ = spearman_partial(rel, "domain_hhi", "ens_auc", controls)
    ax2.set_xlabel("domain_hhi (ドメイン集中度)")
    ax2.set_ylabel("ens_auc (予測しやすさ)")
    ax2.set_title("(b) ドメイン集中度 vs 予測精度\n"
                  f"Spearman rho={rho_main:+.2f} (p={p_main:.2f}), "
                  f"偏相関 rho={prho_main:+.2f} (p={ppm:.2f})\n点サイズ=n_pair",
                  fontsize=9)

    plt.tight_layout()
    figpath = f"{OUT}/fig_project_domain.pdf"
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    print(f"\nsaved fig: {figpath}")
    print(f"saved csv: {OUT}/project_domain_composition.csv, {OUT}/project_domain_corr.csv")


if __name__ == "__main__":
    main()
