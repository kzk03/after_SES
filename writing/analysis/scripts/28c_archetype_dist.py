"""
28c: archetype 構成分布 → IRL/RF 予測優劣 (delta_auc) / アンサンブル AUC (ens_auc)

担当課題: 開発者タイプ (dev archetype) / ペアタイプ (pair archetype) の
ディレクトリ内構成比が予測の優劣を変えるか。

入力: results/directory_master.csv
出力:
  - results/fig_dir_archetype_vs_delta.pdf (8パネル散布図)
  - results/archetype_dist_corr.csv (Spearman + 偏相関の一覧)
  - results/archetype_dist_extremes.csv (delta 上位/下位 5 dir の構成比表)

規約: n_pair (規模) が AUC の最強交絡。必ず log10(n_pair)+base_rate 統制の
偏相関 (partial Spearman) を併記し、統制で消える相関は「弱い/交絡」と判定する。
"""
import json

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = "/Users/kazuki-h/Master_research/writing/analysis/results"
MASTER = f"{RESULTS}/directory_master.csv"

DEV_COLS = ["dev_A1_share", "dev_A2_share", "dev_A3_share", "dev_A4_share"]
PAIR_COLS = ["pair_P1_share", "pair_P2_share", "pair_P3_share", "pair_P4_share"]
SHARE_COLS = DEV_COLS + PAIR_COLS

DEV_LABELS = {
    "dev_A1_share": "dev A1 低活動新人",
    "dev_A2_share": "dev A2 レビュー専念",
    "dev_A3_share": "dev A3 高承諾ベテラン",
    "dev_A4_share": "dev A4 常時活動",
}
PAIR_LABELS = {
    "pair_P1_share": "pair P1 未経験",
    "pair_P2_share": "pair P2 経験/低承諾",
    "pair_P3_share": "pair P3 経験/高承諾",
    "pair_P4_share": "pair P4 中程度関与",
}
ALL_LABELS = {**DEV_LABELS, **PAIR_LABELS}


def partial_spearman(df, x, y, controls):
    """x と y の偏 Spearman 相関 (controls を統制)。
    ランク変換 → controls を説明変数とする線形回帰の残差同士の Pearson 相関。
    返り値: (rho, p, n)
    """
    sub = df[[x, y] + controls].dropna()
    n = len(sub)
    if n < 5:
        return np.nan, np.nan, n
    # rank 変換
    R = sub.rank()
    Xc = np.column_stack([np.ones(n)] + [R[c].to_numpy() for c in controls])
    def resid(vec):
        beta, *_ = np.linalg.lstsq(Xc, vec, rcond=None)
        return vec - Xc @ beta
    rx = resid(R[x].to_numpy())
    ry = resid(R[y].to_numpy())
    # 残差の Pearson = 偏相関。p 値は自由度 n-2-len(controls)
    if np.allclose(rx.std(), 0) or np.allclose(ry.std(), 0):
        return np.nan, np.nan, n
    r = np.corrcoef(rx, ry)[0, 1]
    dof = n - 2 - len(controls)
    if dof <= 0:
        return r, np.nan, n
    t = r * np.sqrt(dof / max(1e-12, (1 - r**2)))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, n


def corr_block(df, targets, label):
    """全 share 列 × 各 target で Spearman と偏 Spearman を計算し DataFrame で返す。"""
    rows = []
    df = df.copy()
    df["log_npair"] = np.log10(df["n_pair"])
    controls = ["log_npair", "base_rate"]
    for tcol in targets:
        for scol in SHARE_COLS:
            sub = df[[scol, tcol]].dropna()
            rho, p = stats.spearmanr(sub[scol], sub[tcol])
            prho, pp, pn = partial_spearman(df, scol, tcol, controls)
            rows.append({
                "subset": label,
                "target": tcol,
                "share": scol,
                "share_label": ALL_LABELS[scol],
                "n": len(sub),
                "spearman_rho": round(rho, 3),
                "spearman_p": round(p, 4),
                "partial_rho": round(prho, 3) if not np.isnan(prho) else np.nan,
                "partial_p": round(pp, 4) if not np.isnan(pp) else np.nan,
                "partial_n": pn,
            })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(MASTER)
    rel = df[df["reliable"]].copy()       # 46 dir
    aucd = df[df["auc_defined"]].copy()   # 286 dir
    print(f"reliable={len(rel)}  auc_defined={len(aucd)}")

    # ---------- 1 & 2: Spearman + 偏相関 ----------
    tgt = ["delta_auc", "ens_auc"]
    corr_rel = corr_block(rel, tgt, "reliable46")
    corr_aux = corr_block(aucd, tgt, "auc_defined286")
    corr = pd.concat([corr_rel, corr_aux], ignore_index=True)
    corr_path = f"{RESULTS}/archetype_dist_corr.csv"
    corr.to_csv(corr_path, index=False)

    print("\n=== reliable46: share vs delta_auc ===")
    show = corr_rel[corr_rel["target"] == "delta_auc"].sort_values("spearman_rho")
    print(show[["share_label", "spearman_rho", "spearman_p",
                "partial_rho", "partial_p"]].to_string(index=False))
    print("\n=== reliable46: share vs ens_auc ===")
    show2 = corr_rel[corr_rel["target"] == "ens_auc"].sort_values("spearman_rho")
    print(show2[["share_label", "spearman_rho", "spearman_p",
                 "partial_rho", "partial_p"]].to_string(index=False))

    print("\n=== auc_defined286: share vs delta_auc ===")
    show3 = corr_aux[corr_aux["target"] == "delta_auc"].sort_values("spearman_rho")
    print(show3[["share_label", "spearman_rho", "spearman_p",
                 "partial_rho", "partial_p"]].to_string(index=False))

    # ---------- 4: extremes ----------
    rel_sorted = rel.sort_values("delta_auc")
    bottom5 = rel_sorted.head(5)   # 最も負 = RF 優位
    top5 = rel_sorted.tail(5)      # 最も正 = IRL 優位
    cols_show = (["directory", "top_dir", "n_pair", "base_rate",
                  "irl_auc", "rf_auc", "delta_auc"] + SHARE_COLS)
    ext = pd.concat([
        top5.assign(group="IRL優位(delta+)"),
        bottom5.assign(group="RF優位(delta-)"),
    ])[["group"] + cols_show].copy()
    for c in ["base_rate", "irl_auc", "rf_auc", "delta_auc"] + SHARE_COLS:
        ext[c] = ext[c].round(3)
    ext_path = f"{RESULTS}/archetype_dist_extremes.csv"
    ext.to_csv(ext_path, index=False)
    print("\n=== extremes (reliable, delta_auc) ===")
    print(ext.to_string(index=False))

    # 上位/下位の構成比平均差
    diff = (top5[SHARE_COLS].mean() - bottom5[SHARE_COLS].mean())
    print("\n=== mean share diff (IRL優位top5 - RF優位bottom5) ===")
    for c in SHARE_COLS:
        print(f"  {ALL_LABELS[c]:24s}: {diff[c]:+.3f}")

    # ---------- 5: 8 パネル散布図 (delta_auc) ----------
    df_p = rel.copy()
    df_p["log_npair"] = np.log10(df_p["n_pair"])
    controls = ["log_npair", "base_rate"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    order = DEV_COLS + PAIR_COLS
    for ax, scol in zip(axes.flat, order):
        sub = df_p[[scol, "delta_auc", "n_pair"]].dropna()
        rho, p = stats.spearmanr(sub[scol], sub["delta_auc"])
        prho, pp, _ = partial_spearman(df_p, scol, "delta_auc", controls)
        sizes = 18 + 60 * (np.log10(sub["n_pair"]) - np.log10(20)) / (
            np.log10(df_p["n_pair"].max()) - np.log10(20) + 1e-9)
        ax.scatter(sub[scol], sub["delta_auc"], s=sizes, alpha=0.7,
                   edgecolor="k", linewidth=0.4, color="#4C72B0")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        # 線形トレンド
        if len(sub) >= 3 and sub[scol].std() > 0:
            b, a = np.polyfit(sub[scol], sub["delta_auc"], 1)
            xs = np.linspace(sub[scol].min(), sub[scol].max(), 50)
            ax.plot(xs, a + b * xs, color="#C44E52", lw=1.3)
        ax.set_xlabel(ALL_LABELS[scol] + " (構成比)")
        ax.set_ylabel("delta_auc (正=IRL優位)")
        star = "*" if (not np.isnan(pp) and pp < 0.05) else ""
        ax.set_title(
            f"{ALL_LABELS[scol]}\n"
            f"Spearman rho={rho:+.2f} (p={p:.3f})\n"
            f"偏相関 rho={prho:+.2f} (p={pp:.3f}){star}",
            fontsize=9)
    fig.suptitle("ディレクトリ内 archetype 構成比 vs delta_auc (reliable 46 dir; "
                 "点サイズ=log n_pair)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = f"{RESULTS}/fig_dir_archetype_vs_delta.pdf"
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved fig: {fig_path}")

    # ---------- W1/§6 事前知見の検証サマリ (json) ----------
    def get(subset_df, share, target):
        r = corr[(corr.subset == subset_df) & (corr.share == share)
                 & (corr.target == target)].iloc[0]
        return dict(rho=r.spearman_rho, p=r.spearman_p,
                    prho=r.partial_rho, pp=r.partial_p)

    prior = {
        "W1_pairP2P3_経験あり多い→RF優位(delta負)を期待": {
            "P2_vs_delta_reliable": get("reliable46", "pair_P2_share", "delta_auc"),
            "P3_vs_delta_reliable": get("reliable46", "pair_P3_share", "delta_auc"),
            "P2_vs_delta_aux286": get("auc_defined286", "pair_P2_share", "delta_auc"),
            "P3_vs_delta_aux286": get("auc_defined286", "pair_P3_share", "delta_auc"),
        },
        "W1_pairP1_未経験多い→IRL寄り(delta正)を期待": {
            "P1_vs_delta_reliable": get("reliable46", "pair_P1_share", "delta_auc"),
            "P1_vs_delta_aux286": get("auc_defined286", "pair_P1_share", "delta_auc"),
        },
        "W1_devA4_常時活動多い→予測困難(ens_auc低)を期待": {
            "A4_vs_ens_reliable": get("reliable46", "dev_A4_share", "ens_auc"),
            "A4_vs_ens_aux286": get("auc_defined286", "dev_A4_share", "ens_auc"),
        },
    }
    with open(f"{RESULTS}/archetype_dist_prior_check.json", "w") as f:
        json.dump(prior, f, ensure_ascii=False, indent=2)
    print("\n=== prior check ===")
    print(json.dumps(prior, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
