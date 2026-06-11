#!/usr/bin/env python3
"""30a: プロジェクト粒度の予測特性 characterization.

入力:
  results/project_master.csv      (69 project; reliable=n_pair>=30 が 28)
  results/project_attrition_timing.csv (top_dir×3遷移)

やること:
  1. 予測性ランキング (Ens AUC で並べ top/bottom)
  2. IRL/RF 優劣ランキング (delta_auc)
  3. dir レベル知見 (P1未経験↑→IRL優位 / P2経験低承諾↑→RF優位) が
     project 粒度でも成立するか: Spearman + log10(n_pair)+base_rate 統制偏相関
  4. project 脱落 timing 概観
  5. 図 (PDF): auc_ranking / delta_sorted / archetype_stacked / attrition_timing
"""
import json
import os

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

RES = "/Users/kazuki-h/Master_research/writing/analysis/results"
MASTER = os.path.join(RES, "project_master.csv")
ATTR = os.path.join(RES, "project_attrition_timing.csv")


def partial_spearman(df, x, y, controls):
    """log/raw control を統制した偏 Spearman 相関 (rank → 残差 → Pearson)。

    n が小さいので t 検定で p を出す。戻り: (r, p, n)。
    """
    sub = df[[x, y] + controls].dropna()
    n = len(sub)
    if n < 5:
        return np.nan, np.nan, n
    # 各変数を rank 化
    rx = stats.rankdata(sub[x].values)
    ry = stats.rankdata(sub[y].values)
    C = np.column_stack([np.ones(n)] + [stats.rankdata(sub[c].values) for c in controls])
    # 残差化
    bx, *_ = np.linalg.lstsq(C, rx, rcond=None)
    by, *_ = np.linalg.lstsq(C, ry, rcond=None)
    ex = rx - C @ bx
    ey = ry - C @ by
    r, _ = stats.pearsonr(ex, ey)
    k = len(controls)
    dof = n - 2 - k
    if dof <= 0:
        return r, np.nan, n
    t = r * np.sqrt(dof / max(1e-12, 1 - r * r))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, n


def main():
    df = pd.read_csv(MASTER)
    df["log10_npair"] = np.log10(df["n_pair"])
    rel = df[df["reliable"]].copy()
    print(f"reliable projects: {len(rel)} / {len(df)}")

    outputs = {}

    # ============ 1. 予測性ランキング ============
    cols_rank = ["top_dir", "ens_auc", "irl_auc", "rf_auc", "base_rate", "n_pair",
                 "n_dev", "n_dir", "delta_auc"]
    rank = rel[cols_rank].sort_values("ens_auc", ascending=False).reset_index(drop=True)
    rank.insert(0, "rank", range(1, len(rank) + 1))
    p1 = os.path.join(RES, "30a_project_auc_ranking.csv")
    rank.round(4).to_csv(p1, index=False)
    outputs["auc_ranking"] = p1
    print("\n=== 1. Ens AUC ランキング (top5 / bottom5) ===")
    print(rank.head(5)[["rank", "top_dir", "ens_auc", "irl_auc", "rf_auc",
                        "base_rate", "n_pair"]].to_string(index=False))
    print("...")
    print(rank.tail(5)[["rank", "top_dir", "ens_auc", "irl_auc", "rf_auc",
                        "base_rate", "n_pair"]].to_string(index=False))

    # ============ 2. IRL/RF 優劣ランキング ============
    cols_d = ["top_dir", "delta_auc", "irl_auc", "rf_auc", "ens_auc", "base_rate",
              "n_pair", "pair_P1_share", "pair_P2_share", "dev_A4_share"]
    dsort = rel[cols_d].sort_values("delta_auc", ascending=False).reset_index(drop=True)
    dsort["優位"] = np.where(dsort["delta_auc"] > 0, "IRL", "RF")
    p2 = os.path.join(RES, "30a_project_delta_ranking.csv")
    dsort.round(4).to_csv(p2, index=False)
    outputs["delta_ranking"] = p2
    irl_fav = list(dsort[dsort["delta_auc"] > 0]["top_dir"])
    rf_fav = list(dsort[dsort["delta_auc"] < 0]["top_dir"])
    print("\n=== 2. delta_auc (IRL-RF) ランキング ===")
    print(f"IRL優位 ({len(irl_fav)}): {irl_fav}")
    print(f"RF優位  ({len(rf_fav)}): {rf_fav}")
    print("top5 IRL優位:")
    print(dsort.head(5)[["top_dir", "delta_auc", "irl_auc", "rf_auc",
                         "pair_P1_share", "pair_P2_share"]].to_string(index=False))
    print("top5 RF優位 (delta最小):")
    print(dsort.tail(5)[["top_dir", "delta_auc", "irl_auc", "rf_auc",
                         "pair_P1_share", "pair_P2_share"]].to_string(index=False))

    # ============ 3. dir 知見の project 粒度検証 ============
    print("\n=== 3. 相関検証 (reliable n=28) ===")
    corr_rows = []
    pairs = [
        ("pair_P1_share", "delta_auc"),
        ("pair_P2_share", "delta_auc"),
        ("dev_A4_share", "delta_auc"),
        ("pair_P1_share", "ens_auc"),
        ("pair_P2_share", "ens_auc"),
        ("dev_A4_share", "ens_auc"),
    ]
    controls = ["log10_npair", "base_rate"]
    for x, y in pairs:
        sub = rel[[x, y]].dropna()
        rho, pval = stats.spearmanr(sub[x], sub[y])
        pr, pp, pn = partial_spearman(rel, x, y, controls)
        corr_rows.append({
            "x": x, "y": y, "n": len(sub),
            "spearman_rho": round(rho, 4), "spearman_p": round(pval, 4),
            "partial_rho_ctrl": round(pr, 4), "partial_p": round(pp, 4),
        })
        print(f"  {x:16s} vs {y:9s}: spearman rho={rho:+.3f} (p={pval:.3f}) | "
              f"partial rho={pr:+.3f} (p={pp:.3f})  [ctrl log10_npair+base_rate]")
    corr = pd.DataFrame(corr_rows)
    p3 = os.path.join(RES, "30a_project_correlations.csv")
    corr.to_csv(p3, index=False)
    outputs["correlations"] = p3

    # 外れ値感度: delta_auc 最小 (octavia/magnum/devstack 等の極端値) を1つ抜いて再計算
    print("  -- 外れ値感度 (P1_share vs delta_auc, |delta|最大の1点除外) --")
    rel_sorted = rel.reindex(rel["delta_auc"].abs().sort_values(ascending=False).index)
    drop_one = rel_sorted.iloc[1:]
    rho_d, p_d = stats.spearmanr(drop_one["pair_P1_share"], drop_one["delta_auc"])
    outlier = rel_sorted.iloc[0]["top_dir"]
    print(f"     除外 project={outlier}: spearman rho={rho_d:+.3f} (p={p_d:.3f}, n={len(drop_one)})")
    corr_rows.append({"x": "pair_P1_share", "y": "delta_auc",
                      "n": len(drop_one),
                      "spearman_rho": round(rho_d, 4), "spearman_p": round(p_d, 4),
                      "partial_rho_ctrl": f"outlier_drop({outlier})", "partial_p": ""})
    pd.DataFrame(corr_rows).to_csv(p3, index=False)

    # ============ 4. project 脱落 timing ============
    at = pd.read_csv(ATTR)
    rel_set = set(rel["top_dir"])
    at_rel = at[at["top_dir"].isin(rel_set)].copy()
    print(f"\n=== 4. 脱落 timing (reliable project, n_row={len(at_rel)}) ===")
    # 遷移ごとの平均脱落率
    by_trans = at_rel.groupby("transition")["attrition_rate"].agg(["mean", "median", "count"])
    order = ["序盤", "中盤", "終盤"]
    by_trans = by_trans.reindex([o for o in order if o in by_trans.index])
    print("遷移別 平均脱落率:")
    print(by_trans.round(3).to_string())
    # ピーク脱落 timing を project ごとに
    pivot = at_rel.pivot_table(index="top_dir", columns="transition",
                               values="attrition_rate", aggfunc="mean")
    pivot = pivot.reindex(columns=[o for o in order if o in pivot.columns])
    pivot["peak_timing"] = pivot[[c for c in order if c in pivot.columns]].idxmax(axis=1)
    p4 = os.path.join(RES, "30a_project_attrition_pivot.csv")
    pivot.round(4).to_csv(p4)
    outputs["attrition_pivot"] = p4
    peak_counts = pivot["peak_timing"].value_counts()
    print("ピーク脱落 timing の project 数:")
    print(peak_counts.to_string())

    # ============ 5. 図 ============
    figs = []

    # (a) AUC ランキング 横棒 (Ens 降順)
    fig, ax = plt.subplots(figsize=(8, 9))
    g = rank.sort_values("ens_auc")  # 下から
    y = np.arange(len(g))
    h = 0.26
    ax.barh(y + h, g["irl_auc"], h, label="IRL", color="#4C72B0")
    ax.barh(y, g["rf_auc"], h, label="RF", color="#DD8452")
    ax.barh(y - h, g["ens_auc"], h, label="Ensemble", color="#55A868")
    ax.set_yticks(y)
    ax.set_yticklabels(g["top_dir"])
    ax.axvline(0.5, ls="--", c="gray", lw=0.8)
    ax.set_xlim(0.4, 0.95)
    ax.set_xlabel("AUC")
    ax.set_title("プロジェクト別 予測性能 (reliable 28, Ens AUC 降順)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fa = os.path.join(RES, "fig_project_auc_ranking.pdf")
    fig.savefig(fa, format="pdf")
    plt.close(fig)
    figs.append(fa)

    # (b) delta_auc ソート棒 (色=IRL/RF 優位)
    fig, ax = plt.subplots(figsize=(8, 9))
    g = dsort.sort_values("delta_auc")
    colors = np.where(g["delta_auc"] > 0, "#4C72B0", "#DD8452")
    y = np.arange(len(g))
    ax.barh(y, g["delta_auc"], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(g["top_dir"])
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("delta_auc = IRL_AUC - RF_AUC  (正→IRL優位)")
    ax.set_title("プロジェクト別 IRL/RF 優劣 (青=IRL優位 / 橙=RF優位)")
    fig.tight_layout()
    fb = os.path.join(RES, "fig_project_delta_sorted.pdf")
    fig.savefig(fb, format="pdf")
    plt.close(fig)
    figs.append(fb)

    # (c) pair archetype (P1-P4) 構成積み上げ棒 (delta 順)
    fig, ax = plt.subplots(figsize=(8, 9))
    g = dsort.merge(rel[["top_dir", "pair_P3_share", "pair_P4_share"]],
                    on="top_dir", how="left")
    g = g.sort_values("delta_auc").reset_index(drop=True)
    y = np.arange(len(g))
    left = np.zeros(len(g))
    pcolors = {"pair_P1_share": "#4C72B0", "pair_P2_share": "#DD8452",
               "pair_P3_share": "#55A868", "pair_P4_share": "#C44E52"}
    plabels = {"pair_P1_share": "P1 未経験", "pair_P2_share": "P2 経験/低承諾",
               "pair_P3_share": "P3 経験/高承諾", "pair_P4_share": "P4 中程度関与"}
    for col in ["pair_P1_share", "pair_P2_share", "pair_P3_share", "pair_P4_share"]:
        ax.barh(y, g[col], left=left, color=pcolors[col], label=plabels[col])
        left += g[col].fillna(0).values
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} ({d:+.02f})" for t, d in zip(g["top_dir"], g["delta_auc"])])
    ax.set_xlabel("ペア archetype 構成比")
    ax.set_xlim(0, 1)
    ax.set_title("プロジェクト別 ペアarchetype構成 (下=RF優位 → 上=IRL優位)")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fc = os.path.join(RES, "fig_project_archetype_stacked.pdf")
    fig.savefig(fc, format="pdf")
    plt.close(fig)
    figs.append(fc)

    # (d) attrition timing ヒートマップ (project × 3遷移)
    hm = pivot[[c for c in order if c in pivot.columns]].copy()
    # delta_auc 順に並べる (可読性: rel の delta_auc でソート)
    dmap = rel.set_index("top_dir")["delta_auc"]
    hm = hm.reindex(dmap.reindex(hm.index).sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(5.5, 9))
    im = ax.imshow(hm.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(hm.shape[1]))
    ax.set_xticklabels(hm.columns)
    ax.set_yticks(range(hm.shape[0]))
    ax.set_yticklabels(hm.index, fontsize=8)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if v < 0.6 else "white")
    fig.colorbar(im, ax=ax, label="脱落率", fraction=0.046, pad=0.04)
    ax.set_title("プロジェクト別 脱落timing (delta_auc降順)")
    fig.tight_layout()
    fd = os.path.join(RES, "fig_project_attrition_timing.pdf")
    fig.savefig(fd, format="pdf")
    plt.close(fig)
    figs.append(fd)

    print("\n=== outputs ===")
    for k, v in outputs.items():
        print(f"  {k}: {v}")
    print("=== figures ===")
    for f in figs:
        print(f"  {f}")

    # サマリ JSON (機械可読)
    summary = {
        "n_reliable": int(len(rel)),
        "ens_auc_top": rank.head(3)[["top_dir", "ens_auc"]].to_dict("records"),
        "ens_auc_bottom": rank.tail(3)[["top_dir", "ens_auc"]].to_dict("records"),
        "irl_favored": irl_fav,
        "rf_favored": rf_fav,
        "correlations": corr_rows,
        "attrition_peak_counts": peak_counts.to_dict(),
        "attrition_by_transition_mean": by_trans["mean"].round(4).to_dict(),
    }
    pj = os.path.join(RES, "30a_project_summary.json")
    with open(pj, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    outputs["summary_json"] = pj
    print(f"  summary: {pj}")


if __name__ == "__main__":
    main()
