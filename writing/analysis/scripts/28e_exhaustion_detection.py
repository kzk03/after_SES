#!/usr/bin/env python
"""28e: ディレクトリ枯渇(レビュアー離脱)検知フレーミング (rank-based).

人数の絶対予測 (headcount) はしない。すべてランキング品質で枯渇検知を担保する:
  (a) per-dir Ens AUC = dir内で継続者と離脱者を正しく順位付けできるか
  (b) attrition recall@K lift = ens_recall_at_attrited - random_baseline_recall
      (脱落者を低スコア側に並べられるか; n_active_prev>=10 の遷移セル)
  (c) dir予測継続度 vs 実 attrition_weighted の Spearman (順位の妥当性)

入力:
  results/directory_master.csv
  results/horizon_dir_attrition.csv
出力:
  results/dir_exhaustion_danger_cells.csv
  results/dir_exhaustion_dir_ranking.csv
  results/fig_dir_exhaustion_detection.pdf
"""
import json
import os

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/Users/kazuki-h/Master_research/writing/analysis"
RES = os.path.join(BASE, "results")
MASTER = os.path.join(RES, "directory_master.csv")
ATTR = os.path.join(RES, "horizon_dir_attrition.csv")


def partial_spearman(x, y, covars):
    """偏 Spearman 相関: x,y を covars (list of arrays) で順位回帰して残差相関。"""
    df = pd.DataFrame({"x": x, "y": y})
    for i, c in enumerate(covars):
        df[f"c{i}"] = c
    df = df.dropna()
    if len(df) < 8:
        return np.nan, np.nan, len(df)
    rk = df.rank()
    cols = [f"c{i}" for i in range(len(covars))]
    Z = np.column_stack([np.ones(len(rk))] + [rk[c].values for c in cols])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    rx = resid(rk["x"].values)
    ry = resid(rk["y"].values)
    if np.std(rx) < 1e-9 or np.std(ry) < 1e-9:
        return np.nan, np.nan, len(df)
    r, p = stats.pearsonr(rx, ry)  # rank残差のpearson = 偏spearman
    return r, p, len(df)


def main():
    m = pd.read_csv(MASTER)
    a = pd.read_csv(ATTR)

    # ============================================================
    # 1. 枯渇検知の rank-based 定義 + recall lift 算出
    # ============================================================
    a["recall_lift"] = a["ens_recall_at_attrited"] - a["random_baseline_recall"]

    # 解析可能な遷移セル (両クラス揃い recall 算出済み, n_active_prev>=10)
    cells = a[(a["n_active_prev"] >= 10) & a["ens_recall_at_attrited"].notna()].copy()

    # ============================================================
    # 2. reliable dir の per-dir Ens AUC 分布 + 危険セル recall lift 集計
    # ============================================================
    # per-dir Ens AUC: master の reliable dir
    rel = m[m["reliable"] & m["ens_auc"].notna()].copy()
    rel_auc = rel["ens_auc"]
    auc_summary = {
        "n_reliable_dir_with_auc": int(len(rel)),
        "ens_auc_median": float(rel_auc.median()),
        "ens_auc_mean": float(rel_auc.mean()),
        "ens_auc_q25": float(rel_auc.quantile(0.25)),
        "ens_auc_q75": float(rel_auc.quantile(0.75)),
        "frac_ens_auc_gt_0.5": float((rel_auc > 0.5).mean()),
        "frac_ens_auc_ge_0.7": float((rel_auc >= 0.7).mean()),
    }

    # 危険セル: attrition_rate>=0.5 (n_active_prev>=10, recall算出済み)
    danger = cells[cells["attrition_rate"] >= 0.5].copy()
    danger = danger.sort_values("recall_lift", ascending=False).reset_index(drop=True)
    danger_summary = {
        "n_danger_cells": int(len(danger)),
        "n_positive_lift": int((danger["recall_lift"] > 0).sum()),
        "frac_positive_lift": float((danger["recall_lift"] > 0).mean()),
        "median_recall_lift": float(danger["recall_lift"].median()),
        "mean_recall_lift": float(danger["recall_lift"].mean()),
        "median_ens_recall": float(danger["ens_recall_at_attrited"].median()),
        "median_random_recall": float(danger["random_baseline_recall"].median()),
    }

    # 全 (非危険含む) セルの recall lift も参考集計
    all_cell_summary = {
        "n_cells_ge10": int(len(cells)),
        "median_recall_lift_all": float(cells["recall_lift"].median()),
        "frac_positive_lift_all": float((cells["recall_lift"] > 0).mean()),
    }

    # ============================================================
    # 3. 機能する dir vs 失敗する dir の同定
    # ============================================================
    # 機能 = per-dir Ens AUC が高い & そのdirのセルで recall lift 正
    # 失敗例 = 低 Ens AUC セル (devstack 6-9->9-12 等)
    # セル単位で「機能/失敗」を判定: AUC>=0.6 かつ lift>=0 を機能
    cells["works"] = (cells["ens_auc"] >= 0.6) & (cells["recall_lift"] >= 0)
    cells["fails"] = cells["ens_auc"] < 0.4  # 構造的失敗 (ランダム以下に近い順位)
    works_cells = cells[cells["works"]].sort_values("ens_auc", ascending=False)
    fail_cells = cells[cells["fails"]].sort_values("ens_auc")
    work_fail_summary = {
        "n_cells_ge10": int(len(cells)),
        "n_works_cells": int(cells["works"].sum()),
        "n_fail_cells": int(cells["fails"].sum()),
        "works_examples": works_cells[["directory", "h_prev", "h_curr", "ens_auc", "recall_lift"]]
        .head(8)
        .round(3)
        .to_dict("records"),
        "fail_examples": fail_cells[
            ["directory", "h_prev", "h_curr", "ens_auc", "attrition_rate", "recall_lift"]
        ]
        .head(8)
        .round(3)
        .to_dict("records"),
    }
    # devstack 6-9->9-12 構造的失敗の明示確認
    ds_fail = a[
        (a["directory"] == "devstack") & (a["h_prev"] == "6-9m") & (a["h_curr"] == "9-12m")
    ]
    if len(ds_fail):
        work_fail_summary["devstack_6-9_to_9-12"] = {
            "ens_auc": float(ds_fail["ens_auc"].iloc[0]),
            "ens_recall": float(ds_fail["ens_recall_at_attrited"].iloc[0]),
            "random_recall": float(ds_fail["random_baseline_recall"].iloc[0]),
            "attrition_rate": float(ds_fail["attrition_rate"].iloc[0]),
            "recall_lift": float(ds_fail["recall_lift"].iloc[0]),
        }

    # ============================================================
    # 4. dir予測継続度 vs 実 attrition のランキング妥当性 (Spearman)
    # ============================================================
    # dirごとの「予測される枯渇度」を rank-based proxy で構成:
    #   セルレベルの ens_recall_at_attrited を dir ごとに集約 (脱落者を低スコアに並べる力)
    #   = そのdirで枯渇 (脱落) を検出できる度合い
    # それを実 attrition_weighted (master) と Spearman で対応付け。
    # さらに「dir平均予測継続度」proxy: base_rate ではなく予測順位品質を使うため、
    #   per-dir の attrition検出力(平均 recall_lift) を主指標にする。

    # dir 単位で recall_lift / ens_recall を平均 (n_active_prev>=10 セル)
    dir_detect = (
        cells.groupby("directory")
        .agg(
            mean_recall_lift=("recall_lift", "mean"),
            mean_ens_recall=("ens_recall_at_attrited", "mean"),
            mean_cell_attrition=("attrition_rate", "mean"),
            n_cells=("recall_lift", "size"),
        )
        .reset_index()
    )
    dir_rank = m.merge(dir_detect, on="directory", how="left")

    # (4a) 予測継続度 proxy vs 実 attrition_weighted
    #   「予測される離脱率」proxy として (1 - mean_ens_recall) ではなく、
    #   ランキングの妥当性を見たいので: 実 attrition が高いdirほど
    #   検出器が脱落者を低スコアに並べられている (mean_ens_recall 高) ことを期待。
    #   主検証: 各dirの実 attrition_weighted を、検出器の attrition検出力で順位付けできるか。
    sub4 = dir_rank.dropna(subset=["attrition_weighted", "mean_ens_recall"])
    sp_r1, sp_p1 = (
        stats.spearmanr(sub4["mean_ens_recall"], sub4["attrition_weighted"])
        if len(sub4) >= 8
        else (np.nan, np.nan)
    )

    # (4b) 本筋: dir平均予測継続度 (per-dir Ens AUC を継続者識別力の proxy として)
    #   実 attrition_weighted が高い (枯渇) dir を、低い AUC や低 recall で
    #   事前に並べられるか。AUC自体は規模交絡が強いので偏相関で統制。
    sub4b = dir_rank.dropna(subset=["attrition_weighted", "ens_auc", "n_pair", "base_rate"]).copy()
    sub4b = sub4b[sub4b["reliable"]]
    # 生 Spearman: ens_auc vs attrition_weighted
    if len(sub4b) >= 8:
        raw_r, raw_p = stats.spearmanr(sub4b["ens_auc"], sub4b["attrition_weighted"])
    else:
        raw_r, raw_p = np.nan, np.nan
    # 偏 Spearman (log10 n_pair, base_rate 統制)
    pr, pp, pn = partial_spearman(
        sub4b["ens_auc"].values,
        sub4b["attrition_weighted"].values,
        [np.log10(sub4b["n_pair"].values), sub4b["base_rate"].values],
    )

    # (4c) recall_lift vs attrition_weighted (dir レベル, 偏相関も)
    sub4c = dir_rank.dropna(
        subset=["attrition_weighted", "mean_recall_lift", "n_pair", "base_rate"]
    ).copy()
    if len(sub4c) >= 8:
        rl_raw_r, rl_raw_p = stats.spearmanr(
            sub4c["mean_recall_lift"], sub4c["attrition_weighted"]
        )
        rl_pr, rl_pp, rl_pn = partial_spearman(
            sub4c["mean_recall_lift"].values,
            sub4c["attrition_weighted"].values,
            [np.log10(sub4c["n_pair"].values), sub4c["base_rate"].values],
        )
    else:
        rl_raw_r = rl_raw_p = rl_pr = rl_pp = np.nan
        rl_pn = len(sub4c)

    spearman_summary = {
        "4a_mean_ens_recall_vs_attrition_weighted": {
            "spearman_r": float(sp_r1),
            "p": float(sp_p1),
            "n": int(len(sub4)),
        },
        "4b_ens_auc_vs_attrition_weighted_reliable": {
            "raw_spearman_r": float(raw_r),
            "raw_p": float(raw_p),
            "partial_spearman_r_ctrl_logn_baserate": float(pr),
            "partial_p": float(pp),
            "n": int(pn),
        },
        "4c_recall_lift_vs_attrition_weighted": {
            "raw_spearman_r": float(rl_raw_r),
            "raw_p": float(rl_raw_p),
            "partial_spearman_r_ctrl_logn_baserate": float(rl_pr),
            "partial_p": float(rl_pp),
            "n": int(rl_pn),
        },
    }

    # ============================================================
    # 出力 CSV
    # ============================================================
    danger_out = danger[
        [
            "directory",
            "h_prev",
            "h_curr",
            "n_active_prev",
            "n_attrited",
            "attrition_rate",
            "ens_auc",
            "ens_recall_at_attrited",
            "random_baseline_recall",
            "recall_lift",
        ]
    ].copy()
    danger_path = os.path.join(RES, "dir_exhaustion_danger_cells.csv")
    danger_out.to_csv(danger_path, index=False)

    dir_rank_out = dir_rank[
        [
            "directory",
            "top_dir",
            "n_pair",
            "base_rate",
            "reliable",
            "ens_auc",
            "attrition_weighted",
            "mean_recall_lift",
            "mean_ens_recall",
            "mean_cell_attrition",
            "n_cells",
        ]
    ].copy()
    dir_rank_path = os.path.join(RES, "dir_exhaustion_dir_ranking.csv")
    dir_rank_out.to_csv(dir_rank_path, index=False)

    # ============================================================
    # 5. 図 (左: 危険セル recall lift ソート棒, 右: 予測継続度 vs 実 attrition 散布)
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 危険セル recall lift ソート棒
    ax = axes[0]
    dlab = danger.copy()
    dlab["label"] = dlab["directory"] + " (" + dlab["h_prev"] + "→" + dlab["h_curr"] + ")"
    colors = ["#2c7fb8" if v > 0 else "#d7301f" for v in dlab["recall_lift"]]
    ypos = np.arange(len(dlab))[::-1]
    ax.barh(ypos, dlab["recall_lift"], color=colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels(dlab["label"], fontsize=7)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("recall lift (Ens 脱落者検出率 − ランダム期待値)")
    ax.set_title(
        f"危険セル({len(dlab)}件; 離脱率0.5以上)の脱落者検出 recall lift\n"
        f"正の lift: {danger_summary['n_positive_lift']}/{len(dlab)}件 "
        f"(中央値 {danger_summary['median_recall_lift']:+.3f})",
        fontsize=10,
    )

    # 右: dir予測継続度 (per-dir Ens AUC) vs 実 attrition_weighted
    ax = axes[1]
    s = sub4b
    sizes = 20 + 40 * (np.log10(s["n_pair"]) - np.log10(s["n_pair"]).min())
    ax.scatter(
        s["ens_auc"],
        s["attrition_weighted"],
        s=sizes,
        c="#41ab5d",
        edgecolor="black",
        lw=0.4,
        alpha=0.8,
    )
    # 回帰線 (rank-based ではないが視覚補助)
    if len(s) >= 3:
        z = np.polyfit(s["ens_auc"], s["attrition_weighted"], 1)
        xs = np.linspace(s["ens_auc"].min(), s["ens_auc"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="#737373", ls="--", lw=1.2)
    ax.set_xlabel("per-dir Ens AUC (継続/離脱の順位付け品質)")
    ax.set_ylabel("実 attrition_weighted (実枯渇度)")
    ax.set_title(
        f"予測ランキング品質 vs 実枯渇度 (reliable dir, n={len(s)})\n"
        f"raw ρ={raw_r:+.3f} (p={raw_p:.3f}) / "
        f"偏ρ(規模·base_rate統制)={pr:+.3f} (p={pp:.3f})",
        fontsize=10,
    )
    # 注釈: devstack 構造的失敗例
    if "devstack_6-9_to_9-12" in work_fail_summary:
        ax.annotate(
            "構造的失敗例: devstack 6-9→9-12\n(セル Ens AUC≈%.2f)"
            % work_fail_summary["devstack_6-9_to_9-12"]["ens_auc"],
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=7.5,
            color="#d7301f",
            va="bottom",
        )

    fig.suptitle(
        "ディレクトリ枯渇(レビュアー離脱)検知: 人数ではなく rank-based 品質で担保",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(RES, "fig_dir_exhaustion_detection.pdf")
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ============================================================
    # まとめ出力 (JSON)
    # ============================================================
    summary = {
        "per_dir_auc": auc_summary,
        "danger_cells": danger_summary,
        "all_cells": all_cell_summary,
        "works_vs_fails": work_fail_summary,
        "ranking_validity_spearman": spearman_summary,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n[written]", danger_path)
    print("[written]", dir_rank_path)
    print("[written]", fig_path)


if __name__ == "__main__":
    main()
