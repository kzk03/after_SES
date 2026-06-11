# -*- coding: utf-8 -*-
"""バイオリズム検証 (dev レベル版, max 集約): 論文 per-dev AUC と整合.

論文 archetype_irl_rf.py の compute_per_dev_metrics に合わせて、
各 (dev, horizon) で max 集約 (any-positive):

  label_dev[h] = max(label_pair[h] for pair in dev's pairs)  # 0 or 1
  pred_dev[h]  = max(pred_pair[h] for pair in dev's pairs)   # 0-1 連続値

これにより:
  - label_dev[h] = 1: その horizon で dev のどれかのペアが承諾
  - label_dev[h] = 0: その horizon で dev の全ペアが拒否 (or 依頼なし)

biorhythm 解釈: 「dev が h で active か inactive か」の振動。

入力: data/pair_trajectories.csv
出力:
  results/biorhythm_dev_level.csv
  results/fig_biorhythm_dev_traj.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, binomtest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def main() -> None:
    pair = pd.read_csv(DATA / "pair_trajectories.csv")

    label_cols = [f"label_{h}" for h in HORIZONS]
    irl_cols = [f"irl_{h}" for h in HORIZONS]
    rf_cols = [f"rf_{h}" for h in HORIZONS]
    avg_cols = [f"avg_{h}" for h in HORIZONS]

    # dev 単位に集約 (論文と同じ max 集約 = any-positive)
    agg_cols = {c: "max" for c in label_cols + irl_cols + rf_cols + avg_cols}
    agg_cols["directory"] = "count"
    dev = pair.groupby("developer").agg(agg_cols).reset_index()
    dev = dev.rename(columns={"directory": "n_pairs"})
    print(f"unique devs: {len(dev)}")
    print(f"n_pairs per dev: median={dev['n_pairs'].median():.0f}, max={dev['n_pairs'].max()}")

    # ── 連続値軌跡相関 (Spearman) ──────────────
    def safe_spearman(p, l):
        mask = ~(np.isnan(p) | np.isnan(l))
        if mask.sum() < 3:
            return np.nan
        a, b = p[mask], l[mask]
        if np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        rho, _ = spearmanr(a, b)
        return float(rho)

    L = dev[label_cols].to_numpy()
    IRL = dev[irl_cols].to_numpy()
    RF = dev[rf_cols].to_numpy()
    AVG = dev[avg_cols].to_numpy()

    dev["sp_irl"] = [safe_spearman(IRL[i], L[i]) for i in range(len(dev))]
    dev["sp_rf"] = [safe_spearman(RF[i], L[i]) for i in range(len(dev))]
    dev["sp_avg"] = [safe_spearman(AVG[i], L[i]) for i in range(len(dev))]

    # dev レベル label の変動 (std across horizons)
    dev["label_std"] = np.nanstd(L, axis=1)
    dev["label_range"] = np.nanmax(L, axis=1) - np.nanmin(L, axis=1)

    # max 集約後の label は既に 0/1。NaN は保持
    L_bin_nan = L  # 既に 0/1 (NaN は NaN のまま)

    # スイッチ回数 (二値化軌跡)
    def n_switches(labels):
        valid = ~np.isnan(labels)
        if valid.sum() < 2:
            return 0
        v = labels[valid]
        return int((np.diff(v) != 0).sum())
    dev["bin_switches"] = [n_switches(L_bin_nan[i]) for i in range(len(dev))]

    # 二値化軌跡 → label_class
    def classify(labels):
        valid = ~np.isnan(labels)
        if valid.sum() < 2:
            return "insufficient"
        v = labels[valid]
        if (v == 1).all():
            return "consistent_accept"
        if (v == 0).all():
            return "consistent_reject"
        return "fluctuating"
    dev["label_class"] = [classify(L_bin_nan[i]) for i in range(len(dev))]

    # 方向一致率 (二値化版)
    def direction_agreement(label, pred):
        mask = ~(np.isnan(label) | np.isnan(pred))
        if mask.sum() < 2:
            return 0, 0, np.nan
        l = label[mask]; p = pred[mask]
        n_sw, n_mt = 0, 0
        for i in range(len(l) - 1):
            d_l = l[i + 1] - l[i]
            if d_l == 0:
                continue
            n_sw += 1
            d_p = p[i + 1] - p[i]
            if (d_l > 0 and d_p > 0) or (d_l < 0 and d_p < 0):
                n_mt += 1
        if n_sw == 0:
            return 0, 0, np.nan
        return n_sw, n_mt, n_mt / n_sw

    # 連続値版方向一致率 (label の連続値 trajectory に対して)
    for name, M in [("irl", IRL), ("rf", RF), ("avg", AVG)]:
        ns, nm, rates = [], [], []
        for i in range(len(dev)):
            s, m, r = direction_agreement(L[i], M[i])
            ns.append(s); nm.append(m); rates.append(r)
        dev[f"{name}_n_sw"] = ns
        dev[f"{name}_n_mt"] = nm
        dev[f"{name}_dir_agree"] = rates

    # クラス分布 (max 集約後 0/1)
    print("\n=== dev レベル label_class 分布 (max 集約, any-positive) ===")
    print(dev["label_class"].value_counts())

    # fluctuating dev
    flu = dev[dev["label_class"] == "fluctuating"]
    print(f"\n=== fluctuating dev (n={len(flu)}) のスイッチ回数 ===")
    print(flu["bin_switches"].value_counts().sort_index())

    # 連続値 Spearman 軌跡相関
    print("\n=== dev 全体: 連続値軌跡 Spearman 相関 ===")
    print(dev[["sp_irl", "sp_rf", "sp_avg"]].describe(percentiles=[0.25, 0.5, 0.75]).round(3))

    # 方向一致率 (fluctuating のみ)
    print("\n=== fluctuating dev の方向一致率 (連続値 label 変化方向) ===")
    print(flu[["irl_dir_agree", "rf_dir_agree", "avg_dir_agree"]].describe(percentiles=[0.5]).round(3))

    # pooled 方向一致率
    print("\n=== pooled 方向一致率 (連続値 label 変化方向) ===")
    pooled = {}
    for name in ["irl", "rf", "avg"]:
        sw = flu[f"{name}_n_sw"].sum()
        mt = flu[f"{name}_n_mt"].sum()
        rate = mt / sw if sw > 0 else np.nan
        pooled[name] = (int(sw), int(mt), rate)
        # 二項検定 vs chance 0.5
        if sw > 0:
            res = binomtest(int(mt), int(sw), p=0.5, alternative="two-sided")
            pval = res.pvalue
        else:
            pval = np.nan
        print(f"  {name}: sw={sw}, mt={mt}, rate={rate:.3f}, "
              f"vs chance 0.5: p={pval:.3g}")

    # McNemar 検定: IRL vs RF
    # 各 dev × switch transition で IRL_match, RF_match を比較
    # 集約: IRL のみ正解, RF のみ正解, 両方正解, 両方誤りの 4 区分
    print("\n=== McNemar 検定 (IRL vs RF, 全 switch transition pooled) ===")
    a_only_irl = 0  # IRL match & RF mismatch
    a_only_rf = 0   # RF match & IRL mismatch
    a_both = 0
    a_neither = 0
    for i in range(len(flu)):
        l = flu.iloc[i][label_cols].to_numpy(dtype=float)
        irl = flu.iloc[i][irl_cols].to_numpy(dtype=float)
        rf = flu.iloc[i][rf_cols].to_numpy(dtype=float)
        mask = ~(np.isnan(l) | np.isnan(irl) | np.isnan(rf))
        l, irl, rf = l[mask], irl[mask], rf[mask]
        for j in range(len(l) - 1):
            dl = l[j + 1] - l[j]
            if dl == 0:
                continue
            di = irl[j + 1] - irl[j]
            dr = rf[j + 1] - rf[j]
            irl_ok = (dl > 0 and di > 0) or (dl < 0 and di < 0)
            rf_ok = (dl > 0 and dr > 0) or (dl < 0 and dr < 0)
            if irl_ok and rf_ok:
                a_both += 1
            elif irl_ok and not rf_ok:
                a_only_irl += 1
            elif rf_ok and not irl_ok:
                a_only_rf += 1
            else:
                a_neither += 1
    print(f"  IRL only correct: {a_only_irl}")
    print(f"  RF  only correct: {a_only_rf}")
    print(f"  Both correct:     {a_both}")
    print(f"  Both wrong:       {a_neither}")
    # McNemar test on discordant pairs
    if a_only_irl + a_only_rf > 0:
        from scipy.stats import chi2
        chi2_stat = (abs(a_only_irl - a_only_rf) - 1) ** 2 / (a_only_irl + a_only_rf)
        pval = 1 - chi2.cdf(chi2_stat, df=1)
        print(f"  McNemar χ² = {chi2_stat:.3f}, p = {pval:.3g}")

    # 保存
    dev.to_csv(OUT / "biorhythm_dev_level.csv", index=False)
    print(f"\nsaved: {OUT/'biorhythm_dev_level.csv'}")

    # ── 図1: dev レベル軌跡相関の boxplot + Spearman 連続値版 ─────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 左: 連続値 Spearman 全 dev
    ax = axes[0]
    data = [dev["sp_irl"].dropna(), dev["sp_rf"].dropna(), dev["sp_avg"].dropna()]
    names = ["IRL", "RF", "Ens"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2), widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    means = [d.mean() for d in data]
    ax.scatter(range(1, 4), means, marker="D", color="white",
               edgecolor="black", s=90, zorder=10, label="mean")
    for i, m in enumerate(means):
        ax.text(i + 1.15, m, f"{m:.2f}", fontsize=10, va="center")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Spearman 軌跡相関 (連続値)", fontsize=11)
    ax.set_title(f"dev レベル 連続値軌跡相関 (n={len(dev)})", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # 右: fluctuating dev の方向一致率
    ax = axes[1]
    flu_data = [flu["irl_dir_agree"].dropna(), flu["rf_dir_agree"].dropna(),
                flu["avg_dir_agree"].dropna()]
    bp = ax.boxplot(flu_data, tick_labels=names, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", linewidth=2), widths=0.55)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    flu_means = [d.mean() for d in flu_data]
    ax.scatter(range(1, 4), flu_means, marker="D", color="white",
               edgecolor="black", s=90, zorder=10, label="mean")
    for i, m in enumerate(flu_means):
        ax.text(i + 1.15, m, f"{m:.2f}", fontsize=10, va="center")
    ax.axhline(0.5, color="gray", lw=0.7, ls=":", label="chance (0.5)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("方向一致率", fontsize=11)
    ax.set_title(
        f"dev レベル 方向一致率 (fluctuating dev n={len(flu)})",
        fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("バイオリズム検証 (dev レベル, ペアを集約)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_biorhythm_dev_traj.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_biorhythm_dev_traj.pdf'}")


if __name__ == "__main__":
    main()
