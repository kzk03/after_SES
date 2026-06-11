# -*- coding: utf-8 -*-
"""バイオリズム検証 (v3, Direction Agreement): horizon 間 label 切替を予測が方向で当てられているか.

ペアの label 軌跡 [label_0-3m, label_3-6m, label_6-9m, label_9-12m] を見て、
- consistent_accept: 全 4 区間で 1
- consistent_reject: 全 4 区間で 0
- fluctuating: 0 と 1 が混在 (「バイオリズム」ある reviewer)

fluctuating ペアについて、連続 horizon 間で label が切り替わった場合に、
予測も同じ方向に動いたか (方向一致) を判定する。

方向一致率 = (label 切替で予測も同方向の遷移数) / (label 切替の総数)

入力:
  data/pair_trajectories.csv

出力:
  results/biorhythm_label_class.csv             pair × class + 方向一致率
  results/biorhythm_traj_correlation.csv        class × model 別 直接比較
  results/fig_biorhythm_direction_agreement.pdf 主力図 (方向一致率の boxplot + bar)
  results/fig_biorhythm_label_examples.pdf      fluctuating ペア軌跡の例
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
EPS = 1e-9  # 予測差の閾値 (絶対値これ以下は「動かない」とみなす)


def classify_label_traj(labels: np.ndarray) -> str:
    valid = ~np.isnan(labels)
    if valid.sum() < 2:
        return "insufficient"
    vals = labels[valid]
    if (vals == 1).all():
        return "consistent_accept"
    if (vals == 0).all():
        return "consistent_reject"
    return "fluctuating"


def direction_agreement(label: np.ndarray, pred: np.ndarray) -> tuple:
    """各連続 horizon ペアで label 切替の方向と予測の方向が一致するかを判定.

    Returns: (n_switches, n_match, agreement_rate)
      n_switches: label が変化した連続 pair の総数
      n_match: 予測も同方向に動いた数
      agreement_rate: n_match / n_switches (n_switches=0 なら NaN)
    """
    mask = ~(np.isnan(label) | np.isnan(pred))
    if mask.sum() < 2:
        return 0, 0, np.nan
    l = label[mask]
    p = pred[mask]
    n_switches = 0
    n_match = 0
    for i in range(len(l) - 1):
        d_label = l[i + 1] - l[i]
        if d_label == 0:
            continue
        n_switches += 1
        d_pred = p[i + 1] - p[i]
        if abs(d_pred) < EPS:
            continue  # 予測動かない場合は match にカウントしない
        if (d_label > 0 and d_pred > 0) or (d_label < 0 and d_pred < 0):
            n_match += 1
    if n_switches == 0:
        return 0, 0, np.nan
    return n_switches, n_match, n_match / n_switches


def main() -> None:
    df = pd.read_csv(DATA / "pair_trajectories.csv")

    label_cols = [f"label_{h}" for h in HORIZONS]
    irl_cols = [f"irl_{h}" for h in HORIZONS]
    rf_cols = [f"rf_{h}" for h in HORIZONS]
    avg_cols = [f"avg_{h}" for h in HORIZONS]

    L = df[label_cols].to_numpy()
    IRL = df[irl_cols].to_numpy()
    RF = df[rf_cols].to_numpy()
    AVG = df[avg_cols].to_numpy()

    df["label_class"] = [classify_label_traj(row) for row in L]

    # スイッチ回数
    def n_switches_for_label(labels):
        valid = ~np.isnan(labels)
        if valid.sum() < 2:
            return 0
        v = labels[valid]
        return int((np.diff(v) != 0).sum())

    df["label_switches"] = [n_switches_for_label(L[i]) for i in range(len(df))]

    # 方向一致率を 3 モデル分計算
    for name, M in [("irl", IRL), ("rf", RF), ("avg", AVG)]:
        ns, nm, rates = [], [], []
        for i in range(len(df)):
            s, m, r = direction_agreement(L[i], M[i])
            ns.append(s)
            nm.append(m)
            rates.append(r)
        df[f"{name}_n_switches"] = ns
        df[f"{name}_n_match"] = nm
        df[f"{name}_dir_agree"] = rates

    # クラス分布
    cls = df["label_class"].value_counts()
    print("=== label_class 分布 ===")
    print(cls)
    print()

    # スイッチ回数別の分布 (fluctuating のみ)
    flu = df[df["label_class"] == "fluctuating"]
    print(f"\n=== fluctuating ペアのスイッチ回数 (n={len(flu)}) ===")
    print(flu["label_switches"].value_counts().sort_index())

    # 方向一致率 (fluctuating)
    print("\n=== fluctuating ペアの方向一致率 ===")
    metrics = ["irl_dir_agree", "rf_dir_agree", "avg_dir_agree"]
    print(flu[metrics].agg(["mean", "median", "count"]).round(3))

    # スイッチ回数別の方向一致率
    print("\n=== スイッチ回数別 方向一致率 平均 (fluctuating のみ) ===")
    sw = flu.groupby("label_switches")[metrics].agg(["mean", "count"]).round(3)
    print(sw.to_string())

    # 全体の集計 (pooled の n_match / n_switches)
    print("\n=== Pooled 方向一致率 (全 fluctuating ペア合算) ===")
    pooled = {}
    for name in ["irl", "rf", "avg"]:
        total_sw = flu[f"{name}_n_switches"].sum()
        total_mt = flu[f"{name}_n_match"].sum()
        rate = total_mt / total_sw if total_sw > 0 else np.nan
        pooled[name] = (int(total_sw), int(total_mt), rate)
        print(f"  {name}: switches={total_sw}, matches={total_mt}, rate={rate:.3f}")

    df.to_csv(OUT / "biorhythm_label_class.csv", index=False)
    print(f"\nsaved: {OUT/'biorhythm_label_class.csv'}")

    # 集計表
    summary_rows = [
        {
            "model": name.upper(),
            "n_pairs_fluctuating": int(len(flu)),
            "pooled_switches": pooled[name][0],
            "pooled_matches": pooled[name][1],
            "pooled_rate": pooled[name][2],
            "per_pair_mean_rate": flu[f"{name}_dir_agree"].mean(),
            "per_pair_median_rate": flu[f"{name}_dir_agree"].median(),
        }
        for name in ["irl", "rf", "avg"]
    ]
    pd.DataFrame(summary_rows).to_csv(
        OUT / "biorhythm_traj_correlation.csv", index=False
    )
    print(f"saved: {OUT/'biorhythm_traj_correlation.csv'}")

    # ── 図1: 主力 — pooled 方向一致率 bar + per-pair distribution ────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # 左: pooled rate bar (per switch count)
    names = ["IRL", "RF", "Ens"]
    keys = ["irl", "rf", "avg"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    # スイッチ回数別 pooled
    switches = sorted(flu["label_switches"].unique())
    width = 0.25
    x = np.arange(len(switches))
    for i, (key, name, color) in enumerate(zip(keys, names, colors)):
        rates = []
        for sw_cnt in switches:
            sub = flu[flu["label_switches"] == sw_cnt]
            t_sw = sub[f"{key}_n_switches"].sum()
            t_mt = sub[f"{key}_n_match"].sum()
            rates.append(t_mt / t_sw if t_sw > 0 else 0)
        ax1.bar(x + (i - 1) * width, rates, width * 0.9, color=color,
                alpha=0.85, label=name)
        for j, v in enumerate(rates):
            ax1.text(j + (i - 1) * width, v + 0.01, f"{v:.2f}",
                     ha="center", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{sw}回\n(n={(flu['label_switches']==sw).sum()})"
                          for sw in switches])
    ax1.set_xlabel("label スイッチ回数 (per pair)", fontsize=11)
    ax1.set_ylabel("方向一致率 (pooled)", fontsize=11)
    ax1.set_title("スイッチ回数別 方向一致率", fontsize=12)
    ax1.axhline(0.5, color="gray", lw=0.7, ls=":", label="ランダム (0.5)")
    ax1.set_ylim(0, 1.0)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)

    # 右: per-pair direction agreement の boxplot
    data = [flu[f"{k}_dir_agree"].dropna().to_numpy() for k in keys]
    bp = ax2.boxplot(data, labels=names, patch_artist=True, showfliers=False,
                      medianprops=dict(color="black", linewidth=2),
                      widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    # 平均値を赤点で重ねる
    means = [flu[f"{k}_dir_agree"].mean() for k in keys]
    ax2.scatter(np.arange(1, 4), means, marker="D", color="white",
                edgecolor="black", s=80, zorder=10, label="mean")
    for i, m in enumerate(means):
        ax2.text(i + 1.15, m, f"{m:.2f}", fontsize=10, va="center")
    ax2.set_ylabel("方向一致率 (per pair)", fontsize=11)
    ax2.set_title("方向一致率の分布 (fluctuating ペア)", fontsize=12)
    ax2.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("バイオリズム検証: horizon 間 label 切替を予測が方向で当てたか",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_biorhythm_direction_agreement.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_biorhythm_direction_agreement.pdf'}")

    # ── 図2: 軌跡例 ──
    # IRL の方向一致率 - RF のそれ で差が大きいペアを抜き出す
    flu2 = flu.dropna(subset=["irl_dir_agree", "rf_dir_agree"]).copy()
    flu2["delta_dir"] = flu2["irl_dir_agree"] - flu2["rf_dir_agree"]
    top_irl = flu2.nlargest(6, "delta_dir")
    top_rf = flu2.nsmallest(6, "delta_dir")
    examples = pd.concat([top_irl, top_rf])

    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, (_, r) in zip(axes.flat, examples.iterrows()):
        labels = [r[c] for c in label_cols]
        irl = [r[c] for c in irl_cols]
        rf = [r[c] for c in rf_cols]
        ax.plot(HORIZONS, labels, "o-", color="black", lw=2, label="label",
                markersize=8)
        ax.plot(HORIZONS, irl, "s--", color="#1f77b4", lw=1.5, label="IRL",
                markersize=7, alpha=0.85)
        ax.plot(HORIZONS, rf, "^--", color="#d62728", lw=1.5, label="RF",
                markersize=7, alpha=0.85)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"{str(r['developer'])[:20]}\n{str(r['directory'])[:30]}\n"
            f"IRL_dir={r['irl_dir_agree']:.2f}, RF_dir={r['rf_dir_agree']:.2f}",
            fontsize=9,
        )
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=True)
    fig.suptitle(
        "fluctuating ペアの軌跡例 (上6: IRL>RF 方向一致, 下6: RF>IRL)",
        fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "fig_biorhythm_label_examples.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_biorhythm_label_examples.pdf'}")


if __name__ == "__main__":
    main()
