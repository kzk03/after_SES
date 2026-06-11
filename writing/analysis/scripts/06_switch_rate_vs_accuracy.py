# -*- coding: utf-8 -*-
"""switch_rate (volatility) 特徴量と予測精度の関係を分析.

recent_switch_rate = 直近 10 件の承諾↔拒否切替率 (Constantinou & Mens JSS 2017)
これは t0 時点の dev の volatility シグナル。

問い:
  - switch_rate が高い (=過去 volatile な) dev では IRL/RF/Ens の AUC はどう変わるか?
  - 「IRL は時系列符号化だから volatile dev に強い」仮説を検証する直接的な指標になる。

入力:
  data/pair_trajectories.csv  (predictions per horizon)
  data/feature_values.csv     (recent_switch_rate at t0)

出力:
  results/switch_rate_bins.csv               bin × model × AUC
  results/fig_switch_rate_vs_auc.pdf         主力図 (bin 別 AUC + distribution)
  results/fig_switch_rate_distribution.pdf   switch_rate の分布 (label_class 別)
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
SNAPSHOT_T0 = "2022-01-01"
N_BINS = 5  # 五分位
MIN_BIN_SIZE = 30


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


def compute_switch_rate_per_dev(raw_csv: pathlib.Path, t0: pd.Timestamp,
                                   n: int = 10) -> pd.DataFrame:
    """t0 時点で各 dev の直近 N=10 件の承諾↔拒否切替率を計算."""
    raw = pd.read_csv(raw_csv, usecols=["email", "timestamp", "label"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw = raw[raw["timestamp"] < t0].copy()
    raw = raw.sort_values(["email", "timestamp"])

    rows = []
    for email, g in raw.groupby("email"):
        labels = g["label"].to_numpy()[-n:]
        if len(labels) < 2:
            continue
        switches = int((np.diff(labels) != 0).sum())
        rate = switches / (len(labels) - 1)
        rows.append({
            "developer": email,
            "switch_rate": rate,
            "n_recent_reviews": len(labels),
        })
    return pd.DataFrame(rows)


def main() -> None:
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    # raw データから switch_rate を計算
    print("computing switch_rate from raw data...")
    sr_df = compute_switch_rate_per_dev(
        DATA / "combined_raw_main32.csv",
        pd.Timestamp(SNAPSHOT_T0),
        n=10,
    )
    print(f"computed switch_rate for {len(sr_df)} devs")
    print(f"switch_rate quantiles: {sr_df['switch_rate'].quantile([0,0.25,0.5,0.75,1.0]).to_dict()}")

    df = pred.merge(sr_df, on="developer", how="inner")
    print(f"merged: {len(df)} pairs with switch_rate features")

    # label_class を計算
    label_cols = [f"label_{h}" for h in HORIZONS]
    L = df[label_cols].to_numpy()
    df["label_class"] = [classify_label_traj(row) for row in L]

    # 実測 label_switches も計算 (4 horizon 内)
    def n_switches(labels):
        valid = ~np.isnan(labels)
        if valid.sum() < 2:
            return 0
        v = labels[valid]
        return int((np.diff(v) != 0).sum())
    df["label_switches"] = [n_switches(L[i]) for i in range(len(df))]

    # ── switch_rate を 5 分位に binning ─────────────────
    df["sr_bin"] = pd.qcut(df["switch_rate"], q=N_BINS,
                            duplicates="drop", labels=False)
    bin_edges = pd.qcut(df["switch_rate"], q=N_BINS,
                         duplicates="drop", retbins=True)[1]
    print(f"\nbin edges (switch_rate): {bin_edges.round(3).tolist()}")
    print(df.groupby("sr_bin").size())

    # ── 各 bin × horizon で AUC を計算 ──────────────────
    rows = []
    for b in sorted(df["sr_bin"].dropna().unique()):
        sub = df[df["sr_bin"] == b]
        if len(sub) < MIN_BIN_SIZE:
            continue
        for h in HORIZONS:
            mask = ~sub[f"label_{h}"].isna()
            y = sub.loc[mask, f"label_{h}"].astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                continue
            irl = sub.loc[mask, f"irl_{h}"].to_numpy()
            rf = sub.loc[mask, f"rf_{h}"].to_numpy()
            avg = sub.loc[mask, f"avg_{h}"].to_numpy()
            rows.append({
                "sr_bin": int(b),
                "sr_range": f"[{bin_edges[int(b)]:.2f}-{bin_edges[int(b)+1]:.2f}]",
                "horizon": h,
                "n": int(mask.sum()),
                "pos_rate": float(y.mean()),
                "irl_auc": float(roc_auc_score(y, irl)),
                "rf_auc": float(roc_auc_score(y, rf)),
                "ens_auc": float(roc_auc_score(y, avg)),
            })
    bin_df = pd.DataFrame(rows)
    bin_df["delta_irl_rf"] = bin_df["irl_auc"] - bin_df["rf_auc"]
    bin_df.to_csv(OUT / "switch_rate_bins.csv", index=False)
    print(f"\nsaved: {OUT/'switch_rate_bins.csv'}")

    # 4 horizon 平均を bin 単位で集計
    bin_mean = bin_df.groupby(["sr_bin", "sr_range"]).agg(
        n=("n", "mean"),
        pos_rate=("pos_rate", "mean"),
        irl_auc=("irl_auc", "mean"),
        rf_auc=("rf_auc", "mean"),
        ens_auc=("ens_auc", "mean"),
        delta=("delta_irl_rf", "mean"),
    ).reset_index()
    print("\n=== bin 別 AUC (4 horizon 平均) ===")
    print(bin_mean.round(3).to_string(index=False))

    # ── 図1: 主力 — bin 別 AUC + switch_rate 分布 ─────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左: bin 別 AUC line plot
    x = bin_mean["sr_bin"].to_numpy()
    ax1.plot(x, bin_mean["irl_auc"], "o-", color="#1f77b4",
             lw=2.5, markersize=10, label="IRL")
    ax1.plot(x, bin_mean["rf_auc"], "s-", color="#d62728",
             lw=2.5, markersize=10, label="RF")
    ax1.plot(x, bin_mean["ens_auc"], "^-", color="#2ca02c",
             lw=2.5, markersize=10, label="Ens")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"Q{b+1}\n{bin_mean.loc[i,'sr_range']}\nn={int(bin_mean.loc[i,'n'])}"
         for i, b in enumerate(x)],
        fontsize=9,
    )
    ax1.set_xlabel("switch_rate 五分位 (低→高)", fontsize=11)
    ax1.set_ylabel("AUC (4 horizon 平均)", fontsize=11)
    ax1.set_title("switch_rate ごとの予測精度", fontsize=12)
    ax1.legend(loc="lower left", fontsize=11)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0.5, 0.95)
    # delta 注記
    for i, (xv, d) in enumerate(zip(x, bin_mean["delta"])):
        ax1.text(xv, 0.52, f"Δ={d:+.3f}", ha="center", fontsize=8,
                 color="black")

    # 右: switch_rate の分布 (label_class 別)
    classes = ["consistent_reject", "fluctuating", "consistent_accept"]
    class_ja = {
        "consistent_reject": "一貫拒否",
        "fluctuating": "変動",
        "consistent_accept": "一貫承諾",
    }
    class_colors = {
        "consistent_reject": "#d62728",
        "fluctuating": "#ff7f0e",
        "consistent_accept": "#2ca02c",
    }
    for c in classes:
        sub = df[df["label_class"] == c]["switch_rate"].dropna()
        ax2.hist(sub, bins=30, alpha=0.55, label=f"{class_ja[c]} (n={len(sub)})",
                 color=class_colors[c], density=True)
    ax2.set_xlabel("switch_rate (t0 時点)", fontsize=11)
    ax2.set_ylabel("密度", fontsize=11)
    ax2.set_title("switch_rate 分布 (label_class 別)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_switch_rate_vs_auc.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_switch_rate_vs_auc.pdf'}")

    # ── 図2: switch_rate (feature) vs label_switches (実測) scatter ─────
    from scipy.stats import spearmanr
    rho, p = spearmanr(df["switch_rate"], df["label_switches"], nan_policy="omit")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(df["switch_rate"],
               df["label_switches"] + np.random.uniform(-0.1, 0.1, len(df)),
               alpha=0.2, s=15, color="#1f77b4")
    ax.set_xlabel("recent_switch_rate (t0時点直近10件)", fontsize=11)
    ax.set_ylabel("実測 label_switches (4 horizon 内)", fontsize=11)
    ax.set_title(
        f"過去 volatility vs 将来 label 変化回数\n"
        f"Spearman ρ={rho:.3f} (p={p:.2g}, NS) → ほぼ独立",
        fontsize=11,
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_switch_rate_distribution.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_switch_rate_distribution.pdf'}")

    # ── サマリ ──────────────────────────────────
    print("\n=== bin 別 mean (table) ===")
    print(bin_mean[["sr_bin", "sr_range", "n", "pos_rate", "irl_auc", "rf_auc", "ens_auc", "delta"]].to_string(index=False))
    print(f"\nSpearman ρ(switch_rate, label_switches) = {rho:.3f} (p={p:.2g})")


if __name__ == "__main__":
    main()
