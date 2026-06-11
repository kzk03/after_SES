# -*- coding: utf-8 -*-
"""ディレクトリ (repo, depth=2) 単位の予測性 (AUC) を分析.

- 各 dir で IRL/RF/Ensemble の AUC を計算
- dir ごとの top-dir (project, depth=1) と組み合わせて project レベル分析も
- AUC ランキング (予測しやすい/しにくい dir)
- IRL vs RF が dir ごとにどう優劣分かれるか

出力:
  results/dir_auc_ranking.csv: dir × model × AUC
  results/project_auc_summary.csv: project (top-dir) 単位の集計
  results/fig_dir_auc_irl_vs_rf.pdf: dir 別 IRL vs RF AUC 散布図
  results/fig_top_predictable_dirs.pdf: top/bottom 予測性 dir
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
MIN_DIR_SIZE = 30  # 最低 30 ペア持つ dir のみ分析対象 (AUC安定性)


def safe_auc(y_true, y_score) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    print(f"pred shape: {pred.shape}")
    print(f"unique dirs: {pred['directory'].nunique()}")

    pred["project"] = pred["directory"].str.split("/").str[0]
    print(f"unique projects: {pred['project'].nunique()}")

    # ── per-dir AUC ───────────────────────────────────────────
    rows = []
    for d in pred["directory"].unique():
        sub = pred[pred["directory"] == d]
        if len(sub) < MIN_DIR_SIZE:
            continue
        for h in HORIZONS:
            mask = ~sub[f"label_{h}"].isna()
            y_true = sub.loc[mask, f"label_{h}"].astype(int).to_numpy()
            if len(np.unique(y_true)) < 2:
                continue
            irl_auc = safe_auc(y_true, sub.loc[mask, f"irl_{h}"].to_numpy())
            rf_auc = safe_auc(y_true, sub.loc[mask, f"rf_{h}"].to_numpy())
            ens_auc = safe_auc(y_true, sub.loc[mask, f"avg_{h}"].to_numpy())
            rows.append({
                "directory": d,
                "project": d.split("/")[0],
                "horizon": h,
                "n_pairs": int(mask.sum()),
                "pos_rate": float(y_true.mean()),
                "irl_auc": irl_auc,
                "rf_auc": rf_auc,
                "ens_auc": ens_auc,
                "delta_irl_rf": (irl_auc - rf_auc) if (irl_auc and rf_auc) else None,
            })

    dir_df = pd.DataFrame(rows)
    dir_df.to_csv(OUT / "dir_auc_ranking.csv", index=False)
    print(f"saved: {OUT/'dir_auc_ranking.csv'}")
    print(f"per-dir rows: {len(dir_df)} (dirs with >={MIN_DIR_SIZE} pairs)")

    # ── 全ホライズン平均 で dir をランク付け ─────────────
    dir_mean = dir_df.groupby("directory").agg(
        n_pairs=("n_pairs", "mean"),
        ens_auc=("ens_auc", "mean"),
        irl_auc=("irl_auc", "mean"),
        rf_auc=("rf_auc", "mean"),
        delta=("delta_irl_rf", "mean"),
    ).reset_index()
    dir_mean = dir_mean.sort_values("ens_auc", ascending=False)
    dir_mean.to_csv(OUT / "dir_auc_mean.csv", index=False)
    print(f"saved: {OUT/'dir_auc_mean.csv'}")

    # ── project (top-dir) サマリ ───────────────────────
    proj_summary = dir_df.groupby(["project", "horizon"]).agg(
        n_dirs=("directory", "nunique"),
        n_pairs=("n_pairs", "sum"),
        irl_auc_mean=("irl_auc", "mean"),
        rf_auc_mean=("rf_auc", "mean"),
        ens_auc_mean=("ens_auc", "mean"),
        delta_mean=("delta_irl_rf", "mean"),
    ).reset_index()
    proj_summary.to_csv(OUT / "project_auc_summary.csv", index=False)
    print(f"saved: {OUT/'project_auc_summary.csv'}")

    # ── 図1: IRL vs RF 散布図 (各点=dir, 全ホライズン平均) ─────
    fig, ax = plt.subplots(figsize=(7, 6.5))
    sc = ax.scatter(dir_mean["rf_auc"], dir_mean["irl_auc"],
                    s=dir_mean["n_pairs"] * 2, alpha=0.55,
                    c=dir_mean["delta"], cmap="RdBu",
                    vmin=-0.15, vmax=0.15,
                    edgecolors="black", linewidths=0.4)
    lims = (0.3, 1.0)
    ax.plot(lims, lims, "k--", lw=1, alpha=0.6)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_xlabel("RF AUC (per-directory, 4区間平均)", fontsize=12)
    ax.set_ylabel("IRL AUC (per-directory, 4区間平均)", fontsize=12)
    ax.set_title(
        f"ディレクトリ別 IRL vs RF AUC (n_dirs={len(dir_mean)}, "
        f"min {MIN_DIR_SIZE} pairs)", fontsize=12)
    plt.colorbar(sc, label="IRL − RF (Δ)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_dir_auc_irl_vs_rf.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_dir_auc_irl_vs_rf.pdf'}")

    # ── 図2: top/bottom 予測性 dir ─────────────────────
    top_n = 10
    top_dirs = dir_mean.head(top_n)[::-1]
    bot_dirs = dir_mean.tail(top_n)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    y1 = np.arange(len(top_dirs))
    ax1.barh(y1, top_dirs["ens_auc"], color="#1f77b4", alpha=0.85)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(top_dirs["directory"], fontsize=10)
    ax1.set_xlim(0.5, 1.0)
    ax1.set_title(f"予測しやすい dir TOP{top_n}", fontsize=12)
    ax1.set_xlabel("Ens. AUC (4区間平均)", fontsize=11)
    ax1.grid(axis="x", alpha=0.3)

    y2 = np.arange(len(bot_dirs))
    ax2.barh(y2, bot_dirs["ens_auc"], color="#d62728", alpha=0.85)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(bot_dirs["directory"], fontsize=10)
    ax2.set_xlim(0.3, 1.0)
    ax2.set_title(f"予測しにくい dir BOTTOM{top_n}", fontsize=12)
    ax2.set_xlabel("Ens. AUC (4区間平均)", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_top_predictable_dirs.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_top_predictable_dirs.pdf'}")

    # ── 主結果サマリ ───────────────────────────────────
    print("\n=== Per-dir AUC ranking (top 10) ===")
    print(dir_mean.head(10)[
        ["directory", "n_pairs", "ens_auc", "irl_auc", "rf_auc", "delta"]
    ].to_string(index=False))
    print("\n=== Per-dir AUC ranking (bottom 10) ===")
    print(dir_mean.tail(10)[
        ["directory", "n_pairs", "ens_auc", "irl_auc", "rf_auc", "delta"]
    ].to_string(index=False))
    print(f"\nIRL > RF の dir 数: {(dir_mean['delta'] > 0).sum()} / {len(dir_mean)}")
    print(f"RF > IRL の dir 数: {(dir_mean['delta'] < 0).sum()} / {len(dir_mean)}")


if __name__ == "__main__":
    main()
