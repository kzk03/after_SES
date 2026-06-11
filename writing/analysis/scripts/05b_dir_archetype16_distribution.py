# -*- coding: utf-8 -*-
"""4×4 アーキタイプ (開発者 A1-A4 × ペア P1-P4) セル別予測性分析.

§6 (dir × archetype) を発展させ、開発者archetype とペアarchetype を
独立軸ではなく **16 セルの組合せ** として扱う。

  - per-cell の Ens AUC / IRL AUC / RF AUC / Δ=IRL-RF を heatmap で可視化
  - TOP10 vs BOTTOM10 dir で 16 セル占有率がどう違うかも heatmap で示す

n_pair 制約は基本かけない (4×4=16 セルに対し全 3,677 ペアを pool)。
ただし TOP/BOTTOM dir 占有率比較のみ §6 と同じ n_pair>=30 dir に絞る。

入力:
  data/pair_trajectories.csv         (developer, directory, archetype=A1-A4, predictions)
  data/pair_archetype.csv            (developer, directory, dominant=A1-A4)

出力:
  results/archetype16_cell_stats.csv      4×4 セル別の n_pair, AUC, base rate
  results/fig_archetype16_auc_heatmap.pdf  Ens AUC を heatmap で
  results/fig_archetype16_delta_heatmap.pdf Δ=IRL-RF を heatmap で
  results/fig_archetype16_top_bot_diff.pdf  TOP10-BOTTOM10 dir の 16 セル占有率差
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
MIN_DIR_SIZE = 30
MIN_CELL_N = 20  # AUC 表示のセル最小ペア数

DEV_LABELS = ["A1\n低活動新人", "A2\nレビュー専念", "A3\n高承諾ベテラン", "A4\n常時活動"]
PAIR_LABELS = ["P1\n未経験", "P2\n経験低承諾", "P3\n経験高承諾", "P4\n中程度関与"]
CODES = ["A1", "A2", "A3", "A4"]


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def pooled_metrics(sub: pd.DataFrame) -> dict:
    """sub (pair 行集合) から 4 ホライズン pool で n, base_rate, AUCs を返す."""
    y, ens, irl, rf = [], [], [], []
    for h in HORIZONS:
        mask = ~sub[f"label_{h}"].isna()
        y.extend(sub.loc[mask, f"label_{h}"].astype(int).tolist())
        ens.extend(sub.loc[mask, f"avg_{h}"].tolist())
        irl.extend(sub.loc[mask, f"irl_{h}"].tolist())
        rf.extend(sub.loc[mask, f"rf_{h}"].tolist())
    n_obs = len(y)
    if n_obs == 0:
        return {"n_obs": 0, "base_rate": np.nan,
                "ens_auc": np.nan, "irl_auc": np.nan, "rf_auc": np.nan}
    return {
        "n_obs": n_obs,
        "base_rate": float(np.mean(y)),
        "ens_auc": safe_auc(y, ens),
        "irl_auc": safe_auc(y, irl),
        "rf_auc": safe_auc(y, rf),
    }


def annotate_heatmap(ax, mat: np.ndarray, fmt: str = "{:.2f}",
                     mask: np.ndarray | None = None,
                     ns: np.ndarray | None = None) -> None:
    """セルに値 (と n) を注記."""
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v) or (mask is not None and not mask[i, j]):
                text = "n/a"
            else:
                text = fmt.format(v)
            sub = f"\nn={int(ns[i, j])}" if ns is not None else ""
            color = "white" if abs(v) > 0.85 * np.nanmax(np.abs(mat)) else "black"
            ax.text(j, i, text + sub, ha="center", va="center",
                    fontsize=9, color=color)


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)

    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv")
    pa = pa.rename(columns={"dominant": "pair_archetype"})[
        ["developer", "directory", "pair_archetype"]
    ]
    df = pred.merge(pa, on=["developer", "directory"], how="left")
    df = df.dropna(subset=["archetype", "pair_archetype"]).copy()
    print(f"全ペア (両 archetype あり): {len(df)}")

    # ── (1) 16 セル × 全ペア per-cell 統計 ─────────────────
    n_mat = np.zeros((4, 4), dtype=int)
    base_mat = np.full((4, 4), np.nan)
    ens_mat = np.full((4, 4), np.nan)
    irl_mat = np.full((4, 4), np.nan)
    rf_mat = np.full((4, 4), np.nan)
    delta_mat = np.full((4, 4), np.nan)
    obs_mat = np.zeros((4, 4), dtype=int)

    rows = []
    for i, a in enumerate(CODES):
        for j, p in enumerate(CODES):
            sub = df[(df["archetype"] == a) & (df["pair_archetype"] == p)]
            n_mat[i, j] = len(sub)
            if len(sub) == 0:
                rows.append({"dev_arch": a, "pair_arch": p, "n_pair": 0,
                             "n_obs": 0, "base_rate": np.nan,
                             "ens_auc": np.nan, "irl_auc": np.nan,
                             "rf_auc": np.nan, "delta_irl_rf": np.nan})
                continue
            m = pooled_metrics(sub)
            obs_mat[i, j] = m["n_obs"]
            base_mat[i, j] = m["base_rate"]
            if m["n_obs"] >= MIN_CELL_N and m["ens_auc"] is not None:
                ens_mat[i, j] = m["ens_auc"]
            if m["n_obs"] >= MIN_CELL_N and m["irl_auc"] is not None:
                irl_mat[i, j] = m["irl_auc"]
            if m["n_obs"] >= MIN_CELL_N and m["rf_auc"] is not None:
                rf_mat[i, j] = m["rf_auc"]
            if not (np.isnan(irl_mat[i, j]) or np.isnan(rf_mat[i, j])):
                delta_mat[i, j] = irl_mat[i, j] - rf_mat[i, j]
            rows.append({
                "dev_arch": a, "pair_arch": p,
                "n_pair": n_mat[i, j], "n_obs": m["n_obs"],
                "base_rate": m["base_rate"],
                "ens_auc": m["ens_auc"], "irl_auc": m["irl_auc"],
                "rf_auc": m["rf_auc"],
                "delta_irl_rf": (m["irl_auc"] - m["rf_auc"])
                if m["irl_auc"] is not None and m["rf_auc"] is not None
                else None,
            })

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(OUT / "archetype16_cell_stats.csv", index=False)
    print(f"saved: {OUT/'archetype16_cell_stats.csv'}")
    print(stats_df.to_string(index=False))

    # ── (2) Ens AUC heatmap ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(ens_mat, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    annotate_heatmap(ax, ens_mat, fmt="{:.2f}", ns=n_mat)
    ax.set_xticks(range(4))
    ax.set_xticklabels(PAIR_LABELS, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(DEV_LABELS, fontsize=9)
    ax.set_xlabel("ペア archetype", fontsize=11)
    ax.set_ylabel("開発者 archetype", fontsize=11)
    ax.set_title(f"4×4 セル別 Ens AUC (全 {len(df)} ペア, 4 ホライズン pool)",
                 fontsize=12)
    plt.colorbar(im, ax=ax, label="Ens AUC", shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "fig_archetype16_auc_heatmap.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_archetype16_auc_heatmap.pdf'}")

    # ── (3) Δ=IRL-RF heatmap ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6.5))
    vmax = max(0.05, np.nanmax(np.abs(delta_mat))) if np.any(~np.isnan(delta_mat)) else 0.05
    im = ax.imshow(delta_mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    annotate_heatmap(ax, delta_mat, fmt="{:+.3f}", ns=n_mat)
    ax.set_xticks(range(4))
    ax.set_xticklabels(PAIR_LABELS, fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(DEV_LABELS, fontsize=9)
    ax.set_xlabel("ペア archetype", fontsize=11)
    ax.set_ylabel("開発者 archetype", fontsize=11)
    ax.set_title(f"4×4 セル別 Δ=IRL−RF AUC (正=IRL 優位, 負=RF 優位)",
                 fontsize=12)
    plt.colorbar(im, ax=ax, label="Δ AUC", shrink=0.85)
    fig.tight_layout()
    fig.savefig(OUT / "fig_archetype16_delta_heatmap.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_archetype16_delta_heatmap.pdf'}")

    # ── (4) TOP10 vs BOTTOM10 dir 占有率差 (n_pair>=30 dir 内) ──
    dir_sizes = df.groupby("directory").size()
    keep_dirs = dir_sizes[dir_sizes >= MIN_DIR_SIZE].index
    sub23 = df[df["directory"].isin(keep_dirs)].copy()

    dir_aucs = []
    for d in sub23["directory"].unique():
        s = sub23[sub23["directory"] == d]
        m = pooled_metrics(s)
        if m["ens_auc"] is None:
            continue
        dir_aucs.append({"directory": d, "n_pair": len(s),
                         "ens_auc": m["ens_auc"]})
    dir_auc_df = pd.DataFrame(dir_aucs).sort_values("ens_auc", ascending=False)
    top_dirs = set(dir_auc_df.head(10)["directory"])
    bot_dirs = set(dir_auc_df.tail(10)["directory"])

    def cell_share(sub: pd.DataFrame) -> np.ndarray:
        mat = np.zeros((4, 4), dtype=float)
        for i, a in enumerate(CODES):
            for j, p in enumerate(CODES):
                mat[i, j] = ((sub["archetype"] == a) &
                             (sub["pair_archetype"] == p)).sum()
        return mat / mat.sum() if mat.sum() > 0 else mat

    top_share = cell_share(sub23[sub23["directory"].isin(top_dirs)])
    bot_share = cell_share(sub23[sub23["directory"].isin(bot_dirs)])
    diff_share = top_share - bot_share

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, mat, title, cmap, vlim in [
        (axes[0], top_share, "TOP10 dir (予測しやすい)", "YlOrBr", (0, 0.25)),
        (axes[1], bot_share, "BOTTOM10 dir (予測しにくい)", "YlOrBr", (0, 0.25)),
        (axes[2], diff_share, "TOP − BOTTOM (差)", "RdBu_r", None),
    ]:
        if vlim is None:
            v = max(0.05, np.nanmax(np.abs(mat)))
            im = ax.imshow(mat, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        else:
            im = ax.imshow(mat, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                           aspect="auto")
        annotate_heatmap(ax, mat, fmt="{:+.2f}" if "差" in title else "{:.2f}")
        ax.set_xticks(range(4))
        ax.set_xticklabels(PAIR_LABELS, fontsize=8)
        ax.set_yticks(range(4))
        ax.set_yticklabels(DEV_LABELS, fontsize=8)
        ax.set_xlabel("ペア archetype", fontsize=10)
        ax.set_ylabel("開発者 archetype", fontsize=10)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"16 セル占有率: TOP10 vs BOTTOM10 dir (n_pair≥{MIN_DIR_SIZE}, {len(sub23)} ペア)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_archetype16_top_bot_diff.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_archetype16_top_bot_diff.pdf'}")

    # ── 数値サマリ ───────────────────────────────
    print("\n=== 16 セル AUC (n_obs >= {}) ===".format(MIN_CELL_N))
    print("\nEns AUC:")
    print(pd.DataFrame(ens_mat, index=CODES, columns=CODES).round(3))
    print("\nΔ=IRL-RF:")
    print(pd.DataFrame(delta_mat, index=CODES, columns=CODES).round(3))
    print("\nTOP10 dir share:")
    print(pd.DataFrame(top_share, index=CODES, columns=CODES).round(3))
    print("\nBOTTOM10 dir share:")
    print(pd.DataFrame(bot_share, index=CODES, columns=CODES).round(3))
    print("\nTOP - BOTTOM:")
    print(pd.DataFrame(diff_share, index=CODES, columns=CODES).round(3))


if __name__ == "__main__":
    main()
