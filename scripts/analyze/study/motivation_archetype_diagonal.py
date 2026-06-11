# -*- coding: utf-8 -*-
"""Motivation figure: archetype × horizon × method の AUC ヒートマップ.

stacking が使う対角 4 セル (train_h, eval_h) と同じ予測を用いて
IRL_diag / RF_diag / Uniform / Stack の archetype 別 AUC を比較し、
stacking が必要な failure regime を可視化する。

入力:
  - per_cell_archetype_auc.csv (B-36, 対角行を抽出)
  - per_archetype.csv (B-37 stacking)

出力:
  - fig_motivation_archetype_diagonal.pdf (4-panel heatmap)
  - fig_motivation_archetype_gain.pdf (stack - best_single の gain heatmap)
  - motivation_archetype_diagonal.csv (マージ済テーブル)
"""
from __future__ import annotations

import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHETYPES = ["A1", "A2", "A3", "A4"]
ARCH_LABELS = {
    "A1": "A1: Sporadic\nNewcomer",
    "A2": "A2: Strict\nReviewer-Only",
    "A3": "A3: Approving\nVeteran",
    "A4": "A4: Saturated\nActive",
}
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_diagonal_cells(per_cell_csv: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(per_cell_csv)
    df["train_h"] = df["train"].str.replace("train_", "", regex=False)
    df["eval_h"] = df["eval"].str.replace("eval_", "", regex=False)
    diag = df[df["train_h"] == df["eval_h"]].copy()
    diag = diag.rename(columns={"eval_h": "horizon"})
    return diag[["horizon", "archetype", "n_pairs", "n_pos",
                 "auc_irl", "auc_rf", "auc_avg"]]


def load_stack(per_archetype_csv: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(per_archetype_csv)
    return df.rename(columns={"target": "horizon"})[
        ["horizon", "archetype", "auc_stack"]
    ]


def make_matrix(df: pd.DataFrame, value_col: str) -> np.ndarray:
    mat = np.zeros((len(ARCHETYPES), len(HORIZONS)))
    for i, a in enumerate(ARCHETYPES):
        for j, h in enumerate(HORIZONS):
            row = df[(df["archetype"] == a) & (df["horizon"] == h)]
            mat[i, j] = row[value_col].iloc[0] if len(row) else np.nan
    return mat


def plot_heatmap(ax, mat, title, vmin=0.5, vmax=0.9, cmap="RdYlGn",
                 annotate=True, annot_fmt="{:.2f}"):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS, fontsize=9)
    ax.set_yticks(range(len(ARCHETYPES)))
    ax.set_yticklabels([ARCH_LABELS[a] for a in ARCHETYPES], fontsize=8)
    ax.set_title(title, fontsize=10)
    if annotate:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "black" if 0.6 <= v <= 0.8 else "white"
                ax.text(j, i, annot_fmt.format(v), ha="center", va="center",
                        fontsize=8, color=color)
    return im


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-cell",
                   default="outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/per_cell_archetype_auc.csv",
                   type=pathlib.Path)
    p.add_argument("--per-arch-stack",
                   default="outputs/analysis/multi_horizon_stacking_volat_ep150/per_archetype.csv",
                   type=pathlib.Path)
    p.add_argument("--out-dir",
                   default="outputs/analysis/multi_horizon_stacking_volat_ep150",
                   type=pathlib.Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    diag = load_diagonal_cells(args.per_cell)
    stack = load_stack(args.per_arch_stack)
    merged = diag.merge(stack, on=["horizon", "archetype"], how="left")
    merged["best_single"] = merged[["auc_irl", "auc_rf"]].max(axis=1)
    merged["gain_vs_best"] = merged["auc_stack"] - merged["best_single"]
    merged["gain_vs_avg"] = merged["auc_stack"] - merged["auc_avg"]
    merged.to_csv(args.out_dir / "motivation_archetype_diagonal.csv", index=False)
    print(merged.to_string(index=False))

    # Panel 1: 4-method heatmap
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    methods = [
        ("auc_irl", "IRL_cold (diagonal)"),
        ("auc_rf", "RF_dir (diagonal)"),
        ("auc_avg", "Uniform ensemble"),
        ("auc_stack", "Multi-Horizon Stacking"),
    ]
    for ax, (col, title) in zip(axes, methods):
        mat = make_matrix(merged, col)
        im = plot_heatmap(ax, mat, title, vmin=0.5, vmax=0.9)
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="AUC")
    fig.suptitle("Archetype × Horizon AUC (diagonal): IRL_cold/RF_dir failure → Stacking 救出",
                 fontsize=11, y=1.02)
    fig.savefig(args.out_dir / "fig_motivation_archetype_diagonal.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Panel 2: stack gain over best single
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    gain_mat = make_matrix(merged, "gain_vs_best")
    im = plot_heatmap(ax, gain_mat, "Stacking AUC − max(IRL_diag, RF_diag)",
                      vmin=-0.05, vmax=0.20, cmap="RdBu_r",
                      annot_fmt="{:+.2f}")
    fig.colorbar(im, ax=ax, label="AUC gain")
    fig.savefig(args.out_dir / "fig_motivation_archetype_gain.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\n→ {args.out_dir}/fig_motivation_archetype_diagonal.pdf")
    print(f"→ {args.out_dir}/fig_motivation_archetype_gain.pdf")


if __name__ == "__main__":
    main()
