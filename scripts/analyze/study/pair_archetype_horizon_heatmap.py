# -*- coding: utf-8 -*-
"""Pair archetype × horizon の 2 heatmap.

  (1) Δ = AUC_IRL − AUC_RF (paper §3.4 motivation 風)
  (2) Stacking gain = AUC_stack − max(IRL, RF) (paper §4 主結果風)

pair archetype は k4_pair_alpha_with_metrics.csv (B-31) の dominant 列を使用.
"""
from __future__ import annotations

import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
ARCHS = ["A1", "A2", "A3", "A4"]
# B-31 で命名されたラベル (path 親和度ベース) — 必要なら検証で変更
PAIR_LABELS = {
    "A1": "P1: Unfamiliar Dir",
    "A2": "P2: Familiar-Reject",
    "A3": "P3: Familiar-Accept",
    "A4": "P4: Mid-Engagement",
}


def load_diagonal(base: pathlib.Path) -> pd.DataFrame:
    """4 horizon の対角セルを wide DataFrame に."""
    wides = []
    for h in HORIZONS:
        p = base / "lstm_baseline" / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        d = pd.read_csv(p).rename(columns={
            "label": f"label_{h}",
            "irl_dir_prob_calibrated": f"irl_{h}",
            "rf_dir_prob": f"rf_{h}",
        })[["developer", "directory", f"label_{h}", f"irl_{h}", f"rf_{h}"]]
        wides.append(d)
    out = wides[0]
    for w in wides[1:]:
        out = out.merge(w, on=["developer", "directory"], how="outer")
    return out


def cv_stack_auc(X, y, irl_s, rf_s, n_splits=5, seed=42):
    if len(set(y)) < 2 or len(y) < 30:
        return None
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    stack_preds = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                         random_state=seed)
        clf.fit(X[tr], y[tr])
        stack_preds[te] = clf.predict_proba(X[te])[:, 1]
    return {
        "auc_irl": roc_auc_score(y, irl_s),
        "auc_rf": roc_auc_score(y, rf_s),
        "auc_avg": roc_auc_score(y, (irl_s + rf_s) / 2),
        "auc_stack": roc_auc_score(y, stack_preds),
    }


def compute_table(base: pathlib.Path, alpha_csv: pathlib.Path) -> pd.DataFrame:
    w = load_diagonal(base)
    alpha = pd.read_csv(alpha_csv)[["developer", "directory", "dominant"]]
    w = w.merge(alpha, on=["developer", "directory"], how="inner")
    feature_cols = [f"irl_{h}" for h in HORIZONS] + [f"rf_{h}" for h in HORIZONS]

    rows = []
    for arch in ARCHS:
        sub_a = w[w["dominant"] == arch]
        for h in HORIZONS:
            lc = f"label_{h}"
            sub = sub_a.dropna(subset=feature_cols + [lc])
            if len(sub) < 30:
                continue
            X = sub[feature_cols].to_numpy()
            y = sub[lc].astype(int).to_numpy()
            irl = sub[f"irl_{h}"].to_numpy()
            rf = sub[f"rf_{h}"].to_numpy()
            r = cv_stack_auc(X, y, irl, rf)
            if r is None:
                continue
            rows.append({
                "archetype": arch, "horizon": h,
                "n": len(sub), "n_pos": int(y.sum()), **r,
            })
    # ALL
    for h in HORIZONS:
        lc = f"label_{h}"
        sub = w.dropna(subset=feature_cols + [lc])
        X = sub[feature_cols].to_numpy()
        y = sub[lc].astype(int).to_numpy()
        irl = sub[f"irl_{h}"].to_numpy()
        rf = sub[f"rf_{h}"].to_numpy()
        r = cv_stack_auc(X, y, irl, rf)
        rows.append({"archetype": "ALL", "horizon": h,
                     "n": len(sub), "n_pos": int(y.sum()), **r})
    out = pd.DataFrame(rows)
    out["diff"] = out["auc_irl"] - out["auc_rf"]
    out["best_single"] = out[["auc_irl", "auc_rf"]].max(axis=1)
    out["gain_vs_best"] = out["auc_stack"] - out["best_single"]
    out["gain_vs_avg"] = out["auc_stack"] - out["auc_avg"]
    return out


def heatmap_panel(ax, mat, irl_mat, rf_mat, ann_extra, title, cmap, vmin, vmax,
                  cbar_label, fmt_main="{:+.03f}"):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                continue
            color = "white" if abs(v - (vmin + vmax) / 2) > (vmax - vmin) * 0.35 \
                    else "black"
            ax.text(j, i - 0.18, fmt_main.format(v),
                    ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold")
            if ann_extra is not None:
                ax.text(j, i + 0.22, ann_extra[i][j],
                        ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS, fontsize=10)
    ax.set_yticks(range(len(ARCHS)))
    ax.set_yticklabels([PAIR_LABELS[a] for a in ARCHS], fontsize=9)
    ax.set_title(title, fontsize=11)
    return im


def to_mat(table: pd.DataFrame, col: str) -> np.ndarray:
    m = np.full((len(ARCHS), len(HORIZONS)), np.nan)
    for i, a in enumerate(ARCHS):
        for j, h in enumerate(HORIZONS):
            r = table[(table["archetype"] == a) & (table["horizon"] == h)]
            if len(r):
                m[i, j] = r[col].iloc[0]
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=pathlib.Path,
                   default="outputs/main32_mce_volat_ep150")
    p.add_argument("--alpha", type=pathlib.Path,
                   default="outputs/analysis/archetype_pair_level_volat_ep150_coldonly/k4_pair_alpha_with_metrics.csv")
    p.add_argument("--out-dir", type=pathlib.Path,
                   default="outputs/analysis/multi_horizon_stacking_volat_ep150")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("computing per-pair-archetype × horizon stacking + AUC...")
    tbl = compute_table(args.base, args.alpha)
    tbl.to_csv(args.out_dir / "pair_archetype_horizon_table.csv", index=False)
    print(tbl.to_string(index=False))

    diff_mat = to_mat(tbl[tbl["archetype"] != "ALL"], "diff")
    irl_mat = to_mat(tbl[tbl["archetype"] != "ALL"], "auc_irl")
    rf_mat = to_mat(tbl[tbl["archetype"] != "ALL"], "auc_rf")
    gain_mat = to_mat(tbl[tbl["archetype"] != "ALL"], "gain_vs_best")
    stack_mat = to_mat(tbl[tbl["archetype"] != "ALL"], "auc_stack")

    ann_irl_rf = [[f"IRL {irl_mat[i,j]:.02f} / RF {rf_mat[i,j]:.02f}"
                   for j in range(len(HORIZONS))] for i in range(len(ARCHS))]
    ann_best_stack = [[f"best {max(irl_mat[i,j], rf_mat[i,j]):.02f} → stk {stack_mat[i,j]:.02f}"
                       for j in range(len(HORIZONS))] for i in range(len(ARCHS))]

    # ---- Figure 1: Δ heatmap (motivation) ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = heatmap_panel(ax, diff_mat, irl_mat, rf_mat, ann_irl_rf,
                       title="(A) Pair archetype × horizon: Δ = AUC$_{IRL}$ − AUC$_{RF}$\n"
                             "青 = RF 優位, 赤 = IRL 優位",
                       cmap="RdBu_r", vmin=-0.15, vmax=0.15,
                       cbar_label="Δ")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="Δ (IRL − RF)")
    cbar.ax.tick_params(labelsize=8)
    out1 = args.out_dir / "fig_pair_archetype_diff_heatmap.pdf"
    fig.savefig(out1, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out1}")

    # ---- Figure 2: Stacking gain heatmap ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = heatmap_panel(ax, gain_mat, irl_mat, rf_mat, ann_best_stack,
                       title="(B) Pair archetype × horizon: Stacking gain over best single\n"
                             "(gain = AUC$_{stack}$ − max(IRL, RF))",
                       cmap="RdBu_r", vmin=-0.05, vmax=0.20,
                       cbar_label="Stacking gain", fmt_main="{:+.03f}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, label="gain vs best single")
    cbar.ax.tick_params(labelsize=8)
    out2 = args.out_dir / "fig_pair_archetype_stacking_gain.pdf"
    fig.savefig(out2, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
