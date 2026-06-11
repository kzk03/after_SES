# -*- coding: utf-8 -*-
"""論文 Table 1 の diamond 可視化 — Eugster soccer 風.

各 dev を α-simplex を 2D 投影した diamond 上に配置し、
4 horizon 別に Δ (AUC_IRL − AUC_RF) を色で表示 (4 panel)。

各 (dev, horizon) ペアの AUC は、その horizon の全 dir 行を pool して計算。
n は horizon ごとに「dir 間で label 変動がある dev」に限定される。

入力:
  - pair_predictions.csv (train_h/eval_h, h=4)
  - k4_dev_alpha_with_metrics.csv (α + dominant)
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
ARCH_LABELS = {
    "A1": "A1\nSporadic\nNewcomer",
    "A2": "A2\nStrict\nReviewer-Only",
    "A3": "A3\nApproving\nVeteran",
    "A4": "A4\nSaturated\nActive",
}
CORNERS = {"A1": (0.0, -1.0), "A2": (1.0, 0.0),
           "A3": (0.0, 1.0), "A4": (-1.0, 0.0)}


def per_dev_horizon_auc(pred_root: pathlib.Path) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        p = pred_root / f"lstm_baseline/train_{h}/eval_{h}/pair_predictions.csv"
        d = pd.read_csv(p)
        for dev, g in d.groupby("developer"):
            y = g["label"].to_numpy().astype(int)
            if len(y) < 2 or len(set(y)) < 2:
                continue
            try:
                a_irl = roc_auc_score(y, g["irl_dir_prob_calibrated"])
                a_rf = roc_auc_score(y, g["rf_dir_prob"])
            except Exception:
                continue
            rows.append({"developer": dev, "horizon": h,
                         "n": len(g), "n_pos": int(y.sum()),
                         "auc_irl": a_irl, "auc_rf": a_rf,
                         "diff": a_irl - a_rf})
    return pd.DataFrame(rows)


def draw_diamond(ax, show_labels=True) -> None:
    pts = [CORNERS[a] for a in ["A1", "A2", "A3", "A4", "A1"]]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "k-", lw=1.2, zorder=2)
    for name, (cx, cy) in CORNERS.items():
        ax.scatter([cx], [cy], s=260, facecolor="white",
                   edgecolor="black", lw=1.2, zorder=4)
        ax.text(cx, cy, name[-1], ha="center", va="center",
                fontsize=10, fontweight="bold", zorder=5)
        if not show_labels:
            continue
        off = 0.10
        if name == "A1":
            ax.text(cx, cy - off, ARCH_LABELS[name],
                    ha="center", va="top", fontsize=7.5)
        elif name == "A3":
            ax.text(cx, cy + off, ARCH_LABELS[name],
                    ha="center", va="bottom", fontsize=7.5)
        elif name == "A2":
            ax.text(cx + off, cy, ARCH_LABELS[name],
                    ha="left", va="center", fontsize=7.5, rotation=270)
        elif name == "A4":
            ax.text(cx - off, cy, ARCH_LABELS[name],
                    ha="right", va="center", fontsize=7.5, rotation=90)
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.45, 1.40)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main() -> None:
    pred_root = pathlib.Path("outputs/main32_mce_volat_ep150")
    alpha_csv = pathlib.Path(
        "outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv"
    )
    out = pathlib.Path(
        "outputs/analysis/multi_horizon_stacking_volat_ep150/fig_diamond_per_horizon_diff.pdf"
    )

    df = per_dev_horizon_auc(pred_root)
    print(f"per (dev, horizon) AUC rows: {len(df)}")
    print(df.groupby("horizon")["developer"].nunique())

    alpha = pd.read_csv(alpha_csv, index_col=0).reset_index().rename(
        columns={"index": "developer"})
    alpha.columns = ["developer"] + list(alpha.columns[1:])
    m = df.merge(alpha[["developer", "A1", "A2", "A3", "A4", "dominant"]],
                 on="developer", how="inner")
    m["x"] = m["A2"] - m["A4"]
    m["y"] = m["A3"] - m["A1"]
    print(f"merged: {len(m)} (dev, horizon) samples, "
          f"covering {m['developer'].nunique()} unique dev")
    m.to_csv(out.with_suffix(".csv"), index=False)

    # 4 panel 版 (horizon 別)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))
    sc = None
    for ax, h in zip(axes, HORIZONS):
        draw_diamond(ax, show_labels=True)
        sub = m[m["horizon"] == h]
        sc = ax.scatter(sub["x"], sub["y"], c=sub["diff"],
                        cmap="RdBu_r", vmin=-0.4, vmax=0.4,
                        s=50, alpha=0.85, edgecolor="black", lw=0.4,
                        zorder=3)
        mean_diff = sub["diff"].mean()
        irl_wins = (sub["diff"] > 0).sum()
        rf_wins = (sub["diff"] < 0).sum()
        ax.set_title(f"Horizon {h}\n"
                     f"n={len(sub)} dev, mean Δ={mean_diff:+.03f}\n"
                     f"IRL wins {irl_wins} / RF wins {rf_wins}",
                     fontsize=9.5)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.85, pad=0.02,
                        label="Δ = AUC$_{IRL}$ − AUC$_{RF}$ (per-dev, per-horizon)")
    cbar.ax.tick_params(labelsize=9)
    fig.suptitle(
        "Per-dev AUC の archetype × horizon 補完性 — diamond 投影\n"
        "(青 = RF 優位、赤 = IRL 優位)",
        fontsize=12, y=1.04,
    )
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")

    # ---- 全 horizon 混合の単一 diamond (|Δ| 小 = 灰色帯は非表示) ----
    GRAY_THRESHOLD = 0.05  # |Δ| ≤ 0.05 (≒ tied) は非表示
    out_single = out.with_name("fig_diamond_pooled.pdf")
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    draw_diamond(ax, show_labels=True)
    markers = {"0-3m": "o", "3-6m": "s", "6-9m": "^", "9-12m": "D"}
    m_shown = m[m["diff"].abs() > GRAY_THRESHOLD]
    n_hidden = len(m) - len(m_shown)
    for h in HORIZONS:
        sub = m_shown[m_shown["horizon"] == h]
        sc = ax.scatter(sub["x"], sub["y"], c=sub["diff"],
                        cmap="RdBu_r", vmin=-0.4, vmax=0.4,
                        marker=markers[h], s=46, alpha=0.85,
                        edgecolor="black", lw=0.3, zorder=3,
                        label=f"{h} (n={len(sub)})")
    mean_diff = m["diff"].mean()
    irl_wins = (m["diff"] > GRAY_THRESHOLD).sum()
    rf_wins = (m["diff"] < -GRAY_THRESHOLD).sum()
    ax.set_title(
        f"All horizons pooled: n={len(m)} samples, {m['developer'].nunique()} dev\n"
        f"|Δ|>{GRAY_THRESHOLD} のみ表示: IRL wins {irl_wins} / RF wins {rf_wins} "
        f"(tied {n_hidden} 非表示)",
        fontsize=10,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.04,
                        label="Δ = AUC$_{IRL}$ − AUC$_{RF}$")
    cbar.ax.tick_params(labelsize=8)
    # legend (marker 形状 = horizon)
    legend_handles = [
        plt.Line2D([0], [0], marker=markers[h], color="gray",
                   markerfacecolor="lightgray", markeredgecolor="black",
                   markersize=7, lw=0,
                   label=f"{h} (n={(m['horizon']==h).sum()})")
        for h in HORIZONS
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              bbox_to_anchor=(1.32, 0.30), fontsize=7.5,
              frameon=False, title="Horizon", title_fontsize=8.5)
    fig.savefig(out_single, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_single}")


if __name__ == "__main__":
    main()
