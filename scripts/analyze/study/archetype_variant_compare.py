# -*- coding: utf-8 -*-
"""3 特徴量 variant 横断の archetype × method 比較.

主結果に使う variant を選ぶ補助図:
  - Panel 1: 3 variant × 4 archetype の IRL vs RF 棒グラフ (差と Wilcoxon p)
  - Panel 2: 3 variant × stacking gain heatmap (archetype × horizon)
  - Panel 3: 3 variant × 3 panel (IRL, RF, IRL-RF diff) の diamond 散布
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

VARIANTS = [
    ("volat_ep150", "outputs/analysis/multi_horizon_stacking_volat_ep150",
     "volat_ep150 (st=32)"),
    ("v2_phase2", "outputs/analysis/motivation_v2_phase2",
     "v2_phase2 (st=28)"),
    ("v3_feat26", "outputs/analysis/motivation_v3_feat26",
     "v3_feat26 (st=21)"),
]
STACK_DIRS = {
    "volat_ep150": "outputs/analysis/multi_horizon_stacking_volat_ep150",
    "v2_phase2": "outputs/analysis/multi_horizon_stacking_v2_phase2_gb",
    "v3_feat26": "outputs/analysis/multi_horizon_stacking_v3_feat26_gb",
}
ARCHS = ["A1", "A2", "A3", "A4"]
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def figure1_bar_compare(out: pathlib.Path) -> None:
    """3 variant × 5 (4 archetype + ALL) の IRL vs RF mean AUC."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for ax, (vkey, vdir, vlabel) in zip(axes, VARIANTS):
        d = pd.read_csv(pathlib.Path(vdir) / "per_dev_auc_inclusive.csv").dropna(
            subset=["auc_irl", "auc_rf", "dominant"]
        )
        rows = []
        for a in ARCHS + ["ALL"]:
            sub = d if a == "ALL" else d[d["dominant"] == a]
            if len(sub) < 5:
                continue
            try:
                _, pv = wilcoxon(sub["auc_irl"], sub["auc_rf"])
            except Exception:
                pv = np.nan
            rows.append({
                "label": a, "n": len(sub),
                "irl": sub["auc_irl"].mean(), "rf": sub["auc_rf"].mean(),
                "diff": sub["auc_irl"].mean() - sub["auc_rf"].mean(),
                "p": pv,
            })
        r = pd.DataFrame(rows)
        x = np.arange(len(r))
        w = 0.38
        ax.bar(x - w / 2, r["irl"], w, color="#E64B35", alpha=0.85, label="IRL_cold")
        ax.bar(x + w / 2, r["rf"], w, color="#4DBBD5", alpha=0.85, label="RF_dir")
        for i, row in r.iterrows():
            top = max(row["irl"], row["rf"]) + 0.025
            sig = stars(row["p"])
            color = "red" if row["p"] < 0.05 else "black"
            ax.text(i, top,
                    f"{sig}\nΔ={row['diff']:+.03f}",
                    ha="center", va="bottom", fontsize=8, color=color)
            ax.text(i, 0.02, f"n={row['n']}", ha="center", va="bottom",
                    fontsize=7, color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(r["label"], fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color="gray", ls="--", lw=0.6, alpha=0.5)
        ax.set_title(vlabel, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Per-dev AUC (mean)", fontsize=10)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("3 variant 横断: archetype 別 per-dev IRL vs RF AUC "
                 "(* = Wilcoxon p<0.05)", fontsize=12, y=1.04)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def figure2_stack_gain_heatmap(out: pathlib.Path) -> None:
    """3 variant の archetype × horizon stacking gain heatmap."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    ims = []
    for ax, (vkey, _, vlabel) in zip(axes, VARIANTS):
        d = pd.read_csv(pathlib.Path(STACK_DIRS[vkey]) / "per_archetype.csv")
        d = d[d["archetype"].isin(ARCHS)]
        mat = np.full((len(ARCHS), len(HORIZONS)), np.nan)
        for i, a in enumerate(ARCHS):
            for j, h in enumerate(HORIZONS):
                row = d[(d["archetype"] == a) & (d["target"] == h)]
                if len(row):
                    mat[i, j] = row["gain_vs_best"].iloc[0]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.05, vmax=0.25, aspect="auto")
        ims.append(im)
        for i in range(len(ARCHS)):
            for j in range(len(HORIZONS)):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "white" if abs(v) > 0.13 else "black"
                ax.text(j, i, f"{v:+.02f}", ha="center", va="center",
                        fontsize=9, color=color)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels(HORIZONS, fontsize=9)
        ax.set_yticks(range(len(ARCHS)))
        ax.set_yticklabels(ARCHS, fontsize=9)
        ax.set_title(f"{vlabel}\nmean gain={np.nanmean(mat):+.03f}", fontsize=10)
    fig.colorbar(ims[-1], ax=axes, shrink=0.85, pad=0.02,
                 label="Stacking AUC − best single (per archetype × horizon)")
    fig.suptitle("3 variant 横断: stacking gain heatmap "
                 "(archetype × horizon、対角セル)", fontsize=12, y=1.05)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def figure3_diamond_3row(out: pathlib.Path) -> None:
    """3 variant × 3 panel diamond 散布 (IRL / RF / diff)."""
    CORNERS = {"A1": (0, -1), "A2": (1, 0), "A3": (0, 1), "A4": (-1, 0)}

    def draw_dia(ax):
        pts = [CORNERS[a] for a in ["A1", "A2", "A3", "A4", "A1"]]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "k-", lw=1)
        for name, (cx, cy) in CORNERS.items():
            ax.scatter([cx], [cy], s=180, facecolor="white",
                       edgecolor="black", lw=1, zorder=4)
            ax.text(cx, cy, name[-1], ha="center", va="center",
                    fontsize=8, fontweight="bold", zorder=5)
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig, axes = plt.subplots(3, 3, figsize=(11, 11))
    for row, (vkey, vdir, vlabel) in enumerate(VARIANTS):
        d = pd.read_csv(pathlib.Path(vdir) / "per_dev_auc_inclusive.csv")
        d["x"] = d["A2"] - d["A4"]
        d["y"] = d["A3"] - d["A1"]
        sub = d.dropna(subset=["auc_irl", "auc_rf"]).copy()
        sub["diff"] = sub["auc_irl"] - sub["auc_rf"]
        cfgs = [
            ("auc_irl", "IRL_cold", "RdYlGn", 0.3, 1.0),
            ("auc_rf", "RF_dir", "RdYlGn", 0.3, 1.0),
            ("diff", "IRL − RF", "RdBu_r", -0.3, 0.3),
        ]
        for col_idx, (col, title, cmap, vmn, vmx) in enumerate(cfgs):
            ax = axes[row, col_idx]
            draw_dia(ax)
            sc = ax.scatter(sub["x"], sub["y"], c=sub[col],
                            cmap=cmap, vmin=vmn, vmax=vmx,
                            s=28, alpha=0.75, edgecolor="black", lw=0.2,
                            zorder=3)
            med = sub[col].median()
            if col == "diff":
                pct_irl_wins = (sub[col] > 0).mean() * 100
                ax.set_title(f"{title}\nmed={med:+.03f}, IRL wins {pct_irl_wins:.0f}%",
                             fontsize=9)
            else:
                ax.set_title(f"{title}\nmed AUC={med:.3f}", fontsize=9)
            if col_idx == 0:
                ax.text(-1.45, 0, vlabel, fontsize=11, fontweight="bold",
                        rotation=90, ha="center", va="center")
            if row == 2:
                # colorbar at bottom of last row
                pos = ax.get_position()
                cax = fig.add_axes(
                    [pos.x0, pos.y0 - 0.05, pos.width, 0.012])
                fig.colorbar(sc, cax=cax, orientation="horizontal")
                cax.tick_params(labelsize=7)
    fig.suptitle("3 variant × diamond 散布: per-dev AUC (IRL / RF / diff)",
                 fontsize=12, y=0.995)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main() -> None:
    out_dir = pathlib.Path("outputs/analysis/variant_comparison_motivation")
    out_dir.mkdir(parents=True, exist_ok=True)
    figure1_bar_compare(out_dir / "fig_variant_archetype_bar.pdf")
    figure2_stack_gain_heatmap(out_dir / "fig_variant_stack_gain_heatmap.pdf")
    figure3_diamond_3row(out_dir / "fig_variant_diamond_3row.pdf")


if __name__ == "__main__":
    main()
