# -*- coding: utf-8 -*-
"""論文 図4 (dev archetype × horizon の Δ ヒートマップ) を単独で出力.

入力: outputs/analysis/multi_horizon_stacking_volat_ep150/motivation_archetype_diagonal.csv
出力: writing/fig/fig_dev_archetype_heatmap.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHS = ["A1", "A2", "A3", "A4"]
ARCH_FULL = {
    "A1": "A1: Sporadic Newcomer",
    "A2": "A2: Strict Reviewer-Only",
    "A3": "A3: Approving Veteran",
    "A4": "A4: Saturated Active",
}
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def main() -> None:
    src = pathlib.Path(
        "outputs/analysis/multi_horizon_stacking_volat_ep150/motivation_archetype_diagonal.csv"
    )
    df = pd.read_csv(src)
    df = df[df["archetype"].isin(ARCHS)].copy()
    df["diff"] = df["auc_irl"] - df["auc_rf"]

    def mat(col):
        m = np.zeros((len(ARCHS), len(HORIZONS)))
        for i, a in enumerate(ARCHS):
            for j, h in enumerate(HORIZONS):
                r = df[(df["archetype"] == a) & (df["horizon"] == h)]
                m[i, j] = r[col].iloc[0] if len(r) else np.nan
        return m

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    dmat = mat("diff")
    irl = mat("auc_irl")
    rf = mat("auc_rf")
    im = ax.imshow(dmat, cmap="RdBu_r", vmin=-0.15, vmax=0.15, aspect="auto")
    for i in range(len(ARCHS)):
        for j in range(len(HORIZONS)):
            v = dmat[i, j]
            color = "white" if abs(v) > 0.09 else "black"
            ax.text(j, i - 0.18, f"{v:+.03f}",
                    ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold")
            ax.text(j, i + 0.22,
                    f"IRL {irl[i,j]:.02f} / RF {rf[i,j]:.02f}",
                    ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels(HORIZONS, fontsize=10)
    ax.set_yticks(range(len(ARCHS)))
    ax.set_yticklabels([ARCH_FULL[a] for a in ARCHS], fontsize=9)
    ax.set_title("Δ = AUC$_{IRL}$ − AUC$_{RF}$ (青=RF優位／赤=IRL優位)",
                 fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                        label="Δ (IRL − RF)")
    cbar.ax.tick_params(labelsize=8)

    out = pathlib.Path("writing/fig/fig_dev_archetype_heatmap.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
