# -*- coding: utf-8 -*-
"""Re-render elbow plots (dev and pair) with larger bold fonts for paper.

Reads rss_elbow.csv and writes fig_{dev,pair}_elbow.pdf into writing/fig/.
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd


def render_elbow(csv_path: pathlib.Path, ycol: str, out_path: pathlib.Path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["k"], df[ycol], "o-", color="#1f77b4", lw=3, markersize=12)
    ax.set_xlabel("$K$", fontsize=22, fontweight="bold")
    ax.set_ylabel("RSS / N", fontsize=22, fontweight="bold")
    ax.tick_params(axis="both", labelsize=18)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    root = pathlib.Path(__file__).resolve().parents[3]
    dev_csv = root / "outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/rss_elbow.csv"
    pair_csv = root / "outputs/analysis/archetype_pair_level_volat_ep150_coldonly/rss_elbow.csv"
    out_dir = root / "writing/fig"
    render_elbow(dev_csv, "rss_per_dev", out_dir / "fig_dev_elbow.pdf")
    render_elbow(pair_csv, "rss_per_pair", out_dir / "fig_pair_elbow.pdf")


if __name__ == "__main__":
    main()
