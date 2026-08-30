# -*- coding: utf-8 -*-
"""Re-render AOSP elbow plots (dev and pair) matching the paper style.

Reads outputs/aosp32/archetype/{dev,pair}/rss_elbow.csv and writes
fig_aosp_{dev,pair}_elbow.pdf into writing/fig/.

No title/caption and no y-axis label text (kept out of the image on
purpose; captions belong in the LaTeX \\caption{}). Y tick labels are
rounded to 1 significant figure.
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


def render_elbow(csv_path: pathlib.Path, ycol: str, out_path: pathlib.Path) -> None:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["k"], df[ycol], "o-", color="#1f77b4", lw=3, markersize=12)
    ax.set_xlabel("$K$", fontsize=22, fontweight="bold")
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
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
    dev_csv = root / "outputs/aosp32/archetype/dev/rss_elbow.csv"
    pair_csv = root / "outputs/aosp32/archetype/pair/rss_elbow.csv"
    out_dir = root / "writing/fig"
    out_dir.mkdir(parents=True, exist_ok=True)
    render_elbow(dev_csv, "rss_per_unit", out_dir / "fig_aosp_dev_elbow.pdf")
    render_elbow(pair_csv, "rss_per_unit", out_dir / "fig_aosp_pair_elbow.pdf")


if __name__ == "__main__":
    main()
