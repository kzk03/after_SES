# -*- coding: utf-8 -*-
"""SES 発表用: pair-level (dev × dir) archetype α の分布プロット一括生成。

「結果」スライドの棒グラフ (archetype_profile_eugster.py --pair) と、
「K-means との違い」概念図 (P1~P4 を正方形の四隅に置いた模式図) が
見た目が揃っていないという指摘への対応。配置 (正方形 / 菱形) ×
表現 (散布図 / hexbin) の組み合わせを全部生成して比較する。

入力: k4_pair_alpha_with_metrics.csv (A1-A4=alpha, dominant=argmax archetype)
出力: --outdir 配下に fig_pair_{layout}_{style}.{pdf,png}
"""
from __future__ import annotations

import argparse
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 「結果」スライドの棒グラフと揃える配色・ラベル
# (archetype_profile_eugster.py / archetype_profile_eugster_aosp.py と同一)
PAIR_NAMES_BY_DATASET = {
    "openstack": {
        "P1": "他プロジェクト経験者",
        "P2": "関与低下型",
        "P3": "高承諾型",
        "P4": "低活動・低承諾型",
    },
    "aosp": {
        "P1": "高活動・未経験ディレクトリ型",
        "P2": "ディレクトリ経験者・高関与型",
        "P3": "低承諾型",
        "P4": "高承諾・汎用型",
    },
}
PAIR_NAMES = PAIR_NAMES_BY_DATASET["openstack"]  # main() で --dataset に応じて上書き

# OpenStack と AOSP の archetype profile(32特徴量)を cosine 類似度で対応づけた結果、
# P1<->P1(sim=0.939)・P2<->P2(sim=0.868)は番号通りだが、
# P3<->P4(sim=0.796)・P4<->P3(sim=0.867)は番号がねじれて対応する。
# OpenStack 側は既存資料があるため変更せず、AOSP 側の P3/P4 の色を入れ替えて揃える。
PAIR_COLORS_BY_DATASET = {
    "openstack": {
        "P1": "#E64B35",
        "P2": "#4DBBD5",
        "P3": "#00A087",
        "P4": "#F39B7F",
    },
    "aosp": {
        "P1": "#E64B35",
        "P2": "#4DBBD5",
        "P3": "#F39B7F",
        "P4": "#00A087",
    },
}
PAIR_COLORS = PAIR_COLORS_BY_DATASET["openstack"]  # main() で --dataset に応じて上書き

# 「K-means との違い」概念図と同じ位置関係 (P1 左上 / P2 右上 / P3 左下 / P4 右下)
CORNERS = {
    "square": {
        "P1": (-1.0, 1.0), "P2": (1.0, 1.0),
        "P3": (-1.0, -1.0), "P4": (1.0, -1.0),
    },
    "rhombus": {
        "P1": (0.0, 1.0), "P2": (1.0, 0.0),
        "P3": (0.0, -1.0), "P4": (-1.0, 0.0),
    },
}
FRAME_ORDER = {
    "square": ["P1", "P2", "P4", "P3", "P1"],
    "rhombus": ["P1", "P2", "P3", "P4", "P1"],
}
LABEL_POS = {
    "square": {
        "P1": (-1.12, 1.12, "right", "bottom"),
        "P2": (1.12, 1.12, "left", "bottom"),
        "P3": (-1.12, -1.12, "right", "top"),
        "P4": (1.12, -1.12, "left", "top"),
    },
    "rhombus": {
        "P1": (0.0, 1.15, "center", "bottom"),
        "P2": (1.15, 0.0, "left", "center"),
        "P3": (0.0, -1.15, "center", "top"),
        "P4": (-1.15, 0.0, "right", "center"),
    },
}


def project(df: pd.DataFrame, layout: str) -> tuple[np.ndarray, np.ndarray]:
    """各 pair の alpha (A1..A4, 和=1) を頂点座標の加重平均で 2D 投影する。"""
    corners = CORNERS[layout]
    x = np.zeros(len(df))
    y = np.zeros(len(df))
    for i in range(1, 5):
        cx, cy = corners[f"P{i}"]
        x += df[f"A{i}"].to_numpy() * cx
        y += df[f"A{i}"].to_numpy() * cy
    return x, y


def draw_frame(ax, layout: str, counts: dict[str, int] | None = None,
               show_labels: bool = True) -> None:
    corners = CORNERS[layout]
    pts = [corners[p] for p in FRAME_ORDER[layout]]
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "k--", lw=1.0, zorder=2, alpha=0.6)
    for p, (cx, cy) in corners.items():
        ax.scatter([cx], [cy], s=220, facecolor="white",
                   edgecolor=PAIR_COLORS[p], lw=2.2, zorder=5)
        if show_labels:
            ax.text(cx, cy, p, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=PAIR_COLORS[p], zorder=6)
    if show_labels:
        for p, (lx, ly, ha, va) in LABEL_POS[layout].items():
            label = f"{p}\n{PAIR_NAMES[p]}"
            if counts is not None:
                label += f" (n={counts[p]})"
            ax.text(lx, ly, label, ha=ha, va=va,
                    fontsize=9, color=PAIR_COLORS[p], fontweight="bold")
    pad = 0.5
    ax.set_xlim(-1 - pad, 1 + pad)
    ax.set_ylim(-1 - pad, 1 + pad)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def savefig(fig, out: pathlib.Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), format="pdf", bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), format="png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"saved: {out.with_suffix('.pdf')}, {out.with_suffix('.png')}")


def plot_scatter(df: pd.DataFrame, layout: str, out: pathlib.Path,
                  point_size: float = 13, point_alpha: float = 0.4,
                  show_labels: bool = True) -> None:
    x, y = project(df, layout)
    counts = {f"P{i}": int((df["dominant"] == f"A{i}").sum()) for i in range(1, 5)}
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    draw_frame(ax, layout, counts=counts, show_labels=show_labels)
    for i in range(1, 5):
        p = f"P{i}"
        mask = (df["dominant"] == f"A{i}").to_numpy()
        ax.scatter(x[mask], y[mask], s=point_size, alpha=point_alpha,
                   color=PAIR_COLORS[p], edgecolor="none", zorder=3)
    fig.tight_layout()
    savefig(fig, out)


def _mode_reduce(vals: np.ndarray):
    v, c = np.unique(vals, return_counts=True)
    return v[np.argmax(c)]


def plot_hexbin_mode(df: pd.DataFrame, layout: str, out: pathlib.Path,
                      gridsize: int = 18, mincnt: int = 3) -> None:
    """各 hexbin セルを『最頻 dominant archetype』の色で塗り、
    セル内の最頻archetype比率を透明度に反映する(単色ベタ塗りを避ける)。"""
    x, y = project(df, layout)
    dom_idx = df["dominant"].str[1].astype(int).to_numpy()  # A1->1 .. A4->4

    fig, ax = plt.subplots(figsize=(6.8, 7.1))

    hb_mode = ax.hexbin(x, y, C=dom_idx, gridsize=gridsize,
                         extent=(-1.5, 1.5, -1.5, 1.5), mincnt=mincnt,
                         reduce_C_function=_mode_reduce, visible=False)
    hb_purity = ax.hexbin(
        x, y, C=dom_idx, gridsize=gridsize, extent=(-1.5, 1.5, -1.5, 1.5),
        mincnt=mincnt,
        reduce_C_function=lambda v: (v == _mode_reduce(v)).mean(),
        visible=False,
    )
    offsets = hb_mode.get_offsets()
    modes = hb_mode.get_array()
    purity = hb_purity.get_array()

    facecolors = np.array([mcolors.to_rgba(PAIR_COLORS[f"P{int(m)}"], alpha=0.35 + 0.55 * pu)
                            for m, pu in zip(modes, purity)])
    from matplotlib.collections import PolyCollection
    hb_mode.remove()
    hb_purity.remove()
    # hexagon path (matplotlib hexbin と同じ向き: pointy-top ではなく flat-top)
    hb_ref = ax.hexbin(x, y, gridsize=gridsize, extent=(-1.5, 1.5, -1.5, 1.5))
    hexagon = hb_ref.get_paths()[0]
    hb_ref.remove()
    coll = PolyCollection([hexagon.vertices + off for off in offsets],
                           facecolors=facecolors, edgecolors="white", linewidths=0.3,
                           zorder=3)
    ax.add_collection(coll)

    draw_frame(ax, layout)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PAIR_COLORS[f"P{i}"]) for i in range(1, 5)]
    labels = [f"P{i}" for i in range(1, 5)]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.03),
              ncol=4, fontsize=8, frameon=False, handletextpad=0.3, columnspacing=1.0)
    ax.set_title(
        f"開発者×ディレクトリ ペアの archetype 分布 (hexbin, n={len(df)})\n"
        f"色=セル内最頻archetype, 濃淡=その純度",
        fontsize=10.5,
    )
    savefig(fig, out)


DEFAULT_PATHS = {
    "openstack": (
        pathlib.Path(
            "outputs/analysis/archetype_pair_level_volat_ep150_coldonly/"
            "k4_pair_alpha_with_metrics.csv"
        ),
        pathlib.Path("outputs/analysis/archetype_pair_level_volat_ep150_coldonly/present"),
        13, 0.4,
    ),
    "aosp": (
        pathlib.Path("outputs/aosp32/archetype/pair/k4_pair_alpha_with_metrics.csv"),
        pathlib.Path("outputs/aosp32/archetype/pair/present"),
        6, 0.25,
    ),
}


def main() -> None:
    global PAIR_NAMES, PAIR_COLORS
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["openstack", "aosp"], default="openstack")
    p.add_argument("--alpha-csv", type=pathlib.Path, default=None)
    p.add_argument("--outdir", type=pathlib.Path, default=None)
    p.add_argument(
        "--include-hexbin", action="store_true",
        help="参考用のhexbin版(dominant多数決で塗る)も出力する。"
             "分離度を誇張して見せるので、発表資料には非推奨(検証記録参照)。",
    )
    args = p.parse_args()

    PAIR_NAMES = PAIR_NAMES_BY_DATASET[args.dataset]
    PAIR_COLORS = PAIR_COLORS_BY_DATASET[args.dataset]
    default_alpha, default_outdir, point_size, point_alpha = DEFAULT_PATHS[args.dataset]
    alpha_csv = args.alpha_csv or default_alpha
    outdir = args.outdir or default_outdir

    df = pd.read_csv(alpha_csv).dropna(subset=["A1", "A2", "A3", "A4", "dominant"])
    print(f"n_pair={len(df)}")

    for layout in ["square", "rhombus"]:
        plot_scatter(df, layout, outdir / f"fig_pair_{layout}_scatter",
                     point_size=point_size, point_alpha=point_alpha)
        if args.include_hexbin:
            plot_hexbin_mode(df, layout, outdir / f"fig_pair_{layout}_hexbin")


if __name__ == "__main__":
    main()
