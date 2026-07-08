# -*- coding: utf-8 -*-
"""原型構成比の時系列図 ＋ 原型名×IRL/RF/Ens の t0 別テーブル生成.

- 構成比: 各 t0 で dominant 割当された dev/pair の原型別シェア（%）を積み上げ棒で
  描く。「レビュアー生態系の変遷」図。基準 t0=2023-01 への対応付けで名前を揃え、
  対応付けが崩壊している点（match_margin<0, writing/t0_archetype_discussion.md §2）
  はハッチで区別する。構成比は archetype 特徴が raw データ依存のため
  モデル系列に依存しない（論文系列の値が正）。
- テーブル: 各 t0 × 原型名 × モデル(IRL/RF/Ens) の pooled AUC と 95%CI を
  markdown で出力（論文の表5/7相当の t0 掃引版）。

入力: t0_archetype_{level}_k4_comparison*.csv（対応付け・品質列つき）
      <tag>/archetype/{level}/k4_pooled_auc*.csv（CI）
出力: writing/fig/fig_t0_archetype_composition{suffix}.pdf
      writing/t0_archetype_tables{suffix}.md
      outputs/<root>/t0_archetype_composition{suffix}.csv

使い方:
  uv run python scripts/analyze/study/t0_archetype_composition.py \
      [--root outputs/t0_sweep] [--levels dev pair]
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import matplotlib
matplotlib.use("Agg")
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]

# 基準 t0=2023-01 の base_archetype → 意味名（プロファイル読み,
# writing/t0_archetype_discussion.md §1/§6）。dev は全系列共通（プロファイル同一）。
DEV_NAMES = {"A1": "低活動・周辺", "A2": "休眠・受け手", "A3": "コア・高承諾", "A4": "活動的・低承諾"}
# pair は base プロファイルの index 順が系列で入れ替わる（構造は一致, §2 追記）
PAIR_NAMES = {
    "outputs/t0_sweep":            {"A1": "未経験×中活動", "A2": "path経験×低承諾", "A3": "経験濃×高承諾", "A4": "低活動×未経験"},
    "outputs/t0_sweep_align_fs24": {"A1": "未経験×中活動", "A2": "path経験×低承諾", "A3": "経験濃×高承諾", "A4": "低活動×未経験"},
    "outputs/t0_sweep_align_fs36": {"A1": "経験濃×高承諾", "A2": "path経験×低承諾", "A3": "未経験×中活動", "A4": "低活動×未経験"},
}
NAME_COLORS = {  # 名前ごとに固定色（dev/pair 別パレット）
    "低活動・周辺": "#8da0cb", "休眠・受け手": "#b3b3b3", "コア・高承諾": "#66c2a5", "活動的・低承諾": "#fc8d62",
    "未経験×中活動": "#8da0cb", "path経験×低承諾": "#b3b3b3", "経験濃×高承諾": "#66c2a5", "低活動×未経験": "#fc8d62",
}
MODEL_LABEL = {"cold_irl": "IRL", "rf_dir": "RF", "ens": "Ens"}


def names_for(root: str, level: str) -> dict:
    return DEV_NAMES if level == "dev" else PAIR_NAMES[root]


def load_comparison(root: str, level: str) -> pd.DataFrame | None:
    hits = sorted(glob.glob(str(ROOT / root / f"t0_archetype_{level}_k4_comparison*.csv")))
    if not hits:
        return None
    df = pd.read_csv(hits[0])
    if "match_ok" not in df.columns:  # 旧形式は品質列なし → 全て True 扱い
        df["match_ok"] = True
    nm = names_for(root, level)
    df["name"] = df["base_archetype"].map(nm)
    return df


def load_ci(root: str, tag: str, level: str) -> pd.DataFrame | None:
    d = ROOT / root / tag / "archetype" / level
    for f in (d / "k4_pooled_auc_bootstrap.csv", d / "k4_pooled_auc.csv"):
        if f.exists():
            return pd.read_csv(f)
    return None


def fig_composition(root: str, levels: list[str], suffix: str) -> None:
    fig, axes = plt.subplots(1, len(levels), figsize=(6.5 * len(levels), 4.6))
    axes = [axes] if len(levels) == 1 else list(axes)
    rows_out = []
    for ax, level in zip(axes, levels):
        df = load_comparison(root, level)
        if df is None:
            ax.set_title(f"{level}: データなし")
            continue
        t0s = sorted(df["t0"].unique())
        names = [names_for(root, level)[a] for a in sorted(df["base_archetype"].unique())]
        bottoms = {t: 0.0 for t in t0s}
        totals = {t: df[df["t0"] == t]["n_samples"].sum() for t in t0s}
        x = list(range(len(t0s)))
        for nm in names:
            sub = df[df["name"] == nm].set_index("t0")
            shares, hatches = [], []
            for t in t0s:
                if t in sub.index:
                    shares.append(100.0 * sub.loc[t, "n_samples"] / totals[t])
                    hatches.append(None if sub.loc[t, "match_ok"] else "//")
                    rows_out.append({
                        "level": level, "t0": t, "name": nm,
                        "n": int(sub.loc[t, "n_samples"]),
                        "share_pct": round(100.0 * sub.loc[t, "n_samples"] / totals[t], 1),
                        "match_ok": bool(sub.loc[t, "match_ok"]),
                    })
                else:
                    shares.append(0.0)
                    hatches.append(None)
            b = [bottoms[t] for t in t0s]
            bars = ax.bar(x, shares, bottom=b, label=nm, color=NAME_COLORS[nm],
                          edgecolor="white", linewidth=0.6)
            for bar, h in zip(bars, hatches):
                if h:
                    bar.set_hatch(h)
                    bar.set_edgecolor("red")
            for t, s in zip(t0s, shares):
                bottoms[t] += s
        ax.set_xticks(x)
        ax.set_xticklabels([t[:7] for t in t0s], rotation=30)
        ax.set_ylim(0, 100)
        ax.set_ylabel("dominant 割当シェア (%)")
        ax.set_title(f"{level}-level 原型構成比（基準2023-01対応付け, //=同定不能）")
        ax2 = ax.twinx()
        ax2.plot(x, [totals[t] for t in t0s], "k--o", ms=4, alpha=.6)
        ax2.set_ylabel("n（予測つき dominant 割当, 横断CSV基準）")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle(f"原型構成比の時系列（{root}）")
    fig.tight_layout()
    p = ROOT / f"writing/fig/fig_t0_archetype_composition{suffix}.pdf"
    fig.savefig(p)
    print(f"saved: {p}")
    out = ROOT / root / f"t0_archetype_composition{suffix}.csv"
    pd.DataFrame(rows_out).to_csv(out, index=False)
    print(f"saved: {out}")


def md_tables(root: str, levels: list[str], suffix: str) -> None:
    lines = [
        f"# t0掃引 × 原型別 AUC テーブル（{root}, k=4, B定義, pooled）",
        "",
        "> 自動生成: `scripts/analyze/study/t0_archetype_composition.py`。",
        "> 原型名は基準 t0=2023-01 への最近傍対応付け（Hungarian）で揃えたもの。",
        "> ⚠ = match_margin<0（同定不能, 名前の同一性が保証されない点）。",
        f"> **AUC の `[lo, hi]` は pooled bootstrap の95%信頼区間**であり、",
        f"> 複数系列（inter/align 等）をまとめた範囲ではない（本ファイルは {root} 単体）。",
        "> 局所 index（その t0 での生 A1..A4）を併記。",
        "> **注意: dev-level の AUC は②方式**（(dev,horizon) を max 集約 =「その型の",
        "> 開発者がそもそも継続するか」の判別）。**論文の表7 は③方式**（ペア単位",
        "> サンプルを開発者原型で切り出しプール）で集計単位が異なる。③方式の表は",
        "> `t0_archetype_tables_devpp*.md`（`t0_archetype_dev_pairpooled.py`）を参照。",
        "> pair-level は論文（表5）と同じ③方式。",
        "",
    ]
    for level in levels:
        df = load_comparison(root, level)
        if df is None:
            continue
        lines.append(f"## {level}-level\n")
        for t0 in sorted(df["t0"].unique()):
            tag = t0.replace("-", "")
            ci = load_ci(root, tag, level)
            sub = df[df["t0"] == t0].sort_values("base_archetype")
            lines.append(f"### t0 = {t0}\n")
            lines.append("| 原型 | 局所idx | n | IRL AUC [95%CI] | RF AUC [95%CI] | Ens AUC [95%CI] | IRL−RF |")
            lines.append("|---|---|---|---|---|---|---|")
            for _, r in sub.iterrows():
                cells = {}
                for m in ["cold_irl", "rf_dir", "ens"]:
                    if ci is not None:
                        row = ci[(ci["archetype"] == r["local_archetype"]) & (ci["model"] == m)]
                        if len(row):
                            v = row.iloc[0]
                            cells[m] = f"{v['auc']:.3f} [{v['auc_ci_lo']:.3f}, {v['auc_ci_hi']:.3f}]"
                            continue
                    cells[m] = {"cold_irl": f"{r['IRL']:.3f}", "rf_dir": f"{r['RF']:.3f}",
                                "ens": f"{r['Ens']:.3f}"}[m]
                warn = "" if r["match_ok"] else " ⚠"
                lines.append(
                    f"| {r['name']}{warn} | {r['local_archetype']} | {int(r['n_samples'])} "
                    f"| {cells['cold_irl']} | {cells['rf_dir']} | {cells['ens']} "
                    f"| {r['IRL_minus_RF']:+.3f} |")
            lines.append("")
    p = ROOT / f"writing/t0_archetype_tables{suffix}.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/t0_sweep")
    ap.add_argument("--levels", nargs="+", default=["dev", "pair"])
    args = ap.parse_args()
    name = pathlib.Path(args.root).name.removeprefix("t0_sweep").removeprefix("_")
    suffix = f"_{name}" if name else ""
    fig_composition(args.root, args.levels, suffix)
    md_tables(args.root, args.levels, suffix)


if __name__ == "__main__":
    main()
