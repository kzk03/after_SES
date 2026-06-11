# -*- coding: utf-8 -*-
"""予測区間 × 16 アーキタイプセルで IRL の独自貢献を閾値フリーで定量化.

§1 (disagreement, horizon 別) と §6.5 (16 セル, horizon プール) を交差させ、
4 予測区間 × 16 セル = 64 マスで以下の閾値フリー指標を計算:

  - Δ AUC = IRL_AUC − RF_AUC          (IRL 単独の rank 品質)
  - Ens lift AUC = Ens_AUC − RF_AUC   (IRL を RF にブレンドした時の rank 改善)
  - Δ AP  = IRL_AP − RF_AP            (正例側の rank 品質; AUC より recall@top に敏感)
  - Ens lift AP = Ens_AP − RF_AP

cold IRL は未校正 (RULES 準拠) のため確率の絶対値や閾値 0.5 は使わない。
すべて rank-based 指標で評価する。

入力:
  data/pair_trajectories.csv
  data/pair_archetype.csv

出力:
  results/horizon_archetype_cell_stats.csv
  results/horizon_archetype_top_irl_contrib.csv
  results/fig_horizon_archetype_auc_delta.pdf   (A) Δ AUC = IRL − RF
  results/fig_horizon_archetype_ens_lift_auc.pdf (A) Ens AUC − RF AUC
  results/fig_horizon_archetype_ap_delta.pdf    (C) Δ AP = IRL − RF
  results/fig_horizon_archetype_ens_lift_ap.pdf (C) Ens AP − RF AP
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
HORIZON_LABEL_JP = {
    "0-3m": "0-3 か月",
    "3-6m": "3-6 か月",
    "6-9m": "6-9 か月",
    "9-12m": "9-12 か月",
}
MIN_CELL_N = 20  # rank 系指標を計算するセル最小ペア数
MIN_POS_N = 2     # AP / AUC が定義可能な最小正例数 (両クラス必要)

DEV_LABELS = ["A1\n低活動新人", "A2\nレビュー専念", "A3\n高承諾ベテラン", "A4\n常時活動"]
PAIR_LABELS = ["P1\n未経験", "P2\n経験低承諾", "P3\n経験高承諾", "P4\n中程度関与"]
CODES = ["A1", "A2", "A3", "A4"]


def safe_auc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_ap(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, s))


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """(developer, directory, dev_arch, pair_arch) × horizon の long 形式."""
    rows = []
    for h in HORIZONS:
        sub = df[["developer", "directory", "archetype", "pair_archetype",
                  f"label_{h}", f"irl_{h}", f"rf_{h}", f"avg_{h}"]].copy()
        sub = sub.rename(columns={
            f"label_{h}": "label",
            f"irl_{h}": "irl",
            f"rf_{h}": "rf",
            f"avg_{h}": "ens",
        })
        sub["horizon"] = h
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    long = long.dropna(subset=["label", "irl", "rf"]).copy()
    long["label"] = long["label"].astype(int)
    return long


def per_cell_stats(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        sub_h = long[long["horizon"] == h]
        for a in CODES:
            for p in CODES:
                sub = sub_h[(sub_h["archetype"] == a) & (sub_h["pair_archetype"] == p)]
                n = len(sub)
                n_pos = int(sub["label"].sum())
                base = float(sub["label"].mean()) if n > 0 else np.nan
                row = {
                    "horizon": h, "dev_arch": a, "pair_arch": p,
                    "n": n, "n_pos": n_pos, "base_rate": base,
                    "irl_auc": np.nan, "rf_auc": np.nan, "ens_auc": np.nan,
                    "delta_auc": np.nan, "ens_lift_auc": np.nan,
                    "irl_ap": np.nan, "rf_ap": np.nan, "ens_ap": np.nan,
                    "delta_ap": np.nan, "ens_lift_ap": np.nan,
                }
                if n >= MIN_CELL_N and n_pos >= MIN_POS_N and (n - n_pos) >= MIN_POS_N:
                    y = sub["label"].values
                    irl, rf, ens = sub["irl"].values, sub["rf"].values, sub["ens"].values
                    row["irl_auc"] = safe_auc(y, irl)
                    row["rf_auc"] = safe_auc(y, rf)
                    row["ens_auc"] = safe_auc(y, ens)
                    row["delta_auc"] = row["irl_auc"] - row["rf_auc"]
                    row["ens_lift_auc"] = row["ens_auc"] - row["rf_auc"]
                    # AP の baseline はクラス比率に応じて変動するため、生 AP の差で比較
                    row["irl_ap"] = safe_ap(y, irl)
                    row["rf_ap"] = safe_ap(y, rf)
                    row["ens_ap"] = safe_ap(y, ens)
                    row["delta_ap"] = row["irl_ap"] - row["rf_ap"]
                    row["ens_lift_ap"] = row["ens_ap"] - row["rf_ap"]
                rows.append(row)
    return pd.DataFrame(rows)


def annotate_cells(ax, mat: np.ndarray, ns: np.ndarray, fmt: str,
                   max_abs: float | None = None) -> None:
    if max_abs is None:
        max_abs = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            n = int(ns[i, j]) if not np.isnan(ns[i, j]) else 0
            v = mat[i, j]
            if n == 0 or np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center",
                        fontsize=8, color="gray")
                continue
            color = "white" if max_abs > 0 and abs(v) > 0.6 * max_abs else "black"
            ax.text(j, i, f"{fmt.format(v)}\nn={n}",
                    ha="center", va="center", fontsize=8, color=color)


def plot_grid(stats: pd.DataFrame, value_col: str,
              out_path: pathlib.Path, title: str, cbar_label: str,
              fmt: str = "{:+.2f}") -> None:
    """4 horizon × 4×4 ヒートマップを 1 枚にまとめる."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2))
    mats, mats_n = [], []
    for h in HORIZONS:
        m = np.full((4, 4), np.nan)
        m_n = np.full((4, 4), np.nan)
        for i, a in enumerate(CODES):
            for j, p in enumerate(CODES):
                row = stats[(stats["horizon"] == h) & (stats["dev_arch"] == a)
                            & (stats["pair_arch"] == p)]
                if not row.empty:
                    m[i, j] = row[value_col].values[0]
                    m_n[i, j] = row["n"].values[0]
        mats.append(m)
        mats_n.append(m_n)
    vals = np.concatenate([m.ravel() for m in mats])
    vmax = float(np.nanmax(np.abs(vals))) if np.isfinite(vals).any() else 0.05
    vmax = max(vmax, 0.02)

    for ax, h, m, m_n in zip(axes, HORIZONS, mats, mats_n):
        im = ax.imshow(m, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_xticklabels(PAIR_LABELS, fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(DEV_LABELS, fontsize=9)
        ax.set_title(HORIZON_LABEL_JP[h], fontsize=12)
        annotate_cells(ax, m, m_n, fmt=fmt, max_abs=vmax)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label(cbar_label, fontsize=10)
    fig.suptitle(title, fontsize=14, y=1.02)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"saved: {out_path}")


def extract_top(stats: pd.DataFrame, value_col: str, k: int = 10,
                ascending: bool = False) -> pd.DataFrame:
    sub = stats.dropna(subset=[value_col]).copy()
    sub = sub.sort_values(value_col, ascending=ascending)
    cols = ["horizon", "dev_arch", "pair_arch", "n", "n_pos", "base_rate",
            "irl_auc", "rf_auc", "ens_auc", "delta_auc", "ens_lift_auc",
            "irl_ap", "rf_ap", "ens_ap", "delta_ap", "ens_lift_ap"]
    return sub[cols].head(k)


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)

    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv")
    pa = pa.rename(columns={"dominant": "pair_archetype"})[
        ["developer", "directory", "pair_archetype"]
    ]
    df = pred.merge(pa, on=["developer", "directory"], how="left")
    df = df.dropna(subset=["archetype", "pair_archetype"]).copy()
    print(f"n_pair after arch merge: {len(df)}")

    long = build_long(df)
    print(f"n_obs across 4 horizons: {len(long)}")

    stats = per_cell_stats(long)
    stats.to_csv(OUT / "horizon_archetype_cell_stats.csv", index=False)
    print(f"saved: {OUT/'horizon_archetype_cell_stats.csv'}")

    # 古い閾値依存ファイルがあれば削除
    for old in ["horizon_archetype_top_rescue_cells.csv",
                "fig_horizon_archetype_irl_rescue.pdf"]:
        p = OUT / old
        if p.exists():
            p.unlink()
            print(f"removed legacy: {p}")

    # (A) AUC ベース 2 枚
    plot_grid(stats, "delta_auc",
              OUT / "fig_horizon_archetype_auc_delta.pdf",
              "予測区間 × 16 アーキタイプセル別 Δ AUC (IRL − RF)",
              "Δ AUC = IRL − RF (正 = IRL alone が rank で優位、n≥20 のセルのみ)",
              fmt="{:+.2f}")
    plot_grid(stats, "ens_lift_auc",
              OUT / "fig_horizon_archetype_ens_lift_auc.pdf",
              "予測区間 × 16 アーキタイプセル別 Ensemble lift over RF (AUC)",
              "Ens AUC − RF AUC (正 = IRL を混ぜると rank が改善、n≥20)",
              fmt="{:+.2f}")
    # (C) AP ベース 2 枚
    plot_grid(stats, "delta_ap",
              OUT / "fig_horizon_archetype_ap_delta.pdf",
              "予測区間 × 16 アーキタイプセル別 Δ AP (IRL − RF)",
              "Δ AP = IRL − RF (正 = IRL alone の正例 rank 品質が優位、n≥20)",
              fmt="{:+.2f}")
    plot_grid(stats, "ens_lift_ap",
              OUT / "fig_horizon_archetype_ens_lift_ap.pdf",
              "予測区間 × 16 アーキタイプセル別 Ensemble lift over RF (AP)",
              "Ens AP − RF AP (正 = IRL を混ぜると正例 rank 品質が改善、n≥20)",
              fmt="{:+.2f}")

    # トップ抽出
    top_delta_auc = extract_top(stats, "delta_auc", 10, ascending=False)
    top_lift_auc = extract_top(stats, "ens_lift_auc", 10, ascending=False)
    top_delta_ap = extract_top(stats, "delta_ap", 10, ascending=False)
    top_lift_ap = extract_top(stats, "ens_lift_ap", 10, ascending=False)
    pd.concat({
        "delta_auc": top_delta_auc,
        "ens_lift_auc": top_lift_auc,
        "delta_ap": top_delta_ap,
        "ens_lift_ap": top_lift_ap,
    }, names=["criterion"]).to_csv(OUT / "horizon_archetype_top_irl_contrib.csv")
    print(f"saved: {OUT/'horizon_archetype_top_irl_contrib.csv'}")

    print("\n=== Top 5 Δ AUC (IRL alone 優位) ===")
    print(top_delta_auc.head(5).to_string(index=False))
    print("\n=== Top 5 Ens lift AUC (IRL を混ぜると AUC 改善) ===")
    print(top_lift_auc.head(5).to_string(index=False))
    print("\n=== Top 5 Δ AP (IRL alone 正例 rank 優位) ===")
    print(top_delta_ap.head(5).to_string(index=False))
    print("\n=== Top 5 Ens lift AP (IRL を混ぜると AP 改善) ===")
    print(top_lift_ap.head(5).to_string(index=False))

    # 集約サマリ (有効セルのみ)
    valid = stats.dropna(subset=["delta_auc"]).copy()
    print(f"\n=== 有効セル (n≥{MIN_CELL_N} かつ両クラス≥{MIN_POS_N}): "
          f"{len(valid)} / 64 ===")
    for col, label in [("delta_auc", "Δ AUC"),
                       ("ens_lift_auc", "Ens lift AUC"),
                       ("delta_ap", "Δ AP"),
                       ("ens_lift_ap", "Ens lift AP")]:
        pos = (valid[col] > 0).sum()
        neg = (valid[col] < 0).sum()
        med = valid[col].median()
        print(f"  {label:18s}: 正 {pos:>2d} / 負 {neg:>2d} (median={med:+.3f}, "
              f"max={valid[col].max():+.3f}, min={valid[col].min():+.3f})")

    # 予測区間別の集約
    print("\n=== 予測区間別の中央値 (有効セルのみ) ===")
    agg = valid.groupby("horizon").agg(
        n_cells=("delta_auc", "size"),
        delta_auc_med=("delta_auc", "median"),
        ens_lift_auc_med=("ens_lift_auc", "median"),
        delta_ap_med=("delta_ap", "median"),
        ens_lift_ap_med=("ens_lift_ap", "median"),
    ).reset_index()
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
