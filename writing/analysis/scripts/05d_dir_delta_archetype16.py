# -*- coding: utf-8 -*-
"""IRL と RF の AUC 差 (Δ=IRL−RF) が大きい dir の 16 セル分布調査.

設計:
  - AUC 計算は **全ペア** で行う (両 archetype 不要)。dir 数を不当に減らさない。
  - 16 セル占有率は **両 archetype あるペアのみ** で集計し、カバー率を併記。
  - n_pair>=20 dir (= 46 dir) を対象とする (n>=30 だと 23 dir に絞られすぎる)。
  - 末尾で「**dir 内の Δ>0 と特定セル単体での Δ>0 は別物**」を検証するため、
    各 dir の最頻セル (例 A4×P1) **単体** での IRL/RF AUC も比較する。

出力:
  results/archetype16_dir_delta.csv             dir 別 AUC, Δ, base_rate, 占有率カバー, 最頻セル
  results/fig_dir_delta_archetype16_top5.pdf    Δ大 dir × 16 セル占有率グリッド
  results/archetype16_cell_auc_in_top_dirs.csv  Δ大 dir 内の最頻セル単体での IRL/RF AUC 比較
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
MIN_DIR_SIZE = 2  # AUC が計算できる最低限 (両クラスのラベル存在の前提のみ)
N_PICK = 5
ROBUST_N = 20  # 信頼性の参照ライン (この値以上を「安定」と呼ぶ)

CODES = ["A1", "A2", "A3", "A4"]
DEV_LABELS = ["A1", "A2", "A3", "A4"]
PAIR_LABELS = ["P1", "P2", "P3", "P4"]


def safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def pool_metrics(sub: pd.DataFrame) -> dict:
    y, irl, rf, ens = [], [], [], []
    for h in HORIZONS:
        mask = ~sub[f"label_{h}"].isna()
        y.extend(sub.loc[mask, f"label_{h}"].astype(int).tolist())
        irl.extend(sub.loc[mask, f"irl_{h}"].tolist())
        rf.extend(sub.loc[mask, f"rf_{h}"].tolist())
        ens.extend(sub.loc[mask, f"avg_{h}"].tolist())
    return {
        "n_obs": len(y),
        "base_rate": float(np.mean(y)) if y else np.nan,
        "irl_auc": safe_auc(y, irl),
        "rf_auc": safe_auc(y, rf),
        "ens_auc": safe_auc(y, ens),
    }


def cell_share(sub_with_arch: pd.DataFrame) -> np.ndarray:
    mat = np.zeros((4, 4), dtype=float)
    for i, a in enumerate(CODES):
        for j, p in enumerate(CODES):
            mat[i, j] = ((sub_with_arch["archetype"] == a) &
                         (sub_with_arch["pair_archetype"] == p)).sum()
    return mat / mat.sum() if mat.sum() > 0 else mat


def annotate(ax, mat, fmt="{:.2f}"):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            text = "·" if v == 0 else fmt.format(v)
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="black")


def plot_grid(picks_irl, picks_rf, df_arch, out_path):
    n_col = max(len(picks_irl), len(picks_rf))
    fig, axes = plt.subplots(2, n_col, figsize=(3.4 * n_col, 7.8))
    if n_col == 1:
        axes = axes.reshape(2, 1)

    for row_idx, (label, picks, cmap) in enumerate([
        ("IRL 優位 (Δ=IRL−RF > 0)", picks_irl, "Blues"),
        ("RF 優位 (Δ=IRL−RF < 0)", picks_rf, "Reds"),
    ]):
        for col_idx in range(n_col):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(picks):
                ax.axis("off")
                continue
            row = picks[col_idx]
            sub = df_arch[df_arch["directory"] == row["directory"]]
            mat = cell_share(sub)
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(0.35, mat.max()),
                           aspect="auto")
            annotate(ax, mat)
            ax.set_xticks(range(4))
            ax.set_xticklabels(PAIR_LABELS, fontsize=8)
            ax.set_yticks(range(4))
            ax.set_yticklabels(DEV_LABELS, fontsize=8)
            ax.set_xlabel("pair arch", fontsize=9)
            if col_idx == 0:
                ax.set_ylabel("dev arch", fontsize=9)
            ax.set_title(
                f"{row['directory']}\n"
                f"IRL={row['irl_auc']:.2f}  RF={row['rf_auc']:.2f}  Δ={row['delta']:+.2f}\n"
                f"n_all={int(row['n_pair'])}  n_arch={int(row['n_pair_arch'])}  "
                f"br={row['base_rate']:.2f}",
                fontsize=9,
            )
        axes[row_idx, 0].text(-1.15, 1.5, label, rotation=90,
                              fontsize=12, fontweight="bold",
                              va="center", ha="center",
                              transform=axes[row_idx, 0].transAxes)

    fig.suptitle(
        f"Δ=IRL−RF が大きい dir 各 {N_PICK} 件の 16 セル占有率\n"
        f"(全 dir 対象, n_pair>={ROBUST_N} で絞り込み, "
        f"AUC は全ペアで計算, 占有率は両 archetype あるペアで計算)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)

    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv")
    pa = pa.rename(columns={"dominant": "pair_archetype"})[
        ["developer", "directory", "pair_archetype"]
    ]
    full = pred.merge(pa, on=["developer", "directory"], how="left")
    print(f"全ペア: {len(full)} (うち両 archetype あり: "
          f"{full.dropna(subset=['archetype','pair_archetype']).shape[0]})")

    print(f"全 dir 数 (元データ): {full['directory'].nunique()}")
    dir_sizes = full.groupby("directory").size()
    keep_dirs = dir_sizes[dir_sizes >= MIN_DIR_SIZE].index
    sub = full[full["directory"].isin(keep_dirs)].copy()
    print(f"n_pair>={MIN_DIR_SIZE} dir: {len(keep_dirs)}, 対象ペア: {len(sub)}")

    rows = []
    for d in sub["directory"].unique():
        s_all = sub[sub["directory"] == d]
        s_arch = s_all.dropna(subset=["archetype", "pair_archetype"])
        m = pool_metrics(s_all)
        if m["irl_auc"] is None or m["rf_auc"] is None:
            continue
        cs = cell_share(s_arch) if len(s_arch) else np.zeros((4, 4))
        if cs.sum() > 0:
            i_max, j_max = np.unravel_index(cs.argmax(), cs.shape)
            top_cell = f"{CODES[i_max]}×{CODES[j_max]}"
            top_share = float(cs[i_max, j_max])
        else:
            top_cell, top_share = "n/a", np.nan
        rows.append({
            "directory": d,
            "n_pair": len(s_all),
            "n_pair_arch": len(s_arch),
            "arch_coverage": len(s_arch) / max(len(s_all), 1),
            "base_rate": m["base_rate"],
            "irl_auc": m["irl_auc"],
            "rf_auc": m["rf_auc"],
            "ens_auc": m["ens_auc"],
            "delta": m["irl_auc"] - m["rf_auc"],
            "top_cell": top_cell,
            "top_cell_share": top_share,
        })
    dir_df = pd.DataFrame(rows).sort_values("delta", ascending=False)
    dir_df.to_csv(OUT / "archetype16_dir_delta.csv", index=False)
    print(f"saved: {OUT/'archetype16_dir_delta.csv'} ({len(dir_df)} dir で AUC 算出可)")

    # ── Δ vs n_pair 散布図 (信頼性の視覚化) ──
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(dir_df["n_pair"], dir_df["delta"], alpha=0.5,
               s=30, color="steelblue", edgecolors="black", linewidths=0.4)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax.axvline(ROBUST_N, color="red", linestyle="--", linewidth=1,
               label=f"n_pair={ROBUST_N} (信頼性ライン)")
    # 極端な点にラベル
    extremes = pd.concat([
        dir_df.head(8), dir_df.tail(8),
        dir_df[dir_df["n_pair"] >= 50].head(5),
        dir_df[dir_df["n_pair"] >= 50].tail(5),
    ]).drop_duplicates(subset=["directory"])
    for _, r in extremes.iterrows():
        ax.annotate(r["directory"], (r["n_pair"], r["delta"]),
                    fontsize=7, alpha=0.7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("n_pair (log scale)", fontsize=11)
    ax.set_ylabel("Δ = IRL AUC − RF AUC", fontsize=11)
    ax.set_title(f"Δ=IRL−RF vs dir サイズ (全 {len(dir_df)} dir)\n"
                 f"n_pair が小さい dir では Δ が極端な値を取りやすい (ノイズ)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_dir_delta_vs_n.pdf", format="pdf",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUT/'fig_dir_delta_vs_n.pdf'}")

    # ── 無フィルタの Top/Bottom ──
    print(f"\n=== 全 {len(dir_df)} dir Δ 降順 (上位 15) ===")
    print(dir_df.head(15)[["directory", "n_pair", "n_pair_arch", "base_rate",
                            "irl_auc", "rf_auc", "delta", "top_cell"]]
          .to_string(index=False))
    print(f"\n=== 全 {len(dir_df)} dir Δ 昇順 (下位 15) ===")
    print(dir_df.tail(15).iloc[::-1][["directory", "n_pair", "n_pair_arch",
                                       "base_rate", "irl_auc", "rf_auc",
                                       "delta", "top_cell"]]
          .to_string(index=False))

    # ── n_pair>=ROBUST_N での Top/Bottom (グリッド描画用) ──
    robust_df = dir_df[dir_df["n_pair"] >= ROBUST_N]
    print(f"\n=== n_pair>={ROBUST_N} の {len(robust_df)} dir で絞り込んだ Top/Bottom ===")
    picks_irl = robust_df.head(N_PICK).to_dict("records")
    picks_rf = robust_df.tail(N_PICK).iloc[::-1].to_dict("records")

    print("\n=== IRL 優位 Top {} ===".format(N_PICK))
    for r in picks_irl:
        print(f"  {r['directory']:28s} Δ={r['delta']:+.3f}  "
              f"IRL={r['irl_auc']:.3f} RF={r['rf_auc']:.3f}  "
              f"n_all={int(r['n_pair']):3d}/n_arch={int(r['n_pair_arch']):3d}  "
              f"br={r['base_rate']:.2f}  top={r['top_cell']} "
              f"({r['top_cell_share']:.0%})")
    print("\n=== RF 優位 Top {} ===".format(N_PICK))
    for r in picks_rf:
        print(f"  {r['directory']:28s} Δ={r['delta']:+.3f}  "
              f"IRL={r['irl_auc']:.3f} RF={r['rf_auc']:.3f}  "
              f"n_all={int(r['n_pair']):3d}/n_arch={int(r['n_pair_arch']):3d}  "
              f"br={r['base_rate']:.2f}  top={r['top_cell']} "
              f"({r['top_cell_share']:.0%})")

    plot_grid(picks_irl, picks_rf, sub,
              OUT / f"fig_dir_delta_archetype16_top{N_PICK}.pdf")

    # ── 検証: dir 内の Δ>0 は本当に「特定セル単体での Δ>0」に駆動されているか? ──
    print("\n=== 検証: 各 Δ大 dir で『最頻セル単体』および『A4×P1, A4×P2』単体での IRL/RF AUC ===")
    cells_to_check = [("A4", "A1"), ("A4", "A2"), ("A3", "A3"), ("A2", "A4")]
    val_rows = []
    targets = picks_irl + picks_rf
    for r in targets:
        s_arch = sub[(sub["directory"] == r["directory"])].dropna(
            subset=["archetype", "pair_archetype"]
        )
        for (a, p) in cells_to_check:
            cell_sub = s_arch[(s_arch["archetype"] == a)
                              & (s_arch["pair_archetype"] == p)]
            if len(cell_sub) < 3:
                continue
            m = pool_metrics(cell_sub)
            val_rows.append({
                "directory": r["directory"],
                "dir_delta": r["delta"],
                "cell": f"{a}×{p}",
                "n_pair_cell": len(cell_sub),
                "n_obs_cell": m["n_obs"],
                "base_rate_cell": m["base_rate"],
                "irl_auc_cell": m["irl_auc"],
                "rf_auc_cell": m["rf_auc"],
                "delta_cell": (m["irl_auc"] - m["rf_auc"])
                if (m["irl_auc"] is not None and m["rf_auc"] is not None)
                else None,
            })
    val_df = pd.DataFrame(val_rows)
    val_df.to_csv(OUT / "archetype16_cell_auc_in_top_dirs.csv", index=False)
    print(f"saved: {OUT/'archetype16_cell_auc_in_top_dirs.csv'}")
    print(val_df.to_string(index=False))

    # 集計: A4×P1 セル単体での Δ の符号は dir 全体の Δ と一致するか?
    print("\n=== A4×P1 セル単体の Δ (dir 内, n>=10) vs dir 全体の Δ ===")
    a4p1 = val_df[(val_df["cell"] == "A4×A1") & (val_df["n_pair_cell"] >= 10)]
    if len(a4p1) > 0:
        print(a4p1[["directory", "dir_delta", "n_pair_cell",
                    "irl_auc_cell", "rf_auc_cell", "delta_cell"]].to_string(index=False))
    else:
        print("(対象セル該当なし)")


if __name__ == "__main__":
    main()
