# -*- coding: utf-8 -*-
"""IRL × RF アンサンブル prototype (B-28 の補完性発見を即値検証)。

B-28 の知見:
  - A2 (Strict Reviewer-Only): cold IRL が AUC で RF を +25pt 圧倒
  - A4 (Saturated Active): per-dev では IRL 勝ち、pooled では RF わずか優位
  - A1 (Sporadic Newcomer): RF わずか優位
  - A3 (Approving Veteran): 互角

→ archetype 別にブレンド重みを変えることで両方の優位を取り込めるか検証。

検証する 4 種類の ensemble:
  1. uniform:      w_irl = 0.5 (archetype 関係なく半々)
  2. archetype_hard: dominant archetype で固定重み (A2→0.8, A1→0.2, A3/A4→0.5)
  3. archetype_soft: α 加重で重み計算 (A2 寄りなら IRL 寄り、A1 寄りなら RF 寄り)
  4. oracle:       per-dev で IRL/RF どちらが良いか label を使って後決め (上限値)

入力:
  outputs/analysis/archetype_irl_rf_deep/k4_dev_alpha_with_metrics.csv (α + 性能)
  outputs/main32_mce_v2_phase2/lstm_baseline/train_0-3m/eval_*/pair_predictions.csv (per-window prob)
  outputs/main32_mce_kd_warmstart_v2/lstm_baseline/train_0-3m/eval_*/pair_predictions.csv (RF prob 同梱)

出力:
  outputs/analysis/archetype_ensemble/
    ensemble_per_archetype_auc.csv   各 ensemble の archetype 別 pooled AUC + CI
    ensemble_overall_auc.csv         全 dev pool での AUC 比較
    fig_ensemble_auc_by_archetype.pdf

使い方:
  uv run python scripts/analyze/study/archetype_ensemble.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_long(cold_dir: pathlib.Path, warm_dir: pathlib.Path) -> pd.DataFrame:
    """全 dev × 4 window × {cold IRL, warm IRL, RF} の (label, prob) を long 形式で。"""
    rows = []
    for w in PERIODS:
        cold = pd.read_csv(cold_dir / f"lstm_baseline/train_0-3m/eval_{w}/pair_predictions.csv")
        warm = pd.read_csv(warm_dir / f"lstm_baseline/train_0-3m/eval_{w}/pair_predictions.csv")
        for df, model, prob_col in [
            (cold, "cold_irl", "irl_dir_prob_calibrated"),
            (warm, "warm_irl", "irl_dir_prob_calibrated"),
            (warm, "rf_dir",   "rf_dir_prob"),
        ]:
            agg = df.groupby("developer").agg(
                prob=(prob_col, "max"),
                label=("label", "max"),
            ).reset_index()
            agg["window"] = w
            agg["model"] = model
            rows.append(agg)
    return pd.concat(rows, ignore_index=True)


def pooled_auc_with_ci(labels: np.ndarray, probs: np.ndarray,
                       rng: np.random.Generator, n_boot: int = 1000) -> Tuple[float, float, float]:
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        return np.nan, np.nan, np.nan
    boots = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bl = labels[idx]
        if len(np.unique(bl)) < 2:
            continue
        try:
            boots.append(roc_auc_score(bl, probs[idx]))
        except ValueError:
            continue
    lo = float(np.percentile(boots, 2.5)) if len(boots) > 50 else np.nan
    hi = float(np.percentile(boots, 97.5)) if len(boots) > 50 else np.nan
    return auc, lo, hi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_irl_rf_deep/k4_dev_alpha_with_metrics.csv"))
    p.add_argument("--cold-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_v2_phase2"))
    p.add_argument("--warm-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_kd_warmstart_v2"))
    p.add_argument("--out-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_ensemble"))
    p.add_argument("--irl-source", choices=["cold", "warm"], default="cold",
                   help="ensemble 対象の IRL (cold=v2_phase2, warm=warm_v2)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(777)

    print(f"[load] alpha file: {args.alpha_csv}")
    alpha_df = pd.read_csv(args.alpha_csv, index_col=0)
    arch_cols = [c for c in alpha_df.columns if c.startswith("A") and c[1:].isdigit()]
    k = len(arch_cols)
    print(f"  k={k}, devs in alpha file: {len(alpha_df)}")

    print(f"[load] per-window predictions ...")
    long = load_long(args.cold_dir, args.warm_dir)

    # archetype 情報を merge
    long = long.merge(alpha_df[arch_cols + ["dominant"]].reset_index().rename(
        columns={"index": "developer"}), on="developer", how="inner"
    )
    n_devs = long["developer"].nunique()
    print(f"  merged: {len(long)} rows, {n_devs} devs with archetype")

    # wide: 各 (dev, window) で cold/warm/rf prob を列に持つ
    wide = long.pivot_table(index=["developer", "window"], columns="model",
                              values="prob").reset_index()
    label_df = long[long["model"] == "cold_irl"][["developer", "window", "label"]]
    wide = wide.merge(label_df, on=["developer", "window"], how="left")
    wide = wide.merge(alpha_df[arch_cols + ["dominant"]].reset_index().rename(
        columns={"index": "developer"}), on="developer", how="left"
    )
    wide = wide.dropna(subset=["cold_irl", "warm_irl", "rf_dir", "label"])
    print(f"[wide] rows: {len(wide)} (after dropna)")

    # ── ensemble 計算 ──
    irl_col = "cold_irl" if args.irl_source == "cold" else "warm_irl"
    rf_col = "rf_dir"

    # 1. uniform
    wide["ens_uniform"] = 0.5 * wide[irl_col] + 0.5 * wide[rf_col]

    # 2. archetype_hard: dominant archetype で固定重み
    #    B-28 から: A1=RF寄り(0.25), A2=IRL寄り(0.75), A3=互角(0.5), A4=やや RF 寄り(0.45)
    hard_weights = {"A1": 0.25, "A2": 0.75, "A3": 0.50, "A4": 0.45}
    # k != 4 でも汎用化: 持っていない archetype は 0.5
    wide["w_hard"] = wide["dominant"].map(lambda a: hard_weights.get(a, 0.5))
    wide["ens_hard"] = wide["w_hard"] * wide[irl_col] + (1 - wide["w_hard"]) * wide[rf_col]

    # 3. archetype_soft: α 加重で重み計算
    #    w_soft_i = Σ_j α_ij * w_hard[Aj]  (= dev の archetype 帰属の重み付き平均)
    aj_weights = np.array([hard_weights.get(c, 0.5) for c in arch_cols])
    alpha_vals = wide[arch_cols].to_numpy()  # (n, k)
    wide["w_soft"] = alpha_vals @ aj_weights
    wide["ens_soft"] = wide["w_soft"] * wide[irl_col] + (1 - wide["w_soft"]) * wide[rf_col]

    # 4. oracle: per-dev で IRL/RF どちらが良いか label を見て決定 (上限)
    #    各 dev × window で max(prob_irl, prob_rf) を label が 1 なら大きい方、0 なら小さい方
    #    実用的には min(|prob - label|) を選ぶ → oracle blend
    wide["ens_oracle"] = np.where(
        wide["label"] == 1,
        np.maximum(wide[irl_col], wide[rf_col]),
        np.minimum(wide[irl_col], wide[rf_col]),
    )

    # ── archetype 別 pooled AUC ──
    rows = []
    models = [("cold_irl", "Cold IRL"), ("warm_irl", "Warm IRL"), ("rf_dir", "RF_Dir"),
               ("ens_uniform", "Ensemble (uniform 0.5)"),
               ("ens_hard",    "Ensemble (archetype hard)"),
               ("ens_soft",    "Ensemble (archetype soft)"),
               ("ens_oracle",  "Oracle (上限)")]

    for arch in arch_cols:
        sub = wide[wide["dominant"] == arch]
        for mcol, mname in models:
            auc, lo, hi = pooled_auc_with_ci(
                sub["label"].to_numpy().astype(int),
                sub[mcol].to_numpy().astype(float), rng,
            )
            rows.append({"archetype": arch, "model": mname,
                          "n_samples": len(sub), "auc": auc,
                          "auc_ci_lo": lo, "auc_ci_hi": hi})
    arch_df = pd.DataFrame(rows)
    arch_df.to_csv(args.out_dir / "ensemble_per_archetype_auc.csv", index=False)
    print(f"\n[per-archetype AUC]\n{arch_df.round(3).to_string(index=False)}")

    # ── overall pooled AUC (全 archetype 合算) ──
    overall_rows = []
    for mcol, mname in models:
        auc, lo, hi = pooled_auc_with_ci(
            wide["label"].to_numpy().astype(int),
            wide[mcol].to_numpy().astype(float), rng,
        )
        overall_rows.append({"model": mname, "n_samples": len(wide),
                              "auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi})
    ov_df = pd.DataFrame(overall_rows)
    ov_df.to_csv(args.out_dir / "ensemble_overall_auc.csv", index=False)
    print(f"\n[overall AUC pooled all archetypes]\n{ov_df.round(3).to_string(index=False)}")

    # ── 可視化 ──
    fig, ax = plt.subplots(figsize=(9, 5.5))
    arch_list = arch_cols
    colors = {"Cold IRL": "#1f77b4", "Warm IRL": "#ff7f0e", "RF_Dir": "#2ca02c",
              "Ensemble (uniform 0.5)": "#9467bd",
              "Ensemble (archetype hard)": "#8c564b",
              "Ensemble (archetype soft)": "#e377c2",
              "Oracle (上限)": "#7f7f7f"}
    width = 0.11
    x = np.arange(len(arch_list))
    for i, (mcol, mname) in enumerate(models):
        sub = arch_df[arch_df["model"] == mname].set_index("archetype").reindex(arch_list)
        ax.bar(x + (i - 3) * width, sub["auc"].values, width,
               label=mname, color=colors[mname],
               yerr=[(sub["auc"] - sub["auc_ci_lo"]).values,
                      (sub["auc_ci_hi"] - sub["auc"]).values],
               capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(n={int(arch_df[arch_df['archetype']==a]['n_samples'].iloc[0])})"
                         for a in arch_list])
    ax.set_ylabel("Pooled AUC (95% CI)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7)
    ax.set_title(f"Ensemble per archetype (irl_source={args.irl_source})")
    ax.legend(loc="lower right", fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"fig_ensemble_auc_by_archetype_{args.irl_source}.pdf",
                 format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
