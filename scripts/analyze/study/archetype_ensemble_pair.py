# -*- coding: utf-8 -*-
"""Pair-level Ensemble — archetype_ensemble.py の pair-level 版。

dev-level (B-29) で +1.9pt overall AUC 改善した archetype 重み ensemble を pair-level に拡張。
input は archetype_pair_level.py の出力 (k=4 pair_alpha_with_metrics.csv) を使う。

archetype 重みは pair-level の pooled AUC 結果から目視設定:
  A1 (Unfamiliar Dir): cold/warm 拮抗、RF わずか劣 → IRL 寄り (0.55)
  A2 (Familiar / Reject): RF わずか優 → RF 寄り (0.40)
  A3 (Familiar / Accept): cold IRL 優 → IRL 寄り (0.65)
  A4 (Mid): warm IRL 優、cold 劣 → 中庸 (0.50)

入力:
  outputs/analysis/archetype_pair_level_v21/k4_pair_alpha_with_metrics.csv
  outputs/main32_mce_v2_phase2/... pair_predictions.csv
  outputs/main32_mce_kd_warmstart_v2/... pair_predictions.csv

出力:
  outputs/analysis/archetype_ensemble_pair/
    ensemble_per_archetype_auc.csv
    ensemble_overall_auc.csv
    fig_ensemble_auc_by_archetype.pdf

使い方:
  uv run python scripts/analyze/study/archetype_ensemble_pair.py
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_pair_long(cold_dir, warm_dir):
    rows = []
    for w in PERIODS:
        cold = pd.read_csv(cold_dir / f"lstm_baseline/train_0-3m/eval_{w}/pair_predictions.csv")
        warm = pd.read_csv(warm_dir / f"lstm_baseline/train_0-3m/eval_{w}/pair_predictions.csv")
        for df, model, prob_col in [
            (cold, "cold_irl", "irl_dir_prob_calibrated"),
            (warm, "warm_irl", "irl_dir_prob_calibrated"),
            (warm, "rf_dir", "rf_dir_prob"),
        ]:
            d = df[["developer", "directory", "label", prob_col]].copy()
            d.rename(columns={prob_col: "prob"}, inplace=True)
            d["window"] = w
            d["model"] = model
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def pooled_auc_with_ci(labels, probs, rng, n_boot=500) -> Tuple[float, float, float]:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_pair_level_v21/k4_pair_alpha_with_metrics.csv"))
    p.add_argument("--cold-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_v2_phase2"))
    p.add_argument("--warm-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_kd_warmstart_v2"))
    p.add_argument("--out-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_ensemble_pair"))
    p.add_argument("--irl-source", choices=["cold", "warm"], default="cold")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(777)

    print(f"[load] alpha file: {args.alpha_csv}")
    alpha_df = pd.read_csv(args.alpha_csv)
    arch_cols = [c for c in alpha_df.columns if c.startswith("A") and len(c) == 2 and c[1:].isdigit()]
    print(f"  k={len(arch_cols)}, n_pair: {len(alpha_df)}")

    print(f"[load] per-window predictions ...")
    long = load_pair_long(args.cold_dir, args.warm_dir)

    # wide: 各 (dev, dir, window) で cold/warm/rf prob を列に
    wide = long.pivot_table(index=["developer", "directory", "window"],
                              columns="model", values="prob").reset_index()
    label_df = long[long["model"] == "cold_irl"][["developer", "directory", "window", "label"]]
    wide = wide.merge(label_df, on=["developer", "directory", "window"], how="left")
    wide = wide.merge(alpha_df[["developer", "directory"] + arch_cols + ["dominant"]],
                       on=["developer", "directory"], how="inner")
    wide = wide.dropna(subset=["cold_irl", "warm_irl", "rf_dir", "label"])
    print(f"[wide] rows: {len(wide)} (after merge + dropna)")

    irl_col = "cold_irl" if args.irl_source == "cold" else "warm_irl"
    rf_col = "rf_dir"

    # archetype hard weights — pair-level pooled AUC から目視
    # A1: cold 0.73 vs RF 0.68 → IRL 寄り
    # A2: cold 0.75 vs RF 0.77 → RF 寄り
    # A3: cold 0.79 vs RF 0.77 → IRL 寄り
    # A4: cold 0.70 vs RF 0.63 → IRL 寄り
    hard_weights = {"A1": 0.55, "A2": 0.40, "A3": 0.65, "A4": 0.60}

    wide["ens_uniform"] = 0.5 * wide[irl_col] + 0.5 * wide[rf_col]
    wide["w_hard"] = wide["dominant"].map(lambda a: hard_weights.get(a, 0.5))
    wide["ens_hard"] = wide["w_hard"] * wide[irl_col] + (1 - wide["w_hard"]) * wide[rf_col]

    aj_weights = np.array([hard_weights.get(c, 0.5) for c in arch_cols])
    alpha_vals = wide[arch_cols].to_numpy()
    wide["w_soft"] = alpha_vals @ aj_weights
    wide["ens_soft"] = wide["w_soft"] * wide[irl_col] + (1 - wide["w_soft"]) * wide[rf_col]

    wide["ens_oracle"] = np.where(
        wide["label"] == 1,
        np.maximum(wide[irl_col], wide[rf_col]),
        np.minimum(wide[irl_col], wide[rf_col]),
    )

    # per-archetype pooled AUC
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

    # overall
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
    print(f"\n[overall AUC]\n{ov_df.round(3).to_string(index=False)}")

    # 可視化
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"Cold IRL": "#1f77b4", "Warm IRL": "#ff7f0e", "RF_Dir": "#2ca02c",
               "Ensemble (uniform 0.5)": "#9467bd",
               "Ensemble (archetype hard)": "#8c564b",
               "Ensemble (archetype soft)": "#e377c2",
               "Oracle (上限)": "#7f7f7f"}
    width = 0.11
    x = np.arange(len(arch_cols))
    for i, (mcol, mname) in enumerate(models):
        sub = arch_df[arch_df["model"] == mname].set_index("archetype").reindex(arch_cols)
        ax.bar(x + (i - 3) * width, sub["auc"].values, width,
               label=mname, color=colors[mname],
               yerr=[(sub["auc"] - sub["auc_ci_lo"]).values,
                      (sub["auc_ci_hi"] - sub["auc"]).values],
               capsize=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n(n={int(arch_df[arch_df['archetype']==a]['n_samples'].iloc[0])})"
                         for a in arch_cols])
    ax.set_ylabel("Pair-level Pooled AUC (95% CI)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7)
    ax.set_title(f"Pair-level ensemble per archetype (irl_source={args.irl_source})")
    ax.legend(loc="lower right", fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / f"fig_ensemble_pair_{args.irl_source}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
