# -*- coding: utf-8 -*-
"""Pair-level Ensemble — archetype 重みを dev-level KFold で grid search 最適化。

既存 archetype_ensemble_pair.py は archetype hard 重み (A1=0.55/A2=0.40/A3=0.65/A4=0.60)
を目視で設定していたが、cold-only データで uniform 0.5 と同等 (overall 0.811) になり
最適化の余地が残っていた。

本スクリプトは dev レベル 5-fold CV (pair 内 leakage 防止) で archetype 別に
最適 w を grid search し、validation 上の pooled AUC で評価する。

入出力:
  入力: --alpha-csv (archetype_pair_level の k=4 出力)
        --cold-dir / --warm-dir (pair_predictions.csv)
  出力: --out-dir/
        ensemble_overall_auc_cv.csv   ... model ごとの CV mean AUC
        ensemble_per_archetype_auc_cv.csv
        optimized_weights_per_fold.csv ... fold ごとの最適 w
        optimized_weights_summary.csv  ... archetype 別 w の mean ± std
        fig_ensemble_pair_gridsearch.pdf

使い方:
  uv run python scripts/analyze/study/archetype_ensemble_pair_gridsearch.py \
      --alpha-csv outputs/analysis/archetype_pair_level_coldonly/k4_pair_alpha_with_metrics.csv \
      --cold-dir outputs/main32_mce_v2_phase2 \
      --warm-dir outputs/main32_mce_kd_warmstart_v2 \
      --out-dir outputs/analysis/archetype_ensemble_pair_gridsearch_coldonly
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Dict, List

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]
W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
RANDOM_SEED = 777


def load_pair_long(cold_dir: pathlib.Path, warm_dir: pathlib.Path) -> pd.DataFrame:
    rows = []
    for w in PERIODS:
        cold = pd.read_csv(cold_dir / f"lstm_baseline/train_0-3m/eval_{w}/pair_predictions.csv")
        warm = pd.read_csv(warm_dir / f"lstm_baseline/train_0-3m/eval_{w}/pair_predictions.csv")
        # cold pipeline の rf_dir_prob を主とする (warm pipeline ではなく) — handover §1-7 参照
        for df, model, prob_col in [
            (cold, "cold_irl", "irl_dir_prob_calibrated"),
            (warm, "warm_irl", "irl_dir_prob_calibrated"),
            (cold, "rf_dir", "rf_dir_prob"),
        ]:
            if prob_col not in df.columns:
                continue
            d = df[["developer", "directory", "label", prob_col]].copy()
            d.rename(columns={prob_col: "prob"}, inplace=True)
            d["window"] = w
            d["model"] = model
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return np.nan


def grid_search_weight(labels: np.ndarray, irl: np.ndarray, rf: np.ndarray) -> float:
    """archetype 内 train pair で pooled AUC を最大化する w を返す。"""
    best_w, best_auc = 0.5, -np.inf
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return 0.5
    for w in W_GRID:
        auc = safe_auc(labels, w * irl + (1.0 - w) * rf)
        if not np.isnan(auc) and auc > best_auc:
            best_auc, best_w = auc, float(w)
    return best_w


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_pair_level_coldonly/k4_pair_alpha_with_metrics.csv"))
    p.add_argument("--cold-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_v2_phase2"))
    p.add_argument("--warm-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_kd_warmstart_v2"))
    p.add_argument("--out-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_ensemble_pair_gridsearch_coldonly"))
    p.add_argument("--irl-source", choices=["cold", "warm"], default="cold")
    p.add_argument("--n-folds", type=int, default=5)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    alpha_df = pd.read_csv(args.alpha_csv)
    arch_cols = [c for c in alpha_df.columns if c.startswith("A") and len(c) == 2 and c[1:].isdigit()]
    print(f"[load] alpha: k={len(arch_cols)}, n_pair_active={len(alpha_df)}")

    long = load_pair_long(args.cold_dir, args.warm_dir)
    wide = long.pivot_table(index=["developer", "directory", "window"],
                            columns="model", values="prob").reset_index()
    label_df = long[long["model"] == "cold_irl"][["developer", "directory", "window", "label"]]
    wide = wide.merge(label_df, on=["developer", "directory", "window"], how="left")
    wide = wide.merge(alpha_df[["developer", "directory"] + arch_cols + ["dominant"]],
                      on=["developer", "directory"], how="inner")
    wide = wide.dropna(subset=["cold_irl", "warm_irl", "rf_dir", "label"]).reset_index(drop=True)
    print(f"[merge] rows={len(wide)} (pair × window)")

    irl_col = "cold_irl" if args.irl_source == "cold" else "warm_irl"

    # dev レベルで KFold (pair leakage 防止)
    devs = wide["developer"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(devs)
    kf = KFold(n_splits=args.n_folds, shuffle=False)

    fold_weights: List[Dict[str, float]] = []
    fold_arch_auc: List[Dict[str, Dict[str, float]]] = []
    fold_overall_auc: List[Dict[str, float]] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(devs)):
        tr_devs = set(devs[tr_idx])
        va_devs = set(devs[va_idx])
        tr = wide[wide["developer"].isin(tr_devs)]
        va = wide[wide["developer"].isin(va_devs)]
        print(f"\n[fold {fold_idx}] train_dev={len(tr_devs)}, val_dev={len(va_devs)}, "
              f"train_rows={len(tr)}, val_rows={len(va)}")

        # archetype 別に train で w を最適化
        best_w = {}
        for arch in arch_cols:
            sub = tr[tr["dominant"] == arch]
            best_w[arch] = grid_search_weight(
                sub["label"].to_numpy().astype(int),
                sub[irl_col].to_numpy().astype(float),
                sub["rf_dir"].to_numpy().astype(float),
            )
        # archetype 横断の overall best_w (hard 比較用)
        best_w["__overall__"] = grid_search_weight(
            tr["label"].to_numpy().astype(int),
            tr[irl_col].to_numpy().astype(float),
            tr["rf_dir"].to_numpy().astype(float),
        )
        fold_weights.append({"fold": fold_idx, **best_w})
        print(f"  best_w: " + ", ".join([f"{k}={v:.2f}" for k, v in best_w.items()]))

        # val で各 ensemble を評価
        v_label = va["label"].to_numpy().astype(int)
        v_irl = va[irl_col].to_numpy().astype(float)
        v_rf = va["rf_dir"].to_numpy().astype(float)
        v_dom = va["dominant"].to_numpy()
        v_alpha = va[arch_cols].to_numpy()

        # uniform / target overall / archetype hard / archetype soft
        ens_uniform = 0.5 * v_irl + 0.5 * v_rf
        ens_overall_opt = best_w["__overall__"] * v_irl + (1 - best_w["__overall__"]) * v_rf
        w_hard_vec = np.array([best_w[d] for d in v_dom])
        ens_hard = w_hard_vec * v_irl + (1 - w_hard_vec) * v_rf
        aj = np.array([best_w[c] for c in arch_cols])
        w_soft_vec = v_alpha @ aj
        ens_soft = w_soft_vec * v_irl + (1 - w_soft_vec) * v_rf
        ens_oracle = np.where(v_label == 1, np.maximum(v_irl, v_rf), np.minimum(v_irl, v_rf))

        fold_overall_auc.append({
            "fold": fold_idx,
            "cold_irl": safe_auc(v_label, v_irl),
            "rf_dir": safe_auc(v_label, v_rf),
            "ens_uniform": safe_auc(v_label, ens_uniform),
            "ens_overall_opt": safe_auc(v_label, ens_overall_opt),
            "ens_hard_cv": safe_auc(v_label, ens_hard),
            "ens_soft_cv": safe_auc(v_label, ens_soft),
            "oracle": safe_auc(v_label, ens_oracle),
        })

        arch_row: Dict[str, Dict[str, float]] = {}
        for arch in arch_cols:
            mask = v_dom == arch
            if mask.sum() < 10:
                continue
            arch_row[arch] = {
                "cold_irl": safe_auc(v_label[mask], v_irl[mask]),
                "rf_dir": safe_auc(v_label[mask], v_rf[mask]),
                "ens_uniform": safe_auc(v_label[mask], ens_uniform[mask]),
                "ens_hard_cv": safe_auc(v_label[mask], ens_hard[mask]),
                "ens_soft_cv": safe_auc(v_label[mask], ens_soft[mask]),
                "oracle": safe_auc(v_label[mask], ens_oracle[mask]),
                "n": int(mask.sum()),
            }
        fold_arch_auc.append(arch_row)

    # 集約
    w_df = pd.DataFrame(fold_weights)
    w_df.to_csv(args.out_dir / "optimized_weights_per_fold.csv", index=False)
    summary_rows = []
    for c in arch_cols + ["__overall__"]:
        vals = w_df[c].to_numpy()
        summary_rows.append({"archetype": c, "w_mean": float(vals.mean()),
                             "w_std": float(vals.std(ddof=1)),
                             "w_min": float(vals.min()), "w_max": float(vals.max())})
    w_sum = pd.DataFrame(summary_rows)
    w_sum.to_csv(args.out_dir / "optimized_weights_summary.csv", index=False)
    print(f"\n[optimized weights summary]\n{w_sum.round(3).to_string(index=False)}")

    ov = pd.DataFrame(fold_overall_auc)
    ov_summary = pd.DataFrame({
        "model": ov.columns.drop("fold"),
        "auc_mean": [ov[c].mean() for c in ov.columns.drop("fold")],
        "auc_std": [ov[c].std(ddof=1) for c in ov.columns.drop("fold")],
    })
    ov_summary.to_csv(args.out_dir / "ensemble_overall_auc_cv.csv", index=False)
    print(f"\n[overall AUC, {args.n_folds}-fold CV]\n{ov_summary.round(4).to_string(index=False)}")

    # per-archetype 集約
    arch_rows = []
    for arch in arch_cols:
        models = ["cold_irl", "rf_dir", "ens_uniform", "ens_hard_cv", "ens_soft_cv", "oracle"]
        for m in models:
            vals = [f[arch][m] for f in fold_arch_auc if arch in f and m in f[arch]]
            if not vals:
                continue
            arch_rows.append({
                "archetype": arch, "model": m,
                "auc_mean": float(np.mean(vals)), "auc_std": float(np.std(vals, ddof=1)),
                "n_folds_with_data": len(vals),
            })
    arch_summary = pd.DataFrame(arch_rows)
    arch_summary.to_csv(args.out_dir / "ensemble_per_archetype_auc_cv.csv", index=False)
    print(f"\n[per-archetype AUC, CV mean]\n{arch_summary.round(4).to_string(index=False)}")

    # 図
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_models = ["cold_irl", "rf_dir", "ens_uniform", "ens_hard_cv", "ens_soft_cv", "oracle"]
    labels = ["Cold IRL", "RF_Dir", "Uniform 0.5", "Hard (CV-opt)", "Soft (CV-opt)", "Oracle"]
    colors = ["#1f77b4", "#2ca02c", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    x = np.arange(len(arch_cols))
    width = 0.13
    for i, (m, lab, col) in enumerate(zip(plot_models, labels, colors)):
        means = [arch_summary[(arch_summary["archetype"] == a) & (arch_summary["model"] == m)]["auc_mean"].iloc[0]
                 if not arch_summary[(arch_summary["archetype"] == a) & (arch_summary["model"] == m)].empty else np.nan
                 for a in arch_cols]
        stds = [arch_summary[(arch_summary["archetype"] == a) & (arch_summary["model"] == m)]["auc_std"].iloc[0]
                if not arch_summary[(arch_summary["archetype"] == a) & (arch_summary["model"] == m)].empty else 0
                for a in arch_cols]
        ax.bar(x + (i - 2.5) * width, means, width, yerr=stds, capsize=2,
               label=lab, color=col)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_cols)
    ax.set_ylabel(f"Val pooled AUC ({args.n_folds}-fold CV mean ± std)")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7)
    ax.set_title(f"Pair-level ensemble (grid-search) — irl_source={args.irl_source}")
    ax.legend(loc="lower right", fontsize=8, ncols=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_ensemble_pair_gridsearch.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {args.out_dir}")


if __name__ == "__main__":
    main()
