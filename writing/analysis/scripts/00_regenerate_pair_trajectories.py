# -*- coding: utf-8 -*-
"""pair_trajectories.csv を RULES 準拠で再生成.

- IRL: irl_dir_prob (未キャリブレーション, "cold IRL")
- RF: rf_dir_prob
- Ensemble: 上記2つの単純平均

入力: outputs/main32_mce_volat_ep150/lstm_baseline/train_<h>/eval_<h>/pair_predictions.csv (diagonal 4 cells)
出力: writing/analysis/data/pair_trajectories.csv (上書き)
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/main32_mce_volat_ep150/lstm_baseline"
ALPHA_CSV = ROOT / "outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/k4_dev_alpha_with_metrics.csv"
OUT_CSV = ROOT / "writing/analysis/data/pair_trajectories.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def main() -> None:
    dfs = []
    for h in HORIZONS:
        p = BASE / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        d = pd.read_csv(p)
        d = d.rename(columns={
            "label": f"label_{h}",
            "irl_dir_prob": f"irl_{h}",
            "rf_dir_prob": f"rf_{h}",
        })[["developer", "directory", f"label_{h}", f"irl_{h}", f"rf_{h}"]]
        dfs.append(d)
        print(f"{h}: {len(d)} pairs")

    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["developer", "directory"], how="outer")
    print(f"merged (outer): {len(merged)} pairs")

    # ensemble = simple average
    for h in HORIZONS:
        merged[f"avg_{h}"] = (merged[f"irl_{h}"] + merged[f"rf_{h}"]) / 2

    # archetype を付与
    alpha = pd.read_csv(ALPHA_CSV, index_col=0)
    alpha.index.name = "developer"
    alpha = alpha.reset_index()[["developer", "dominant"]].rename(
        columns={"dominant": "archetype"}
    )
    merged = merged.merge(alpha, on="developer", how="left")

    merged.to_csv(OUT_CSV, index=False)
    print(f"saved: {OUT_CSV}")
    print(f"final shape: {merged.shape}")

    # 簡易 sanity: pooled AUC を計算
    from sklearn.metrics import roc_auc_score
    print("\n=== Pooled AUC sanity check ===")
    for h in HORIZONS:
        sub = merged[~merged[f"label_{h}"].isna()].copy()
        y = sub[f"label_{h}"].astype(int).to_numpy()
        irl_auc = roc_auc_score(y, sub[f"irl_{h}"])
        rf_auc = roc_auc_score(y, sub[f"rf_{h}"])
        ens_auc = roc_auc_score(y, sub[f"avg_{h}"])
        print(f"  {h}: n={len(sub)}, IRL={irl_auc:.3f}, RF={rf_auc:.3f}, "
              f"Ens={ens_auc:.3f}")


if __name__ == "__main__":
    main()
