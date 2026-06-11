"""(dev, dir) pair ごとに 4 horizon の prediction 軌跡 vs label 軌跡を比較する。

入力: diagonal 4 cell の pair_predictions.csv
処理:
  1. (developer, directory) で 4 cell を横方向 merge → 4 horizon の label/IRL/RF を 1 行に
  2. label 軌跡パターンで分類 (一貫承諾/一貫拒否/変動型)
  3. 各 pair で prediction 軌跡 と label 軌跡 の Spearman 相関
  4. archetype × label パターンで集計

論文の問い:
  「変動型 pair (label が 4 窓で switch する) を IRL は予測軌跡で捉えているか？」
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load_diagonal(base: Path, variant: str = "lstm_baseline") -> pd.DataFrame:
    """4 horizon の predictions を横に merge した wide table を返す。"""
    dfs = []
    for h in HORIZONS:
        p = base / variant / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        d = pd.read_csv(p)
        d = d.rename(columns={
            "label": f"label_{h}",
            "irl_dir_prob_calibrated": f"irl_{h}",
            "rf_dir_prob": f"rf_{h}",
        })[["developer", "directory", f"label_{h}", f"irl_{h}", f"rf_{h}"]]
        dfs.append(d)
    # outer merge を順に
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, on=["developer", "directory"], how="outer")
    return merged


def classify_label_traj(labels: np.ndarray) -> str:
    """label 軌跡を分類 (NaN は欠損として無視)。"""
    valid = ~np.isnan(labels)
    if valid.sum() < 2:
        return "insufficient"
    vals = labels[valid]
    if (vals == 1).all():
        return "consistent_accept"
    if (vals == 0).all():
        return "consistent_reject"
    return "fluctuating"  # mix


def traj_corr(pred: np.ndarray, label: np.ndarray):
    """軌跡相関 (NaN を許容、有効点 ≥ 3 で計算)。"""
    mask = ~(np.isnan(pred) | np.isnan(label))
    if mask.sum() < 3:
        return np.nan, np.nan
    p = pred[mask]; l = label[mask]
    if np.std(p) == 0 or np.std(l) == 0:
        return np.nan, np.nan
    s, _ = spearmanr(p, l)
    pe, _ = pearsonr(p, l)
    return s, pe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path,
                    default=Path("outputs/main32_mce_volat"))
    ap.add_argument("--alpha", type=Path,
                    default=Path("outputs/analysis/archetype_irl_rf_volat_coldonly/k4_dev_alpha_with_metrics.csv"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("outputs/analysis/prediction_trajectory_volat"))
    args = ap.parse_args()

    print(f"[load] diagonal cells from {args.base}")
    w = load_diagonal(args.base)
    print(f"  merged pairs: {len(w)}")

    label_cols = [f"label_{h}" for h in HORIZONS]
    irl_cols = [f"irl_{h}" for h in HORIZONS]
    rf_cols  = [f"rf_{h}"  for h in HORIZONS]
    avg_cols = [f"avg_{h}" for h in HORIZONS]
    for h in HORIZONS:
        w[f"avg_{h}"] = (w[f"irl_{h}"] + w[f"rf_{h}"]) / 2

    # archetype merge
    alpha = pd.read_csv(args.alpha, index_col=0)
    alpha.index.name = "developer"
    alpha = alpha.reset_index()[["developer", "dominant"]].rename(
        columns={"dominant": "archetype"}
    )
    w = w.merge(alpha, on="developer", how="left")

    # label trajectory class
    L = w[label_cols].to_numpy()
    w["label_class"] = [classify_label_traj(row) for row in L]

    # trajectory correlations
    IRL = w[irl_cols].to_numpy()
    RF  = w[rf_cols].to_numpy()
    AVG = w[avg_cols].to_numpy()

    irl_sp, irl_pe = [], []
    rf_sp,  rf_pe  = [], []
    avg_sp, avg_pe = [], []
    for i in range(len(w)):
        s, p = traj_corr(IRL[i], L[i]); irl_sp.append(s); irl_pe.append(p)
        s, p = traj_corr(RF[i],  L[i]); rf_sp.append(s);  rf_pe.append(p)
        s, p = traj_corr(AVG[i], L[i]); avg_sp.append(s); avg_pe.append(p)
    w["traj_sp_irl"] = irl_sp; w["traj_pe_irl"] = irl_pe
    w["traj_sp_rf"]  = rf_sp;  w["traj_pe_rf"]  = rf_pe
    w["traj_sp_avg"] = avg_sp; w["traj_pe_avg"] = avg_pe

    args.out_dir.mkdir(parents=True, exist_ok=True)
    w.to_csv(args.out_dir / "pair_trajectories.csv", index=False)

    print(f"\n[label_class counts]")
    print(w["label_class"].value_counts())

    # 集計
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 200)
    print("\n=== mean Spearman trajectory correlation, by label_class ===")
    g = w.groupby("label_class")[["traj_sp_irl", "traj_sp_rf", "traj_sp_avg"]].agg(
        ["mean", "median", "count"]
    )
    print(g.to_string())

    print("\n=== fluctuating pairs only, by archetype ===")
    flu = w[w["label_class"] == "fluctuating"]
    print(f"  fluctuating pairs: {len(flu)}, with archetype: {flu['archetype'].notna().sum()}")
    g2 = flu.groupby("archetype")[["traj_sp_irl", "traj_sp_rf", "traj_sp_avg"]].agg(
        ["mean", "median", "count"]
    )
    print(g2.to_string())

    print("\n=== mean per-horizon prediction by label_class (sanity) ===")
    for cls in ["consistent_accept", "consistent_reject", "fluctuating"]:
        sub = w[w["label_class"] == cls]
        if sub.empty:
            continue
        print(f"\n[{cls}] n={len(sub)}")
        for col_pref, name in [("label", "label"), ("irl", "IRL"), ("rf", "RF"), ("avg", "AVG")]:
            cols = [f"{col_pref}_{h}" for h in HORIZONS]
            means = sub[cols].mean()
            print(f"  {name:5}:  " + "  ".join(f"{h}={m:.3f}" for h, m in zip(HORIZONS, means)))

    print(f"\nwrote: {args.out_dir}/pair_trajectories.csv")


if __name__ == "__main__":
    main()
