"""train/eval 行動シフトと IRL 優位の関係を全 archetype で検証する。

A2 で発見した「mid shift bucket で IRL が RF に大勝、特に 9-12m で +23pp」が、
A2 固有現象か全 archetype 共通かを判別する。

各 archetype × 予測窓 × shift 3 分位 で:
  - IRL AUC
  - RF AUC
  - IRL - RF
を計算。

入力:
  - data/combined_raw_main32.csv
  - outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv
  - outputs/main32_mce_volat_ep150/lstm_baseline/train_*m/eval_*m/pair_predictions.csv

出力:
  - outputs/analysis/shift_bucket_cross_archetype/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN_END = pd.Timestamp("2022-01-01")
EVAL_END = pd.Timestamp("2023-01-01")
WINDOWS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load(reviews_csv, arche_csv):
    rev = pd.read_csv(reviews_csv, parse_dates=["timestamp"])
    rev["label"] = rev["label"].astype(int)
    arche = pd.read_csv(arche_csv)
    dev_arch = (arche.groupby("developer")["dominant"]
                .agg(lambda s: s.value_counts().idxmax())
                .reset_index().rename(columns={"developer": "email", "dominant": "archetype"}))
    return rev, dev_arch


def per_dev_shift(rev, dev_arch, min_train_n=5):
    train = rev[rev["timestamp"] < TRAIN_END]
    eval_ = rev[(rev["timestamp"] >= TRAIN_END) & (rev["timestamp"] < EVAL_END)]
    tg = train.groupby("email")["label"].agg(["mean", "size"]).reset_index()
    tg.columns = ["email", "train_rate", "train_n"]
    eg = eval_.groupby("email")["label"].agg(["mean", "size"]).reset_index()
    eg.columns = ["email", "eval_rate", "eval_n"]
    m = tg.merge(eg, on="email", how="inner").merge(dev_arch, on="email", how="inner")
    m["shift"] = m["eval_rate"] - m["train_rate"]
    return m[m["train_n"] >= min_train_n]


def load_predictions(base):
    base_dir = Path(base) / "lstm_baseline"
    frames = []
    for h in WINDOWS:
        p = base_dir / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p).dropna(
            subset=["irl_dir_prob_calibrated", "rf_dir_prob", "label"]).copy()
        d["window"] = h
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def cross_archetype_analysis(merged, predictions, min_n_per_bucket=30):
    """archetype × window × shift_bucket の IRL vs RF AUC。"""
    rows = []
    for arch in ["A1", "A2", "A3", "A4"]:
        devs_in_arch = set(merged[merged["archetype"] == arch]["email"])
        # archetype 内で shift を 3 分位に切る (archetype 別に閾値設定)
        sh = merged[merged["archetype"] == arch][["email", "shift"]].copy()
        sh["abs_shift"] = sh["shift"].abs()
        try:
            sh["shift_bucket"] = pd.qcut(sh["abs_shift"], 3,
                                          labels=["low", "mid", "high"],
                                          duplicates="drop")
        except Exception:
            continue
        bucket_map = dict(zip(sh["email"], sh["shift_bucket"]))

        for h in WINDOWS:
            pred = predictions[(predictions["window"] == h) &
                               (predictions["developer"].isin(devs_in_arch))].copy()
            pred["shift_bucket"] = pred["developer"].map(bucket_map)
            for q in ["low", "mid", "high"]:
                sub = pred[pred["shift_bucket"] == q]
                if len(sub) < min_n_per_bucket or sub["label"].nunique() < 2:
                    continue
                y = sub["label"].astype(int).to_numpy()
                try:
                    ai = roc_auc_score(y, sub["irl_dir_prob_calibrated"])
                    ar = roc_auc_score(y, sub["rf_dir_prob"])
                except ValueError:
                    continue
                rows.append({
                    "archetype": arch, "window": h, "shift_bucket": q,
                    "n_pairs": len(sub), "n_pos": int(y.sum()),
                    "auc_irl": ai, "auc_rf": ar,
                    "irl_minus_rf_pp": (ai - ar) * 100,
                })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", type=Path, default=Path("data/combined_raw_main32.csv"))
    ap.add_argument("--archetype-csv", type=Path,
                    default=Path("outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv"))
    ap.add_argument("--predictions-base", type=Path,
                    default=Path("outputs/main32_mce_volat_ep150"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/analysis/shift_bucket_cross_archetype"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rev, dev_arch = load(args.reviews, args.archetype_csv)
    merged = per_dev_shift(rev, dev_arch)
    print(f"[loaded] archetype-assigned devs with train data = {len(merged):,}")

    pred = load_predictions(args.predictions_base)
    if pred is None:
        raise SystemExit("predictions not found")

    result = cross_archetype_analysis(merged, pred)
    result.to_csv(args.output_dir / "cross_archetype_shift_bucket.csv", index=False)

    print("\n=== Archetype x Window x Shift bucket の IRL-RF gap (pp) ===")
    pivot = result.pivot_table(
        index=["archetype", "window"], columns="shift_bucket",
        values="irl_minus_rf_pp",
    ).round(1)
    print(pivot.to_string())

    print("\n=== 同じ pivot の n_pairs (subgroup サイズ確認) ===")
    pivot_n = result.pivot_table(
        index=["archetype", "window"], columns="shift_bucket",
        values="n_pairs",
    ).fillna(0).astype(int)
    print(pivot_n.to_string())

    # mid bucket だけ archetype 別に集約
    print("\n=== mid shift bucket のみ: archetype 別 IRL-RF gap (pp) by window ===")
    mid = result[result["shift_bucket"] == "mid"]
    mid_pivot = mid.pivot(index="archetype", columns="window",
                          values="irl_minus_rf_pp").round(1)
    print(mid_pivot.to_string())

    # A2 9-12m mid と他 archetype 9-12m mid の比較
    print("\n=== 9-12m × mid bucket のみ (全 archetype) ===")
    sub = result[(result["window"] == "9-12m") & (result["shift_bucket"] == "mid")]
    print(sub[["archetype", "n_pairs", "n_pos", "auc_irl", "auc_rf",
               "irl_minus_rf_pp"]].to_string(index=False))

    with open(args.output_dir / "summary.txt", "w") as f:
        f.write("Shift bucket cross-archetype analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write("Pivot: archetype x window x shift_bucket -> IRL-RF (pp)\n")
        f.write(pivot.to_string() + "\n\n")
        f.write("Pivot: n_pairs\n")
        f.write(pivot_n.to_string() + "\n")

    print(f"\n[done] outputs at {args.output_dir}")


if __name__ == "__main__":
    main()
