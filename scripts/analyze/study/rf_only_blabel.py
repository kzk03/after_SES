# -*- coding: utf-8 -*-
"""RF単体のB定義(承諾)AUCを t0掃引ディレクトリから集計する.

IRL 出力がまだ無い系列でも RF だけ先に B定義で見るための薄いラッパ
（verify_blabel_headline.compute_headline は IRL+RF 両方を前提とするため）。
combined_raw の読み込みが重いので、複数 sweep root をまとめて渡せる。

使い方:
  uv run python scripts/analyze/study/rf_only_blabel.py \
      outputs/t0_sweep_align_fs36 outputs/t0_sweep_inter_fs36 [...]
出力: 各 root 直下に rf_only_blabel.csv（t0×horizon+pooled の RF AUC-ROC）
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_blabel_headline import FS, HZ, load_df_with_dirs, recompute_blabel  # noqa: E402

T0S = ["2022-01-01", "2022-10-01", "2023-01-01", "2023-04-01", "2024-01-01"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("sweep_roots", nargs="+", help="t0掃引の出力root (例 outputs/t0_sweep_align_fs36)")
    args = p.parse_args()

    df = load_df_with_dirs()
    for root_s in args.sweep_roots:
        root = pathlib.Path(root_s)
        rows = []
        for t0s in T0S:
            t0 = pd.Timestamp(t0s)
            tag = t0s.replace("-", "")
            pooled = []
            for h in HZ:
                fp = root / tag / "rf" / f"fs_{h.replace('-', '_to_')}m/pair_predictions.csv"
                if not fp.exists():
                    print(f"[skip] {root_s} t0={t0s} {h}: {fp} なし")
                    break
                rf = pd.read_csv(fp)
                dir_devs = recompute_blabel(df, t0, FS[h], 3, assign=False)
                rf["blabel"] = [
                    1 if dev in dir_devs.get(d, set()) else 0
                    for dev, d in zip(rf["developer"], rf["directory"])
                ]
                rows.append({
                    "t0": t0s, "horizon": h, "n": len(rf),
                    "posrate": rf["blabel"].mean(),
                    "RF_roc": roc_auc_score(rf["blabel"], rf["rf_dir_prob"]),
                })
                pooled.append(rf[["blabel", "rf_dir_prob"]])
            if len(pooled) == len(HZ):
                P = pd.concat(pooled, ignore_index=True)
                rows.append({
                    "t0": t0s, "horizon": "pooled", "n": len(P),
                    "posrate": P["blabel"].mean(),
                    "RF_roc": roc_auc_score(P["blabel"], P["rf_dir_prob"]),
                })
        out = pd.DataFrame(rows)
        out_path = root / "rf_only_blabel.csv"
        out.to_csv(out_path, index=False)
        print(f"\n=== {root_s} (B定義, RF単体) ===")
        print(out[out["horizon"] == "pooled"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
