# -*- coding: utf-8 -*-
"""原型別・評価期間別 AUC 計算 (論文 表8/表9 方式).

t0_archetype.py の archetypal_analysis 結果 (k{K}_dev_alpha_with_metrics.csv /
k{K}_pair_alpha_with_metrics.csv の dominant 列) をそのまま再利用し、
archetype (dev または pair 単位で分類済み) × horizon(4期間) × model で
pair レベルの生データ (集約前の long、B定義ラベル) をプールして AUC を計算する。

t0_archetype.py の per_unit_auc は unit (dev または (dev,dir)) 単位で
horizon をまとめて集約してから AUC を取るため、論文の表8/9 (archetype ×
評価期間別 AUC) とは軸が異なる。本スクリプトはその集計軸のズレを解消する
ためのもの。archetypal_analysis (重い反復最適化) 自体は再実行しない。

使い方:
  uv run python scripts/analyze/study/archetype_period_auc.py \
      --level dev --k 4 \
      --alpha-csv outputs/aosp32/archetype/dev/k4_dev_alpha_with_metrics.csv \
      --irl-base outputs/aosp32/dynstartB_ep150/lstm_baseline \
      --rf-base outputs/aosp32/rf_gbdt_1snap_v2 \
      --prediction-time 2023-01-01 --delta-months 3 \
      --data data/combined_raw_aosp32.csv \
      --raw-json-list data/combined_raw_aosp32.raw_json_list.txt \
      --output outputs/aosp32/archetype/dev/k4_period_auc.csv
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from t0_archetype import build_cold_long  # noqa: E402
from verify_blabel_headline import HZ, load_df_with_dirs  # noqa: E402


def archetype_period_auc(
    long: pd.DataFrame, unit_to_arch: dict, unit_cols: list, k: int,
) -> pd.DataFrame:
    """archetype × horizon × model (cold_irl/rf_dir) で pooled AUC を計算。

    n はそのarchetype×horizonでAUC計算に実際に使われた (dev,dir)ペア
    レコード数(pairレベル生データの行数)。dev-levelでは1人が複数dirを
    持つためunitの数(devの数)より大きくなる。論文の表8/9のn・AUCと同じ軸。
    """
    a = long.copy()
    if unit_cols == ["developer"]:
        a["archetype"] = a["developer"].map(lambda d: unit_to_arch.get((d,)))
    else:
        a["archetype"] = [
            unit_to_arch.get((dev, d))
            for dev, d in zip(a["developer"], a["directory"])
        ]
    a = a.dropna(subset=["archetype"])
    a["archetype"] = a["archetype"].astype(int)

    rows = []
    for j in range(k):
        sub_j = a[a["archetype"] == j]
        for h in HZ:
            row = {"archetype": f"A{j + 1}", "horizon": h}
            n_rows = None
            for model in ["cold_irl", "rf_dir"]:
                sub = sub_j[
                    (sub_j["horizon"] == h) & (sub_j["model"] == model)
                ][["label", "prob"]].to_numpy()
                if n_rows is None:
                    n_rows = len(sub)
                if len(sub) < 10 or len(np.unique(sub[:, 0])) < 2:
                    auc = np.nan
                else:
                    try:
                        auc = float(
                            roc_auc_score(sub[:, 0].astype(int), sub[:, 1].astype(float))
                        )
                    except ValueError:
                        auc = np.nan
                row[f"auc_{model}"] = auc
            row["n"] = n_rows
            rows.append(row)
    df = pd.DataFrame(rows)
    df["delta_pp"] = (df["auc_cold_irl"] - df["auc_rf_dir"]) * 100
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=["dev", "pair"], required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument(
        "--alpha-csv", required=True,
        help="k{K}_dev_alpha_with_metrics.csv または k{K}_pair_alpha_with_metrics.csv",
    )
    p.add_argument("--irl-base", required=True)
    p.add_argument("--rf-base", required=True)
    p.add_argument("--prediction-time", required=True)
    p.add_argument("--delta-months", type=int, default=3)
    p.add_argument("--data", default="data/combined_raw_main32.csv")
    p.add_argument("--raw-json-list", default="data/combined_raw_main32.raw_json_list.txt")
    p.add_argument("--responded-in-window", action="store_true")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"[load] raw + dirs (t0={args.prediction_time}) ...")
    df_dirs = load_df_with_dirs(args.data, args.raw_json_list)
    df_dirs["timestamp"] = pd.to_datetime(df_dirs["timestamp"])

    print("[load] cold predictions + B定義ラベル付け ...")
    long = build_cold_long(
        df_dirs, args.irl_base, args.rf_base,
        args.prediction_time, args.delta_months,
        responded_in_window=args.responded_in_window,
    )
    print(f"[load] long rows: {len(long)}")

    print(f"[load] alpha csv: {args.alpha_csv}")
    if args.level == "dev":
        alpha_df = pd.read_csv(args.alpha_csv, index_col=0)
        unit_cols = ["developer"]
        unit_to_arch = {
            (email,): int(dom[1:]) - 1
            for email, dom in zip(alpha_df.index, alpha_df["dominant"])
        }
    else:
        alpha_df = pd.read_csv(args.alpha_csv)
        unit_cols = ["developer", "directory"]
        unit_to_arch = {
            (row["developer"], row["directory"]): int(row["dominant"][1:]) - 1
            for _, row in alpha_df.iterrows()
        }
    print(f"[load] units with archetype assignment: {len(unit_to_arch)}")

    result = archetype_period_auc(long, unit_to_arch, unit_cols, args.k)
    result.to_csv(args.output, index=False)
    print(f"[done] {args.output}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
