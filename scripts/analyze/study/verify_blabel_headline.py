# -*- coding: utf-8 -*-
"""手順5（B定義=承諾ラベル）の復元・検証スクリプト.

REPRODUCE.md 手順5の「cold 予測を align（4,534ペア）し、B定義ラベルを
combined_raw から再計算して AUC」を再構成する。

焼き込み label との違い:
  - pair_predictions.csv の label = get_directory_developers(df, W_h) による
    「W_h 内に reviewer として登場したか（アサイン定義、label 不問）」
  - B定義(承諾) = 未来窓 W_h を df[df.label==1]（response_window 内にコメント/
    投票で反応）でフィルタしてから get_directory_developers で actual_devs を作る

t0 を --prediction-time で変えれば、同一のラベル定義で任意時点の AUC を出せる。
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
from review_predictor.IRL.features.directory_contributors import (  # noqa: E402
    get_directory_developers,
)
from review_predictor.IRL.features.path_features import (  # noqa: E402
    attach_dirs_to_df,
    load_change_dir_map_multi,
)

HZ = ["0-3", "3-6", "6-9", "9-12"]
FS = {"0-3": 0, "3-6": 3, "6-9": 6, "9-12": 9}


def recompute_blabel(
    df_dirs: pd.DataFrame, t0: pd.Timestamp, fs: int, delta: int, assign: bool,
    responded_in_window: bool = False,
) -> dict:
    """{directory: set(devs)} を未来窓 W_h で作る。

    assign=False かつ responded_in_window=False (デフォルト): label==1（14日以内反応）のみ。
    assign=False かつ responded_in_window=True: first_response_time が not null のもの（反応あり）。
    assign=True: フィルタなし（アサイン定義）。
    """
    fut_start = t0 + pd.DateOffset(months=fs)
    fut_end = t0 + pd.DateOffset(months=fs + delta)
    if assign:
        src = df_dirs
    elif responded_in_window:
        src = df_dirs[df_dirs["first_response_time"].notna()]
    else:
        src = df_dirs[df_dirs["label"] == 1]
    return get_directory_developers(src, fut_start, fut_end)


def load_df_with_dirs(
    data: str = "data/combined_raw_main32.csv",
    raw_json_list: str = "data/combined_raw_main32.raw_json_list.txt",
    dir_depth: int = 2,
) -> pd.DataFrame:
    """combined_raw を読み、raw_json から 'dirs' 列を付与した df を返す（重いので使い回す）。"""
    df = pd.read_csv(data, low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    raw_json = pathlib.Path(raw_json_list).read_text().split()
    cdm = load_change_dir_map_multi(raw_json, depth=dir_depth)
    return attach_dirs_to_df(df, cdm)


def compute_headline(
    df_dirs: pd.DataFrame, irl_base: str, rf_base: str, t0,
    delta: int = 3, assign: bool = False,
) -> pd.DataFrame:
    """IRL(diagonal)/RF の cold 予測を align し、B定義(or アサイン)ラベルで
    horizon 別＋pooled の AUC-ROC/PR を返す。"""
    t0 = pd.Timestamp(t0)
    rows, pooled = [], []
    for h in HZ:
        irl = pd.read_csv(f"{irl_base}/train_{h}m/eval_{h}m/pair_predictions.csv")
        rf = pd.read_csv(f"{rf_base}/fs_{h.replace('-', '_to_')}m/pair_predictions.csv")
        m = irl[["developer", "directory", "irl_dir_prob"]].merge(
            rf[["developer", "directory", "rf_dir_prob"]],
            on=["developer", "directory"], how="inner")
        dir_devs = recompute_blabel(df_dirs, t0, FS[h], delta, assign)
        m["label"] = [
            1 if dev in dir_devs.get(d, set()) else 0
            for dev, d in zip(m["developer"], m["directory"])
        ]
        y = m["label"].to_numpy()
        pi, pr = m["irl_dir_prob"].to_numpy(), m["rf_dir_prob"].to_numpy()
        pe = (pi + pr) / 2
        rows.append({
            "horizon": h, "n": len(m), "pos": int(y.sum()), "posrate": y.mean(),
            "IRL_roc": roc_auc_score(y, pi), "RF_roc": roc_auc_score(y, pr),
            "Ens_roc": roc_auc_score(y, pe),
            "IRL_pr": average_precision_score(y, pi),
            "RF_pr": average_precision_score(y, pr),
            "Ens_pr": average_precision_score(y, pe),
        })
        mm = m.copy(); mm["horizon"] = h; pooled.append(mm)
    P = pd.concat(pooled, ignore_index=True)
    y = P["label"].to_numpy()
    pi, pr = P["irl_dir_prob"].to_numpy(), P["rf_dir_prob"].to_numpy()
    pe = (pi + pr) / 2
    rows.append({
        "horizon": "pooled", "n": len(P), "pos": int(y.sum()), "posrate": y.mean(),
        "IRL_roc": roc_auc_score(y, pi), "RF_roc": roc_auc_score(y, pr),
        "Ens_roc": roc_auc_score(y, pe),
        "IRL_pr": average_precision_score(y, pi),
        "RF_pr": average_precision_score(y, pr),
        "Ens_pr": average_precision_score(y, pe),
    })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/combined_raw_main32.csv")
    p.add_argument("--raw-json-list", default="data/combined_raw_main32.raw_json_list.txt")
    p.add_argument("--irl-base", default="outputs/dynstartB_ep150/lstm_baseline")
    p.add_argument("--rf-base", default="outputs/dynstart_rf_1snap_s777")
    p.add_argument("--prediction-time", default="2023-01-01")
    p.add_argument("--delta-months", type=int, default=3)
    p.add_argument("--dir-depth", type=int, default=2)
    p.add_argument("--assign", action="store_true",
                   help="承諾でなくアサイン定義（label 不問）で集計（焼き込み label 相当）")
    p.add_argument("--out-csv", default=None)
    args = p.parse_args()

    df = load_df_with_dirs(args.data, args.raw_json_list, args.dir_depth)
    out = compute_headline(df, args.irl_base, args.rf_base, args.prediction_time,
                           delta=args.delta_months, assign=args.assign)
    label_kind = "ASSIGN(アサイン)" if args.assign else "B定義(承諾)"
    print(f"\n=== t0={args.prediction_time}  label={label_kind} ===")
    pd.set_option("display.width", 160)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nmd headline (t0=2023-01, B定義): "
          "IRL_roc pooled=0.789 / RF_1snap=0.801 / Ens=0.823 ; "
          "0-3m IRL=0.826 RF=0.791 Ens=0.838")
    if args.out_csv:
        out.to_csv(args.out_csv, index=False)
        print(f"\nsaved: {args.out_csv}")


if __name__ == "__main__":
    main()
