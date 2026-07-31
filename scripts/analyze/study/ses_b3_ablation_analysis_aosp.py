# -*- coding: utf-8 -*-
"""SES査読B-3 退化版(final_step_only)ベースライン分析のAOSP版.

ses_b3_ablation_analysis.py のAOSP版。GRU/Transformerの退化版はAOSPで
未実施のため、IRL本体・LSTM退化版・RFの3モデルのみを比較する。

サーバ上でAOSPの生データ（combined_raw相当・raw_json_list相当）のパスを
--data / --raw-json-list に指定して実行する。パスは環境に合わせて要確認。

実行例:
    uv run python scripts/analyze/study/ses_b3_ablation_analysis_aosp.py \
      --data data/combined_raw_aosp32.csv \
      --raw-json-list data/combined_raw_aosp32.raw_json_list.txt
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from verify_blabel_headline import FS, HZ, load_df_with_dirs, recompute_blabel  # noqa: E402

MODELS = {
    "irl_main": "outputs/aosp32/dynstartB_ep150/lstm_baseline",
    "lstm_fo": "outputs/aosp32/b3_ablation/lstm_final_only",
    "gru_fo": "outputs/aosp32/b3_ablation/gru_final_only",
}


def build_wide_table(df_dirs: pd.DataFrame, rf_base: str, t0: pd.Timestamp) -> pd.DataFrame:
    """label + 全モデル確率 + RF をペアレベルで横結合した表を返す。"""
    frames = []
    for h in HZ:
        dir_devs = recompute_blabel(df_dirs, t0, FS[h], 3, assign=False)
        rf = pd.read_csv(f"{rf_base}/fs_{h.replace('-', '_to_')}m/pair_predictions.csv")
        m = rf[["developer", "directory", "rf_dir_prob"]].copy()
        for name, base in MODELS.items():
            path = f"{base}/train_{h}m/eval_{h}m/pair_predictions.csv"
            if not pathlib.Path(path).exists():
                print(f"  [skip] {name} has no eval for horizon {h} ({path})")
                continue
            irl = pd.read_csv(path)
            m = m.merge(
                irl[["developer", "directory", "irl_dir_prob"]].rename(
                    columns={"irl_dir_prob": f"p_{name}"}),
                on=["developer", "directory"], how="inner")
        m["horizon"] = h
        m["label"] = [
            1 if dev in dir_devs.get(d, set()) else 0
            for dev, d in zip(m["developer"], m["directory"])
        ]
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True,
                   help="AOSP版 combined_raw CSVのパス（要確認・環境依存）")
    p.add_argument("--raw-json-list", required=True,
                   help="AOSP版 raw_json_listのパス（要確認・環境依存）")
    p.add_argument("--rf-base", default="outputs/aosp32/rf_gbdt_1snap_v2")
    p.add_argument("--prediction-time", default="2023-01-01")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv",
                   default="outputs/aosp32/b3_ablation/pooled_pairs_blabel_t0_20230101.csv")
    args = p.parse_args()

    t0 = pd.Timestamp(args.prediction_time)
    df = load_df_with_dirs(args.data, args.raw_json_list)
    W = build_wide_table(df, args.rf_base, t0)
    W.to_csv(args.out_csv, index=False)
    print(f"wide表 {W.shape} saved: {args.out_csv}")

    cols = {"irl_main": "p_irl_main", "lstm_fo": "p_lstm_fo", "rf": "rf_dir_prob"}
    y = W["label"].to_numpy()

    print("\n=== pooled AUC ===")
    for name, c in cols.items():
        raw = roc_auc_score(y, W[c])
        print(f"  {name:9s}: {raw:.3f}")

    rng = np.random.default_rng(args.seed)
    n = len(W)
    P = {k: W[c].to_numpy() for k, c in cols.items()}
    pairs = [("irl_main", "lstm_fo"), ("irl_main", "rf"), ("lstm_fo", "rf")]
    deltas = {pr: np.empty(args.n_boot) for pr in pairs}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aucs = {k: roc_auc_score(yb, v[idx]) for k, v in P.items()}
        for a, bb in pairs:
            deltas[(a, bb)][b] = aucs[a] - aucs[bb]
    print(f"\n=== ペア・ブートストラップ ΔAUC (pooled, B={args.n_boot}, seed={args.seed}) ===")
    for (a, bb), d in deltas.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        point = roc_auc_score(y, P[a]) - roc_auc_score(y, P[bb])
        print(f"  {a} − {bb}: Δ={point:+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  P(Δ<=0)={(d <= 0).mean():.4f}")

    print("\n=== horizon別 AUC ===")
    rows = []
    for h in HZ:
        mh = W[W["horizon"] == h]
        yh = mh["label"].to_numpy()
        rows.append({"horizon": h, **{k: roc_auc_score(yh, mh[c]) for k, c in cols.items()}})
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if "p_gru_fo" in W.columns:
        print("\n=== gru_fo (現時点で評価済みの horizon のみ・部分集計) ===")
        for h in HZ:
            mh = W[(W["horizon"] == h) & W["p_gru_fo"].notna()]
            if len(mh) == 0:
                print(f"  {h}: 未評価")
                continue
            yh = mh["label"].to_numpy()
            auc_gru = roc_auc_score(yh, mh["p_gru_fo"])
            auc_irl = roc_auc_score(yh, mh["p_irl_main"])
            auc_rf = roc_auc_score(yh, mh["rf_dir_prob"])
            print(f"  {h}: n={len(mh)}  irl_main={auc_irl:.3f}  gru_fo={auc_gru:.3f}  rf={auc_rf:.3f}")


if __name__ == "__main__":
    main()
