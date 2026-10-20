# -*- coding: utf-8 -*-
"""AOSP の IRL / LSTM退化版 / RF を、表14（アンサンブル）と同じ方法で集計する（論文誌 査読 2-条6 対応）.

表14 の AOSP 値は pair_predictions.csv のラベル・予測を 4 評価期間で統合（pooled）して
AUC-ROC を算出している。表5（アブレーション）の AOSP 値をこれに揃えるため、同じ入力・同じ方法で
IRL・LSTM・RF の統合 AUC と、差（X − IRL）のペア bootstrap 95% CI（2,000 回, seed 0）を出す。

入力:
  IRL : outputs/aosp32/dynstartB_ep150/lstm_baseline/train_{h}m/eval_{h}m/pair_predictions.csv
  LSTM: outputs/aosp32/b3_ablation/lstm_final_only/train_{h}m/eval_{h}m/pair_predictions.csv
  RF  : outputs/aosp32/rf_gbdt_1snap_v2/fs_{h}m/pair_predictions.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HZ = ["0-3", "3-6", "6-9", "9-12"]

p = argparse.ArgumentParser()
p.add_argument("--n-boot", type=int, default=2000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out-csv", default="outputs/aosp32/b3_ablation/headline_pooled_table5.csv")
a = p.parse_args()

parts = []
for h in HZ:
    irl = pd.read_csv(f"outputs/aosp32/dynstartB_ep150/lstm_baseline/train_{h}m/eval_{h}m/pair_predictions.csv")[["developer", "directory", "label", "irl_dir_prob"]]
    lstm = pd.read_csv(f"outputs/aosp32/b3_ablation/lstm_final_only/train_{h}m/eval_{h}m/pair_predictions.csv")[["developer", "directory", "irl_dir_prob"]].rename(columns={"irl_dir_prob": "lstm"})
    rf = pd.read_csv(f"outputs/aosp32/rf_gbdt_1snap_v2/fs_{h.replace('-', '_to_')}m/pair_predictions.csv")[["developer", "directory", "rf_dir_prob"]]
    m = irl.merge(lstm, on=["developer", "directory"]).merge(rf, on=["developer", "directory"])
    m["horizon"] = h
    parts.append(m)
P = pd.concat(parts, ignore_index=True).rename(columns={"irl_dir_prob": "irl", "rf_dir_prob": "rf"})
y = P["label"].to_numpy()
cols = {"IRL": "irl", "LSTM": "lstm", "RF": "rf"}
auc = {k: roc_auc_score(y, P[c]) for k, c in cols.items()}

rng = np.random.default_rng(a.seed)
X = {k: P[c].to_numpy() for k, c in cols.items()}
diffs = {"LSTM": [], "RF": []}
for _ in range(a.n_boot):
    i = rng.integers(0, len(y), len(y))
    base = roc_auc_score(y[i], X["IRL"][i])
    for k in diffs:
        diffs[k].append(roc_auc_score(y[i], X[k][i]) - base)

rows = [{"model": "IRL", "auc_pooled": auc["IRL"], "diff_vs_irl_pp": 0.0, "ci_lo_pp": np.nan, "ci_hi_pp": np.nan}]
for k in diffs:
    lo, hi = np.percentile(diffs[k], [2.5, 97.5]) * 100
    rows.append({"model": k, "auc_pooled": auc[k], "diff_vs_irl_pp": (auc[k] - auc["IRL"]) * 100, "ci_lo_pp": lo, "ci_hi_pp": hi})
out = pd.DataFrame(rows)
out["n"] = len(y)
out.to_csv(a.out_csv, index=False)
print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("saved:", a.out_csv)
