# -*- coding: utf-8 -*-
"""AOSP の IRL / LSTM退化版 / RF を評価期間別に集計する（論文誌 査読 2-条5 対応）.

表3・表4（ペア単位・開発者単位の評価期間別AUC）の AOSP 側に LSTM の列を足すための集計。
入力は aosp_ablation_pooled.py と同じ pair_predictions.csv。
  ペア単位  : 各評価期間のペアでそのまま AUC
  開発者単位: 各 (developer, horizon) でラベル・予測確率をディレクトリ横断の max で集約（dev_level_auc_ablation.py と同じ）
あわせて参考値として LSTM+RF / IRL+RF の単純平均アンサンブルの統合（4期間 pooled）AUC を出す。
"""
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

HZ = ["0-3", "3-6", "6-9", "9-12"]
MODELS = ["irl", "lstm", "rf"]

p = argparse.ArgumentParser()
p.add_argument("--out-csv", default="outputs/aosp32/b3_ablation/auc_by_horizon.csv")
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
P["ens_irl_rf"] = (P["irl"] + P["rf"]) / 2
P["ens_lstm_rf"] = (P["lstm"] + P["rf"]) / 2

rows = []
D = P.groupby(["developer", "horizon"], as_index=False)[["label", *MODELS]].max()
for level, df in [("pair", P), ("dev", D)]:
    for h in HZ:
        d = df[df["horizon"] == h]
        rows.append({"level": level, "horizon": h, "n": len(d), "pos": int(d["label"].sum()),
                     **{m: roc_auc_score(d["label"], d[m]) for m in MODELS}})
    mean = pd.DataFrame(rows[-4:])[MODELS].mean()
    rows.append({"level": level, "horizon": "mean4", **mean.to_dict()})
y = P["label"]
rows.append({"level": "pair", "horizon": "pooled", "n": len(P), "pos": int(y.sum()),
             **{m: roc_auc_score(y, P[m]) for m in [*MODELS, "ens_irl_rf", "ens_lstm_rf"]}})
out = pd.DataFrame(rows)
out.to_csv(a.out_csv, index=False)
print(out.round(4).to_string(index=False))
print("saved:", a.out_csv)
