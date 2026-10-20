# -*- coding: utf-8 -*-
"""OpenStack の IRL / RF / LSTM退化版 の開発者単位 AUC を評価期間別に集計する（論文誌 査読 2-条5 対応）.

開発者単位 = 各 (developer, horizon) でラベル・予測確率をディレクトリ横断の max で集約（t0_archetype.py の per_unit_auc と同じ）。
入力は b3_ablation で保存済みの B定義（承諾）ラベル付きペア表。
"""
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

COLS = {"IRL": "p_irl_main", "RF": "rf_dir_prob", "LSTM": "p_lstm_fo", "GRU": "p_gru_fo", "Transformer": "p_tr_fo"}

p = argparse.ArgumentParser()
p.add_argument("--csv", default="outputs/ses_review/b3_ablation/pooled_pairs_blabel_t0_20230101.csv")
p.add_argument("--out-csv", default="outputs/ses_review/b3_ablation/dev_level_auc_by_horizon.csv")
a = p.parse_args()

df = pd.read_csv(a.csv)
dev = df.groupby(["developer", "horizon"], as_index=False)[["label", *COLS.values()]].max()
rows = []
for h in ["0-3", "3-6", "6-9", "9-12"]:
    d = dev[dev["horizon"] == h]
    rows.append({"horizon": h, "N": len(d), "pos": int(d["label"].sum()),
                 **{k: roc_auc_score(d["label"], d[c]) for k, c in COLS.items()}})
out = pd.DataFrame(rows)
mean = out[list(COLS)].mean()
out.loc[len(out)] = {"horizon": "mean4", "N": None, "pos": None, **mean.to_dict()}
out.to_csv(a.out_csv, index=False)
print(out.round(4).to_string(index=False))
print("saved:", a.out_csv)
