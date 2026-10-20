# -*- coding: utf-8 -*-
"""AOSP の t0 掃引を、論文の AOSP 表（表3・4・14）と同じラベルで集計し直す（論文誌 査読 1-5 対応）.

論文の AOSP 表は pair_predictions.csv の label 列（生データからの承諾ラベル再計算はしていない）で
AUC を出している。t0 掃引の headline_blabel.csv は承諾ラベルで再計算した値なので、比較のため
同じ pair_predictions.csv の label 列で IRL / RF / 単純平均アンサンブルの AUC を出し直す。
基準として主結果（t0=2023-01, outputs/aosp32/dynstartB_ep150 + rf_gbdt_1snap_v2）も同じ方法で並べる。
"""
import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score

HZ = ["0-3", "3-6", "6-9", "9-12"]

p = argparse.ArgumentParser()
p.add_argument("--sweep-root", default="outputs/t0_sweep_inter_fs36_aosp32_ext")
p.add_argument("--t0s", nargs="+", default=["20220701", "20230701"])
p.add_argument("--out-csv", default="outputs/t0_sweep_inter_fs36_aosp32_ext/t0_comparison_predlabel.csv")
a = p.parse_args()

bases = {"20230101(main)": ("outputs/aosp32/dynstartB_ep150/lstm_baseline", "outputs/aosp32/rf_gbdt_1snap_v2")}
for t in a.t0s:
    bases[t] = (f"{a.sweep_root}/{t}/irl/lstm_baseline", f"{a.sweep_root}/{t}/rf")

rows = []
for t0, (irl_base, rf_base) in bases.items():
    parts = []
    for h in HZ:
        irl = pd.read_csv(f"{irl_base}/train_{h}m/eval_{h}m/pair_predictions.csv")[["developer", "directory", "label", "irl_dir_prob"]]
        rf = pd.read_csv(f"{rf_base}/fs_{h.replace('-', '_to_')}m/pair_predictions.csv")[["developer", "directory", "label", "rf_dir_prob"]]
        m = irl.merge(rf, on=["developer", "directory"], suffixes=("", "_rf"))
        mismatch = int((m["label"] != m["label_rf"]).sum())
        m["horizon"] = h
        parts.append(m)
        print(f"{t0} {h}: irl={len(irl)} rf={len(rf)} merged={len(m)} label_mismatch={mismatch}")
    P = pd.concat(parts, ignore_index=True)
    P["ens"] = (P["irl_dir_prob"] + P["rf_dir_prob"]) / 2
    for h in [*HZ, "pooled"]:
        d = P if h == "pooled" else P[P["horizon"] == h]
        y = d["label"]
        rows.append({"t0": t0, "horizon": h, "n": len(d), "pos": int(y.sum()), "posrate": y.mean(),
                     "IRL": roc_auc_score(y, d["irl_dir_prob"]), "RF": roc_auc_score(y, d["rf_dir_prob"]),
                     "Ens": roc_auc_score(y, d["ens"])})
out = pd.DataFrame(rows)
out["IRL_minus_RF_pp"] = 100 * (out["IRL"] - out["RF"])
out["Ens_minus_best_pp"] = 100 * (out["Ens"] - out[["IRL", "RF"]].max(axis=1))
out.to_csv(a.out_csv, index=False)
print(out.round(3).to_string(index=False))
print("saved:", a.out_csv)
