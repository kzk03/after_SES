# -*- coding: utf-8 -*-
"""弱負例重み付きRF（RF_WEAK_NEG_WEIGHT=0.5）の B定義 headline を集計する（論文誌 査読 1-4 対応）.

B定義ラベルは raw_json が無くても再現できるよう、b3_ablation で保存済みの
pooled_pairs_blabel_t0_20230101.csv（label, rf_dir_prob, p_irl_main を含む）を使う。
重み付きRFの確率は outputs/dynstart_rf_1snap_s777_weakneg05/fs_*/pair_predictions.csv から結合する。

出力: ペア単位（horizon別・pooled）と開発者単位（(dev,horizon) で max 集約）の AUC-ROC、
およびペア単位 pooled の差分の bootstrap 95% CI。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HZ = ["0-3", "3-6", "6-9", "9-12"]


def load(blabel_csv: str, weak_base: str) -> pd.DataFrame:
    base = pd.read_csv(blabel_csv)[
        ["developer", "directory", "horizon", "label", "p_irl_main", "rf_dir_prob"]
    ]
    parts = []
    for h in HZ:
        w = pd.read_csv(f"{weak_base}/fs_{h.replace('-', '_to_')}m/pair_predictions.csv")
        w = w[["developer", "directory", "rf_dir_prob"]].rename(columns={"rf_dir_prob": "rfw"})
        w["horizon"] = h
        parts.append(w)
    m = base.merge(pd.concat(parts), on=["developer", "directory", "horizon"], how="inner")
    assert len(m) == len(base), (len(m), len(base))
    m = m.rename(columns={"p_irl_main": "irl", "rf_dir_prob": "rf"})
    m["ens"] = (m["irl"] + m["rf"]) / 2
    m["ensw"] = (m["irl"] + m["rfw"]) / 2
    return m


COLS = ["irl", "rf", "rfw", "ens", "ensw"]


def aucs(d: pd.DataFrame) -> dict:
    return {c: roc_auc_score(d["label"], d[c]) for c in COLS}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--blabel-csv", default="outputs/ses_review/b3_ablation/pooled_pairs_blabel_t0_20230101.csv")
    p.add_argument("--weak-base", default="outputs/dynstart_rf_1snap_s777_weakneg05")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", default="outputs/dynstart_rf_1snap_s777_weakneg05/headline_blabel_weakneg.csv")
    a = p.parse_args()

    m = load(a.blabel_csv, a.weak_base)
    rows = []
    for unit in ["pair", "dev"]:
        if unit == "pair":
            d_all = m
        else:
            d_all = m.groupby(["developer", "horizon"], as_index=False)[["label", *COLS]].max()
        for h in [*HZ, "pooled"]:
            d = d_all if h == "pooled" else d_all[d_all["horizon"] == h]
            rows.append({"unit": unit, "horizon": h, "n": len(d), "pos": int(d["label"].sum()), **aucs(d)})
    out = pd.DataFrame(rows)

    # ペア単位 pooled の差分 bootstrap（ペア行の復元抽出）
    rng = np.random.default_rng(a.seed)
    y = m["label"].to_numpy()
    P = {c: m[c].to_numpy() for c in COLS}
    diffs = {"rfw-rf": [], "ensw-ens": [], "ensw-max(irl,rfw)": []}
    for _ in range(a.n_boot):
        i = rng.integers(0, len(y), len(y))
        s = {c: roc_auc_score(y[i], P[c][i]) for c in COLS}
        diffs["rfw-rf"].append(s["rfw"] - s["rf"])
        diffs["ensw-ens"].append(s["ensw"] - s["ens"])
        diffs["ensw-max(irl,rfw)"].append(s["ensw"] - max(s["irl"], s["rfw"]))
    for k, v in diffs.items():
        lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"pair pooled {k}: CI95 [{lo:+.4f}, {hi:+.4f}]")
        out.loc[len(out)] = {"unit": "pair_boot", "horizon": k, "n": a.n_boot, "pos": np.nan,
                             "irl": np.nan, "rf": lo, "rfw": hi, "ens": np.nan, "ensw": np.nan}
    out.to_csv(a.out_csv, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved: {a.out_csv}  (pair_boot 行は rf=CI下限, rfw=CI上限)")


if __name__ == "__main__":
    main()
