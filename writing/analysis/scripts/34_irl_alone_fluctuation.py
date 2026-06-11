# -*- coding: utf-8 -*-
"""IRL 単体の「断続性」に対する傾向検証 — 継続(active→) vs 復帰(inactive→) で IRL−RF が反転するか.

問い: アンサンブル上乗せだけでなく、IRL 単体にも「断続変動レビュアーで効く」傾向があるか。
方法: 各区間遷移で、前区間 active (label_prev=1) な部分集合での継続予測と、前区間 inactive
(label_prev=0) な部分集合での復帰予測を分け、IRL/RF/Ens の AUC を算出。
反転 = (復帰の IRL−RF) − (継続の IRL−RF) を開発者クラスタ bootstrap で検定 (seed 777, B=2000)。

入力: data/pair_trajectories.csv
出力: results/irl_alone_fluctuation.csv (per-transition + pooled), 標準出力に反転検定
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
SEED = 777
B = 2000
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
TRANS = [("0-3m", "3-6m"), ("3-6m", "6-9m"), ("6-9m", "9-12m")]


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def build():
    df = pd.read_csv(DATA / "pair_trajectories.csv")
    rows = []
    for hp, hc in TRANS:
        sub = df[["developer", "directory",
                  f"label_{hp}", f"label_{hc}",
                  f"irl_{hc}", f"rf_{hc}", f"avg_{hc}"]].copy()
        sub.columns = ["developer", "directory", "label_prev", "label_curr",
                       "irl", "rf", "ens"]
        sub = sub.dropna(subset=["label_prev", "label_curr", "irl", "rf"])
        sub["trans"] = f"{hp}->{hc}"
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def cluster_boot_delta(g):
    """開発者クラスタ bootstrap で (irl_auc-rf_auc) と (ens_auc-rf_auc) の分布を返す (高速版)."""
    y = g["label_curr"].to_numpy()
    s_irl = g["irl"].to_numpy()
    s_rf = g["rf"].to_numpy()
    s_ens = g["ens"].to_numpy()
    # 開発者ごとの行インデックスを事前計算
    groups = [np.asarray(v) for v in g.groupby("developer").indices.values()]
    n_dev = len(groups)
    rng = np.random.default_rng(SEED)
    d_irl, d_ens = [], []
    for _ in range(B):
        pick = rng.integers(0, n_dev, size=n_dev)
        rows = np.concatenate([groups[i] for i in pick])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        a_rf = auc(yy, s_rf[rows])
        d_irl.append(auc(yy, s_irl[rows]) - a_rf)
        d_ens.append(auc(yy, s_ens[rows]) - a_rf)
    return np.array(d_irl), np.array(d_ens)


def main():
    df = build()
    recs = []
    # --- per-transition ---
    for task, cond in [("continue", df["label_prev"] == 1),
                       ("return", df["label_prev"] == 0)]:
        d = df[cond]
        for tr in [t for t in d["trans"].unique()] + ["pooled"]:
            g = d if tr == "pooled" else d[d["trans"] == tr]
            y = g["label_curr"].values
            rec = {"task": task, "trans": tr, "n": len(g),
                   "base_rate": round(float(g["label_curr"].mean()), 4),
                   "irl": auc(y, g["irl"].values), "rf": auc(y, g["rf"].values),
                   "ens": auc(y, g["ens"].values)}
            rec["irl_minus_rf"] = rec["irl"] - rec["rf"]
            rec["ens_minus_rf"] = rec["ens"] - rec["rf"]
            recs.append(rec)
    res = pd.DataFrame(recs)
    res.to_csv(OUT / "irl_alone_fluctuation.csv", index=False)

    # --- 反転検定: pooled の (復帰 IRL-RF) - (継続 IRL-RF) ---
    cont = df[df["label_prev"] == 1]
    ret = df[df["label_prev"] == 0]
    dic, dec = cluster_boot_delta(cont)
    dir_, der = cluster_boot_delta(ret)
    # 同一 bootstrap seed で対応づけ済み (両者 SEED=777 で devs 別なので独立近似)
    n = min(len(dic), len(dir_))
    flip_irl = dir_[:n] - dic[:n]   # (復帰delta) - (継続delta) for IRL
    flip_ens = der[:n] - dec[:n]

    def ci(a):
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(np.mean(a))

    # --- delta サマリ CSV + 図 ---
    summ = []
    for name, arr in [("継続 IRL-RF", dic), ("復帰 IRL-RF", dir_),
                      ("継続 ENS-RF", dec), ("復帰 ENS-RF", der),
                      ("反転 IRL", flip_irl), ("反転 ENS", flip_ens)]:
        lo, hi, m = ci(arr)
        summ.append({"quantity": name, "mean_pp": round(m * 100, 2),
                     "ci_lo_pp": round(lo * 100, 2), "ci_hi_pp": round(hi * 100, 2),
                     "p_gt0": round(float(np.mean(arr > 0)), 3)})
    pd.DataFrame(summ).to_csv(OUT / "irl_alone_fluctuation_delta.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import japanize_matplotlib  # noqa: F401
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["継続\n(active→, 安定層)", "復帰\n(inactive→, 断続層)"]
    irl_m = [ci(dic)[2] * 100, ci(dir_)[2] * 100]
    irl_err = [[(ci(dic)[2] - ci(dic)[0]) * 100, (ci(dir_)[2] - ci(dir_)[0]) * 100],
               [(ci(dic)[1] - ci(dic)[2]) * 100, (ci(dir_)[1] - ci(dir_)[2]) * 100]]
    ens_m = [ci(dec)[2] * 100, ci(der)[2] * 100]
    ens_err = [[(ci(dec)[2] - ci(dec)[0]) * 100, (ci(der)[2] - ci(der)[0]) * 100],
               [(ci(dec)[1] - ci(dec)[2]) * 100, (ci(der)[1] - ci(der)[2]) * 100]]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, irl_m, w, yerr=irl_err, capsize=4, label="IRL − RF", color="#4C72B0")
    ax.bar(x + w / 2, ens_m, w, yerr=ens_err, capsize=4, label="Ensemble − RF", color="#55A868")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC 差 (pp, vs RF)")
    ax.set_title("IRL/アンサンブルの相対優位: 安定継続 vs 断続復帰\n"
                 "(開発者クラスタ bootstrap 95%CI; 反転 IRL +4.5pp[+0.5,+8.8] 有意)")
    ax.legend()
    for xi, m in zip(x - w / 2, irl_m):
        ax.annotate(f"{m:+.1f}", (xi, m), ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=9)
    for xi, m in zip(x + w / 2, ens_m):
        ax.annotate(f"{m:+.1f}", (xi, m), ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_irl_alone_fluctuation.pdf", format="pdf")
    print(f"\nsaved fig_irl_alone_fluctuation.pdf / irl_alone_fluctuation_delta.csv")

    print("=== per-transition / pooled AUC (results/irl_alone_fluctuation.csv) ===")
    show = res.copy()
    for c in ["irl", "rf", "ens", "irl_minus_rf", "ens_minus_rf"]:
        show[c] = show[c].round(4)
    print(show.to_string(index=False))

    print("\n=== pooled delta (開発者クラスタ bootstrap, B=2000, seed 777) ===")
    lo, hi, m = ci(dic); print(f"継続 IRL-RF : {m:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  P(>0)={np.mean(dic>0):.3f}")
    lo, hi, m = ci(dir_); print(f"復帰 IRL-RF : {m:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  P(>0)={np.mean(dir_>0):.3f}")
    lo, hi, m = ci(dec); print(f"継続 ENS-RF : {m:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]")
    lo, hi, m = ci(der); print(f"復帰 ENS-RF : {m:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]")
    print("\n--- 反転 (復帰delta − 継続delta) ---")
    lo, hi, m = ci(flip_irl)
    print(f"IRL の反転 : {m:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  P(反転>0)={np.mean(flip_irl>0):.3f}")
    lo, hi, m = ci(flip_ens)
    print(f"ENS の反転 : {m:+.4f}  95%CI[{lo:+.4f},{hi:+.4f}]  P(反転>0)={np.mean(flip_ens>0):.3f}")


if __name__ == "__main__":
    main()
