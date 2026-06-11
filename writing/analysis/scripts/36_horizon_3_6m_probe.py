# -*- coding: utf-8 -*-
"""3-6m 予測区間の特異性調査 — なぜ第2四半期(2022 Apr-Jun)でRF優位・経験依存が突出するか.

ユーザー指摘: fig_experience_axis_simple で 3-6m だけ RF の経験依存(偏ρ=0.44)が突出。
本スクリプトは (1) 区間別の予測像 (AUC/IRL-RF を開発者クラスタbootstrap CI付き),
(2) 3-6m の RF 優位が有意に大きいかの差分検定, (3) どの archetype/経験層で 3-6m の
RF優位が生じるかの局在, を調べる。

入力: data/pair_trajectories.csv, data/pair_archetype.csv, data/pair_repo_experience.csv
出力: results/horizon_3_6m_probe.csv, 標準出力に検定
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "results"
H = ["0-3m", "3-6m", "6-9m", "9-12m"]
SEED, B = 777, 2000


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan


def load_long():
    pt = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv").rename(columns={"dominant": "parch"})
    pa = pa[["developer", "directory", "parch"]]
    exp = pd.read_csv(DATA / "pair_repo_experience.csv")[
        ["developer", "directory", "prior_n", "repo_exp_covered"]]
    rows = []
    for h in H:
        s = pt[["developer", "directory", "archetype",
                f"label_{h}", f"irl_{h}", f"rf_{h}", f"avg_{h}"]].copy()
        s.columns = ["developer", "directory", "darch", "label", "irl", "rf", "ens"]
        s["h"] = h
        rows.append(s)
    lg = pd.concat(rows, ignore_index=True).dropna(subset=["label", "irl", "rf"])
    lg = lg.merge(pa, on=["developer", "directory"], how="left")
    lg = lg.merge(exp, on=["developer", "directory"], how="left")
    lg["label"] = lg["label"].astype(int)
    return lg


def boot_delta(g, n_boot=B):
    """開発者クラスタ bootstrap で IRL-RF, Ens-best の分布."""
    groups = [np.asarray(v) for v in g.groupby("developer").indices.values()]
    nd = len(groups)
    y = g["label"].to_numpy(); si = g["irl"].to_numpy(); sr = g["rf"].to_numpy(); se = g["ens"].to_numpy()
    rng = np.random.default_rng(SEED)
    di, dl = [], []
    for _ in range(n_boot):
        rows = np.concatenate([groups[i] for i in rng.integers(0, nd, nd)])
        yy = y[rows]
        if len(np.unique(yy)) < 2:
            continue
        ar = auc(yy, sr[rows]); ai = auc(yy, si[rows]); ae = auc(yy, se[rows])
        di.append(ai - ar); dl.append(ae - max(ai, ar))
    return np.array(di), np.array(dl)


def main():
    lg = load_long()
    recs = []
    boots = {}
    for h in H:
        g = lg[lg["h"] == h]
        y = g["label"].values
        di, dl = boot_delta(g)
        boots[h] = di
        recs.append({
            "h": h, "n": len(g), "base_rate": round(y.mean(), 4),
            "irl": round(auc(y, g["irl"]), 4), "rf": round(auc(y, g["rf"]), 4),
            "ens": round(auc(y, g["ens"]), 4),
            "irl_minus_rf": round(auc(y, g["irl"]) - auc(y, g["rf"]), 4),
            "irl_rf_ci": f"[{np.percentile(di,2.5)*100:+.1f},{np.percentile(di,97.5)*100:+.1f}]",
            "ens_lift": round(np.mean(dl), 4),
        })
    res = pd.DataFrame(recs)
    res.to_csv(OUT / "horizon_3_6m_probe.csv", index=False)
    print("=== 区間別 予測像 (開発者クラスタbootstrap) ===")
    print(res.to_string(index=False))

    # 3-6m の IRL-RF が他3区間平均より有意に低い(RF優位が大きい)か
    other = np.concatenate([boots[h] for h in H if h != "3-6m"])
    d36 = boots["3-6m"]
    n = min(len(d36), len(other))
    diff = d36[:n] - other[:n]
    print(f"\n=== 3-6m vs 他区間平均: (IRL-RF)_3-6m − (IRL-RF)_others ===")
    print(f"  diff mean={np.mean(diff)*100:+.2f}pp  95%CI[{np.percentile(diff,2.5)*100:+.2f},"
          f"{np.percentile(diff,97.5)*100:+.2f}]  P(3-6mがより RF優位)={np.mean(diff<0):.3f}")

    # archetype 局在: 区間別 × dev/pair archetype の IRL-RF
    print("\n=== 区間 × dev archetype の IRL-RF (pp) ===")
    for ar in ["A1", "A2", "A3", "A4"]:
        line = [f"{ar}:"]
        for h in H:
            g = lg[(lg["h"] == h) & (lg["darch"] == ar)]
            y = g["label"].values
            d = (auc(y, g["irl"]) - auc(y, g["rf"])) * 100 if len(np.unique(y)) > 1 else np.nan
            line.append(f"{h}={d:+.1f}")
        print("  " + "  ".join(line))
    print("\n=== 区間 × pair archetype の IRL-RF (pp) (parch: A1=P1..A4=P4) ===")
    for ar, p in [("A1", "P1"), ("A2", "P2"), ("A3", "P3"), ("A4", "P4")]:
        line = [f"{p}:"]
        for h in H:
            g = lg[(lg["h"] == h) & (lg["parch"] == ar)]
            y = g["label"].values
            d = (auc(y, g["irl"]) - auc(y, g["rf"])) * 100 if len(np.unique(y)) > 1 else np.nan
            line.append(f"{h}={d:+.1f}")
        print("  " + "  ".join(line))

    # 経験層別: 区間別に 高経験(prior_n>=6) ペアの占有と IRL-RF
    print("\n=== 区間別: 高経験(prior_n>=6)ペアの obs 占有率 と そこでの IRL-RF ===")
    cov = lg[lg["repo_exp_covered"] == True].copy()
    cov["prior_n"] = cov["prior_n"].fillna(0)
    for h in H:
        g = cov[cov["h"] == h]
        hi = g[g["prior_n"] >= 6]
        y = hi["label"].values
        d = (auc(y, hi["irl"]) - auc(y, hi["rf"])) * 100 if len(np.unique(y)) > 1 else np.nan
        print(f"  {h}: 高経験占有={len(hi)/len(g):.2f} (n={len(hi)}) base={y.mean():.3f} IRL-RF={d:+.1f}pp")


if __name__ == "__main__":
    main()
