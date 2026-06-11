# -*- coding: utf-8 -*-
"""プロジェクト単位 (top_dir) の特性 master テーブル + 脱落timing [project粒度].

ディレクトリ(depth=2)版 (27_directory_master_table.py) の project (top_dir) 版。
各 project について 予測性能・規模・archetype 構成・脱落 timing を集約する。
脱落 timing は pair_trajectories のラベルから直接 project 単位で再計算する
(序盤 0-3m→3-6m / 中盤 3-6m→6-9m / 終盤 6-9m→9-12m)。

入力:
  data/pair_trajectories.csv, data/pair_archetype.csv
出力:
  results/project_master.csv
  results/project_attrition_timing.csv (project × 3遷移)
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
TRANS = [("0-3m", "3-6m", "序盤"), ("3-6m", "6-9m", "中盤"), ("6-9m", "9-12m", "終盤")]
PMAP = {"A1": "P1", "A2": "P2", "A3": "P3", "A4": "P4"}
GENERIC = {"gmail.com", "googlemail.com", "163.com", "qq.com", "126.com",
           "outlook.com", "hotmail.com", "yahoo.com", "yahoo.co.jp", "icloud.com",
           "foxmail.com", "protonmail.com", "live.com", "me.com", "sina.com",
           "yandex.ru", "139.com"}


def auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def ap(y, s):
    return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def hhi(sh):
    return float(np.sum(sh ** 2)) if len(sh) else np.nan


def main() -> None:
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv").rename(
        columns={"dominant": "pair_arch"})[["developer", "directory", "pair_arch"]]
    pred = pred.merge(pa, on=["developer", "directory"], how="left")
    pred["top_dir"] = pred["directory"].str.split("/").str[0]
    pred["domain"] = pred["developer"].str.split("@").str[1].str.lower()

    # --- 脱落 timing を project 単位で直接計算 ---
    att_rows = []
    for proj, g in pred.groupby("top_dir"):
        for hp, hc, name in TRANS:
            prev = g[(g[f"label_{hp}"] == 1)]
            n_prev = len(prev)
            if n_prev == 0:
                continue
            n_attr = int((prev[f"label_{hc}"] == 0).sum())
            att_rows.append({"top_dir": proj, "transition": name,
                             "h_prev": hp, "h_curr": hc,
                             "n_active_prev": n_prev, "n_attrited": n_attr,
                             "attrition_rate": round(n_attr / n_prev, 4)})
    att = pd.DataFrame(att_rows)
    att.to_csv(OUT / "project_attrition_timing.csv", index=False)

    # --- long 化 (per pair × horizon) ---
    long_rows = []
    for h in HORIZONS:
        sub = pred[["developer", "directory", "top_dir", "domain", "archetype",
                    "pair_arch", f"label_{h}", f"irl_{h}", f"rf_{h}", f"avg_{h}"]].copy()
        sub.columns = ["developer", "directory", "top_dir", "domain", "dev_arch",
                       "pair_arch", "label", "irl", "rf", "ens"]
        long_rows.append(sub)
    lg = pd.concat(long_rows, ignore_index=True).dropna(subset=["label", "irl", "rf"])
    lg["label"] = lg["label"].astype(int)

    rows = []
    for proj, g in lg.groupby("top_dir"):
        pairs = g[["developer", "directory", "dev_arch", "pair_arch", "domain"]
                  ].drop_duplicates(["developer", "directory"])
        y = g["label"].values
        rec = {
            "top_dir": proj,
            "n_pair": len(pairs), "n_obs": len(g),
            "n_dev": pairs["developer"].nunique(),
            "n_dir": pairs["directory"].nunique(),
            "n_pos": int(g["label"].sum()), "base_rate": round(g["label"].mean(), 4),
            "irl_auc": auc(y, g["irl"].values), "rf_auc": auc(y, g["rf"].values),
            "ens_auc": auc(y, g["ens"].values),
            "irl_ap": ap(y, g["irl"].values), "rf_ap": ap(y, g["rf"].values),
            "ens_ap": ap(y, g["ens"].values),
        }
        rec["delta_auc"] = (rec["irl_auc"] - rec["rf_auc"]
                            if pd.notna(rec["irl_auc"]) else np.nan)
        rec["ens_lift_auc"] = (rec["ens_auc"] - rec["rf_auc"]
                               if pd.notna(rec["ens_auc"]) else np.nan)
        rec["ens_lift_ap"] = (rec["ens_ap"] - rec["rf_ap"]
                              if pd.notna(rec["ens_ap"]) else np.nan)
        # archetype 構成
        for a in ["A1", "A2", "A3", "A4"]:
            rec[f"dev_{a}_share"] = round((pairs["dev_arch"] == a).mean(), 4)
        pk = pairs.dropna(subset=["pair_arch"])
        for a, p in PMAP.items():
            rec[f"pair_{p}_share"] = (round((pk["pair_arch"] == a).mean(), 4)
                                      if len(pk) else np.nan)
        # domain (考察用)
        dom = pairs["domain"].dropna()
        dc = dom.value_counts()
        rec["n_domains"] = int(dom.nunique())
        rec["domain_hhi"] = hhi((dc / dc.sum()).values) if len(dc) else np.nan
        rec["top_domain"] = dc.index[0] if len(dc) else None
        corp = dom[~dom.isin(GENERIC)]
        rec["corporate_share"] = round(len(corp) / len(dom), 4) if len(dom) else np.nan
        # 脱落 timing
        ga = att[att["top_dir"] == proj]
        for _, t in ga.iterrows():
            rec[f"attr_{t['transition']}"] = t["attrition_rate"]
        if len(ga):
            pk_t = ga.loc[ga["attrition_rate"].idxmax()]
            rec["attr_peak"] = pk_t["transition"]
            rec["attr_range"] = round(float(ga["attrition_rate"].max()
                                            - ga["attrition_rate"].min()), 4)
            rec["attr_weighted"] = round(
                ga["n_attrited"].sum() / ga["n_active_prev"].sum(), 4)
        rows.append(rec)

    pm = pd.DataFrame(rows)
    pm["auc_defined"] = pm["irl_auc"].notna()
    pm["reliable"] = pm["auc_defined"] & (pm["n_pair"] >= 30)
    pm = pm.sort_values("n_pair", ascending=False)
    pm.to_csv(OUT / "project_master.csv", index=False)

    # sanity
    yall = lg["label"].values
    print("=== sanity (全 obs プール; ~ IRL .775 / RF .796 / Ens .817) ===")
    print(f"  IRL={auc(yall, lg['irl'].values):.3f} RF={auc(yall, lg['rf'].values):.3f} "
          f"Ens={auc(yall, lg['ens'].values):.3f}")
    print(f"\nsaved project_master.csv: {len(pm)} project "
          f"(AUC定義可 {int(pm['auc_defined'].sum())}, reliable n_pair>=30 "
          f"{int(pm['reliable'].sum())})")
    rel = pm[pm["reliable"]].copy()
    print(f"\n=== reliable {len(rel)} project: 予測性 top/bottom ===")
    show = ["top_dir", "n_pair", "base_rate", "irl_auc", "rf_auc", "ens_auc",
            "delta_auc", "dev_A4_share", "pair_P1_share", "attr_peak", "attr_range"]
    print(rel.sort_values("ens_auc", ascending=False).head(8)[show].to_string(index=False))
    print("  ...")
    print(rel.sort_values("ens_auc").head(6)[show].to_string(index=False))
    print(f"\n=== delta_auc (IRL−RF) で並べた project ===")
    print(rel.sort_values("delta_auc", ascending=False)[
        ["top_dir", "n_pair", "delta_auc", "ens_lift_auc", "dev_A4_share",
         "pair_P1_share", "pair_P2_share"]].to_string(index=False))


if __name__ == "__main__":
    main()
