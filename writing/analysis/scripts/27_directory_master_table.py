# -*- coding: utf-8 -*-
"""ディレクトリ単位の網羅的特性 master テーブル構築 [柱B 深掘りの基盤].

各ディレクトリ (repo, depth=2) について、予測性能・規模・archetype 構成・企業ドメイン
構成・脱落 timing をすべて 1 つの CSV に集約する。後続の facet 分析 (回帰 / ドメイン /
archetype / 枯渇検知) はすべてこの master テーブルを入力にする。

規約 (CLAUDE.md / RULES.md):
  - cold IRL (irl_*) / RF (rf_*) / アンサンブル (avg_*) のみ使用
  - directory = depth=2 path (pair_trajectories の directory 列そのもの)
  - 4 予測区間 (diagonal cell) をプールして per-dir AUC/AP を算出

入力:
  data/pair_trajectories.csv      (per-pair × 4 予測区間, developer=email)
  data/pair_archetype.csv         (pair archetype A1-A4 → P1-P4)
  results/horizon_dir_attrition.csv          (§13 脱落 timing)
  results/interval_attrition_timing_test.csv (§B-42 timing 検定)

出力:
  results/directory_master.csv
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
PMAP = {"A1": "P1", "A2": "P2", "A3": "P3", "A4": "P4"}

# 個人用 / 汎用ドメイン (企業ではない) — corporate 判定で除外
GENERIC_DOMAINS = {
    "gmail.com", "googlemail.com", "163.com", "qq.com", "126.com", "outlook.com",
    "hotmail.com", "yahoo.com", "yahoo.co.jp", "icloud.com", "foxmail.com",
    "protonmail.com", "live.com", "me.com", "sina.com", "yandex.ru", "139.com",
}


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def safe_ap(y, s):
    return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def hhi(shares: np.ndarray) -> float:
    """Herfindahl-Hirschman Index (0=完全分散, 1=独占)."""
    return float(np.sum(shares ** 2))


def main() -> None:
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv").rename(
        columns={"dominant": "pair_arch"})[["developer", "directory", "pair_arch"]]
    pred = pred.merge(pa, on=["developer", "directory"], how="left")
    pred["domain"] = pred["developer"].str.split("@").str[1].str.lower()
    pred["top_dir"] = pred["directory"].str.split("/").str[0]

    # long 化 (per pair × horizon)
    long_rows = []
    for h in HORIZONS:
        sub = pred[["developer", "directory", "top_dir", "domain", "archetype",
                    "pair_arch", f"label_{h}", f"irl_{h}", f"rf_{h}", f"avg_{h}"]].copy()
        sub.columns = ["developer", "directory", "top_dir", "domain", "dev_arch",
                       "pair_arch", "label", "irl", "rf", "ens"]
        sub["horizon"] = h
        long_rows.append(sub)
    lg = pd.concat(long_rows, ignore_index=True).dropna(subset=["label", "irl", "rf"])
    lg["label"] = lg["label"].astype(int)

    # attrition (§13) — dir 単位に集約: peak transition, range, 重み付き attrition, recall lift
    att = pd.read_csv(OUT / "horizon_dir_attrition.csv")
    timing = pd.read_csv(OUT / "interval_attrition_timing_test.csv")
    timing = timing.set_index("directory")

    rows = []
    for d, g in lg.groupby("directory"):
        pairs = g[["developer", "dev_arch", "pair_arch", "domain"]].drop_duplicates(
            "developer")
        n_pair = pairs["developer"].nunique()
        y = g["label"].values
        rec = {
            "directory": d, "top_dir": g["top_dir"].iloc[0],
            "n_pair": n_pair, "n_obs": len(g), "n_dev": n_pair,
            "n_pos": int(g["label"].sum()), "base_rate": round(g["label"].mean(), 4),
            "irl_auc": safe_auc(y, g["irl"].values),
            "rf_auc": safe_auc(y, g["rf"].values),
            "ens_auc": safe_auc(y, g["ens"].values),
            "irl_ap": safe_ap(y, g["irl"].values),
            "rf_ap": safe_ap(y, g["rf"].values),
            "ens_ap": safe_ap(y, g["ens"].values),
        }
        rec["delta_auc"] = (rec["irl_auc"] - rec["rf_auc"]
                            if pd.notna(rec["irl_auc"]) else np.nan)
        rec["ens_lift_auc"] = (rec["ens_auc"] - rec["rf_auc"]
                               if pd.notna(rec["ens_auc"]) else np.nan)
        rec["ens_lift_ap"] = (rec["ens_ap"] - rec["rf_ap"]
                              if pd.notna(rec["ens_ap"]) else np.nan)

        # --- dev archetype 構成比 (pair 単位) ---
        for a in ["A1", "A2", "A3", "A4"]:
            rec[f"dev_{a}_share"] = round((pairs["dev_arch"] == a).mean(), 4)
        # --- pair archetype 構成比 ---
        pa_known = pairs.dropna(subset=["pair_arch"])
        for a, p in PMAP.items():
            rec[f"pair_{p}_share"] = (round((pa_known["pair_arch"] == a).mean(), 4)
                                      if len(pa_known) else np.nan)
        rec["pair_arch_cov"] = round(len(pa_known) / n_pair, 3) if n_pair else np.nan

        # --- 企業ドメイン構成 ---
        dom = pairs["domain"].dropna()
        dom_counts = dom.value_counts()
        shares = (dom_counts / dom_counts.sum()).values if len(dom_counts) else np.array([])
        rec["n_domains"] = int(dom.nunique())
        rec["domain_hhi"] = round(hhi(shares), 4) if len(shares) else np.nan
        rec["top_domain"] = dom_counts.index[0] if len(dom_counts) else None
        rec["top_domain_share"] = round(float(shares[0]), 4) if len(shares) else np.nan
        corp = dom[~dom.isin(GENERIC_DOMAINS)]
        rec["corporate_share"] = round(len(corp) / len(dom), 4) if len(dom) else np.nan
        rec["n_corporate_domains"] = int(corp.nunique())
        # corporate のみでの HHI (企業が複数あっても 1 社集中か分散か)
        if len(corp):
            cc = corp.value_counts()
            rec["corp_hhi"] = round(hhi((cc / cc.sum()).values), 4)
            rec["top_corp_domain"] = cc.index[0]
            rec["top_corp_share_of_all"] = round(cc.iloc[0] / len(dom), 4)
        else:
            rec["corp_hhi"] = np.nan
            rec["top_corp_domain"] = None
            rec["top_corp_share_of_all"] = np.nan

        # --- 脱落 timing (§13 / §B-42) ---
        ga = att[(att["directory"] == d) & (att["n_active_prev"] >= 1)]
        if len(ga):
            wsum_n = ga["n_active_prev"].sum()
            rec["attrition_weighted"] = round(
                (ga["n_attrited"].sum()) / wsum_n, 4) if wsum_n else np.nan
            ga3 = ga[ga["n_active_prev"] >= 10]
            if len(ga3):
                pk = ga3.loc[ga3["attrition_rate"].idxmax()]
                rec["attr_peak_transition"] = f"{pk['h_prev']}→{pk['h_curr']}"
                rec["attr_peak_rate"] = round(float(pk["attrition_rate"]), 4)
            else:
                rec["attr_peak_transition"] = None
                rec["attr_peak_rate"] = np.nan
        else:
            rec["attrition_weighted"] = np.nan
            rec["attr_peak_transition"] = None
            rec["attr_peak_rate"] = np.nan
        if d in timing.index:
            rec["attr_range"] = float(timing.loc[d, "attr_range"])
            rec["attr_timing_p_mc"] = float(timing.loc[d, "p_montecarlo"])
            rec["attr_timing_sig"] = bool(timing.loc[d, "sig_mc"])
        else:
            rec["attr_range"] = np.nan
            rec["attr_timing_p_mc"] = np.nan
            rec["attr_timing_sig"] = False
        rows.append(rec)

    master = pd.DataFrame(rows)
    # reliable フラグ (両クラス & 十分な規模)
    master["auc_defined"] = master["irl_auc"].notna()
    master["reliable"] = master["auc_defined"] & (master["n_pair"] >= 20)
    master = master.sort_values("n_pair", ascending=False)
    master.to_csv(OUT / "directory_master.csv", index=False)

    # --- サニティチェック: 全 dir プール AUC が論文 Table 1 に一致するか ---
    yall = lg["label"].values
    print("=== sanity (全 obs プール AUC; 論文 4区間平均 ~ IRL .775 / RF .794 / Ens .816) ===")
    print(f"  IRL={safe_auc(yall, lg['irl'].values):.3f} "
          f"RF={safe_auc(yall, lg['rf'].values):.3f} "
          f"Ens={safe_auc(yall, lg['ens'].values):.3f}")
    print(f"\nsaved {OUT / 'directory_master.csv'}")
    print(f"  全 {len(master)} dir / AUC定義可 {int(master['auc_defined'].sum())} "
          f"/ reliable(n_pair>=20) {int(master['reliable'].sum())}")
    print(f"  列: {list(master.columns)}")
    rel = master[master["reliable"]]
    print(f"\n=== reliable {len(rel)} dir のうち IRL優位/RF優位/拮抗 ===")
    print(f"  IRL優位 (Δ>0.03): {int((rel['delta_auc']>0.03).sum())}")
    print(f"  RF優位 (Δ<-0.03): {int((rel['delta_auc']<-0.03).sum())}")
    print(f"  拮抗 (|Δ|<=0.03): {int((rel['delta_auc'].abs()<=0.03).sum())}")
    print(f"\n=== ドメイン構成サマリ (reliable) ===")
    print(f"  domain_hhi: median={rel['domain_hhi'].median():.3f} "
          f"min={rel['domain_hhi'].min():.3f} max={rel['domain_hhi'].max():.3f}")
    print(f"  corporate_share: median={rel['corporate_share'].median():.3f}")


if __name__ == "__main__":
    main()
