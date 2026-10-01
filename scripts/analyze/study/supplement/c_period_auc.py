# -*- coding: utf-8 -*-
"""補足資料 C群の前処理: 原型 × 評価期間 × モデル の AUC を用意する.

OpenStack には k4_period_auc.csv が無い (AOSP のみ)。既存の
archetype_period_auc.py は raw JSON (data/raw_json/) を要求するが
ローカルに無いため、B定義ラベル付きのプール済み予測
outputs/ses_review/b3_ablation/pooled_pairs_blabel_t0_20230101.csv
から同じ集計軸で計算する。

AOSP は既存 k4_period_auc.csv に auc_ens 列が無いため、pair_predictions
から ens を計算し、既存 CSV の IRL/RF と一致するか検算した上で合流する。

出力: outputs/analysis/ses_supplement/tab/period_auc_{dataset}_{level}.csv
実行: .venv/bin/python scripts/analyze/study/supplement/c_period_auc.py
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _common import HORIZONS, ROOT, SOURCES, TAB  # noqa: E402

OS_POOLED = ROOT / "outputs/ses_review/b3_ablation/pooled_pairs_blabel_t0_20230101.csv"
OS_HEADLINE = ROOT / "outputs/ses_review/b3_ablation/headline_blabel_irl_main.csv"
AOSP_IRL = ROOT / "outputs/aosp32/dynstartB_ep150/lstm_baseline"
AOSP_RF = ROOT / "outputs/aosp32/rf_gbdt_1snap_v2"
TOL = 5e-4
# dominant 列はペア単位でも "A1..A4" (表示名の P1..P4 は Source.label で付ける)
ARCH_IDS = ["A1", "A2", "A3", "A4"]


def _auc(y, p) -> float:
    if len(y) < 10 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _rows(long: pd.DataFrame, key_cols: list[str], unit_to_arch: dict,
          arch_ids: list[str]) -> pd.DataFrame:
    """archetype × horizon × model の AUC。archetype_period_auc.py:43-90 と同じ軸。"""
    a = long.copy()
    a["archetype"] = [unit_to_arch.get(k) for k in
                      (zip(*[a[c] for c in key_cols]) if len(key_cols) > 1
                       else ((v,) for v in a[key_cols[0]]))]
    a = a.dropna(subset=["archetype"])
    rows = []
    for arch in arch_ids:
        sub_a = a[a["archetype"] == arch]
        for h in HORIZONS:
            sub = sub_a[sub_a["horizon"] == h]
            rows.append({
                "archetype": arch, "horizon": h, "n": len(sub),
                "auc_cold_irl": _auc(sub["label"], sub["p_irl"]),
                "auc_rf_dir": _auc(sub["label"], sub["p_rf"]),
                "auc_ens": _auc(sub["label"], sub["p_ens"]),
            })
    df = pd.DataFrame(rows)
    df["delta_pp"] = (df["auc_cold_irl"] - df["auc_rf_dir"]) * 100
    df["ens_gain_pp"] = (df["auc_ens"]
                         - df[["auc_cold_irl", "auc_rf_dir"]].max(axis=1)) * 100
    return df


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ OpenStack
def load_openstack_long() -> pd.DataFrame:
    df = pd.read_csv(OS_POOLED)
    long = df.rename(columns={"p_irl_main": "p_irl", "rf_dir_prob": "p_rf"})
    # アンサンブルは verify_blabel_headline.py:87 と同じ単純平均
    long["p_ens"] = (long["p_irl"] + long["p_rf"]) / 2
    return long[["developer", "directory", "horizon", "label",
                 "p_irl", "p_rf", "p_ens"]]


def verify_openstack(long: pd.DataFrame) -> None:
    """原型で分ける前に、全体 AUC が headline CSV と一致することを確認する。"""
    head = pd.read_csv(OS_HEADLINE).set_index("horizon")
    print("[verify] OpenStack 全体 AUC vs headline_blabel_irl_main.csv")
    ok = True
    for h in HORIZONS + ["pooled"]:
        sub = long if h == "pooled" else long[long["horizon"] == h]
        got = {"IRL_roc": _auc(sub["label"], sub["p_irl"]),
               "RF_roc": _auc(sub["label"], sub["p_rf"]),
               "Ens_roc": _auc(sub["label"], sub["p_ens"])}
        for col, v in got.items():
            exp = float(head.loc[h, col])
            mark = "OK " if abs(v - exp) < TOL else "NG "
            ok &= abs(v - exp) < TOL
            print(f"  {mark}{h:>6} {col:<8} 再計算={v:.5f} headline={exp:.5f}")
    assert ok, "OpenStack の全体 AUC が headline CSV と一致しない。中断する。"
    print("[verify] 一致を確認")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ AOSP
def load_aosp_long() -> pd.DataFrame:
    """AOSP の pair_predictions (ラベル焼き込み済み) を horizon 方向に結合する。"""
    frames = []
    for h in HORIZONS:
        irl = pd.read_csv(AOSP_IRL / f"train_{h}m" / f"eval_{h}m" / "pair_predictions.csv")
        rf = pd.read_csv(AOSP_RF / f"fs_{h.replace('-', '_to_')}m" / "pair_predictions.csv")
        m = irl[["developer", "directory", "label", "irl_dir_prob"]].merge(
            rf[["developer", "directory", "rf_dir_prob"]],
            on=["developer", "directory"], how="inner", validate="one_to_one")
        m = m.rename(columns={"irl_dir_prob": "p_irl", "rf_dir_prob": "p_rf"})
        m["p_ens"] = (m["p_irl"] + m["p_rf"]) / 2
        m["horizon"] = h
        frames.append(m)
    return pd.concat(frames, ignore_index=True)


def compare_aosp(new: pd.DataFrame, level: str) -> tuple[pd.DataFrame, bool]:
    """既存 k4_period_auc.csv (B定義ラベル) と、焼き込みラベルでの再計算を突き合わせる。"""
    src = SOURCES[("aosp", level)]
    old = pd.read_csv(src.path("k4_period_auc.csv"))
    merged = old.merge(new, on=["archetype", "horizon"], suffixes=("_old", "_new"))
    d_irl = (merged["auc_cold_irl_old"] - merged["auc_cold_irl_new"]).abs().max()
    d_rf = (merged["auc_rf_dir_old"] - merged["auc_rf_dir_new"]).abs().max()
    match = bool(d_irl < TOL and d_rf < TOL)
    print(f"[verify] AOSP/{level}: 既存CSV との最大差 IRL={d_irl:.5f} RF={d_rf:.5f} "
          f"→ {'一致' if match else '不一致（ラベル定義の違い）'}")
    return merged, match


def overall_auc(long: pd.DataFrame, label: str) -> pd.DataFrame:
    """期間別＋全体の AUC (原型で分けないアンサンブル用)。"""
    rows = []
    for h in HORIZONS + ["pooled"]:
        sub = long if h == "pooled" else long[long["horizon"] == h]
        rows.append({
            "horizon": h, "n": len(sub), "pos": int(sub["label"].sum()),
            "auc_irl": _auc(sub["label"], sub["p_irl"]),
            "auc_rf": _auc(sub["label"], sub["p_rf"]),
            "auc_ens": _auc(sub["label"], sub["p_ens"]),
        })
    df = pd.DataFrame(rows)
    df["ens_gain_pp"] = (df["auc_ens"]
                         - df[["auc_irl", "auc_rf"]].max(axis=1)) * 100
    out = TAB / f"overall_auc_{label}.csv"
    df.to_csv(out, index=False)
    print(f"[done] {out.relative_to(ROOT)}")
    return df


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)

    # ── OpenStack ──
    long = load_openstack_long()
    print(f"[load] OpenStack pooled: {len(long):,} 行 "
          f"({long['developer'].nunique():,} dev / "
          f"{long.groupby(['developer', 'directory']).ngroups:,} ペア)")
    verify_openstack(long)
    overall_auc(long, "openstack")
    for level in ["pair", "dev"]:
        src = SOURCES[("openstack", level)]
        alpha = pd.read_csv(src.path(src.alpha_file),
                            index_col=0 if level == "dev" else None)
        if level == "dev":
            unit_to_arch = {(e,): d for e, d in zip(alpha.index, alpha["dominant"])}
            keys = ["developer"]
        else:
            unit_to_arch = {(r.developer, r.directory): r.dominant
                            for r in alpha.itertuples()}
            keys = ["developer", "directory"]
        df = _rows(long, keys, unit_to_arch, ARCH_IDS)
        out = TAB / f"period_auc_openstack_{level}.csv"
        df.to_csv(out, index=False)
        covered = df["n"].sum()
        print(f"[done] {out.relative_to(ROOT)} "
              f"（原型が付いた行 {covered:,} / 全 {len(long):,}）")

    # ── AOSP ──
    ao_long = load_aosp_long()
    print(f"[load] AOSP pair_predictions: {len(ao_long):,} 行 "
          f"({ao_long.groupby(['developer', 'directory']).ngroups:,} ペア × 4期間)")
    overall_auc(ao_long, "aosp")
    for level in ["pair", "dev"]:
        src = SOURCES[("aosp", level)]
        alpha = pd.read_csv(src.path(src.alpha_file),
                            index_col=0 if level == "dev" else None)
        if level == "dev":
            unit_to_arch = {(e,): d for e, d in zip(alpha.index, alpha["dominant"])}
            keys = ["developer"]
        else:
            unit_to_arch = {(r.developer, r.directory): r.dominant
                            for r in alpha.itertuples()}
            keys = ["developer", "directory"]
        recomputed = _rows(ao_long, keys, unit_to_arch, ARCH_IDS)
        merged, match = compare_aosp(recomputed, level)
        old = pd.read_csv(src.path("k4_period_auc.csv"))
        if match:
            # 既存CSV(B定義ラベル)に ens を合流
            out_df = old.merge(
                recomputed[["archetype", "horizon", "auc_ens", "ens_gain_pp"]],
                on=["archetype", "horizon"])
            out_df["ens_source"] = "再計算（既存CSVのIRL/RFと一致を確認）"
        else:
            out_df = old.copy()
            out_df["auc_ens"] = np.nan
            out_df["ens_gain_pp"] = np.nan
            out_df["ens_source"] = "未算出（ラベル定義が既存CSVと異なるため合流しない）"
            recomputed.to_csv(
                TAB / f"period_auc_aosp_{level}_bakedlabel.csv", index=False)
            print(f"       → 焼き込みラベル版を period_auc_aosp_{level}_bakedlabel.csv "
                  f"に別途保存")
        out = TAB / f"period_auc_aosp_{level}.csv"
        out_df.to_csv(out, index=False)
        print(f"[done] {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
