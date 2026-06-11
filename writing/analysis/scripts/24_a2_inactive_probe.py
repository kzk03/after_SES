# -*- coding: utf-8 -*-
"""A2 (レビュー専念型) の IRL 寄与を rank-based で防御する [W5].

W1 (25_archetype_delta_significance.py) で「IRL 単独の A2 優位」は開発者クラスタ
bootstrap で有意でない (Δ̄=+3.2pp, 95%CI [-2.1,+7.8]) と判明した。本スクリプトは
A2 を別の 2 つの厳密なレンズで再検証し、論文に残せる主張を探す:

  Part 1 [アンサンブル寄与の有意性]:
    IRL 単独では勝てなくても、A2 に IRL を混ぜるとランキングが有意に改善するか?
    Ens lift = Ens − RF を AUC / AP で測り、開発者クラスタ bootstrap 95% CI を付与。
    「IRL は A2 に独立な情報を供給する」を主張できるかを判定。
    (A2 dev の 554 ペア中 530 が A2×P4 なので、A2 ≒ A2×P4)

  Part 2 [機序の直接検証 — rank-based]:
    A2 の IRL 寄与が「過去 active だが沈黙したペア」の正しい低位ランクから来るか。
    cold IRL は未校正のため prob 中央値の絶対比較は不可。代わりに t0 の累積活動量
    (active_months_ratio / avg_activity_gap) で A2 を 3 分位に分け、各群で IRL/RF/Ens の
    AUC を測り、IRL の寄与がどの群に集中するかを示す。

入力:
  data/pair_trajectories.csv   (archetype = 開発者 archetype)
  data/pair_archetype.csv      (dominant = ペア archetype A1-A4 → P1-P4)
  data/feature_values.csv      (__email, __snapshot; t0=2022-01-01 を抽出)

出力:
  results/a2_ensemble_lift_ci.csv      A2 / A2×P4 の Ens lift と CI
  results/a2_mechanism_probe.csv       3 分位 × 指標の AUC
  results/a2_probe_summary.json
  results/fig_a2_defense.pdf
"""
from __future__ import annotations

import json
import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
HORIZON_JP = {"0-3m": "0-3 か月", "3-6m": "3-6 か月",
              "6-9m": "6-9 か月", "9-12m": "9-12 か月"}
T0 = "2022-01-01"
N_BOOT = 2000
SEED = 777


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def safe_ap(y, s):
    return float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else np.nan


def metrics_for(d: pd.DataFrame, h: str) -> dict:
    y = d[f"label_{h}"].astype(int).values
    irl, rf, ens = d[f"irl_{h}"].values, d[f"rf_{h}"].values, d[f"avg_{h}"].values
    return {
        "irl_auc": safe_auc(y, irl), "rf_auc": safe_auc(y, rf),
        "ens_auc": safe_auc(y, ens),
        "irl_ap": safe_ap(y, irl), "rf_ap": safe_ap(y, rf),
        "ens_ap": safe_ap(y, ens),
    }


def cluster_boot_lift(sub: pd.DataFrame, value_fn, rng: np.random.Generator,
                      n_boot: int = N_BOOT) -> tuple[float, float, float, float]:
    """開発者クラスタ bootstrap で value_fn(再標本) の点推定・CI・two-sided p を返す."""
    sub = sub.reset_index(drop=True)
    devs = sub["developer"].unique()
    idx_by_dev = {dv: sub.index[sub["developer"] == dv].to_numpy() for dv in devs}
    obs = value_fn(sub)
    vals = np.full(n_boot, np.nan)
    n_dev = len(devs)
    for b in range(n_boot):
        pick = rng.integers(0, n_dev, size=n_dev)
        rows = np.concatenate([idx_by_dev[devs[i]] for i in pick])
        v = value_fn(sub.loc[rows])
        vals[b] = v
    vals = vals[~np.isnan(vals)]
    if len(vals) < 50:
        return obs, np.nan, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    p = float(min(1.0, 2 * min(np.mean(vals <= 0), np.mean(vals >= 0))))
    return float(obs), float(lo), float(hi), p


def make_lift_fn(metric: str):
    """metric in {ens_lift_auc, ens_lift_ap, delta_auc} の 4 予測区間平均を返す関数."""
    def fn(df: pd.DataFrame) -> float:
        vals = []
        for h in HORIZONS:
            d = df.dropna(subset=[f"label_{h}", f"irl_{h}", f"rf_{h}"])
            if len(np.unique(d[f"label_{h}"])) < 2:
                continue
            m = metrics_for(d, h)
            if metric == "ens_lift_auc":
                vals.append(m["ens_auc"] - m["rf_auc"])
            elif metric == "ens_lift_ap":
                vals.append(m["ens_ap"] - m["rf_ap"])
            elif metric == "delta_auc":
                vals.append(m["irl_auc"] - m["rf_auc"])
        return float(np.nanmean(vals)) if vals else np.nan
    return fn


def part1_ensemble_lift(a2: pd.DataFrame, a2p4: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for name, sub in [("A2 全体 (n=%d)" % len(a2), a2),
                      ("A2×P4 (n=%d)" % len(a2p4), a2p4)]:
        for metric, label in [("delta_auc", "Δ AUC (IRL−RF)"),
                              ("ens_lift_auc", "Ens lift AUC"),
                              ("ens_lift_ap", "Ens lift AP")]:
            obs, lo, hi, p = cluster_boot_lift(sub, make_lift_fn(metric), rng)
            sig = bool(np.isfinite(lo) and (lo > 0 or hi < 0))
            rows.append({
                "group": name, "metric": metric, "metric_jp": label,
                "value_pp": round(obs * 100, 2),
                "ci_lo_pp": round(lo * 100, 2), "ci_hi_pp": round(hi * 100, 2),
                "boot_p": round(p, 4), "sig": sig,
            })
            print(f"  {name:18s} {label:16s} = {obs*100:+.1f}pp "
                  f"[{lo*100:+.1f},{hi*100:+.1f}] p={p:.3f} {'★' if sig else ''}")
    return pd.DataFrame(rows)


def part2_mechanism(a2_feat: pd.DataFrame, split_col: str) -> pd.DataFrame:
    """A2 を split_col の 3 分位に分け、各群で IRL/RF/Ens の pooled AUC (4 区間プール)."""
    rows = []
    q = a2_feat[split_col].quantile([1 / 3, 2 / 3]).values
    def bucket(v):
        if v <= q[0]:
            return "低 (Q1)"
        if v <= q[1]:
            return "中 (Q2)"
        return "高 (Q3)"
    a2_feat = a2_feat.copy()
    a2_feat["bucket"] = a2_feat[split_col].apply(bucket)
    # 4 予測区間をプールして long 化
    long_rows = []
    for _, r in a2_feat.iterrows():
        for h in HORIZONS:
            if pd.isna(r[f"label_{h}"]):
                continue
            long_rows.append({"bucket": r["bucket"], "y": int(r[f"label_{h}"]),
                              "irl": r[f"irl_{h}"], "rf": r[f"rf_{h}"],
                              "ens": r[f"avg_{h}"]})
    lg = pd.DataFrame(long_rows)
    for bk in ["低 (Q1)", "中 (Q2)", "高 (Q3)"]:
        sub = lg[lg["bucket"] == bk]
        y = sub["y"].values
        rows.append({
            "split_col": split_col, "bucket": bk,
            "n_obs": len(sub), "n_pos": int(sub["y"].sum()),
            "base_rate": round(sub["y"].mean(), 3),
            "irl_auc": round(safe_auc(y, sub["irl"].values), 3),
            "rf_auc": round(safe_auc(y, sub["rf"].values), 3),
            "ens_auc": round(safe_auc(y, sub["ens"].values), 3),
            "delta_auc": round(safe_auc(y, sub["irl"].values)
                               - safe_auc(y, sub["rf"].values), 3),
            "ens_lift_auc": round(safe_auc(y, sub["ens"].values)
                                  - safe_auc(y, sub["rf"].values), 3),
        })
    return pd.DataFrame(rows)


def plot_defense(lift: pd.DataFrame, mech: pd.DataFrame,
                 out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # 左: Ens lift forest (A2×P4)
    ax = axes[0]
    sub = lift[lift["group"].str.startswith("A2×P4")]
    ys = range(len(sub))
    colors = ["#7f7f7f", "#2ca02c", "#1f77b4"]
    for i, (_, r) in enumerate(sub.iterrows()):
        ax.plot([r["ci_lo_pp"], r["ci_hi_pp"]], [i, i], color=colors[i], lw=2.5)
        ax.scatter([r["value_pp"]], [i], color=colors[i],
                   s=110 if r["sig"] else 50,
                   edgecolor="black" if r["sig"] else "none", linewidth=1.3, zorder=3)
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(sub["metric_jp"].tolist(), fontsize=11)
    ax.set_xlabel("値 (pp)  [95% CI, 塗り＝有意]", fontsize=10)
    ax.set_title("A2×P4 (n=530): IRL 単独 vs アンサンブル寄与", fontsize=12)
    ax.grid(axis="x", alpha=0.25)

    # 右: mechanism probe (active_months_ratio 3分位)
    ax = axes[1]
    m = mech[mech["split_col"] == "active_months_ratio"]
    x = np.arange(len(m))
    w = 0.25
    ax.bar(x - w, m["irl_auc"], w, label="IRL", color="#1f77b4")
    ax.bar(x, m["rf_auc"], w, label="RF", color="#d62728")
    ax.bar(x + w, m["ens_auc"], w, label="Ens", color="#2ca02c")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(m["bucket"].tolist(), fontsize=10)
    ax.set_ylim(0.4, max(0.85, m[["irl_auc", "rf_auc", "ens_auc"]].max().max() + 0.05))
    ax.set_ylabel("pooled AUC (4 予測区間)", fontsize=10)
    ax.set_xlabel("A2 を活動月比で 3 分位", fontsize=10)
    ax.set_title("A2 機序: IRL 寄与はどの活動水準に集中するか", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("A2 (レビュー専念型) への IRL 寄与の検証", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    pred = pd.read_csv(DATA / "pair_trajectories.csv")
    pa = pd.read_csv(DATA / "pair_archetype.csv").rename(
        columns={"dominant": "pair_arch_code"})[
        ["developer", "directory", "pair_arch_code"]]
    df = pred.merge(pa, on=["developer", "directory"], how="left")

    a2 = df[df["archetype"] == "A2"].copy()
    a2p4 = a2[a2["pair_arch_code"] == "A4"].copy()  # A4 code = P4
    print(f"A2 dev pairs={len(a2)}  | A2×P4={len(a2p4)}  "
          f"| A2×P2={int((a2['pair_arch_code']=='A2').sum())}")

    print("\n=== Part 1: A2 への IRL 寄与 (開発者クラスタ bootstrap, 4 区間平均) ===")
    lift = part1_ensemble_lift(a2, a2p4)
    lift.to_csv(OUT / "a2_ensemble_lift_ci.csv", index=False)

    # --- t0 特徴量を join (機序プローブ用) ---
    fv = pd.read_csv(DATA / "feature_values.csv")
    fv = fv[fv["__snapshot"] == T0][
        ["__email", "active_months_ratio", "avg_activity_gap",
         "recent_activity_frequency"]].rename(columns={"__email": "developer"})
    fv = fv.drop_duplicates("developer")
    a2_feat = a2.merge(fv, on="developer", how="inner")
    print(f"\n=== Part 2: 機序プローブ (A2 with t0 features, n={len(a2_feat)}) ===")
    mech_amr = part2_mechanism(a2_feat, "active_months_ratio")
    mech_gap = part2_mechanism(a2_feat, "avg_activity_gap")
    mech = pd.concat([mech_amr, mech_gap], ignore_index=True)
    mech.to_csv(OUT / "a2_mechanism_probe.csv", index=False)
    print("[active_months_ratio 3分位]")
    print(mech_amr[["bucket", "n_obs", "base_rate", "irl_auc", "rf_auc",
                    "ens_auc", "delta_auc", "ens_lift_auc"]].to_string(index=False))
    print("[avg_activity_gap 3分位]")
    print(mech_gap[["bucket", "n_obs", "base_rate", "irl_auc", "rf_auc",
                    "ens_auc", "delta_auc", "ens_lift_auc"]].to_string(index=False))

    # --- 判定サマリ ---
    a2p4_lift_ap = lift[(lift["group"].str.startswith("A2×P4"))
                        & (lift["metric"] == "ens_lift_ap")].iloc[0]
    a2p4_delta = lift[(lift["group"].str.startswith("A2×P4"))
                      & (lift["metric"] == "delta_auc")].iloc[0]
    verdict = {
        "irl_alone_a2p4_delta_auc_pp": a2p4_delta["value_pp"],
        "irl_alone_a2p4_sig": bool(a2p4_delta["sig"]),
        "ens_lift_ap_a2p4_pp": a2p4_lift_ap["value_pp"],
        "ens_lift_ap_a2p4_ci": [a2p4_lift_ap["ci_lo_pp"], a2p4_lift_ap["ci_hi_pp"]],
        "ens_lift_ap_a2p4_sig": bool(a2p4_lift_ap["sig"]),
        "defensible_claim": (
            "IRL 単独の A2 優位は非有意だが、A2×P4 への Ens lift AP が有意なら "
            "『IRL は A2 に独立情報を供給する』として残せる"),
    }
    with open(OUT / "a2_probe_summary.json", "w") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print("\n=== 判定 ===")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))

    plot_defense(lift, mech, OUT / "fig_a2_defense.pdf")


if __name__ == "__main__":
    main()
