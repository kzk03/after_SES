# -*- coding: utf-8 -*-
"""担当33c: 新人レビュアーの継続(LTC)早期予測.

論文の動機は「長期貢献者(LTC)の確保」。新人レビュアーのうち誰が継続するかを
早期に予測できるかを検証する。

新人定義 (2 通り):
  (A) tenure 新人: feature_values t0='2022-01-01' の window_tenure_days 下位25% (<=q25)。
      window_tenure_days は約180日の窓内 tenure (cap~179)、低値=直近に活動を始めた新人。
  (B) archetype 新人: 開発者 archetype A1 (低活動新人)。

分析:
  1. 新人ペアの規模・base_rate を確認 (新人 vs 非新人)。
  2. 新人 vs 非新人で各区間継続(label=1)の予測しやすさ (AUC, IRL/RF/Ens) を比較。
  3. LTC早期検出: 「0-3m活動 かつ 9-12m活動継続」を LTC継続と定義し、
     新人の中で 0-3m スコア (および t0 prior 経験) が 9-12m継続を予測するか。
  4. 新人の中で IRL/RF どちらが効くか、当てられる新人の特徴。
  5. 図 fig_newcomer_ltc.pdf。

規約: cold IRL 未校正 → rank-based (AUC/AP) のみ。bootstrap CI (seed=777)。
AUC は両クラス揃う時のみ。

入力:
  data/pair_trajectories.csv
  data/feature_values.csv
  data/pair_repo_experience.csv
出力:
  results/newcomer_ltc_*.csv
  results/fig_newcomer_ltc.pdf
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
SEED = 777
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
T0 = "2022-01-01"
rng = np.random.default_rng(SEED)


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def safe_ap(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, s))


def boot_auc_ci(y, s, n_boot=2000):
    """AUC の bootstrap 95% CI。両クラス揃わない resample はスキップ。"""
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return (np.nan, np.nan)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(roc_auc_score(yb, s[idx]))
    if len(vals) < 50:
        return (np.nan, np.nan)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def boot_diff_ci(y, s_a, s_b, n_boot=2000):
    """AUC(a)-AUC(b) の bootstrap 95% CI (同一 resample 対)。"""
    y = np.asarray(y)
    s_a = np.asarray(s_a)
    s_b = np.asarray(s_b)
    if len(np.unique(y)) < 2:
        return (np.nan, np.nan, np.nan)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(roc_auc_score(yb, s_a[idx]) - roc_auc_score(yb, s_b[idx]))
    if len(diffs) < 50:
        return (np.nan, np.nan, np.nan)
    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def main():
    OUT.mkdir(exist_ok=True)
    pt = pd.read_csv(DATA / "pair_trajectories.csv")
    fv = pd.read_csv(DATA / "feature_values.csv")
    rexp = pd.read_csv(DATA / "pair_repo_experience.csv")

    # ── t0 特徴 ──
    t0 = fv[fv["__snapshot"] == T0].copy()
    pair_devs = set(pt["developer"].unique())
    t0d = t0[t0["__email"].isin(pair_devs)].copy()
    covered = set(t0d["__email"])
    q25 = t0d["window_tenure_days"].quantile(0.25)

    # 新人定義 A: tenure 下位25%
    newcomer_tenure_devs = set(t0d[t0d["window_tenure_days"] <= q25]["__email"])
    # 新人定義 B: archetype A1
    dev_arch = (
        pt.dropna(subset=["archetype"]).drop_duplicates("developer")
        .set_index("developer")["archetype"]
    )
    newcomer_a1_devs = set(dev_arch[dev_arch == "A1"].index)

    pt["has_t0"] = pt["developer"].isin(covered)
    pt["is_newcomer_tenure"] = pt["developer"].isin(newcomer_tenure_devs)
    pt["is_newcomer_a1"] = pt["developer"].isin(newcomer_a1_devs)
    pt["dev_arch"] = pt["developer"].map(dev_arch)
    # avg(Ensemble)列を作る
    for h in HORIZONS:
        pt[f"avg_{h}"] = (pt[f"irl_{h}"] + pt[f"rf_{h}"]) / 2

    # repo経験を join
    pt = pt.merge(
        rexp[["developer", "directory", "prior_n", "prior_acc",
              "log_prior_n", "prior_recency_days", "repo_exp_covered"]],
        on=["developer", "directory"], how="left",
    )

    # t0 tenure を pair に付与
    t0_tenure = t0d.set_index("__email")["window_tenure_days"]
    pt["t0_tenure"] = pt["developer"].map(t0_tenure)
    t0_totrev = t0d.set_index("__email")["total_reviews"]
    pt["t0_total_reviews"] = pt["developer"].map(t0_totrev)

    print("=" * 60)
    print(f"母集団 pairs={len(pt):,}, devs={pt['developer'].nunique():,}")
    print(f"t0 covered devs={len(covered):,}, window_tenure q25={q25:.0f}")
    print(f"新人(tenure<=q25) devs={len(newcomer_tenure_devs):,} "
          f"pairs={pt['is_newcomer_tenure'].sum():,}")
    print(f"新人(A1) devs={len(newcomer_a1_devs):,} "
          f"pairs={pt['is_newcomer_a1'].sum():,}")

    # ============================================================
    # 1. 規模・base_rate 表 (新人 vs 非新人, 両定義)
    # ============================================================
    scale_rows = []
    for defn, col in [("tenure_q25", "is_newcomer_tenure"), ("archetype_A1", "is_newcomer_a1")]:
        for grp, mask in [
            ("newcomer", pt[col]),
            ("established", (~pt[col]) & pt["has_t0"]),
        ]:
            sub = pt[mask]
            row = {
                "definition": defn,
                "group": grp,
                "n_pairs": int(len(sub)),
                "n_devs": int(sub["developer"].nunique()),
            }
            for h in HORIZONS:
                row[f"base_{h}"] = round(sub[f"label_{h}"].mean(), 4)
            scale_rows.append(row)
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(OUT / "newcomer_ltc_scale.csv", index=False)
    print("\n[1] 規模・base_rate")
    print(scale_df.to_string(index=False))

    # ============================================================
    # 2. 新人 vs 非新人 の継続予測 AUC (区間別, IRL/RF/Ens)
    # ============================================================
    auc_rows = []
    for defn, col in [("tenure_q25", "is_newcomer_tenure"), ("archetype_A1", "is_newcomer_a1")]:
        for grp, mask in [
            ("newcomer", pt[col]),
            ("established", (~pt[col]) & pt["has_t0"]),
        ]:
            sub = pt[mask]
            for h in HORIZONS:
                y = sub[f"label_{h}"].astype(int).values
                if len(np.unique(y)) < 2:
                    continue
                irl = sub[f"irl_{h}"].values
                rf = sub[f"rf_{h}"].values
                ens = sub[f"avg_{h}"].values
                irl_auc = safe_auc(y, irl)
                rf_auc = safe_auc(y, rf)
                ens_auc = safe_auc(y, ens)
                irl_lo, irl_hi = boot_auc_ci(y, irl)
                rf_lo, rf_hi = boot_auc_ci(y, rf)
                dmean, dlo, dhi = boot_diff_ci(y, irl, rf)
                auc_rows.append({
                    "definition": defn, "group": grp, "horizon": h,
                    "n": len(y), "n_pos": int(y.sum()),
                    "base_rate": round(y.mean(), 4),
                    "irl_auc": round(irl_auc, 4), "rf_auc": round(rf_auc, 4),
                    "ens_auc": round(ens_auc, 4),
                    "irl_ap": round(safe_ap(y, irl), 4),
                    "rf_ap": round(safe_ap(y, rf), 4),
                    "irl_ci_lo": round(irl_lo, 4), "irl_ci_hi": round(irl_hi, 4),
                    "rf_ci_lo": round(rf_lo, 4), "rf_ci_hi": round(rf_hi, 4),
                    "irl_minus_rf": round(irl_auc - rf_auc, 4),
                    "diff_ci_lo": round(dlo, 4), "diff_ci_hi": round(dhi, 4),
                    "diff_sig": bool(dlo > 0 or dhi < 0) if not np.isnan(dlo) else False,
                })
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(OUT / "newcomer_ltc_auc.csv", index=False)
    print("\n[2] 新人 vs 非新人 継続予測 AUC")
    print(auc_df[["definition", "group", "horizon", "n", "base_rate",
                  "irl_auc", "rf_auc", "ens_auc", "irl_minus_rf",
                  "diff_ci_lo", "diff_ci_hi", "diff_sig"]].to_string(index=False))

    # ============================================================
    # 3. LTC早期検出
    #    LTC継続 = 0-3m活動(label=1) かつ 9-12m活動(label=1)。
    #    新人の中で、早期スコア(0-3m irl/rf/ens) と t0 prior経験 が
    #    9-12m継続(LTC)を予測できるか。
    #    母集団は「0-3mで活動した新人ペア」(=新規参入を確認できたペア)。
    # ============================================================
    pt["ltc"] = ((pt["label_0-3m"] == 1) & (pt["label_9-12m"] == 1)).astype(int)
    # 0-3m活動ペア内での 9-12m継続 (条件付き継続)
    ltc_rows = []
    early_predictors = {
        "irl_0-3m_score": "irl_0-3m",
        "rf_0-3m_score": "rf_0-3m",
        "ens_0-3m_score": "avg_0-3m",
        "irl_9-12m_score": "irl_9-12m",   # 当該区間スコア(参考: 同区間予測)
        "rf_9-12m_score": "rf_9-12m",
        "t0_prior_n": "log_prior_n",      # repo事前経験
        "t0_prior_recency": "prior_recency_days",  # 負方向期待→符号反転で扱う
        "t0_tenure": "t0_tenure",
    }
    for defn, col in [("tenure_q25", "is_newcomer_tenure"), ("archetype_A1", "is_newcomer_a1"),
                      ("established", None), ("all", "ALL")]:
        if defn == "established":
            base = pt[(~pt["is_newcomer_tenure"]) & pt["has_t0"]]
        elif defn == "all":
            base = pt
        else:
            base = pt[pt[col]]
        # 0-3mで活動したペアに限定し、9-12m継続を当てる
        active = base[base["label_0-3m"] == 1].copy()
        y = active["label_9-12m"].astype(int).values
        if len(np.unique(y)) < 2 or len(y) < 20:
            continue
        for pname, pcol in early_predictors.items():
            s = active[pcol].values.astype(float)
            sign = -1.0 if pname == "t0_prior_recency" else 1.0
            # NaN 多い予測子は有効ペアのみで評価
            valid = ~np.isnan(s)
            yv = y[valid]
            sv = s[valid] * sign
            if len(np.unique(yv)) < 2 or len(yv) < 20:
                continue
            auc = safe_auc(yv, sv)
            lo, hi = boot_auc_ci(yv, sv)
            ltc_rows.append({
                "definition": defn,
                "predictor": pname,
                "n_active03m": int(len(yv)),
                "ltc_rate": round(yv.mean(), 4),
                "auc": round(auc, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "ap": round(safe_ap(yv, sv), 4),
                "sig_above_0.5": bool(lo > 0.5) if not np.isnan(lo) else False,
            })
    ltc_df = pd.DataFrame(ltc_rows)
    ltc_df.to_csv(OUT / "newcomer_ltc_early_detection.csv", index=False)
    print("\n[3] LTC早期検出 (0-3m活動ペア内で 9-12m継続を当てる識別力)")
    print(ltc_df.to_string(index=False))

    # LTC rate 自体 (新人 vs 非新人)
    ltc_rate_rows = []
    for defn, col in [("tenure_q25", "is_newcomer_tenure"), ("archetype_A1", "is_newcomer_a1")]:
        for grp, mask in [("newcomer", pt[col]),
                          ("established", (~pt[col]) & pt["has_t0"])]:
            sub = pt[mask]
            act = sub[sub["label_0-3m"] == 1]
            ltc_rate_rows.append({
                "definition": defn, "group": grp,
                "n_active03m": int(len(act)),
                "ltc_among_active": round(act["label_9-12m"].mean(), 4) if len(act) else np.nan,
                "ltc_among_all": round(sub["ltc"].mean(), 4),
            })
    ltc_rate_df = pd.DataFrame(ltc_rate_rows)
    ltc_rate_df.to_csv(OUT / "newcomer_ltc_rate.csv", index=False)
    print("\n[3b] LTC率 (新人 vs 非新人)")
    print(ltc_rate_df.to_string(index=False))

    # ============================================================
    # 4. 新人内で IRL/RF どちらが効くか (archetype 別の差も)
    #    既知 B-41: A1 は RF 優位。新人(tenure)内で A1/非A1 の差を見る。
    # ============================================================
    nc = pt[pt["is_newcomer_tenure"]].copy()
    detail_rows = []
    for sub_name, sub in [
        ("newcomer_all", nc),
        ("newcomer_A1", nc[nc["dev_arch"] == "A1"]),
        ("newcomer_nonA1", nc[nc["dev_arch"] != "A1"]),
    ]:
        for h in HORIZONS:
            y = sub[f"label_{h}"].astype(int).values
            if len(np.unique(y)) < 2 or len(y) < 20:
                continue
            irl = sub[f"irl_{h}"].values
            rf = sub[f"rf_{h}"].values
            dmean, dlo, dhi = boot_diff_ci(y, irl, rf)
            detail_rows.append({
                "subset": sub_name, "horizon": h, "n": len(y),
                "base_rate": round(y.mean(), 4),
                "irl_auc": round(safe_auc(y, irl), 4),
                "rf_auc": round(safe_auc(y, rf), 4),
                "irl_minus_rf": round(safe_auc(y, irl) - safe_auc(y, rf), 4),
                "diff_ci_lo": round(dlo, 4) if not np.isnan(dlo) else np.nan,
                "diff_ci_hi": round(dhi, 4) if not np.isnan(dhi) else np.nan,
            })
    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(OUT / "newcomer_ltc_model_detail.csv", index=False)
    print("\n[4] 新人内 IRL vs RF (archetype別)")
    print(detail_df.to_string(index=False))

    # ============================================================
    # 5. 図 fig_newcomer_ltc.pdf
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) 新人 vs 非新人の区間別 AUC (tenure定義, Ens)
    ax = axes[0]
    sub_a = auc_df[auc_df["definition"] == "tenure_q25"]
    x = np.arange(len(HORIZONS))
    w = 0.35
    for i, (grp, color) in enumerate([("newcomer", "#d62728"),
                                      ("established", "#1f77b4")]):
        g = sub_a[sub_a["group"] == grp].set_index("horizon").reindex(HORIZONS)
        ax.bar(x + (i - 0.5) * w, g["ens_auc"], w, label=grp, color=color, alpha=0.85)
        # CI as errorbar from irl? use ens CI not stored; show irl/rf range via dots
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(HORIZONS)
    ax.set_ylabel("継続予測 AUC (Ensemble)")
    ax.set_title("(a) 新人 vs 既存 の継続予測しやすさ\n(新人定義: tenure下位25%)")
    ax.legend(loc="upper right")
    ax.set_ylim(0.4, 1.0)

    # (b) 新人継続予測の区間別 IRL/RF/Ens (tenure新人)
    ax = axes[1]
    sub_b = auc_df[(auc_df["definition"] == "tenure_q25") &
                   (auc_df["group"] == "newcomer")].set_index("horizon").reindex(HORIZONS)
    for col, name, color in [("irl_auc", "cold IRL", "#2ca02c"),
                             ("rf_auc", "RF", "#ff7f0e"),
                             ("ens_auc", "Ensemble", "#9467bd")]:
        ax.plot(HORIZONS, sub_b[col], "o-", label=name, color=color, lw=2, ms=8)
    # IRL CI band
    ax.fill_between(HORIZONS, sub_b["irl_ci_lo"], sub_b["irl_ci_hi"],
                    color="#2ca02c", alpha=0.15)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_ylabel("継続予測 AUC")
    ax.set_title("(b) 新人ペアの区間別継続予測\n(IRL CI帯: bootstrap95%)")
    ax.legend(loc="upper right")
    ax.set_ylim(0.4, 1.0)

    # (c) LTC継続(9-12m)を当てる早期スコアの識別力
    ax = axes[2]
    show_defs = ["tenure_q25", "archetype_A1", "established"]
    show_preds = ["irl_0-3m_score", "rf_0-3m_score", "ens_0-3m_score", "t0_prior_n"]
    pred_label = {"irl_0-3m_score": "0-3m IRL", "rf_0-3m_score": "0-3m RF",
                  "ens_0-3m_score": "0-3m Ens", "t0_prior_n": "t0 repo経験"}
    def_label = {"tenure_q25": "新人(tenure)", "archetype_A1": "新人(A1)",
                 "established": "既存"}
    x = np.arange(len(show_preds))
    w = 0.25
    colors = {"tenure_q25": "#d62728", "archetype_A1": "#e377c2", "established": "#1f77b4"}
    for i, d in enumerate(show_defs):
        sd = ltc_df[ltc_df["definition"] == d].set_index("predictor")
        vals, los, his = [], [], []
        for p in show_preds:
            if p in sd.index:
                vals.append(sd.loc[p, "auc"])
                los.append(sd.loc[p, "auc"] - sd.loc[p, "ci_lo"])
                his.append(sd.loc[p, "ci_hi"] - sd.loc[p, "auc"])
            else:
                vals.append(np.nan); los.append(0); his.append(0)
        ax.bar(x + (i - 1) * w, vals, w, label=def_label[d], color=colors[d], alpha=0.85,
               yerr=[los, his], capsize=3, error_kw={"lw": 1})
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([pred_label[p] for p in show_preds], rotation=15)
    ax.set_ylabel("9-12m継続(LTC)の識別 AUC")
    ax.set_title("(c) LTC継続を当てる早期スコア\n(0-3m活動ペア内, 9-12m継続)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0.4, 0.85)

    plt.tight_layout()
    fig.savefig(OUT / "fig_newcomer_ltc.pdf", format="pdf", bbox_inches="tight")
    print(f"\n図保存: {OUT / 'fig_newcomer_ltc.pdf'}")

    # サマリ JSON
    summary = {
        "n_pairs": int(len(pt)),
        "n_devs": int(pt["developer"].nunique()),
        "newcomer_tenure_q25": float(q25),
        "newcomer_tenure_pairs": int(pt["is_newcomer_tenure"].sum()),
        "newcomer_a1_pairs": int(pt["is_newcomer_a1"].sum()),
    }
    with open(OUT / "newcomer_ltc_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
