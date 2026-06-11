#!/usr/bin/env python
"""32c: 期間別予測で言えること — 開発者の期間変動 + 危険信号 (枯渇とは言わず).

主張 (ユーザー指示):
  (1) 母集団の非定常性: 4 予測区間 (0-3m,3-6m,6-9m,9-12m) の正例率 (承諾率) が
      後半ほど低下する (= 活動縮小)。
  (2) 開発者/ペアの期間変動: 連続区間間のラベル遷移を 4 状態
      (継続 1->1 / 離脱 1->0 / 復帰 0->1 / 恒常不在 0->0) で集計し、
      「何 % のペアが区間間でステータスを変えるか」を数字で示す。
  (3) 危険信号フレーミング (headcount でなく rank): dir×期間で
      「次区間に承諾しなくなるペアを Ens スコア低い順で拾えるか (recall lift)」を
      危険信号として提示。高 attrition × 検知良好な dir×期間を列挙
      (B-43 exhaustion の言い換え、過大主張しない)。

入力:
  data/pair_trajectories.csv          (developer,directory,label_{h}/irl/rf/avg, archetype)
  data/pair_archetype.csv             (dominant=ペア archetype A1-A4)
  results/directory_master.csv        (attrition_weighted, ens_auc, reliable 等)
  results/horizon_dir_attrition.csv   (per cell の attrition + ens_recall_at_attrited)

出力:
  results/period_positive_rate.csv          区間別正例率
  results/period_transition_matrix.csv      4 状態遷移の集計 (遷移別 + 全体)
  results/period_pair_status_class.csv      ペア単位の変動分類比率
  results/period_danger_signal_cells.csv    危険信号セル (高 attrition × recall lift)
  results/fig_period_variation_signals.pdf  3 パネル図
"""
import json
import os

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = "/Users/kazuki-h/Master_research/writing/analysis"
RES = os.path.join(BASE, "results")
TRAJ = os.path.join(BASE, "data", "pair_trajectories.csv")
PAIR_ARCH = os.path.join(BASE, "data", "pair_archetype.csv")
MASTER = os.path.join(RES, "directory_master.csv")
ATTR = os.path.join(RES, "horizon_dir_attrition.csv")

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
TRANSITIONS = [("0-3m", "3-6m"), ("3-6m", "6-9m"), ("6-9m", "9-12m")]

# 4 状態遷移ラベル (prev,curr) -> 状態名
STATE_LABELS = {
    (1, 1): "継続 (1->1)",
    (1, 0): "離脱 (1->0)",
    (0, 1): "復帰 (0->1)",
    (0, 0): "恒常不在 (0->0)",
}
STATE_ORDER = ["継続 (1->1)", "離脱 (1->0)", "復帰 (0->1)", "恒常不在 (0->0)"]
STATE_COLORS = {
    "継続 (1->1)": "#2c7fb8",
    "離脱 (1->0)": "#d7301f",
    "復帰 (0->1)": "#41ab5d",
    "恒常不在 (0->0)": "#bdbdbd",
}


def partial_spearman(x, y, covars):
    """偏 Spearman: x,y を covars で順位回帰した残差相関 (= rank 残差の pearson)。"""
    d = pd.DataFrame({"x": x, "y": y})
    for i, c in enumerate(covars):
        d[f"c{i}"] = c
    d = d.dropna()
    if len(d) < 8:
        return np.nan, np.nan, len(d)
    rk = d.rank()
    cols = [f"c{i}" for i in range(len(covars))]
    Z = np.column_stack([np.ones(len(rk))] + [rk[c].values for c in cols])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    rx, ry = resid(rk["x"].values), resid(rk["y"].values)
    if np.std(rx) < 1e-9 or np.std(ry) < 1e-9:
        return np.nan, np.nan, len(d)
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(d)


def main():
    df = pd.read_csv(TRAJ)
    labels = [f"label_{h}" for h in HORIZONS]

    # =========================================================================
    # (1) 母集団の非定常性: 区間別正例率 (承諾率) 推移
    # =========================================================================
    n_pairs = len(df)
    pos_rate = {h: float(df[f"label_{h}"].mean()) for h in HORIZONS}
    n_pos = {h: int(df[f"label_{h}"].sum()) for h in HORIZONS}
    # 単調減少か線形傾向か
    yvals = np.array([pos_rate[h] for h in HORIZONS])
    xidx = np.arange(len(HORIZONS))
    slope, intercept, r_lin, p_lin, _ = stats.linregress(xidx, yvals)
    nonstationarity = {
        "n_pairs": n_pairs,
        "positive_rate_by_horizon": pos_rate,
        "n_positive_by_horizon": n_pos,
        "monotonic_decreasing": bool(np.all(np.diff(yvals) < 0)),
        "first_to_last_drop_abs": float(yvals[0] - yvals[-1]),
        "first_to_last_drop_rel_pct": float((yvals[0] - yvals[-1]) / yvals[0] * 100),
        "linregress_slope_per_quarter": float(slope),
        "linregress_r": float(r_lin),
        "linregress_p": float(p_lin),
    }

    pr_df = pd.DataFrame(
        {
            "horizon": HORIZONS,
            "n_pair": [n_pairs] * 4,
            "n_positive": [n_pos[h] for h in HORIZONS],
            "positive_rate": [pos_rate[h] for h in HORIZONS],
        }
    )
    pr_path = os.path.join(RES, "period_positive_rate.csv")
    pr_df.to_csv(pr_path, index=False)

    # =========================================================================
    # (2) 開発者/ペアの期間変動: 連続区間間の 4 状態遷移行列
    # =========================================================================
    # (2a) 遷移ごと (3 遷移) の状態比率
    trans_rows = []
    for h_prev, h_curr in TRANSITIONS:
        lp = df[f"label_{h_prev}"].astype(int).values
        lc = df[f"label_{h_curr}"].astype(int).values
        n = len(lp)
        for (sp, sc), name in STATE_LABELS.items():
            cnt = int(np.sum((lp == sp) & (lc == sc)))
            trans_rows.append(
                {
                    "transition": f"{h_prev}->{h_curr}",
                    "h_prev": h_prev,
                    "h_curr": h_curr,
                    "state": name,
                    "count": cnt,
                    "frac": cnt / n,
                }
            )
    trans_df = pd.DataFrame(trans_rows)

    # (2b) 全 3 遷移をプール (pair×遷移 を 1 観測とする) した全体比率
    pooled = (
        trans_df.groupby("state")["count"].sum().reindex(STATE_ORDER).reset_index()
    )
    pooled["transition"] = "pooled(3遷移)"
    pooled["frac"] = pooled["count"] / pooled["count"].sum()
    pooled = pooled[["transition", "state", "count", "frac"]]
    trans_out = pd.concat(
        [trans_df[["transition", "state", "count", "frac"]], pooled], ignore_index=True
    )
    tm_path = os.path.join(RES, "period_transition_matrix.csv")
    trans_out.to_csv(tm_path, index=False)

    # 「ステータスを変える」= 離脱 or 復帰 (prev != curr)
    pooled_d = pooled.set_index("state")["frac"]
    frac_changed = float(pooled_d["離脱 (1->0)"] + pooled_d["復帰 (0->1)"])
    # 各遷移での変化率も
    changed_per_trans = {}
    for h_prev, h_curr in TRANSITIONS:
        sub = trans_df[trans_df["transition"] == f"{h_prev}->{h_curr}"].set_index("state")[
            "frac"
        ]
        changed_per_trans[f"{h_prev}->{h_curr}"] = float(
            sub["離脱 (1->0)"] + sub["復帰 (0->1)"]
        )

    # アクティブ条件付き離脱率 (= attrition の素): 区間 prev で active だったペアのうち離脱
    cond_attrition = {}
    for h_prev, h_curr in TRANSITIONS:
        sub = trans_df[trans_df["transition"] == f"{h_prev}->{h_curr}"].set_index("state")[
            "count"
        ]
        active_prev = sub["継続 (1->1)"] + sub["離脱 (1->0)"]
        cond_attrition[f"{h_prev}->{h_curr}"] = (
            float(sub["離脱 (1->0)"] / active_prev) if active_prev > 0 else np.nan
        )
    # 不在ペアの復帰率 (prev で不在のうち curr で復帰)
    cond_return = {}
    for h_prev, h_curr in TRANSITIONS:
        sub = trans_df[trans_df["transition"] == f"{h_prev}->{h_curr}"].set_index("state")[
            "count"
        ]
        absent_prev = sub["復帰 (0->1)"] + sub["恒常不在 (0->0)"]
        cond_return[f"{h_prev}->{h_curr}"] = (
            float(sub["復帰 (0->1)"] / absent_prev) if absent_prev > 0 else np.nan
        )

    variation_summary = {
        "pooled_state_frac": {k: float(v) for k, v in pooled_d.items()},
        "frac_changed_status_pooled": frac_changed,
        "frac_changed_status_per_transition": changed_per_trans,
        "conditional_attrition_rate_per_transition": cond_attrition,
        "conditional_return_rate_per_transition": cond_return,
    }

    # =========================================================================
    # (2c) ペア単位の変動分類 (4 区間ラベル系列を一意に分類)
    #   - 恒常アクティブ: 全 4 区間で 1
    #   - 恒常不在: 全 4 区間で 0
    #   - 単調離脱: 1 で始まり最後に 0、復帰なし (1...10...0 型)
    #   - 変動 (断続): それ以外で 1 と 0 が混在 (復帰/再離脱を含む)
    # =========================================================================
    seq = df[labels].astype(int).values
    n_active = seq.sum(axis=1)
    classes = []
    for row in seq:
        s = "".join(map(str, row))
        if n_active_row := int(row.sum()):
            if row.all():
                classes.append("恒常アクティブ")
            else:
                # 1 が一塊で先頭から連続し、その後 0 が続く -> 単調離脱
                first_zero_after_one = False
                monotone = True
                seen_zero = False
                for v in row:
                    if v == 0:
                        seen_zero = True
                    elif seen_zero:  # 0 の後に 1 -> 復帰あり
                        monotone = False
                        break
                # 先頭が 1 で始まり最後が 0 の単調離脱
                if monotone and row[0] == 1:
                    classes.append("単調離脱")
                    _ = first_zero_after_one
                else:
                    classes.append("変動(断続/復帰)")
        else:
            classes.append("恒常不在")
        _ = n_active_row
    df["_var_class"] = classes
    vc = df["_var_class"].value_counts()
    var_class_frac = (vc / len(df)).to_dict()
    # アクティブ経験あり (恒常不在以外) の中での内訳
    active_ever = df[df["_var_class"] != "恒常不在"]
    var_class_among_active = (
        active_ever["_var_class"].value_counts() / len(active_ever)
    ).to_dict()

    pair_class_df = pd.DataFrame(
        {
            "var_class": list(var_class_frac.keys()),
            "n_pair": [int(vc[k]) for k in var_class_frac.keys()],
            "frac_all": [float(var_class_frac[k]) for k in var_class_frac.keys()],
        }
    )
    pc_path = os.path.join(RES, "period_pair_status_class.csv")
    pair_class_df.to_csv(pc_path, index=False)

    variation_summary["pair_var_class_frac_all"] = {
        k: float(v) for k, v in var_class_frac.items()
    }
    variation_summary["pair_var_class_frac_among_active"] = {
        k: float(v) for k, v in var_class_among_active.items()
    }
    # 「アクティブ経験ありのうち単調離脱でも恒常でもない=断続変動」割合
    variation_summary["frac_intermittent_among_active"] = float(
        var_class_among_active.get("変動(断続/復帰)", 0.0)
    )

    # =========================================================================
    # (3) 危険信号フレーミング (rank-based, headcount でない): recall lift
    #   既存 horizon_dir_attrition.csv (B-43) を再利用。
    #   危険信号セル = 高 attrition (>=0.5) かつ 検知良好 (recall lift>0)。
    # =========================================================================
    a = pd.read_csv(ATTR)
    a["recall_lift"] = a["ens_recall_at_attrited"] - a["random_baseline_recall"]
    cells = a[(a["n_active_prev"] >= 10) & a["ens_recall_at_attrited"].notna()].copy()

    danger = cells[cells["attrition_rate"] >= 0.5].copy()
    danger = danger.sort_values("recall_lift", ascending=False).reset_index(drop=True)

    danger_signal_summary = {
        "n_cells_evaluable": int(len(cells)),
        "median_recall_lift_all_cells": float(cells["recall_lift"].median()),
        "frac_positive_lift_all_cells": float((cells["recall_lift"] > 0).mean()),
        "n_danger_cells": int(len(danger)),
        "n_danger_positive_lift": int((danger["recall_lift"] > 0).sum()),
        "frac_danger_positive_lift": float((danger["recall_lift"] > 0).mean())
        if len(danger)
        else np.nan,
        "median_danger_recall_lift": float(danger["recall_lift"].median())
        if len(danger)
        else np.nan,
        "median_danger_ens_recall": float(danger["ens_recall_at_attrited"].median())
        if len(danger)
        else np.nan,
        "top_signal_cells": danger.head(10)[
            [
                "directory",
                "h_prev",
                "h_curr",
                "n_active_prev",
                "n_attrited",
                "attrition_rate",
                "ens_auc",
                "ens_recall_at_attrited",
                "random_baseline_recall",
                "recall_lift",
            ]
        ]
        .round(3)
        .to_dict("records"),
    }

    danger_out = danger[
        [
            "directory",
            "h_prev",
            "h_curr",
            "n_active_prev",
            "n_attrited",
            "attrition_rate",
            "irl_auc",
            "rf_auc",
            "ens_auc",
            "ens_recall_at_attrited",
            "random_baseline_recall",
            "recall_lift",
        ]
    ].copy()
    ds_path = os.path.join(RES, "period_danger_signal_cells.csv")
    danger_out.to_csv(ds_path, index=False)

    # (3b) ランキング妥当性: per-dir Ens AUC が実 attrition_weighted を順位付けできるか
    #   (規模・base_rate 統制の偏 Spearman)。過大主張しないための統制検証。
    m = pd.read_csv(MASTER)
    sub = m[m["reliable"] & m["ens_auc"].notna()].dropna(
        subset=["attrition_weighted", "n_pair", "base_rate"]
    ).copy()
    raw_r, raw_p = (
        stats.spearmanr(sub["ens_auc"], sub["attrition_weighted"])
        if len(sub) >= 8
        else (np.nan, np.nan)
    )
    pr_r, pr_p, pr_n = partial_spearman(
        sub["ens_auc"].values,
        sub["attrition_weighted"].values,
        [np.log10(sub["n_pair"].values), sub["base_rate"].values],
    )
    danger_signal_summary["ranking_validity"] = {
        "ens_auc_vs_attrition_weighted_raw_spearman_r": float(raw_r),
        "raw_p": float(raw_p),
        "partial_spearman_r_ctrl_logn_baserate": float(pr_r),
        "partial_p": float(pr_p),
        "n_reliable_dir": int(pr_n),
        "note": "弱～中の正相関なら『順位品質は実枯渇度と整合』程度の控えめ主張に留める",
    }

    # =========================================================================
    # 図: 3 パネル
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

    # (a) 4 区間の正例率推移
    ax = axes[0]
    ax.plot(
        xidx,
        yvals,
        marker="o",
        color="#d7301f",
        lw=2,
        markersize=9,
        markerfacecolor="white",
        markeredgewidth=2,
    )
    for i, v in enumerate(yvals):
        ax.annotate(
            f"{v:.3f}",
            (xidx[i], v),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
        )
    ax.set_xticks(xidx)
    ax.set_xticklabels(HORIZONS)
    ax.set_ylim(0, max(yvals) * 1.25)
    ax.set_xlabel("予測区間 (t0 からの将来 3 ヶ月窓)")
    ax.set_ylabel("正例率 (承諾ペア比率)")
    ax.set_title(
        f"(a) 母集団の非定常性: 区間別正例率\n"
        f"{yvals[0]:.3f}→{yvals[-1]:.3f} "
        f"(相対 -{nonstationarity['first_to_last_drop_rel_pct']:.0f}%, "
        f"傾き {slope:+.3f}/区間, p={p_lin:.3f})",
        fontsize=11,
    )
    ax.grid(axis="y", ls=":", alpha=0.5)

    # (b) 区間間ラベル遷移 (積み上げ棒: 遷移ごと + pooled)
    ax = axes[1]
    bar_labels = [f"{p}\n→{c}" for p, c in TRANSITIONS] + ["pooled\n(3遷移)"]
    bx = np.arange(len(bar_labels))
    bottoms = np.zeros(len(bar_labels))
    # 各遷移の状態 frac を行列化
    fmat = {}
    for state in STATE_ORDER:
        vals = []
        for h_prev, h_curr in TRANSITIONS:
            v = trans_df[
                (trans_df["transition"] == f"{h_prev}->{h_curr}")
                & (trans_df["state"] == state)
            ]["frac"].iloc[0]
            vals.append(v)
        vals.append(float(pooled_d[state]))
        fmat[state] = np.array(vals)
    for state in STATE_ORDER:
        ax.bar(
            bx,
            fmat[state],
            bottom=bottoms,
            color=STATE_COLORS[state],
            label=state,
            edgecolor="white",
            lw=0.5,
        )
        # 各セグメントに % ラベル (大きいセグメントのみ)
        for i, v in enumerate(fmat[state]):
            if v >= 0.06:
                ax.text(
                    bx[i],
                    bottoms[i] + v / 2,
                    f"{v*100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if state != "恒常不在 (0->0)" else "black",
                )
        bottoms += fmat[state]
    ax.set_xticks(bx)
    ax.set_xticklabels(bar_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("ペア比率")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title(
        f"(b) 区間間ラベル遷移 (4 状態)\n"
        f"プール: ステータス変化 (離脱+復帰) = {frac_changed*100:.1f}%",
        fontsize=11,
    )

    # (c) 危険信号デモ: dir×期間 recall lift 上位 (水平棒)
    ax = axes[2]
    top = danger.head(12).copy()
    top["lab"] = (
        top["directory"] + " (" + top["h_prev"] + "→" + top["h_curr"] + ")"
    )
    ypos = np.arange(len(top))[::-1]
    colors = ["#2c7fb8" if v > 0 else "#d7301f" for v in top["recall_lift"]]
    ax.barh(ypos, top["recall_lift"], color=colors, edgecolor="black", lw=0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(top["lab"], fontsize=7.5)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("recall lift (Ens 低スコア順での脱落者捕捉率 − ランダム期待値)")
    pos_lift = danger_signal_summary["n_danger_positive_lift"]
    ax.set_title(
        f"(c) 危険信号デモ: 高 attrition セルで\n"
        f"次区間の非承諾ペアを低スコア順に拾えるか\n"
        f"(危険セル {len(danger)} 件中 lift 正 {pos_lift} 件, "
        f"中央値 {danger_signal_summary['median_danger_recall_lift']:+.3f})",
        fontsize=10,
    )

    fig.suptitle(
        "期間別予測で言えること: 開発者は区間ごとに変動し、危険信号 (枯渇に至る前の非承諾化) を rank で拾える",
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(RES, "fig_period_variation_signals.pdf")
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # まとめ出力
    # =========================================================================
    summary = {
        "1_nonstationarity": nonstationarity,
        "2_period_variation": variation_summary,
        "3_danger_signal": danger_signal_summary,
        "outputs": {
            "period_positive_rate_csv": pr_path,
            "period_transition_matrix_csv": tm_path,
            "period_pair_status_class_csv": pc_path,
            "period_danger_signal_cells_csv": ds_path,
            "fig": fig_path,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
