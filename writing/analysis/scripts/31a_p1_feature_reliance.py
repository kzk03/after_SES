"""
31a: P1未経験ペアで IRL が RF に勝つ機序の検証

ユーザー解釈:「IRL は当該 dir の path 特徴を重視せず、開発者状態 (state) 特徴も
見ているから P1 (path特徴ほぼ0) で勝てる場合が多いのでは?」

検証設計:
  IRL も RF も同じ 32 特徴 (state26 + path6) を入力。P1 では path6 がほぼ0。
  よって差は「path 欠落時にどちらが state 系列から信号を残せるか」。

1. P1ペア集合で各 state 特徴の単変量 label 識別力 (AUC, max(a,1-a)) を算出
   → P1 でも開発者状態特徴は label を識別できる (= path 以外に信号がある)
2. IRL prob / RF prob を P1 ペアで state 特徴に回帰・相関
   → IRL prob はどの state 特徴と連動するか / RF prob は連動が弱く base_rate 付近に潰れるか
   → §10 の avg_activity_gap+ / active_months_ratio+ と整合するか
3. 「IRL正解 ∧ RF誤り」P1ペア vs 逆 を抽出し state 特徴プロファイル差を比較
4. 図 fig_p1_feature_reliance.pdf (a)(b)(c)
"""
import japanize_matplotlib  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

BASE = "/Users/kazuki-h/Master_research/writing/analysis"
RES = f"{BASE}/results"

STATE_FEATS = [
    "window_tenure_days", "total_changes", "total_reviews",
    "recent_activity_frequency", "avg_activity_gap", "activity_trend",
    "unique_collaborator_count", "overall_acceptance_rate",
    "recent_acceptance_rate", "recent_load_ratio_30d_all",
    "days_since_last_activity", "acceptance_trend", "reciprocity_score",
    "recent_load_ratio_7d_30d", "core_reviewer_ratio",
    "recent_rejection_streak", "acceptance_rate_last10",
    "active_months_ratio", "response_time_trend", "complex_pr_bias",
    "avg_action_intensity", "avg_change_lines", "avg_response_time",
    "avg_review_size", "repeat_collaboration_rate",
]
# §10 / ユーザー指定の発展性系 state 特徴 (注目するもの)
KEY_FEATS = [
    "active_months_ratio", "avg_activity_gap", "recent_activity_frequency",
    "days_since_last_activity", "total_reviews",
]
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def univariate_auc(x, y):
    """単変量 AUC を方向不問で max(a, 1-a) として返す。"""
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
        return np.nan, np.nan
    a = roc_auc_score(y, x)
    return max(a, 1 - a), a  # (方向不問AUC, 生AUC[>0.5なら正方向])


def main():
    pa = pd.read_csv(f"{BASE}/data/pair_archetype.csv")
    pt = pd.read_csv(f"{BASE}/data/pair_trajectories.csv")
    fv = pd.read_csv(f"{BASE}/data/feature_values.csv")
    t0 = (fv[fv["__snapshot"] == "2022-01-01"]
          .drop(columns=["__snapshot"])
          .rename(columns={"__email": "developer"}))

    # ペアarchetype dominant を結合 (A1 = P1未経験)
    m = pt.merge(pa[["developer", "directory", "dominant"]],
                 on=["developer", "directory"], how="left")
    p1 = m[m["dominant"] == "A1"].copy()

    # 長形式: 各ペアは 4 horizon の (pair, horizon) 観測に展開
    rows = []
    for h in HORIZONS:
        sub = p1[["developer", "directory",
                  f"label_{h}", f"irl_{h}", f"rf_{h}", f"avg_{h}"]].copy()
        sub.columns = ["developer", "directory", "label", "irl", "rf", "avg"]
        sub["h"] = h
        rows.append(sub)
    lng = pd.concat(rows, ignore_index=True).dropna(subset=["label"])
    lng = lng.merge(t0, on="developer", how="left")
    lng = lng.dropna(subset=STATE_FEATS)

    n_obs = len(lng)
    base_rate = lng["label"].mean()
    irl_auc = roc_auc_score(lng["label"], lng["irl"])
    rf_auc = roc_auc_score(lng["label"], lng["rf"])
    ens_auc = roc_auc_score(lng["label"], lng["avg"])
    print(f"P1 pooled: n_obs={n_obs}, base_rate={base_rate:.3f}")
    print(f"  IRL AUC={irl_auc:.3f}  RF AUC={rf_auc:.3f}  ENS AUC={ens_auc:.3f}  "
          f"(IRL-RF={irl_auc-rf_auc:+.3f})")

    # ============================================================
    # 1. 各 state 特徴の単変量 label 識別力 (AUC, max(a,1-a))
    # ============================================================
    feat_rows = []
    for f in STATE_FEATS:
        auc_dir, auc_raw = univariate_auc(lng[f].values, lng["label"].values)
        # IRL prob / RF prob との Spearman 連動
        sp_irl = stats.spearmanr(lng[f], lng["irl"], nan_policy="omit").statistic
        sp_rf = stats.spearmanr(lng[f], lng["rf"], nan_policy="omit").statistic
        feat_rows.append({
            "feature": f,
            "univ_auc": auc_dir,
            "raw_auc": auc_raw,
            "direction": "+" if auc_raw >= 0.5 else "-",  # 大きいほど label=1
            "spearman_irl": sp_irl,
            "spearman_rf": sp_rf,
            "abs_sp_irl": abs(sp_irl),
            "abs_sp_rf": abs(sp_rf),
        })
    fdf = pd.DataFrame(feat_rows).sort_values("univ_auc", ascending=False)
    fdf.to_csv(f"{RES}/p1_feature_univariate_auc.csv", index=False)
    print("\n=== (1) P1 state特徴 単変量 label AUC top10 ===")
    print(fdf[["feature", "univ_auc", "direction",
               "spearman_irl", "spearman_rf"]].head(10).to_string(index=False))

    # 連動の総括: |Spearman| の平均で IRL vs RF
    print(f"\n  mean |Spearman(feat, prob)|: IRL={fdf['abs_sp_irl'].mean():.3f}  "
          f"RF={fdf['abs_sp_rf'].mean():.3f}")

    # ============================================================
    # 2. IRL prob / RF prob を state 特徴に回帰 (標準化線形回帰 R^2 と係数)
    # ============================================================
    X = lng[STATE_FEATS].values
    Xs = StandardScaler().fit_transform(X)
    reg_summary = {}
    coef_rows = []
    for target in ["irl", "rf"]:
        y = lng[target].values
        reg = LinearRegression().fit(Xs, y)
        r2 = reg.score(Xs, y)
        reg_summary[target] = r2
        for f, c in zip(STATE_FEATS, reg.coef_):
            coef_rows.append({"feature": f, "target": target, "std_coef": c})
    cdf = pd.DataFrame(coef_rows)
    cpiv = cdf.pivot(index="feature", columns="target", values="std_coef")
    cpiv.to_csv(f"{RES}/p1_prob_state_regression.csv")
    print("\n=== (2) state特徴 -> prob 標準化線形回帰 ===")
    print(f"  R^2: IRL prob={reg_summary['irl']:.3f}  RF prob={reg_summary['rf']:.3f}")
    print("  (R^2 大 = prob が state 特徴で説明できる = state 連動)")
    print("  top IRL 係数 (絶対値):")
    print(cpiv.reindex(cpiv["irl"].abs().sort_values(ascending=False).index)
          .head(8).round(4).to_string())

    # variance check: RF prob が base_rate 付近に潰れていないか
    print(f"\n  prob std: IRL={lng['irl'].std():.4f}  RF={lng['rf'].std():.4f}")
    print(f"  prob IQR: IRL={lng['irl'].quantile(.75)-lng['irl'].quantile(.25):.4f}  "
          f"RF={lng['rf'].quantile(.75)-lng['rf'].quantile(.25):.4f}")

    # ============================================================
    # 3. IRL正解 ∧ RF誤り vs 逆 の state 特徴プロファイル差
    #    正解/誤り = 各 horizon の中央値しきい値での 0/1 判定で評価
    #    (cold IRL は未校正なので絶対しきい値でなく rank/分位で判定)
    # ============================================================
    # horizon ごとに prob を分位順位 (0-1) に変換してから判定
    lng = lng.copy()
    lng["irl_rank"] = lng.groupby("h")["irl"].rank(pct=True)
    lng["rf_rank"] = lng.groupby("h")["rf"].rank(pct=True)
    # しきい値は各 horizon の base_rate に合わせる (上位 base_rate 割を正と予測)
    def pred_top(g, col):
        thr = 1 - g["label"].mean()  # 上位 base_rate 分位を 1
        return (g[col] >= thr).astype(int)
    lng["irl_pred"] = lng.groupby("h", group_keys=False).apply(
        lambda g: pred_top(g, "irl_rank"))
    lng["rf_pred"] = lng.groupby("h", group_keys=False).apply(
        lambda g: pred_top(g, "rf_rank"))
    lng["irl_correct"] = (lng["irl_pred"] == lng["label"]).astype(int)
    lng["rf_correct"] = (lng["rf_pred"] == lng["label"]).astype(int)

    grp_irl_win = lng[(lng["irl_correct"] == 1) & (lng["rf_correct"] == 0)]
    grp_rf_win = lng[(lng["irl_correct"] == 0) & (lng["rf_correct"] == 1)]
    print(f"\n=== (3) IRL正解∧RF誤り: n={len(grp_irl_win)}  "
          f"IRL誤り∧RF正解: n={len(grp_rf_win)} ===")

    diff_rows = []
    for f in STATE_FEATS:
        a = grp_irl_win[f].dropna()
        b = grp_rf_win[f].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        # 標準化差 (Cohen's d 近似)
        pooled_sd = np.sqrt((a.var() + b.var()) / 2)
        d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        diff_rows.append({
            "feature": f,
            "mean_irl_win": a.mean(),
            "mean_rf_win": b.mean(),
            "cohens_d": d,
            "mwu_p": u.pvalue,
        })
    ddf = pd.DataFrame(diff_rows).sort_values("cohens_d", key=abs, ascending=False)
    ddf.to_csv(f"{RES}/p1_irlwin_vs_rfwin_statediff.csv", index=False)
    print("  state特徴差 (Cohen's d, +なら IRL勝ちペアで高い) top8:")
    print(ddf.head(8)[["feature", "cohens_d", "mwu_p"]].round(4).to_string(index=False))

    # ============================================================
    # 4. 図
    # ============================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # (a) state特徴 単変量 AUC ランキング (上位12)
    top = fdf.head(12).iloc[::-1]
    colors = ["#d62728" if ft in KEY_FEATS else "#7f7f7f"
              for ft in top["feature"]]
    axes[0].barh(top["feature"], top["univ_auc"], color=colors)
    axes[0].axvline(0.5, color="k", ls="--", lw=1, label="無情報 (0.5)")
    axes[0].set_xlabel("単変量 label AUC (方向不問, max(a,1-a))")
    axes[0].set_title("(a) P1での state特徴 単変量識別力\n赤=発展性系 (§10/注目特徴)")
    axes[0].set_xlim(0.45, max(0.7, top["univ_auc"].max() + 0.02))
    axes[0].legend(loc="lower right", fontsize=9)

    # (b) active_months_ratio 分位別の平均 prob (IRL vs RF)
    keyf = "active_months_ratio"
    lng["_q"] = pd.qcut(lng[keyf].rank(method="first"), 5,
                        labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"])
    qg = lng.groupby("_q", observed=True).agg(
        irl=("irl", "mean"), rf=("rf", "mean"), lab=("label", "mean")).reset_index()
    xpos = np.arange(len(qg))
    w = 0.27
    axes[1].bar(xpos - w, qg["irl"], w, label="IRL prob", color="#1f77b4")
    axes[1].bar(xpos, qg["rf"], w, label="RF prob", color="#ff7f0e")
    axes[1].bar(xpos + w, qg["lab"], w, label="実 label率", color="#2ca02c", alpha=0.6)
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels(qg["_q"], fontsize=9)
    axes[1].set_ylabel("平均 prob / label率")
    axes[1].set_xlabel(f"{keyf} 分位")
    axes[1].set_title("(b) active_months_ratio 分位別の平均prob\nIRLは単調連動 / RFは平坦化?")
    axes[1].legend(fontsize=9)

    # (c) IRL勝ち vs RF勝ち の state特徴差 (Cohen's d) 上位
    cd = ddf.head(10).iloc[::-1]
    cc = ["#d62728" if ft in KEY_FEATS else "#7f7f7f" for ft in cd["feature"]]
    axes[2].barh(cd["feature"], cd["cohens_d"], color=cc)
    axes[2].axvline(0, color="k", lw=1)
    axes[2].set_xlabel("Cohen's d  (+: IRL正解∧RF誤りペアで高い)")
    axes[2].set_title("(c) IRL正解∧RF誤り vs 逆\nの t0 state特徴差")

    plt.tight_layout()
    plt.savefig(f"{RES}/fig_p1_feature_reliance.pdf", format="pdf", bbox_inches="tight")
    print(f"\nsaved: {RES}/fig_p1_feature_reliance.pdf")

    # 連動度の要約も保存
    summary = pd.DataFrame([{
        "n_obs": n_obs, "base_rate": base_rate,
        "irl_auc": irl_auc, "rf_auc": rf_auc, "ens_auc": ens_auc,
        "irl_minus_rf_auc": irl_auc - rf_auc,
        "reg_r2_irl": reg_summary["irl"], "reg_r2_rf": reg_summary["rf"],
        "mean_abs_spearman_irl": fdf["abs_sp_irl"].mean(),
        "mean_abs_spearman_rf": fdf["abs_sp_rf"].mean(),
        "prob_std_irl": lng["irl"].std(), "prob_std_rf": lng["rf"].std(),
        "n_irl_win": len(grp_irl_win), "n_rf_win": len(grp_rf_win),
    }])
    summary.to_csv(f"{RES}/p1_feature_reliance_summary.csv", index=False)
    print(f"saved: {RES}/p1_feature_reliance_summary.csv")
    print(summary.round(4).T.to_string())


if __name__ == "__main__":
    main()
