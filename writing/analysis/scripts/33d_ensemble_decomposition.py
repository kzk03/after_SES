"""
33d: アンサンブル利得 +2.2pp の所在を網羅分解。

主結果は Ens(=(irl+rf)/2) が RF 比 +2.2pp AUC(全区間有意)。
この利得がどの部分集団から来るかを網羅分解する。

- ens_lift = Ens_AUC - best(IRL,RF)_AUC  (best基準: 強い主張)
- ens_lift_vs_rf = Ens_AUC - RF_AUC       (論文主結果の基準)
を 開発者archetype(A1-4) × ペアarchetype(P1-4) × base_rate分位 × 予測区間 で分解。

bootstrap CI(seed=777), AUC は両クラス揃う時のみ。
出力: results/ 配下の CSV + fig_ensemble_decomposition.pdf
"""

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 777
RESULTS = "/Users/kazuki-h/Master_research/writing/analysis/results"
DATA = "/Users/kazuki-h/Master_research/writing/analysis/data"
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
N_BOOT = 2000

PAIR_LABEL = {"A1": "P1未経験", "A2": "P2経験低承諾", "A3": "P3経験高承諾", "A4": "P4中程度"}
DEV_LABEL = {"A1": "A1低活動新人", "A2": "A2レビュー専念", "A3": "A3高承諾ベテラン", "A4": "A4常時活動"}


def safe_auc(y, s):
    """両クラス揃い n>=10 のときのみ AUC を返す。"""
    y = np.asarray(y)
    s = np.asarray(s)
    if len(y) < 10 or y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, s)


def boot_auc_diff(y, s_ens, s_base, n_boot=N_BOOT, seed=RANDOM_SEED):
    """Ens_AUC - base_AUC の bootstrap 分布 -> (mean, lo, hi, p_gt0)。"""
    y = np.asarray(y)
    s_ens = np.asarray(s_ens)
    s_base = np.asarray(s_base)
    if len(y) < 10 or y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        a_ens = roc_auc_score(yb, s_ens[idx])
        a_base = roc_auc_score(yb, s_base[idx])
        diffs.append(a_ens - a_base)
    if len(diffs) < 50:
        return np.nan, np.nan, np.nan, np.nan
    diffs = np.array(diffs)
    return (
        float(np.mean(diffs)),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
        float((diffs > 0).mean()),
    )


def long_format(df):
    """horizon を縦持ちに展開した long DF を作る。"""
    rows = []
    for h in HORIZONS:
        sub = df[["developer", "directory", "dev_arch", "pair_arch"]].copy()
        sub["horizon"] = h
        sub["label"] = df[f"label_{h}"].values
        sub["irl"] = df[f"irl_{h}"].values
        sub["rf"] = df[f"rf_{h}"].values
        sub["ens"] = df[f"avg_{h}"].values
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def subgroup_lift(long, group_cols, group_name):
    """部分集団ごとに Ens-best と Ens-RF の lift+CI を計算。"""
    out = []
    for keys, g in long.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = g["label"].values
        a_irl = safe_auc(y, g["irl"].values)
        a_rf = safe_auc(y, g["rf"].values)
        a_ens = safe_auc(y, g["ens"].values)
        if np.isnan(a_ens) or np.isnan(a_rf):
            continue
        best = np.nanmax([a_irl, a_rf])
        base_better = "RF" if (np.isnan(a_irl) or a_rf >= a_irl) else "IRL"
        # bootstrap vs best
        if base_better == "RF":
            m_b, lo_b, hi_b, p_b = boot_auc_diff(y, g["ens"].values, g["rf"].values)
        else:
            m_b, lo_b, hi_b, p_b = boot_auc_diff(y, g["ens"].values, g["irl"].values)
        # bootstrap vs RF (論文基準)
        m_r, lo_r, hi_r, p_r = boot_auc_diff(y, g["ens"].values, g["rf"].values)
        rec = dict(zip(group_cols, keys))
        rec.update(
            {
                "group_type": group_name,
                "n": len(g),
                "base_rate": float(np.mean(y)),
                "auc_irl": a_irl,
                "auc_rf": a_rf,
                "auc_ens": a_ens,
                "best_base": best,
                "base_better_model": base_better,
                "ens_lift_vs_best": a_ens - best,
                "ens_lift_vs_best_boot": m_b,
                "lift_best_ci_lo": lo_b,
                "lift_best_ci_hi": hi_b,
                "p_best_gt0": p_b,
                "ens_lift_vs_rf": a_ens - a_rf,
                "ens_lift_vs_rf_boot": m_r,
                "lift_rf_ci_lo": lo_r,
                "lift_rf_ci_hi": hi_r,
                "p_rf_gt0": p_r,
                "sig_vs_rf": (lo_r > 0) if not np.isnan(lo_r) else False,
            }
        )
        out.append(rec)
    return pd.DataFrame(out)


def main():
    t = pd.read_csv(f"{DATA}/pair_trajectories.csv")
    pa = pd.read_csv(f"{DATA}/pair_archetype.csv")
    t = t.rename(columns={"archetype": "dev_arch"})
    t = t.merge(
        pa[["developer", "directory", "dominant"]].rename(columns={"dominant": "pair_arch"}),
        on=["developer", "directory"],
        how="left",
    )
    re = pd.read_csv(f"{DATA}/pair_repo_experience.csv")
    t = t.merge(
        re[["developer", "directory", "log_prior_n", "prior_acc", "repo_exp_covered"]],
        on=["developer", "directory"],
        how="left",
    )

    long = long_format(t)
    # base_rate 分位は per-pair の prob ではなく集団分割が要るので、
    # 開発者単位の overall acceptance proxy として rf prob 分位を使うとリークになるため
    # base_rate 分位は「ペアの過去承諾率 prior_acc」を用いる(t0前のみ, リークなし)。
    long = long.merge(
        t[["developer", "directory", "prior_acc"]], on=["developer", "directory"], how="left"
    )
    # prior_acc 4分位(被覆ペアのみ)。未被覆は別カテゴリ。
    cov = long["prior_acc"].notna()
    long["pa_q"] = "未被覆"
    try:
        long.loc[cov, "pa_q"] = pd.qcut(
            long.loc[cov, "prior_acc"], 4, labels=["Q1低", "Q2", "Q3", "Q4高"], duplicates="drop"
        ).astype(str)
    except ValueError:
        long.loc[cov, "pa_q"] = pd.cut(
            long.loc[cov, "prior_acc"], 4, labels=["Q1低", "Q2", "Q3", "Q4高"]
        ).astype(str)

    # ===== 0. 全体 lift (区間別 + 全体) =====
    overall_rows = []
    for h in HORIZONS + ["全区間"]:
        g = long if h == "全区間" else long[long["horizon"] == h]
        y = g["label"].values
        a_irl = safe_auc(y, g["irl"].values)
        a_rf = safe_auc(y, g["rf"].values)
        a_ens = safe_auc(y, g["ens"].values)
        best = np.nanmax([a_irl, a_rf])
        m_r, lo_r, hi_r, p_r = boot_auc_diff(y, g["ens"].values, g["rf"].values)
        m_b, lo_b, hi_b, p_b = boot_auc_diff(
            y, g["ens"].values, g["rf"].values if a_rf >= a_irl else g["irl"].values
        )
        overall_rows.append(
            {
                "horizon": h,
                "n": len(g),
                "base_rate": float(np.mean(y)),
                "auc_irl": a_irl,
                "auc_rf": a_rf,
                "auc_ens": a_ens,
                "ens_lift_vs_rf": a_ens - a_rf,
                "lift_rf_ci_lo": lo_r,
                "lift_rf_ci_hi": hi_r,
                "p_rf_gt0": p_r,
                "ens_lift_vs_best": a_ens - best,
                "lift_best_ci_lo": lo_b,
                "lift_best_ci_hi": hi_b,
            }
        )
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(f"{RESULTS}/ens_overall_lift.csv", index=False)

    # ===== 1. 部分集団別分解 =====
    dec_dev_h = subgroup_lift(long, ["dev_arch", "horizon"], "dev_arch_x_horizon")
    dec_pair_h = subgroup_lift(long, ["pair_arch", "horizon"], "pair_arch_x_horizon")
    dec_dev = subgroup_lift(long, ["dev_arch"], "dev_arch")
    dec_pair = subgroup_lift(long, ["pair_arch"], "pair_arch")
    dec_paq = subgroup_lift(long, ["pa_q"], "prior_acc_quartile")
    dec_paq_h = subgroup_lift(long, ["pa_q", "horizon"], "prior_acc_q_x_horizon")
    dec_h = subgroup_lift(long, ["horizon"], "horizon")
    # 2軸クロス: dev x pair
    dec_devpair = subgroup_lift(long, ["dev_arch", "pair_arch"], "dev_x_pair")

    decomposition = pd.concat(
        [dec_h, dec_dev, dec_pair, dec_paq, dec_dev_h, dec_pair_h, dec_paq_h, dec_devpair],
        ignore_index=True,
    )
    decomposition.to_csv(f"{RESULTS}/ens_decomposition.csv", index=False)

    # ===== 2. disagreement(直交誤り) vs ens_lift =====
    # IRL/RF それぞれの per-pair 正解判定は閾値依存になるため、
    # rank-based disagreement を使う: 部分集団内で IRL と RF のスコア順位の
    # spearman 不一致 = 1 - |spearman|。さらに「IRL_only_correct vs RF_only_correct」は
    # 中央値分位(集団内 median split)で疑似ラベル化して直交率を測る。
    from scipy.stats import spearmanr

    dis_rows = []

    def disagree_metrics(g):
        y = g["label"].values
        irl = g["irl"].values
        rf = g["rf"].values
        if len(g) < 20 or y.sum() < 3 or y.sum() > len(y) - 3:
            return None
        rho, _ = spearmanr(irl, rf)
        if np.isnan(rho):
            return None
        # 集団内 median split による疑似予測 (rank-based, 閾値非依存の相対判定)
        irl_pred = (irl > np.median(irl)).astype(int)
        rf_pred = (rf > np.median(rf)).astype(int)
        irl_corr = (irl_pred == y)
        rf_corr = (rf_pred == y)
        irl_only = (irl_corr & ~rf_corr).mean()
        rf_only = (rf_corr & ~irl_corr).mean()
        both_wrong = (~irl_corr & ~rf_corr).mean()
        # 直交度: IRL_only と RF_only が拮抗 & 双方の only が大きいほど Ens に余地
        balance = 1 - abs(irl_only - rf_only) / (irl_only + rf_only + 1e-9)
        complementarity = (irl_only + rf_only) * balance
        a_rf = safe_auc(y, rf)
        a_ens = safe_auc(y, g["ens"].values)
        return {
            "n": len(g),
            "base_rate": float(np.mean(y)),
            "spearman_irl_rf": float(rho),
            "disagreement": float(1 - abs(rho)),
            "irl_only_correct": float(irl_only),
            "rf_only_correct": float(rf_only),
            "both_wrong": float(both_wrong),
            "complementarity": float(complementarity),
            "ens_lift_vs_rf": (a_ens - a_rf) if not (np.isnan(a_ens) or np.isnan(a_rf)) else np.nan,
        }

    for keys, g in long.groupby(["dev_arch", "pair_arch", "horizon"], dropna=False):
        m = disagree_metrics(g)
        if m is None:
            continue
        m.update({"dev_arch": keys[0], "pair_arch": keys[1], "horizon": keys[2]})
        dis_rows.append(m)
    disagree = pd.DataFrame(dis_rows).dropna(subset=["ens_lift_vs_rf"])
    disagree.to_csv(f"{RESULTS}/ens_disagreement.csv", index=False)

    # disagreement と lift の相関(規模重み付けなし + Spearman)
    corr_dis = np.nan
    corr_comp = np.nan
    if len(disagree) >= 8:
        corr_dis = spearmanr(disagree["disagreement"], disagree["ens_lift_vs_rf"])[0]
        corr_comp = spearmanr(disagree["complementarity"], disagree["ens_lift_vs_rf"])[0]

    # ===== 3. 誰が救われるか: per-pair の IRL-RF 乖離 vs Ens 順位改善 =====
    # 各 horizon 内で全体順位を計算し、Ens 順位 - RF 順位の改善を per-pair で見る。
    rescue_rows = []
    for h in HORIZONS:
        g = long[long["horizon"] == h].copy()
        # 全体順位(高prob=高順位)。承諾=正例なので prob 大が良い。
        g["rank_rf"] = g["rf"].rank(pct=True)
        g["rank_ens"] = g["ens"].rank(pct=True)
        # 乖離: IRL と RF の rank 差(集団内)
        g["rank_irl"] = g["irl"].rank(pct=True)
        g["irl_rf_rank_gap"] = (g["rank_irl"] - g["rank_rf"]).abs()
        # 正例について「Ensで順位が上がったか」= rescue(正例の順位上昇は良い)
        # rank_pct 改善
        g["rank_improve"] = g["rank_ens"] - g["rank_rf"]
        rescue_rows.append(g)
    rescue = pd.concat(rescue_rows, ignore_index=True)
    # 乖離4分位 × 正例/負例 で rank_improve 平均
    rescue["gap_q"] = pd.qcut(
        rescue["irl_rf_rank_gap"], 4, labels=["低乖離", "中低", "中高", "高乖離"], duplicates="drop"
    ).astype(str)
    rescue_summary = (
        rescue.groupby(["gap_q", "label"])
        .agg(
            n=("rank_improve", "size"),
            mean_rank_improve=("rank_improve", "mean"),
            mean_gap=("irl_rf_rank_gap", "mean"),
        )
        .reset_index()
    )
    rescue_summary.to_csv(f"{RESULTS}/ens_rescue_by_gap.csv", index=False)

    # 「救われたペア」= 正例 かつ Ens順位が RF順位より閾値以上上昇
    rescued = rescue[(rescue["label"] == 1) & (rescue["rank_improve"] > 0.05)]
    not_rescued = rescue[(rescue["label"] == 1) & (rescue["rank_improve"] <= 0.05)]
    rescue_profile = pd.DataFrame(
        {
            "group": ["救済(正例,rank↑>0.05)", "非救済(正例)"],
            "n": [len(rescued), len(not_rescued)],
            "mean_irl_rf_rank_gap": [
                rescued["irl_rf_rank_gap"].mean(),
                not_rescued["irl_rf_rank_gap"].mean(),
            ],
            "mean_rf_rank": [rescued["rank_rf"].mean(), not_rescued["rank_rf"].mean()],
        }
    )
    rescue_profile.to_csv(f"{RESULTS}/ens_rescue_profile.csv", index=False)

    # ===== 図 =====
    fig = plt.figure(figsize=(16, 11))

    # (a) ヒートマップ: pair_arch × horizon の ens_lift_vs_rf
    ax_a = fig.add_subplot(2, 3, 1)
    piv = dec_pair_h.pivot(index="pair_arch", columns="horizon", values="ens_lift_vs_rf")
    piv = piv.reindex(index=["A1", "A2", "A3", "A4"], columns=HORIZONS)
    piv.index = [PAIR_LABEL.get(i, i) for i in piv.index]
    im = ax_a.imshow(piv.values * 100, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
    ax_a.set_xticks(range(len(HORIZONS)))
    ax_a.set_xticklabels(HORIZONS)
    ax_a.set_yticks(range(len(piv.index)))
    ax_a.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax_a.text(j, i, f"{v*100:+.1f}", ha="center", va="center", fontsize=9)
    ax_a.set_title("(a) ペアarchetype×区間 のEns利得(pp, vs RF)")
    fig.colorbar(im, ax=ax_a, fraction=0.046, label="Ens−RF AUC (pp)")

    # (a2) ヒートマップ: dev_arch × horizon
    ax_a2 = fig.add_subplot(2, 3, 4)
    piv2 = dec_dev_h.pivot(index="dev_arch", columns="horizon", values="ens_lift_vs_rf")
    piv2 = piv2.reindex(index=["A1", "A2", "A3", "A4"], columns=HORIZONS)
    im2 = ax_a2.imshow(piv2.values * 100, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
    ax_a2.set_xticks(range(len(HORIZONS)))
    ax_a2.set_xticklabels(HORIZONS)
    ax_a2.set_yticks(range(len(piv2.index)))
    ax_a2.set_yticklabels(list(piv2.index))
    for i in range(piv2.shape[0]):
        for j in range(piv2.shape[1]):
            v = piv2.values[i, j]
            if not np.isnan(v):
                ax_a2.text(j, i, f"{v*100:+.1f}", ha="center", va="center", fontsize=9)
    ax_a2.set_title("(a2) 開発者archetype×区間 のEns利得(pp, vs RF)")
    fig.colorbar(im2, ax=ax_a2, fraction=0.046, label="Ens−RF AUC (pp)")

    # (b) disagreement vs ens_lift
    ax_b = fig.add_subplot(2, 3, 2)
    sc = ax_b.scatter(
        disagree["disagreement"],
        disagree["ens_lift_vs_rf"] * 100,
        s=disagree["n"] / 3,
        c=disagree["complementarity"],
        cmap="viridis",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.3,
    )
    ax_b.axhline(0, color="gray", ls="--", lw=0.8)
    ax_b.set_xlabel("disagreement (1−|spearman(IRL,RF)|)")
    ax_b.set_ylabel("Ens−RF AUC (pp)")
    ax_b.set_title(
        f"(b) 不一致度 vs Ens利得\nSpearman(disagree)={corr_dis:.2f}, "
        f"Spearman(comp)={corr_comp:.2f}\n(点=dev×pair×区間, サイズ∝n)"
    )
    fig.colorbar(sc, ax=ax_b, fraction=0.046, label="補完度")

    # (c) ens_lift が大きい部分集団 top (dev x pair x horizon, n>=150 のみ = 頑健)
    ax_c = fig.add_subplot(2, 3, 5)
    MIN_N = 150
    top = (
        disagree[disagree["n"] >= MIN_N]
        .sort_values("ens_lift_vs_rf", ascending=False)
        .head(12)
        .copy()
    )
    top["lab"] = top.apply(
        lambda r: f"{r['dev_arch']}×{r['pair_arch']} {r['horizon']} (n={int(r['n'])})", axis=1
    )
    colors = ["tab:green" if v > 0 else "tab:red" for v in top["ens_lift_vs_rf"]]
    ax_c.barh(range(len(top)), top["ens_lift_vs_rf"] * 100, color=colors)
    ax_c.set_yticks(range(len(top)))
    ax_c.set_yticklabels(top["lab"], fontsize=7)
    ax_c.invert_yaxis()
    ax_c.axvline(0, color="gray", lw=0.8)
    ax_c.set_xlabel("Ens−RF AUC (pp)")
    ax_c.set_title(f"(c) Ens利得が大きい部分集団 top(n≥{MIN_N})\n(開発者×ペア×区間)")

    # (d) rescue: IRL-RF rank乖離分位 × 正例 の順位改善
    ax_d = fig.add_subplot(2, 3, 3)
    pos = rescue_summary[rescue_summary["label"] == 1]
    neg = rescue_summary[rescue_summary["label"] == 0]
    order = ["低乖離", "中低", "中高", "高乖離"]
    pos = pos.set_index("gap_q").reindex(order)
    neg = neg.set_index("gap_q").reindex(order)
    x = np.arange(len(order))
    ax_d.bar(x - 0.2, pos["mean_rank_improve"] * 100, 0.4, label="正例(承諾)", color="tab:green")
    ax_d.bar(x + 0.2, neg["mean_rank_improve"] * 100, 0.4, label="負例(拒否)", color="tab:red")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(order)
    ax_d.axhline(0, color="gray", lw=0.8)
    ax_d.set_xlabel("IRL-RF rank乖離分位")
    ax_d.set_ylabel("Ens順位 − RF順位 平均(pct点, ×100)")
    ax_d.set_title("(d) 誰が救われるか:乖離大ペアで\n正例の順位が上がるか")
    ax_d.legend(fontsize=8)

    # (e) 全体区間別 lift bar with CI
    ax_e = fig.add_subplot(2, 3, 6)
    ov = overall[overall["horizon"] != "全区間"].set_index("horizon").reindex(HORIZONS)
    x = np.arange(len(HORIZONS))
    yv = ov["ens_lift_vs_rf"].values * 100
    lo = (ov["ens_lift_vs_rf"].values - ov["lift_rf_ci_lo"].values) * 100
    hi = (ov["lift_rf_ci_hi"].values - ov["ens_lift_vs_rf"].values) * 100
    ax_e.bar(x, yv, color="tab:blue", alpha=0.7)
    ax_e.errorbar(x, yv, yerr=[lo, hi], fmt="none", ecolor="k", capsize=4)
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(HORIZONS)
    ax_e.axhline(0, color="gray", lw=0.8)
    ax_e.set_ylabel("Ens−RF AUC (pp)")
    ax_e.set_title("(e) 全体 区間別 Ens利得(95%CI)")
    for i, v in enumerate(yv):
        ax_e.text(i, v + 0.2, f"{v:+.1f}", ha="center", fontsize=8)

    fig.suptitle("アンサンブル利得(+2.2pp)の網羅分解: どの部分集団が稼ぐか", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{RESULTS}/fig_ensemble_decomposition.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ===== 標準出力サマリ =====
    print("=== 全体 区間別 lift ===")
    print(overall.to_string(index=False))
    print("\n=== ペアarchetype別 lift (vs RF) ===")
    print(
        dec_pair[
            ["pair_arch", "n", "base_rate", "auc_rf", "auc_ens", "ens_lift_vs_rf",
             "lift_rf_ci_lo", "lift_rf_ci_hi", "sig_vs_rf"]
        ].to_string(index=False)
    )
    print("\n=== 開発者archetype別 lift (vs RF) ===")
    print(
        dec_dev[
            ["dev_arch", "n", "base_rate", "auc_rf", "auc_ens", "ens_lift_vs_rf",
             "lift_rf_ci_lo", "lift_rf_ci_hi", "sig_vs_rf"]
        ].to_string(index=False)
    )
    print("\n=== prior_acc分位別 lift (vs RF) ===")
    print(
        dec_paq[
            ["pa_q", "n", "base_rate", "auc_rf", "auc_ens", "ens_lift_vs_rf",
             "lift_rf_ci_lo", "lift_rf_ci_hi", "sig_vs_rf"]
        ].to_string(index=False)
    )
    print("\n=== disagreement vs ens_lift 相関 ===")
    print(f"Spearman(disagreement, ens_lift) = {corr_dis:.3f}")
    print(f"Spearman(complementarity, ens_lift) = {corr_comp:.3f}")
    print("\n=== rescue profile ===")
    print(rescue_profile.to_string(index=False))
    print("\n=== rescue by gap (正例のみ) ===")
    print(pos[["n", "mean_rank_improve", "mean_gap"]].to_string())
    print("\n=== Ens利得 top 部分集団 (dev×pair×区間) ===")
    print(
        disagree.sort_values("ens_lift_vs_rf", ascending=False)
        .head(10)[
            ["dev_arch", "pair_arch", "horizon", "n", "base_rate", "disagreement",
             "ens_lift_vs_rf"]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
