"""
30c_p1_mechanism.py
担当: なぜ P1(当該dir未経験)が多い dir では IRL が RF に優位なのか — 機序深掘り。

中心仮説: RF は当該dir累積経験(path親和度)を主シグナルにするが、P1未経験ペアでは
これらが ~0 で欠落 → RF は識別シグナルを失い base_rate 近傍に潰れる(予測のばらつき消失)。
一方 IRL は開発者の月次状態系列(全体活動)を符号化するので、当該dir経験が無くても
継続/離脱を識別できる。

V1: P1 vs P3 で IRL_AUC, RF_AUC, delta (pooled + 区間別)
V2: 各群で RF/IRL 予測値の std/分布幅。P1 で RF が base_rate 近傍に集中 (低std) するか
V3: P1 群で RF prob と実ラベルの関係が弱い (RF prob ほぼ定数) ことを定量化
V4: P1 高でも RF 優位な project (skyline_apiserver, roles 等) の確認
"""
import json
import os

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = "/Users/kazuki-h/Master_research/writing/analysis"
RESULTS = os.path.join(BASE, "results")
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
PMAP = {"A1": "P1", "A2": "P2", "A3": "P3", "A4": "P4"}


def load():
    traj = pd.read_csv(os.path.join(BASE, "data/pair_trajectories.csv"))
    arch = pd.read_csv(os.path.join(BASE, "data/pair_archetype.csv"))
    m = traj.merge(
        arch[["developer", "directory", "dominant"]], on=["developer", "directory"], how="left"
    )
    m["pair_arch"] = m["dominant"].map(PMAP)
    m["top_dir"] = m["directory"].str.split("/").str[0]
    return m


def to_long(df):
    """pair-wide -> obs-long (1 行 = 1 pair×horizon)."""
    rows = []
    for h in HORIZONS:
        sub = df[["developer", "directory", "top_dir", "pair_arch"]].copy()
        sub["horizon"] = h
        sub["label"] = df[f"label_{h}"].values
        sub["irl"] = df[f"irl_{h}"].values
        sub["rf"] = df[f"rf_{h}"].values
        rows.append(sub)
    return pd.concat(rows, ignore_index=True)


def safe_auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def safe_ap(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, s)


# ---------------------------------------------------------------------------
# V1: P1 vs P3 (and P2,P4) AUC/AP for IRL vs RF, pooled + per-horizon
# ---------------------------------------------------------------------------
def v1(long):
    rows = []
    groups = ["P1", "P2", "P3", "P4"]
    for g in groups:
        sub = long[long["pair_arch"] == g]
        # pooled
        rows.append(
            dict(
                group=g,
                horizon="pooled",
                n_obs=len(sub),
                n_pos=int(sub["label"].sum()),
                base_rate=sub["label"].mean(),
                irl_auc=safe_auc(sub["label"], sub["irl"]),
                rf_auc=safe_auc(sub["label"], sub["rf"]),
                irl_ap=safe_ap(sub["label"], sub["irl"]),
                rf_ap=safe_ap(sub["label"], sub["rf"]),
            )
        )
        for h in HORIZONS:
            s2 = sub[sub["horizon"] == h]
            rows.append(
                dict(
                    group=g,
                    horizon=h,
                    n_obs=len(s2),
                    n_pos=int(s2["label"].sum()),
                    base_rate=s2["label"].mean(),
                    irl_auc=safe_auc(s2["label"], s2["irl"]),
                    rf_auc=safe_auc(s2["label"], s2["rf"]),
                    irl_ap=safe_ap(s2["label"], s2["irl"]),
                    rf_ap=safe_ap(s2["label"], s2["rf"]),
                )
            )
    df = pd.DataFrame(rows)
    df["delta_auc"] = df["irl_auc"] - df["rf_auc"]
    df["delta_ap"] = df["irl_ap"] - df["rf_ap"]
    return df


# ---------------------------------------------------------------------------
# V2/V3: prediction dispersion (RF collapse) per group
# ---------------------------------------------------------------------------
def v2(long):
    rows = []
    for g in ["P1", "P2", "P3", "P4"]:
        sub = long[long["pair_arch"] == g]
        br = sub["label"].mean()
        # how close RF is to base_rate: mean |rf - base_rate|, and std
        rf = sub["rf"].values
        irl = sub["irl"].values
        # rank-normalize irl to [0,1]? keep raw for std but also report IQR
        rows.append(
            dict(
                group=g,
                n_obs=len(sub),
                base_rate=br,
                rf_mean=rf.mean(),
                rf_std=rf.std(),
                rf_iqr=np.percentile(rf, 75) - np.percentile(rf, 25),
                rf_cv=rf.std() / (rf.mean() + 1e-9),
                rf_mad_from_base=np.mean(np.abs(rf - br)),
                irl_mean=irl.mean(),
                irl_std=irl.std(),
                irl_iqr=np.percentile(irl, 75) - np.percentile(irl, 25),
                # IRL raw scores are tiny/uncalibrated; use rank-based dispersion (gini of ranks)
                irl_cv=irl.std() / (irl.mean() + 1e-9),
            )
        )
    return pd.DataFrame(rows)


def v3(long):
    """RF prob -> label association weakness in P1.
    Use point-biserial (= spearman with binary) between score and label,
    and the spread of mean score between pos/neg classes (separation).
    """
    rows = []
    for g in ["P1", "P2", "P3", "P4"]:
        sub = long[long["pair_arch"] == g]
        y = sub["label"].values
        for model, col in [("RF", "rf"), ("IRL", "irl")]:
            s = sub[col].values
            if len(np.unique(y)) < 2:
                rho, p = np.nan, np.nan
            else:
                rho, p = spearmanr(s, y)
            pos = s[y == 1]
            neg = s[y == 0]
            # class separation in std units (Cohen's-d-like on raw score)
            pooled_sd = np.sqrt((pos.var() + neg.var()) / 2) if len(pos) and len(neg) else np.nan
            sep = (pos.mean() - neg.mean()) / (pooled_sd + 1e-12) if pooled_sd else np.nan
            # Mann-Whitney effect: AUC already covers this; report mean gap
            rows.append(
                dict(
                    group=g,
                    model=model,
                    n_obs=len(sub),
                    n_pos=int(y.sum()),
                    score_label_rho=rho,
                    rho_p=p,
                    mean_score_pos=pos.mean() if len(pos) else np.nan,
                    mean_score_neg=neg.mean() if len(neg) else np.nan,
                    class_separation_d=sep,
                )
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# V3b: SIZE-CONTROLLED evidence
#  (a) base_rate-matched contrast P1 vs P4 (both base_rate ~0.21-0.24)
#  (b) within-project PAIRED delta: P1-subset vs experienced(P2+P3)-subset
#      -> controls project-level base_rate / score scale
# ---------------------------------------------------------------------------
def v3b(long):
    out = {}
    # (a) base_rate-matched: P1 vs P4
    rows = []
    for g in ["P1", "P4"]:
        sub = long[long["pair_arch"] == g]
        br = sub["label"].mean()
        rows.append(
            dict(
                group=g,
                base_rate=br,
                rf_std=sub["rf"].std(),
                irl_std=sub["irl"].std(),
                rf_auc=safe_auc(sub["label"], sub["rf"]),
                irl_auc=safe_auc(sub["label"], sub["irl"]),
                delta_auc=safe_auc(sub["label"], sub["irl"]) - safe_auc(sub["label"], sub["rf"]),
                # collapse: fraction of RF preds within +-0.05 of base_rate
                rf_frac_at_base=float(np.mean(np.abs(sub["rf"] - br) < 0.05)),
            )
        )
    out["matched_P1_P4"] = pd.DataFrame(rows)

    # (b) within-project paired
    res = []
    for td, g in long.groupby("top_dir"):
        p1 = g[g["pair_arch"] == "P1"]
        exp = g[g["pair_arch"].isin(["P2", "P3"])]
        if p1["label"].nunique() < 2 or len(p1) < 15:
            continue
        d_p1 = safe_auc(p1["label"], p1["irl"]) - safe_auc(p1["label"], p1["rf"])
        if len(exp) >= 15 and exp["label"].nunique() > 1:
            d_exp = safe_auc(exp["label"], exp["irl"]) - safe_auc(exp["label"], exp["rf"])
        else:
            d_exp = np.nan
        res.append(dict(top_dir=td, p1_n=len(p1), p1_delta=d_p1, exp_delta=d_exp))
    wp = pd.DataFrame(res)
    out["within_project"] = wp
    paired = wp.dropna(subset=["exp_delta"])
    stat, pval = (np.nan, np.nan)
    if len(paired) >= 6:
        stat, pval = wilcoxon(paired["p1_delta"], paired["exp_delta"])
    out["within_project_summary"] = dict(
        n_proj_p1=int(len(wp)),
        mean_p1_delta=float(wp["p1_delta"].mean()),
        median_p1_delta=float(wp["p1_delta"].median()),
        frac_p1_irl_wins=float((wp["p1_delta"] > 0).mean()),
        n_paired=int(len(paired)),
        mean_p1_delta_paired=float(paired["p1_delta"].mean()),
        mean_exp_delta_paired=float(paired["exp_delta"].mean()),
        paired_diff=float((paired["p1_delta"] - paired["exp_delta"]).mean()),
        wilcoxon_p=float(pval),
    )
    return out


# ---------------------------------------------------------------------------
# V4: counterexample projects (high P1 share but RF-favorable)
# ---------------------------------------------------------------------------
def v4(long, pm):
    # per project: P1 share, delta_auc (project level), and within-P1 RF dispersion
    rel = pm[pm["reliable"] == (pm["reliable"].max())].copy() if pm["reliable"].dtype != bool else pm[pm["reliable"]].copy()
    # identify counterexamples: high P1 share, negative delta_auc
    rel = rel.sort_values("pair_P1_share", ascending=False)
    rows = []
    for _, r in rel.iterrows():
        td = r["top_dir"]
        sub = long[(long["top_dir"] == td) & (long["pair_arch"] == "P1")]
        if len(sub) < 5:
            continue
        y = sub["label"].values
        rows.append(
            dict(
                top_dir=td,
                n_pair=int(r["n_pair"]),
                base_rate=r["base_rate"],
                pair_P1_share=r["pair_P1_share"],
                proj_delta_auc=r["delta_auc"],
                p1_n_obs=len(sub),
                p1_base_rate=sub["label"].mean(),
                p1_rf_std=sub["rf"].std(),
                p1_rf_auc=safe_auc(y, sub["rf"].values),
                p1_irl_auc=safe_auc(y, sub["irl"].values),
            )
        )
    df = pd.DataFrame(rows)
    df["p1_delta_auc"] = df["p1_irl_auc"] - df["p1_rf_auc"]
    return df


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_fig(v1df, long, v3bd):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --- (a) AUC bars: IRL/RF for P1 vs P3 ---
    ax = axes[0, 0]
    pooled = v1df[v1df["horizon"] == "pooled"].set_index("group")
    groups = ["P1", "P3"]
    x = np.arange(len(groups))
    w = 0.35
    irl_v = [pooled.loc[g, "irl_auc"] for g in groups]
    rf_v = [pooled.loc[g, "rf_auc"] for g in groups]
    ax.bar(x - w / 2, irl_v, w, label="IRL", color="#1f77b4")
    ax.bar(x + w / 2, rf_v, w, label="RF", color="#ff7f0e")
    ax.axhline(0.5, ls="--", c="gray", lw=1, label="ランダム (0.5)")
    for i in range(len(groups)):
        ax.text(i - w / 2, irl_v[i] + 0.008, f"{irl_v[i]:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, rf_v[i] + 0.008, f"{rf_v[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["P1\n(未経験)", "P3\n(経験あり高承諾)"])
    ax.set_ylabel("AUC (pooled)")
    ax.set_ylim(0.45, 0.9)
    ax.set_title("(a) P1ではIRL>RF, P3ではRF>IRL")
    ax.legend(fontsize=8, loc="lower right")

    # --- (b) RF prediction distribution P1 vs P3 (collapse to base_rate) ---
    ax = axes[0, 1]
    p1 = long[long["pair_arch"] == "P1"]
    p3 = long[long["pair_arch"] == "P3"]
    bins = np.linspace(0, 1, 31)
    ax.hist(p1["rf"], bins=bins, density=True, alpha=0.55, color="#ff7f0e",
            label=f"P1未経験 (std={p1['rf'].std():.3f})")
    ax.hist(p3["rf"], bins=bins, density=True, alpha=0.40, color="#2ca02c",
            label=f"P3経験あり (std={p3['rf'].std():.3f})", histtype="stepfilled")
    ax.axvline(p1["label"].mean(), ls="--", c="#cc5500", lw=1.2,
               label=f"P1 base_rate={p1['label'].mean():.2f}")
    ax.set_xlabel("RF 予測確率")
    ax.set_ylabel("密度")
    ax.set_title("(b) RF予測: P1は低base_rate域に圧縮")
    ax.legend(fontsize=7.5)

    # --- (c) class separation (mean_pos - mean_neg), base_rate-normalized ---
    ax = axes[1, 0]
    sep = {}
    for g in ["P1", "P3"]:
        sub = long[long["pair_arch"] == g]
        y = sub["label"].values
        br = sub["label"].mean()
        sep[g] = {}
        for col, name in [("rf", "RF"), ("irl", "IRL")]:
            pos = sub[col][y == 1]
            neg = sub[col][y == 0]
            sep[g][name] = (pos.mean() - neg.mean()) / br  # base_rate-normalized gap
    gx = np.arange(2)
    rf_s = [sep["P1"]["RF"], sep["P3"]["RF"]]
    irl_s = [sep["P1"]["IRL"], sep["P3"]["IRL"]]
    ax.bar(gx - w / 2, irl_s, w, label="IRL", color="#1f77b4")
    ax.bar(gx + w / 2, rf_s, w, label="RF", color="#ff7f0e")
    for i in range(2):
        ax.text(i - w / 2, irl_s[i] + 0.005, f"{irl_s[i]:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, rf_s[i] + 0.005, f"{rf_s[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(gx)
    ax.set_xticklabels(["P1\n(未経験)", "P3\n(経験あり高承諾)"])
    ax.set_ylabel("クラス分離 (正例-負例平均) / base_rate")
    ax.set_title("(c) P1ではIRLのクラス分離がRFを上回る")
    ax.legend(fontsize=8)

    # --- (d) size-controlled: within-project paired delta_auc ---
    ax = axes[1, 1]
    paired = v3bd["within_project"].dropna(subset=["exp_delta"])
    for _, r in paired.iterrows():
        ax.plot([0, 1], [r["exp_delta"], r["p1_delta"]], color="gray", alpha=0.35, lw=0.8)
    ax.scatter(np.zeros(len(paired)), paired["exp_delta"], color="#2ca02c", s=18, zorder=3,
               label="経験ありペア(P2+P3)")
    ax.scatter(np.ones(len(paired)), paired["p1_delta"], color="#ff7f0e", s=18, zorder=3,
               label="P1未経験ペア")
    ax.plot([0, 1], [paired["exp_delta"].mean(), paired["p1_delta"].mean()],
            color="black", lw=2.2, marker="o", zorder=4, label="平均")
    ax.axhline(0, ls="--", c="gray", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["経験あり\n(P2+P3)", "P1\n(未経験)"])
    ax.set_ylabel("delta_auc (IRL - RF), 同一project内")
    s = v3bd["within_project_summary"]
    ax.set_title(
        f"(d) project内対応: P1でIRLが相対優位\n"
        f"平均差={s['paired_diff']:+.3f}, Wilcoxon p={s['wilcoxon_p']:.3f}, n={s['n_paired']}",
        fontsize=10,
    )
    ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle(
        "P1(当該dir未経験)でIRLがRFに優位な機序: path特徴欠落でRFのクラス分離が弱まりIRLの系列符号化が上回る",
        fontsize=12.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULTS, "fig_p1_mechanism.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    m = load()
    long = to_long(m)
    pm = pd.read_csv(os.path.join(RESULTS, "project_master.csv"))

    v1df = v1(long)
    v2df = v2(long)
    v3df = v3(long)
    v3bd = v3b(long)
    v4df = v4(long, pm)

    v1df.to_csv(os.path.join(RESULTS, "p1_mech_v1_auc.csv"), index=False)
    v2df.to_csv(os.path.join(RESULTS, "p1_mech_v2_dispersion.csv"), index=False)
    v3df.to_csv(os.path.join(RESULTS, "p1_mech_v3_separation.csv"), index=False)
    v3bd["matched_P1_P4"].to_csv(os.path.join(RESULTS, "p1_mech_v3b_matched.csv"), index=False)
    v3bd["within_project"].to_csv(os.path.join(RESULTS, "p1_mech_v3b_within_project.csv"), index=False)
    v4df.to_csv(os.path.join(RESULTS, "p1_mech_v4_counterexamples.csv"), index=False)

    figpath = make_fig(v1df, long, v3bd)

    # ---- console summary ----
    print("===== V1 pooled AUC by pair archetype =====")
    print(
        v1df[v1df["horizon"] == "pooled"][
            ["group", "n_obs", "n_pos", "base_rate", "irl_auc", "rf_auc",
             "delta_auc", "irl_ap", "rf_ap", "delta_ap"]
        ].round(4).to_string(index=False)
    )
    print("\n===== V1 per-horizon delta_auc (IRL-RF) =====")
    piv = v1df[v1df["horizon"] != "pooled"].pivot(index="group", columns="horizon", values="delta_auc")
    print(piv.round(4).to_string())

    print("\n===== V2 prediction dispersion =====")
    print(v2df[["group", "base_rate", "rf_mean", "rf_std", "rf_iqr",
                "rf_mad_from_base", "irl_std", "irl_iqr"]].round(4).to_string(index=False))

    print("\n===== V3 score-label association =====")
    print(v3df[["group", "model", "score_label_rho", "rho_p",
                "mean_score_pos", "mean_score_neg", "class_separation_d"]].round(4).to_string(index=False))

    print("\n===== V3b(a) base_rate-matched P1 vs P4 (both base_rate ~0.21-0.24) =====")
    print(v3bd["matched_P1_P4"].round(4).to_string(index=False))
    print("\n===== V3b(b) within-project PAIRED (size-controlled) =====")
    print(json.dumps(v3bd["within_project_summary"], indent=2, ensure_ascii=False))

    print("\n===== V4 counterexamples (project-level, sorted by P1 share) =====")
    print(v4df[["top_dir", "n_pair", "base_rate", "pair_P1_share", "proj_delta_auc",
                "p1_n_obs", "p1_base_rate", "p1_rf_std", "p1_rf_auc",
                "p1_irl_auc", "p1_delta_auc"]].round(4).to_string(index=False))

    # Save a small JSON of headline numbers
    pooled = v1df[v1df["horizon"] == "pooled"].set_index("group")
    headline = dict(
        p1_irl_auc=float(pooled.loc["P1", "irl_auc"]),
        p1_rf_auc=float(pooled.loc["P1", "rf_auc"]),
        p1_delta_auc=float(pooled.loc["P1", "delta_auc"]),
        p3_irl_auc=float(pooled.loc["P3", "irl_auc"]),
        p3_rf_auc=float(pooled.loc["P3", "rf_auc"]),
        p3_delta_auc=float(pooled.loc["P3", "delta_auc"]),
        p1_rf_std=float(v2df.set_index("group").loc["P1", "rf_std"]),
        p3_rf_std=float(v2df.set_index("group").loc["P3", "rf_std"]),
        p1_rf_rho=float(v3df[(v3df.group == "P1") & (v3df.model == "RF")]["score_label_rho"].iloc[0]),
        p1_irl_rho=float(v3df[(v3df.group == "P1") & (v3df.model == "IRL")]["score_label_rho"].iloc[0]),
        # base_rate-matched: P1 vs P4 both low base_rate
        p4_base_rate=float(v3bd["matched_P1_P4"].set_index("group").loc["P4", "base_rate"]),
        p4_rf_std=float(v3bd["matched_P1_P4"].set_index("group").loc["P4", "rf_std"]),
        p4_delta_auc=float(v3bd["matched_P1_P4"].set_index("group").loc["P4", "delta_auc"]),
        # within-project size-controlled
        within_proj=v3bd["within_project_summary"],
    )
    with open(os.path.join(RESULTS, "p1_mech_headline.json"), "w") as f:
        json.dump(headline, f, indent=2, ensure_ascii=False)
    print("\nHEADLINE:", json.dumps(headline, indent=2, ensure_ascii=False))
    print("\nFIG:", figpath)


if __name__ == "__main__":
    main()
