"""30b_a4_mechanism.py
A4(常時活動型)が多い dir/project は、なぜ予測が難しいのか — 機序深掘り。

既知: dir レベルで dev_A4_share↑ → ens_auc↓ (偏rho −0.32)。本スクリプトはその機序を
ペア粒度で検証する。検証する仮説:
  H1[ラベル曖昧性]   A4ペアの base_rate が 0.5 近傍で本質的に予測困難か。
                     A4 vs 非A4 で base_rate と「ラベルの予測区間間ばらつき(揺らぎ)」を比較。
  H2[累積特徴の飽和] A4は累積統計が頭打ち→識別力が低いか。
                     A4 vs 非A4 で主要特徴量の分布と特徴量→ラベルの単変量 AUC を比較。
  H3[A4×P1局在]     A4の難しさが「A4×P1セル」に局在するか。
                     A4をpair archetype(P1-P4)で分解し各セルの base_rate, IRL/RF/Ens AUC を出す。
  H4[活動の脱文脈性] A4は複数dirに分散→ペア単位シグナルがノイジー。
                     A4開発者の関与dir数や活動分散を非A4と比較。

実行: cd /Users/kazuki-h/Master_research/writing/analysis && uv run python scripts/30b_a4_mechanism.py
"""
import json
import warnings

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

DATA = "/Users/kazuki-h/Master_research/writing/analysis/data"
RES = "/Users/kazuki-h/Master_research/writing/analysis/results"
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
# 開発者 archetype A1-A4 を pair archetype P1-P4 にマップ (A1→P1..A4→P4)
ARCH2P = {"A1": "P1", "A2": "P2", "A3": "P3", "A4": "P4"}


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    m = ~(pd.isna(y) | pd.isna(s))
    y, s = y[m], s[m]
    if len(np.unique(y)) < 2 or len(y) < 8:
        return np.nan, len(y), int(y.sum())
    return roc_auc_score(y, s), len(y), int(y.sum())


# ----------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------
pt = pd.read_csv(f"{DATA}/pair_trajectories.csv")
pa = pd.read_csv(f"{DATA}/pair_archetype.csv")
fv = pd.read_csv(f"{DATA}/feature_values.csv")
fv0 = fv[fv["__snapshot"] == "2022-01-01"].copy()

# developer archetype は pt['archetype'] (A1-A4 / NaN)
pt["devarch"] = pt["archetype"]
pt["isA4"] = pt["devarch"] == "A4"
pt["grp"] = np.where(pt["isA4"], "A4", "non-A4")
# 非A4 のうち archetype 判定済み (A1-A3) のみを「非A4(既知)」として一部解析で使う
pt["has_arch"] = pt["devarch"].isin(["A1", "A2", "A3", "A4"])

# pair archetype を join
pt = pt.merge(
    pa[["developer", "directory", "dominant"]], on=["developer", "directory"], how="left"
)
pt["pair_arch"] = pt["dominant"].map(ARCH2P)

# per-pair base_rate と label fluctuation (予測区間間の揺らぎ)
labcols = [f"label_{h}" for h in HORIZONS]
pt["pair_baserate"] = pt[labcols].mean(axis=1)
pt["pair_labelstd"] = pt[labcols].std(axis=1)  # 4 horizon にわたる揺らぎ
# 0.5 近傍度: |baserate-0.5| が小さいほど曖昧。ambiguity = 1-2*|br-0.5| in [0,1]
pt["ambiguity"] = 1 - 2 * np.abs(pt["pair_baserate"] - 0.5)

# ----------------------------------------------------------------------------
# long format: (pair, horizon) = 1 obs
# ----------------------------------------------------------------------------
rows = []
for h in HORIZONS:
    s = pt[
        ["developer", "directory", "devarch", "grp", "pair_arch", f"label_{h}", f"irl_{h}",
         f"rf_{h}", f"avg_{h}"]
    ].copy()
    s.columns = ["developer", "directory", "devarch", "grp", "pair_arch",
                 "label", "irl", "rf", "avg"]
    s["horizon"] = h
    rows.append(s)
long = pd.concat(rows, ignore_index=True)

# ============================================================================
# H1: ラベル曖昧性
# ============================================================================
print("=" * 70)
print("H1: ラベル曖昧性 (base_rate 0.5近傍 / 揺らぎ)")
print("=" * 70)
h1 = pt.groupby("grp").agg(
    n_pair=("developer", "size"),
    baserate_mean=("pair_baserate", "mean"),
    baserate_median=("pair_baserate", "median"),
    labelstd_mean=("pair_labelstd", "mean"),
    ambiguity_mean=("ambiguity", "mean"),
).reset_index()
print(h1.to_string(index=False))

# obs-level pos rate
obs_pos = long.groupby("grp")["label"].agg(["mean", "count"])
print("\nobs-level pos rate (base_rate):")
print(obs_pos)

# A4 vs nonA4: pair_labelstd の有意差 (Mann-Whitney)
a4_std = pt.loc[pt.isA4, "pair_labelstd"].dropna()
non_std = pt.loc[~pt.isA4, "pair_labelstd"].dropna()
u, p_std = stats.mannwhitneyu(a4_std, non_std, alternative="greater")
print(f"\npair_labelstd A4={a4_std.mean():.3f} vs nonA4={non_std.mean():.3f} "
      f"MWU p(A4>)={p_std:.2e}")
a4_amb = pt.loc[pt.isA4, "ambiguity"].dropna()
non_amb = pt.loc[~pt.isA4, "ambiguity"].dropna()
u2, p_amb = stats.mannwhitneyu(a4_amb, non_amb, alternative="greater")
print(f"ambiguity   A4={a4_amb.mean():.3f} vs nonA4={non_amb.mean():.3f} "
      f"MWU p(A4>)={p_amb:.2e}")

# 「曖昧ペア (0<br<1, 揺らぎあり)」の割合
pt["is_ambig_pair"] = (pt["pair_baserate"] > 0) & (pt["pair_baserate"] < 1)
amb_share = pt.groupby("grp")["is_ambig_pair"].mean()
print("\n中間 base_rate (0<br<1; 揺らぎペア) の割合:")
print(amb_share)

# H1 をデータで条件付け検証: A4 と非A4 を「同じ base_rate 帯」に揃えても AUC が低いか
# (難しさが単に base_rate≈0.5 由来かを切り分け)。曖昧ペア obs のみで AUC 比較。
print("\n[条件付け] base_rate 中間帯 (0<br<1) のペア obs に限定した Ens AUC:")
mid_long = long.merge(pt[["developer", "directory", "is_ambig_pair"]],
                      on=["developer", "directory"], how="left")
for g in ["A4", "non-A4"]:
    sub = mid_long[(mid_long.grp == g) & (mid_long.is_ambig_pair)]
    auc, n, npos = safe_auc(sub["label"], sub["avg"])
    print(f"  {g:7s}: Ens AUC={auc:.3f} (n={n}, pos={npos})")

# ============================================================================
# H2: 累積特徴の飽和 / 識別力
# ============================================================================
print("\n" + "=" * 70)
print("H2: 累積特徴の飽和と識別力")
print("=" * 70)
# 各開発者の t0 特徴量を join (pair の developer 単位)
feat_cols = [
    "total_changes", "total_reviews", "recent_activity_frequency", "active_months_ratio",
    "unique_collaborator_count", "overall_acceptance_rate", "recent_acceptance_rate",
    "avg_activity_gap", "core_reviewer_ratio", "repeat_collaboration_rate",
    "acceptance_trend", "activity_trend",
]
fmap = fv0.set_index("__email")
# obs に developer 特徴量を付与
long_f = long.merge(
    fv0.rename(columns={"__email": "developer"})[["developer"] + feat_cols],
    on="developer", how="left",
)

# 主要累積特徴の A4 vs nonA4 分布 (dev 単位で 1 回ずつ)
dev_feat = pt[["developer", "grp", "devarch"]].drop_duplicates("developer").merge(
    fv0.rename(columns={"__email": "developer"})[["developer"] + feat_cols],
    on="developer", how="left",
)
print("\n主要累積特徴の A4 vs non-A4 (dev 単位 median):")
sat_tbl = dev_feat.groupby("grp")[feat_cols].median()
print(sat_tbl.T.to_string())

# 識別力: 各特徴の「ラベルに対する単変量 AUC」を A4 / nonA4 別に。
# 特徴量が継続/離脱(label)を分けられるか。obs 単位、developer 特徴で予測。
print("\n各特徴の単変量 AUC (label 予測, obs単位; A4 / non-A4 別):")
disc_rows = []
for f in feat_cols:
    rec = {"feature": f}
    for g in ["A4", "non-A4"]:
        sub = long_f[long_f.grp == g]
        x = sub[f].values
        y = sub["label"].values
        m = ~(pd.isna(x) | pd.isna(y))
        x, y = x[m], y[m]
        if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
            rec[g] = np.nan
        else:
            a = roc_auc_score(y, x)
            rec[g] = max(a, 1 - a)  # 方向を問わない識別力
    disc_rows.append(rec)
disc = pd.DataFrame(disc_rows)
disc["gap"] = disc["non-A4"] - disc["A4"]
print(disc.sort_values("gap", ascending=False).to_string(index=False))
print(f"\n平均識別力 (max(auc,1-auc)): A4={disc['A4'].mean():.3f} "
      f"non-A4={disc['non-A4'].mean():.3f}")

# ============================================================================
# H3: A4×P(pair archetype) 分解
# ============================================================================
print("\n" + "=" * 70)
print("H3: A4 開発者ペアを pair archetype で分解 (各セル base_rate / AUC)")
print("=" * 70)
a4_long = long[long.devarch == "A4"].copy()
cell_rows = []
for p in ["P1", "P2", "P3", "P4"]:
    sub = a4_long[a4_long.pair_arch == p]
    if len(sub) == 0:
        continue
    irl_auc, n, npos = safe_auc(sub["label"], sub["irl"])
    rf_auc, _, _ = safe_auc(sub["label"], sub["rf"])
    ens_auc, _, _ = safe_auc(sub["label"], sub["avg"])
    cell_rows.append({
        "cell": f"A4×{p}", "pair_arch": p, "n_obs": n, "n_pos": npos,
        "base_rate": sub["label"].mean(),
        "irl_auc": irl_auc, "rf_auc": rf_auc, "ens_auc": ens_auc,
    })
# 比較対照: 全 A4
irl_auc, n, npos = safe_auc(a4_long["label"], a4_long["irl"])
rf_auc, _, _ = safe_auc(a4_long["label"], a4_long["rf"])
ens_auc, _, _ = safe_auc(a4_long["label"], a4_long["avg"])
cell_rows.append({"cell": "A4 (全体)", "pair_arch": "ALL", "n_obs": n, "n_pos": npos,
                  "base_rate": a4_long["label"].mean(),
                  "irl_auc": irl_auc, "rf_auc": rf_auc, "ens_auc": ens_auc})
# 比較対照: 非A4 全体
non_long = long[long.devarch.isin(["A1", "A2", "A3"])]
irl_auc, n, npos = safe_auc(non_long["label"], non_long["irl"])
rf_auc, _, _ = safe_auc(non_long["label"], non_long["rf"])
ens_auc, _, _ = safe_auc(non_long["label"], non_long["avg"])
cell_rows.append({"cell": "non-A4 (A1-3)", "pair_arch": "ALL", "n_obs": n, "n_pos": npos,
                  "base_rate": non_long["label"].mean(),
                  "irl_auc": irl_auc, "rf_auc": rf_auc, "ens_auc": ens_auc})
h3 = pd.DataFrame(cell_rows)
print(h3.to_string(index=False))

# ============================================================================
# H4: 活動の脱文脈性 (関与 dir 数 / 活動分散)
# ============================================================================
print("\n" + "=" * 70)
print("H4: 活動の脱文脈性 (関与 dir 数 / 分散)")
print("=" * 70)
dev_disp = pt.groupby("developer").agg(
    grp=("grp", "first"),
    devarch=("devarch", "first"),
    n_dir=("directory", "nunique"),
    n_obs_pair=("directory", "size"),
).reset_index()
# 1 dir あたりの平均ペア数 (集中度の逆) と base_rate の dir 間分散
# 各 dev について dir 間 baserate のばらつき (活動が dir 横断でどれだけ均質か)
pair_br = pt.groupby(["developer", "directory"])["pair_baserate"].first().reset_index()
br_disp = pair_br.groupby("developer")["pair_baserate"].std().rename("baserate_dir_std")
dev_disp = dev_disp.merge(br_disp, on="developer", how="left")

h4 = dev_disp.groupby("grp").agg(
    n_dev=("developer", "size"),
    n_dir_mean=("n_dir", "mean"),
    n_dir_median=("n_dir", "median"),
    baserate_dir_std_mean=("baserate_dir_std", "mean"),
).reset_index()
print(h4.to_string(index=False))
a4_nd = dev_disp.loc[dev_disp.grp == "A4", "n_dir"]
non_nd = dev_disp.loc[dev_disp.grp == "non-A4", "n_dir"]
u4, p4 = stats.mannwhitneyu(a4_nd, non_nd, alternative="greater")
print(f"\nn_dir A4 median={a4_nd.median():.0f} vs nonA4 median={non_nd.median():.0f} "
      f"MWU p(A4>)={p4:.2e}")

# H4 をスコア面で: A4 ペアでは「全体活動 (total_reviews)」と「当該 dir での将来 label」の
# 相関が非A4より弱いか (脱文脈性 = 全体シグナルが dir 行動を説明しない)
print("\n[脱文脈性] dev 全体活動 (total_reviews,t0) と pair 将来継続 (pair_baserate) の相関:")
brf = pt.merge(fv0.rename(columns={"__email": "developer"})[["developer", "total_reviews",
              "recent_activity_frequency"]], on="developer", how="left")
for g in ["A4", "non-A4"]:
    sub = brf[brf.grp == g].dropna(subset=["total_reviews", "pair_baserate"])
    r1, p1 = stats.spearmanr(sub["total_reviews"], sub["pair_baserate"])
    r2, p2 = stats.spearmanr(sub["recent_activity_frequency"], sub["pair_baserate"])
    print(f"  {g:7s}: rho(total_reviews,br)={r1:+.3f}(p={p1:.2g})  "
          f"rho(recent_freq,br)={r2:+.3f}(p={p2:.2g})  n={len(sub)}")

# ============================================================================
# 図 fig_a4_mechanism.pdf  (3 パネル)
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 左: A4 vs 非A4 の base_rate 分布 + 揺らぎ
ax = axes[0]
br_a4 = pt.loc[pt.grp == "A4", "pair_baserate"]
br_non = pt.loc[pt.grp == "non-A4", "pair_baserate"]
bins = np.linspace(0, 1, 6)
ax.hist([br_non, br_a4], bins=bins, label=["非A4", "A4"],
        color=["#7fa8d0", "#d9534f"], density=True, rwidth=0.85)
ax.set_xlabel("ペア base_rate (4区間の継続率)")
ax.set_ylabel("密度")
ax.set_title(f"H1 ラベル曖昧性\nA4 揺らぎ={a4_std.mean():.3f} > 非A4={non_std.mean():.3f}\n"
             f"(中間base_rate割合 A4={amb_share['A4']:.2f} 非A4={amb_share['non-A4']:.2f})")
ax.legend()

# 中: A4×pair archetype セル別 AUC
ax = axes[1]
cells = h3[h3.cell.str.startswith("A4×")].copy()
cells = cells.set_index("pair_arch").reindex(["P1", "P2", "P3", "P4"]).dropna(how="all")
x = np.arange(len(cells))
w = 0.25
ax.bar(x - w, cells["irl_auc"], w, label="IRL", color="#4c72b0")
ax.bar(x, cells["rf_auc"], w, label="RF", color="#dd8452")
ax.bar(x + w, cells["ens_auc"], w, label="Ens", color="#55a868")
for i, (_, r) in enumerate(cells.iterrows()):
    ax.text(i, 0.02, f"br={r['base_rate']:.2f}\nn={int(r['n_obs'])}",
            ha="center", va="bottom", fontsize=8)
ax.axhline(0.5, ls="--", c="gray", lw=1)
ax.set_xticks(x)
ax.set_xticklabels([f"A4×{p}" for p in cells.index])
ax.set_ylim(0, 1)
ax.set_ylabel("AUC")
ax.set_title("H3 A4ペアを pair archetype で分解\nセル別 AUC")
ax.legend(loc="upper right", fontsize=8)

# 右: 識別力 (特徴量単変量AUC) + 関与dir数
ax = axes[2]
top_disc = disc.dropna().sort_values("non-A4", ascending=True).tail(8)
y = np.arange(len(top_disc))
ax.barh(y - 0.2, top_disc["A4"], 0.4, label="A4", color="#d9534f")
ax.barh(y + 0.2, top_disc["non-A4"], 0.4, label="非A4", color="#7fa8d0")
ax.axvline(0.5, ls="--", c="gray", lw=1)
ax.set_yticks(y)
ax.set_yticklabels(top_disc["feature"], fontsize=8)
ax.set_xlim(0.45, max(0.75, top_disc["non-A4"].max() + 0.02))
ax.set_xlabel("単変量 AUC (label識別力, max(a,1-a))")
ax.set_title(f"H2 特徴量識別力\n平均 A4={disc['A4'].mean():.3f} < 非A4={disc['non-A4'].mean():.3f}\n"
             f"(H4 関与dir中央値 A4={a4_nd.median():.0f} vs 非A4={non_nd.median():.0f})")
ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
fig.savefig(f"{RES}/fig_a4_mechanism.pdf", format="pdf", bbox_inches="tight")
print(f"\n[saved] {RES}/fig_a4_mechanism.pdf")

# ----------------------------------------------------------------------------
# CSV 出力
# ----------------------------------------------------------------------------
h1.to_csv(f"{RES}/a4_mech_h1_ambiguity.csv", index=False)
disc.to_csv(f"{RES}/a4_mech_h2_discrimination.csv", index=False)
h3.to_csv(f"{RES}/a4_mech_h3_cell_auc.csv", index=False)
h4.to_csv(f"{RES}/a4_mech_h4_dispersion.csv", index=False)

summary = {
    "H1_labelstd_A4": float(a4_std.mean()), "H1_labelstd_nonA4": float(non_std.mean()),
    "H1_labelstd_p": float(p_std),
    "H1_ambig_share_A4": float(amb_share["A4"]), "H1_ambig_share_nonA4": float(amb_share["non-A4"]),
    "H1_baserate_A4": float(pt.loc[pt.isA4, "pair_baserate"].mean()),
    "H1_baserate_nonA4": float(pt.loc[~pt.isA4, "pair_baserate"].mean()),
    "H2_disc_A4": float(disc["A4"].mean()), "H2_disc_nonA4": float(disc["non-A4"].mean()),
    "H4_ndir_A4_median": float(a4_nd.median()), "H4_ndir_nonA4_median": float(non_nd.median()),
    "H4_ndir_p": float(p4),
}
with open(f"{RES}/a4_mech_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"[saved] {RES}/a4_mech_summary.json")
print(json.dumps(summary, ensure_ascii=False, indent=2))
