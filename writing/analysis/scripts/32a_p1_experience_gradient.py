"""
32a: P1機序の深掘り — 経験勾配による自然実験的 ablation
==========================================================

真の path-ablation(再学習)は path 特徴が analysis に無く不可。代わりに
ペアarchetype P1(未経験) → P2(経験低承諾) → P4(中程度) → P3(経験高承諾) を
「当該 dir 経験(=path 信号)の強度勾配」として扱い、RF/IRL の挙動が
経験度でどう変わるかを観察する(自然実験的 ablation)。

中心仮説:
  RF の AUC と予測 spread(std) は経験度が上がるほど上昇し(P1<...<P3)、
  IRL はより平坦(経験に頑健)。これが示せれば
  「RF は当該 dir 経験(path)に依存、IRL は開発者状態系列で経験非依存に識別」を実証。
  delta_auc(IRL-RF) が低経験ほど正(IRL優位)に振れるか。

出力:
  results/p1_experience_gradient_pooled.csv   (archetype別 pooled 指標)
  results/p1_experience_gradient_stateonly.csv (補助: state-only RF vs deployed RF)
  results/fig_p1_experience_gradient.pdf       (3パネル図)
"""
import japanize_matplotlib  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

SEED = 777
DATA = "data"
RES = "results"
HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]

# ペアarchetype の P 対応(タスク規約)と「経験勾配」順序
#   A1→P1 未経験(path≈0) / A2→P2 経験低承諾 / A3→P3 経験高承諾 / A4→P4 中程度
#   経験度勾配(path信号の強度): P1(最弱) < P2 < P4 < P3(最強)
A2P = {"A1": "P1", "A2": "P2", "A3": "P3", "A4": "P4"}
P_LABEL = {
    "P1": "P1\n未経験",
    "P2": "P2\n経験低承諾",
    "P4": "P4\n中程度",
    "P3": "P3\n経験高承諾",
}
GRADIENT = ["P1", "P2", "P4", "P3"]  # 経験弱→強


def safe_auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, s)


def safe_ap(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    if len(np.unique(y)) < 2:
        return np.nan
    return average_precision_score(y, s)


# ---------------------------------------------------------------------------
# 1. ペアarchetype を結合し、4区間を pooled する long テーブルを作る
# ---------------------------------------------------------------------------
pt = pd.read_csv(f"{DATA}/pair_trajectories.csv")
pa = pd.read_csv(f"{DATA}/pair_archetype.csv")

# ペアarchetype の authoritative source は pair_archetype.dominant
m = pt.merge(
    pa[["developer", "directory", "dominant"]], on=["developer", "directory"], how="inner"
)
m["P"] = m["dominant"].map(A2P)
m = m.dropna(subset=["P"])
print(f"[join] pair_trajectories={len(pt)} pair_archetype={len(pa)} inner+mapped={len(m)}")
print("[join] ペアarchetype 分布(pair単位):")
print(m["P"].value_counts())

# long 化: 各 (pair, horizon) を 1 行に。observation 単位で pooled する
recs = []
for h in HORIZONS:
    sub = m[["developer", "directory", "P", f"label_{h}", f"irl_{h}", f"rf_{h}", f"avg_{h}"]].copy()
    sub.columns = ["developer", "directory", "P", "label", "irl", "rf", "avg"]
    sub["horizon"] = h
    recs.append(sub)
long = pd.concat(recs, ignore_index=True)
long = long.dropna(subset=["label", "irl", "rf", "avg"])
print(f"[long] pooled observations (pair x horizon) = {len(long)}")

# ---------------------------------------------------------------------------
# 2. archetype別 pooled 指標
# ---------------------------------------------------------------------------
rows = []
for P in GRADIENT:
    g = long[long["P"] == P]
    y = g["label"].values
    rows.append(
        {
            "P": P,
            "exp_rank": GRADIENT.index(P) + 1,  # 1=未経験 ... 4=経験豊富
            "n_obs": len(g),
            "n_pair": m[m["P"] == P].shape[0],
            "base_rate": y.mean(),
            "n_pos": int(y.sum()),
            "irl_auc": safe_auc(y, g["irl"]),
            "rf_auc": safe_auc(y, g["rf"]),
            "ens_auc": safe_auc(y, g["avg"]),
            "irl_ap": safe_ap(y, g["irl"]),
            "rf_ap": safe_ap(y, g["rf"]),
            "ens_ap": safe_ap(y, g["avg"]),
            "delta_auc": safe_auc(y, g["irl"]) - safe_auc(y, g["rf"]),
            "irl_std": g["irl"].std(),
            "rf_std": g["rf"].std(),
            "ens_std": g["avg"].std(),
        }
    )
pooled = pd.DataFrame(rows)
pooled.to_csv(f"{RES}/p1_experience_gradient_pooled.csv", index=False)
print("\n[pooled] archetype別 指標(経験勾配順 P1->P2->P4->P3):")
print(
    pooled[
        ["P", "n_obs", "base_rate", "irl_auc", "rf_auc", "ens_auc", "delta_auc", "irl_std", "rf_std"]
    ].to_string(index=False)
)

# 単調性(経験 rank との Spearman): RF/IRL の AUC と std が経験度でどう動くか
from scipy.stats import spearmanr

def trend(col):
    rho, p = spearmanr(pooled["exp_rank"], pooled[col])
    return rho, p

print("\n[trend] 経験 rank(P1<P2<P4<P3) との Spearman:")
for col in ["rf_auc", "irl_auc", "rf_std", "irl_std", "delta_auc"]:
    rho, p = trend(col)
    print(f"  {col:10s}: rho={rho:+.3f} p={p:.3f}")

# ---------------------------------------------------------------------------
# 3. 補助: state-only RF (25特徴, 再学習) vs deployed RF (32特徴) を archetype別に比較
#    交絡: t0='2022-01-01' の state25 を全 horizon で共有(時間窓を厳密に守れない簡易版)。
#    あくまで「path 特徴の寄与が経験ありpairで大きいか」の傍証。
# ---------------------------------------------------------------------------
print("\n[state-only RF] 再学習中 ...")
fv = pd.read_csv(f"{DATA}/feature_values.csv")
fv_t0 = fv[fv["__snapshot"] == "2022-01-01"].copy()
feat_cols = [c for c in fv.columns if not c.startswith("__")]
print(f"  state特徴 = {len(feat_cols)} (期待25), t0 snapshot rows={len(fv_t0)}")

# developer(email) -> state25 (t0)。pair の developer は email。
fv_t0 = fv_t0.drop_duplicates("__email").set_index("__email")[feat_cols]

state_rows = []
for h in HORIZONS:
    # この horizon の pair に state25 を結合(developer 単位の特徴)
    sub = m[["developer", "directory", "P", f"label_{h}", f"rf_{h}", f"irl_{h}"]].copy()
    sub.columns = ["developer", "directory", "P", "label", "rf", "irl"]
    sub = sub.dropna(subset=["label"])
    X = sub["developer"].map(lambda e: fv_t0.loc[e].values if e in fv_t0.index else None)
    has = X.notna()
    sub = sub[has].reset_index(drop=True)
    Xmat = np.vstack(X[has].values)
    y = sub["label"].values.astype(int)
    if len(np.unique(y)) < 2:
        continue
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, random_state=SEED, n_jobs=-1,
        oob_score=True, bootstrap=True,
    )
    clf.fit(Xmat, y)
    # OOB 予測で held-out 評価(in-sample のリークを避ける)。
    # 各 horizon を独立に学習。t0 state25 を全 horizon で共有する点が交絡(明記)。
    oob = clf.oob_decision_function_
    so_prob = oob[:, 1] if oob.shape[1] == 2 else oob[:, 0]
    # OOB が NaN の行(bootstrap で常に選ばれた稀ケース)は in-bag 確率で補完
    nanmask = np.isnan(so_prob)
    if nanmask.any():
        ib = clf.predict_proba(Xmat[nanmask])[:, 1]
        so_prob = so_prob.copy()
        so_prob[nanmask] = ib
    sub["state_only_prob"] = so_prob
    for P in GRADIENT:
        gp = sub[sub["P"] == P]
        yy = gp["label"].values
        state_rows.append(
            {
                "horizon": h,
                "P": P,
                "n": len(gp),
                "base_rate": yy.mean() if len(gp) else np.nan,
                "state_only_rf_auc": safe_auc(yy, gp["state_only_prob"]),
                "deployed_rf_auc": safe_auc(yy, gp["rf"]),
                "irl_auc": safe_auc(yy, gp["irl"]),
            }
        )

state_df = pd.DataFrame(state_rows)
# horizon プールではなく archetype別に平均(両クラス揃う horizon のみ)
state_agg = (
    state_df.dropna(subset=["state_only_rf_auc", "deployed_rf_auc"])
    .groupby("P")
    .agg(
        n=("n", "sum"),
        state_only_rf_auc=("state_only_rf_auc", "mean"),
        deployed_rf_auc=("deployed_rf_auc", "mean"),
        irl_auc=("irl_auc", "mean"),
    )
    .reindex(GRADIENT)
    .reset_index()
)
state_agg["path_gain"] = state_agg["deployed_rf_auc"] - state_agg["state_only_rf_auc"]
state_agg.to_csv(f"{RES}/p1_experience_gradient_stateonly.csv", index=False)
print("\n[state-only] deployed(32特徴) - state_only(25特徴, OOB) RF AUC = path特徴の寄与(傍証):")
print(state_agg.to_string(index=False))
print("  (交絡注意: t0 state25 を全 horizon 共有。OOB 評価でリークは回避したが厳密な ablation ではない)")

# ---------------------------------------------------------------------------
# 3b. std の base_rate 統制ロバストネス:
#     予測 std は base_rate(分散 = p(1-p) が p=0.5 で最大)に機械的に依存する。
#     経験勾配での std 上昇が base_rate 差の副産物でないか確認するため、
#     P1(base≈0.24) と P4(base≈0.21) という「base_rate がほぼ同じ低承諾ペア」で
#     RF/IRL std を比較する(経験は P1<P4)。両者で RF std が上がるなら base_rate 非依存。
# ---------------------------------------------------------------------------
lowbr = long[long["P"].isin(["P1", "P4"])]
print("\n[std robustness] base_rate がほぼ同じ低承諾ペア P1 vs P4 (経験 P1<P4):")
for P in ["P1", "P4"]:
    g = lowbr[lowbr["P"] == P]
    print(
        f"  {P}: base_rate={g['label'].mean():.3f} rf_std={g['rf'].std():.3f} "
        f"irl_std={g['irl'].std():.3f} rf_auc={safe_auc(g['label'], g['rf']):.3f} "
        f"irl_auc={safe_auc(g['label'], g['irl']):.3f}"
    )

# ---------------------------------------------------------------------------
# 4. 図 fig_p1_experience_gradient.pdf  (3パネル)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
x = np.arange(len(GRADIENT))
xlabels = [P_LABEL[p] for p in GRADIENT]

# (a) archetype別 IRL/RF/Ens AUC (経験勾配順)
ax = axes[0]
ax.plot(x, pooled["irl_auc"], "o-", color="#d62728", lw=2, ms=8, label="IRL")
ax.plot(x, pooled["rf_auc"], "s-", color="#1f77b4", lw=2, ms=8, label="RF")
ax.plot(x, pooled["ens_auc"], "^--", color="#2ca02c", lw=1.6, ms=7, label="Ensemble")
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_ylabel("AUC (pooled, 4区間)")
ax.set_xlabel("経験勾配(弱 → 強)")
ax.set_title("(a) ペアarchetype別 AUC")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

# (b) archetype別 RF/IRL prob std
ax = axes[1]
w = 0.38
ax.bar(x - w / 2, pooled["irl_std"], w, color="#d62728", alpha=0.85, label="IRL std")
ax.bar(x + w / 2, pooled["rf_std"], w, color="#1f77b4", alpha=0.85, label="RF std")
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_ylabel("予測確率の標準偏差 (spread)")
ax.set_xlabel("経験勾配(弱 → 強)")
ax.set_title("(b) ペアarchetype別 予測 spread")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, axis="y")

# (c) delta_auc by archetype (IRL - RF)
ax = axes[2]
colors = ["#d62728" if v >= 0 else "#1f77b4" for v in pooled["delta_auc"]]
ax.bar(x, pooled["delta_auc"], 0.6, color=colors, alpha=0.85)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(xlabels)
ax.set_ylabel("ΔAUC = IRL − RF")
ax.set_xlabel("経験勾配(弱 → 強)")
ax.set_title("(c) ペアarchetype別 IRL−RF 優位差")
for xi, v in zip(x, pooled["delta_auc"]):
    ax.text(xi, v + (0.002 if v >= 0 else -0.004), f"{v:+.3f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=9)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(f"{RES}/fig_p1_experience_gradient.pdf", format="pdf", bbox_inches="tight")
print(f"\n[fig] saved {RES}/fig_p1_experience_gradient.pdf")
