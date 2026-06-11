"""Part 2: disagreement, rescue, scale caveat."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SEED = 777
DATA = "/Users/kazuki-h/Master_research/writing/analysis/data"
HOR = ["0-3m", "3-6m", "6-9m", "9-12m"]


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(y) < 10 or y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, s)


t = pd.read_csv(f"{DATA}/pair_trajectories.csv").rename(columns={"archetype": "dev_arch"})
pa = pd.read_csv(f"{DATA}/pair_archetype.csv")
t = t.merge(pa[["developer", "directory", "dominant"]].rename(columns={"dominant": "pair_arch"}),
            on=["developer", "directory"], how="left")
rows = []
for h in HOR:
    s = t[["developer", "directory", "dev_arch", "pair_arch"]].copy()
    s["horizon"] = h
    s["label"] = t[f"label_{h}"].values
    s["irl"] = t[f"irl_{h}"].values; s["rf"] = t[f"rf_{h}"].values; s["ens"] = t[f"avg_{h}"].values
    rows.append(s)
long = pd.concat(rows, ignore_index=True)

# === disagreement cells ===
dis = []
for k, g in long.groupby(["dev_arch", "pair_arch", "horizon"], dropna=False):
    y = g.label.values; irl = g.irl.values; rf = g.rf.values
    if len(g) < 20 or y.sum() < 3 or y.sum() > len(y) - 3:
        continue
    rho, _ = spearmanr(irl, rf)
    if np.isnan(rho):
        continue
    ip = (irl > np.median(irl)).astype(int); rp = (rf > np.median(rf)).astype(int)
    ic = ip == y; rc = rp == y
    io = (ic & ~rc).mean(); ro = (rc & ~ic).mean()
    bal = 1 - abs(io - ro) / (io + ro + 1e-9)
    comp = (io + ro) * bal
    a_rf = safe_auc(y, rf); a_ens = safe_auc(y, g.ens.values)
    dis.append(dict(dev=k[0], pair=k[1], hor=k[2], n=len(g), base=y.mean(),
                    dis=1 - abs(rho), comp=comp, lift=a_ens - a_rf))
dis = pd.DataFrame(dis)
print("n disagreement cells:", len(dis))
print("Spearman(dis, lift) =", spearmanr(dis.dis, dis.lift)[0])
print("Spearman(comp, lift) =", spearmanr(dis.comp, dis.lift)[0])
# also size-weighted / partial-control sanity: control for base_rate
from numpy.polynomial import polynomial as P
# partial spearman of dis vs lift controlling base via rank residuals
def partial_spearman(x, y, z):
    rx = pd.Series(x).rank(); ry = pd.Series(y).rank(); rz = pd.Series(z).rank()
    # residualize
    bx = np.polyfit(rz, rx, 1); ex = rx - np.polyval(bx, rz)
    by = np.polyfit(rz, ry, 1); ey = ry - np.polyval(by, rz)
    return spearmanr(ex, ey)[0]
print("Partial Spearman(dis,lift | base) =", partial_spearman(dis.dis.values, dis.lift.values, dis.base.values))
print("Partial Spearman(comp,lift | base) =", partial_spearman(dis.comp.values, dis.lift.values, dis.base.values))

# === rescue by gap ===
print("\n=== rescue by gap (Ens rank - RF rank, pct points x100) ===")
resc = []
for h in HOR:
    g = long[long.horizon == h].copy()
    g["rank_rf"] = g.rf.rank(pct=True); g["rank_ens"] = g.ens.rank(pct=True); g["rank_irl"] = g.irl.rank(pct=True)
    g["gap"] = (g.rank_irl - g.rank_rf).abs()
    g["improve"] = g.rank_ens - g.rank_rf
    resc.append(g)
resc = pd.concat(resc, ignore_index=True)
resc["gap_q"] = pd.qcut(resc.gap, 4, labels=["低", "中低", "中高", "高"], duplicates="drop").astype(str)
summ = resc.groupby(["gap_q", "label"]).improve.mean().unstack() * 100
print(summ.reindex(["低", "中低", "中高", "高"]))

# rescue profile
rescued = resc[(resc.label == 1) & (resc.improve > 0.05)]
notr = resc[(resc.label == 1) & (resc.improve <= 0.05)]
print("\nrescued n=%d mean_gap=%.3f mean_rf_rank=%.3f" % (len(rescued), rescued.gap.mean(), rescued.rank_rf.mean()))
print("notres  n=%d mean_gap=%.3f mean_rf_rank=%.3f" % (len(notr), notr.gap.mean(), notr.rank_rf.mean()))

# === scale caveat: scale-corrected ensemble (rank-average) ===
print("\n=== scale caveat: rank-mean ensemble vs simple-mean ensemble (all-pool) ===")
for h in HOR + ["全区間"]:
    g = long if h == "全区間" else long[long.horizon == h]
    y = g.label.values
    # rank-mean within horizon (or whole pool)
    if h == "全区間":
        rmean = (long.irl.rank(pct=True) + long.rf.rank(pct=True)) / 2
        rmean = rmean.values
    else:
        gg = long[long.horizon == h]
        rmean = ((gg.irl.rank(pct=True) + gg.rf.rank(pct=True)) / 2).values
    a_rf = safe_auc(y, g.rf.values); a_ens = safe_auc(y, g.ens.values); a_rmean = safe_auc(y, rmean)
    print("%-6s simple-ens=%.4f rank-ens=%.4f rf=%.4f | simple-RF=%+.4f rank-RF=%+.4f" %
          (h, a_ens, a_rmean, a_rf, a_ens - a_rf, a_rmean - a_rf))
