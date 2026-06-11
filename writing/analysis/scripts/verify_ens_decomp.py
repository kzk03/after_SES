"""Independent verification of ensemble_decomposition facet."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 777
DATA = "/Users/kazuki-h/Master_research/writing/analysis/data"
HOR = ["0-3m", "3-6m", "6-9m", "9-12m"]
NB = 2000


def safe_auc(y, s):
    y = np.asarray(y); s = np.asarray(s)
    if len(y) < 10 or y.sum() == 0 or y.sum() == len(y):
        return np.nan
    return roc_auc_score(y, s)


def boot_diff(y, se, sb, nb=NB, seed=SEED):
    y = np.asarray(y); se = np.asarray(se); sb = np.asarray(sb)
    if len(y) < 10 or y.sum() == 0 or y.sum() == len(y):
        return (np.nan,) * 4
    rng = np.random.default_rng(seed)
    n = len(y); diffs = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        diffs.append(roc_auc_score(yb, se[idx]) - roc_auc_score(yb, sb[idx]))
    if len(diffs) < 50:
        return (np.nan,) * 4
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


t = pd.read_csv(f"{DATA}/pair_trajectories.csv").rename(columns={"archetype": "dev_arch"})
pa = pd.read_csv(f"{DATA}/pair_archetype.csv")
t = t.merge(pa[["developer", "directory", "dominant"]].rename(columns={"dominant": "pair_arch"}),
            on=["developer", "directory"], how="left")
print("n rows:", len(t), "n devs:", t.developer.nunique())

# long
rows = []
for h in HOR:
    s = t[["developer", "directory", "dev_arch", "pair_arch"]].copy()
    s["horizon"] = h
    s["label"] = t[f"label_{h}"].values
    s["irl"] = t[f"irl_{h}"].values
    s["rf"] = t[f"rf_{h}"].values
    s["ens"] = t[f"avg_{h}"].values
    rows.append(s)
long = pd.concat(rows, ignore_index=True)

# verify ens = (irl+rf)/2
chk = np.abs(long["ens"] - (long["irl"] + long["rf"]) / 2).max()
print("max |ens-(irl+rf)/2| =", chk)

# base rates per horizon
print("\n=== base rates ===")
for h in HOR:
    g = long[long.horizon == h]
    print(h, "base_rate=%.3f n=%d" % (g.label.mean(), len(g)))

# scale check
print("\nirl median=%.4f rf median=%.4f" % (long.irl.median(), long.rf.median()))

# === overall lift ===
print("\n=== OVERALL Ens-RF lift ===")
for h in HOR + ["全区間"]:
    g = long if h == "全区間" else long[long.horizon == h]
    y = g.label.values
    a_irl = safe_auc(y, g.irl.values); a_rf = safe_auc(y, g.rf.values); a_ens = safe_auc(y, g.ens.values)
    m, lo, hi, p = boot_diff(y, g.ens.values, g.rf.values)
    print("%-6s n=%d AUC irl=%.4f rf=%.4f ens=%.4f | Ens-RF=%+.4f CI[%.4f,%.4f] p>0=%.3f | Ens-IRL=%+.4f" %
          (h, len(g), a_irl, a_rf, a_ens, a_ens - a_rf, lo, hi, p, a_ens - a_irl))

# === pair_arch ===
print("\n=== pair_arch lift (vs RF) ===")
for k, g in long.groupby("pair_arch", dropna=False):
    y = g.label.values
    a_irl = safe_auc(y, g.irl.values); a_rf = safe_auc(y, g.rf.values); a_ens = safe_auc(y, g.ens.values)
    if np.isnan(a_ens) or np.isnan(a_rf):
        print(k, "skip"); continue
    m, lo, hi, p = boot_diff(y, g.ens.values, g.rf.values)
    print("%-4s n=%d base=%.3f auc_rf=%.4f auc_ens=%.4f Ens-RF=%+.4f CI[%.4f,%.4f] p>0=%.3f" %
          (str(k), len(g), y.mean(), a_rf, a_ens, a_ens - a_rf, lo, hi, p))

# === dev_arch ===
print("\n=== dev_arch lift (vs RF) ===")
for k, g in long.groupby("dev_arch", dropna=False):
    y = g.label.values
    a_irl = safe_auc(y, g.irl.values); a_rf = safe_auc(y, g.rf.values); a_ens = safe_auc(y, g.ens.values)
    if np.isnan(a_ens) or np.isnan(a_rf):
        print(k, "skip"); continue
    m, lo, hi, p = boot_diff(y, g.ens.values, g.rf.values)
    print("%-6s n=%d base=%.3f auc_rf=%.4f auc_ens=%.4f Ens-RF=%+.4f CI[%.4f,%.4f] p>0=%.3f" %
          (str(k), len(g), y.mean(), a_rf, a_ens, a_ens - a_rf, lo, hi, p))

# === top robust cells dev x pair x horizon, n>=150 ===
print("\n=== top/bottom cells dev x pair x horizon (n>=150) ===")
cells = []
for k, g in long.groupby(["dev_arch", "pair_arch", "horizon"], dropna=False):
    if len(g) < 150:
        continue
    y = g.label.values
    a_rf = safe_auc(y, g.rf.values); a_ens = safe_auc(y, g.ens.values)
    if np.isnan(a_ens) or np.isnan(a_rf):
        continue
    cells.append((k[0], k[1], k[2], len(g), y.mean(), a_rf, a_ens, a_ens - a_rf))
cells = pd.DataFrame(cells, columns=["dev", "pair", "hor", "n", "base", "auc_rf", "auc_ens", "lift"])
print("TOP:")
print(cells.sort_values("lift", ascending=False).head(6).to_string(index=False))
print("BOTTOM:")
print(cells.sort_values("lift").head(6).to_string(index=False))

# === A3 x P1 0-3m specific ===
g = long[(long.dev_arch == "A3") & (long.pair_arch == "A1") & (long.horizon == "0-3m")]
if len(g):
    y = g.label.values
    print("\nA3xP1 0-3m: n=%d base=%.3f auc_rf=%.4f auc_ens=%.4f lift=%+.4f" %
          (len(g), y.mean(), safe_auc(y, g.rf.values), safe_auc(y, g.ens.values),
           safe_auc(y, g.ens.values) - safe_auc(y, g.rf.values)))
