"""Adversarial verification of domain_company facet. Does NOT overwrite any existing output."""
import numpy as np
import pandas as pd
from scipy import stats

MASTER = "/Users/kazuki-h/Master_research/writing/analysis/results/directory_master.csv"


def partial_spearman(df, x_col, y_col, ctrl_cols):
    cols = [x_col, y_col] + ctrl_cols
    sub = df[cols].dropna()
    if len(sub) < (len(ctrl_cols) + 5):
        return np.nan, np.nan, len(sub)
    R = sub.rank()
    Z = np.column_stack([np.ones(len(R))] + [R[c].values for c in ctrl_cols])

    def resid(yvec):
        beta, *_ = np.linalg.lstsq(Z, yvec, rcond=None)
        return yvec - Z @ beta

    rx = resid(R[x_col].values)
    ry = resid(R[y_col].values)
    n = len(sub)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan, np.nan, n
    r = np.corrcoef(rx, ry)[0, 1]
    k = len(ctrl_cols)
    dof = n - 2 - k
    if dof <= 0:
        return r, np.nan, n
    t = r * np.sqrt(dof / max(1e-12, (1 - r * r)))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, n


def partial_pearson_mixed(df, x_col, y_col, ctrl_cols):
    """Partial correlation where x,y are rank-transformed but project dummies kept as 0/1."""
    cols = [x_col, y_col] + ctrl_cols
    sub = df[cols].dropna().copy()
    n = len(sub)
    if n < (len(ctrl_cols) + 5):
        return np.nan, np.nan, n
    # rank-transform x,y and continuous controls; keep dummies as-is
    Zcols = []
    for c in ctrl_cols:
        v = sub[c].values
        if set(np.unique(v)) <= {0.0, 1.0}:
            Zcols.append(v.astype(float))
        else:
            Zcols.append(stats.rankdata(v))
    Z = np.column_stack([np.ones(n)] + Zcols)
    rx_in = stats.rankdata(sub[x_col].values)
    ry_in = stats.rankdata(sub[y_col].values)

    def resid(yvec):
        beta, *_ = np.linalg.lstsq(Z, yvec, rcond=None)
        return yvec - Z @ beta

    rx = resid(rx_in)
    ry = resid(ry_in)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return np.nan, np.nan, n
    r = np.corrcoef(rx, ry)[0, 1]
    k = len(ctrl_cols)
    dof = n - 2 - k
    t = r * np.sqrt(dof / max(1e-12, (1 - r * r)))
    p = 2 * stats.t.sf(abs(t), dof)
    return r, p, n


df = pd.read_csv(MASTER)
df["log_npair"] = np.log10(df["n_pair"])
reliable = df[df["reliable"] == True].copy()  # noqa: E712
aucdef = df[df["auc_defined"] == True].copy()  # noqa: E712

print("N: full=%d reliable=%d aucdef=%d" % (len(df), len(reliable), len(aucdef)))

# project distribution
print("\n=== top_dir counts (aucdef286) ===")
print(aucdef["top_dir"].value_counts().head(12).to_string())
print("\n=== top_dir counts (reliable46) ===")
print(reliable["top_dir"].value_counts().to_string())

# Build project dummies (top projects)
for grp_name, grp in [("aucdef286", aucdef), ("reliable46", reliable)]:
    for proj in ["cinder", "neutron", "nova"]:
        grp["is_%s" % proj] = (grp["top_dir"] == proj).astype(float)

print("\n========== KEY CLAIM: corporate_share -> delta_auc ==========")
for grp_name, grp in [("aucdef286", aucdef), ("reliable46", reliable)]:
    x = grp["corporate_share"]
    y = grp["delta_auc"]
    m = (~x.isna()) & (~y.isna())
    rho, p = stats.spearmanr(x[m], y[m])
    pr_sb, pp_sb, n_sb = partial_spearman(grp, "corporate_share", "delta_auc", ["log_npair", "base_rate"])
    pr_pj, pp_pj, n_pj = partial_pearson_mixed(
        grp, "corporate_share", "delta_auc",
        ["log_npair", "base_rate", "is_cinder", "is_neutron", "is_nova"])
    print(f"[{grp_name}] n={m.sum()}")
    print(f"  raw Spearman           rho={rho:+.3f} p={p:.4f}")
    print(f"  partial(size,base)     rho={pr_sb:+.3f} p={pp_sb:.4f} n={n_sb}")
    print(f"  partial(+3 proj dummy) rho={pr_pj:+.3f} p={pp_pj:.4f} n={n_pj}")

print("\n========== domain_hhi -> ens_auc (claimed to collapse) ==========")
for grp_name, grp in [("aucdef286", aucdef), ("reliable46", reliable)]:
    x = grp["domain_hhi"]
    y = grp["ens_auc"]
    m = (~x.isna()) & (~y.isna())
    rho, p = stats.spearmanr(x[m], y[m])
    pr_sb, pp_sb, _ = partial_spearman(grp, "domain_hhi", "ens_auc", ["log_npair", "base_rate"])
    pr_pj, pp_pj, _ = partial_pearson_mixed(
        grp, "domain_hhi", "ens_auc",
        ["log_npair", "base_rate", "is_cinder", "is_neutron"])
    print(f"[{grp_name}] n={m.sum()}: raw rho={rho:+.3f}(p={p:.4f}) | "
          f"partial(size,base) rho={pr_sb:+.3f}(p={pp_sb:.4f}) | "
          f"+proj rho={pr_pj:+.3f}(p={pp_pj:.4f})")

print("\n========== top_corp_share_of_all -> delta_auc (claimed weak) ==========")
for grp_name, grp in [("aucdef286", aucdef), ("reliable46", reliable)]:
    pr_sb, pp_sb, _ = partial_spearman(grp, "top_corp_share_of_all", "delta_auc", ["log_npair", "base_rate"])
    x = grp["top_corp_share_of_all"]; y = grp["delta_auc"]
    m = (~x.isna()) & (~y.isna())
    rho, p = stats.spearmanr(x[m], y[m])
    print(f"[{grp_name}] n={m.sum()}: raw rho={rho:+.3f}(p={p:.4f}) | partial(size,base) rho={pr_sb:+.3f}(p={pp_sb:.4f})")

print("\n========== corporate_share -> ens_lift_auc ==========")
for grp_name, grp in [("aucdef286", aucdef), ("reliable46", reliable)]:
    pr_sb, pp_sb, _ = partial_spearman(grp, "corporate_share", "ens_lift_auc", ["log_npair", "base_rate"])
    pr_pj, pp_pj, _ = partial_pearson_mixed(
        grp, "corporate_share", "ens_lift_auc",
        ["log_npair", "base_rate", "is_cinder", "is_neutron", "is_nova"])
    x = grp["corporate_share"]; y = grp["ens_lift_auc"]
    m = (~x.isna()) & (~y.isna())
    rho, p = stats.spearmanr(x[m], y[m])
    print(f"[{grp_name}] n={m.sum()}: raw rho={rho:+.3f}(p={p:.4f}) | partial(size,base) rho={pr_sb:+.3f}(p={pp_sb:.4f}) | +proj rho={pr_pj:+.3f}(p={pp_pj:.4f})")

print("\n========== n_domains -> ens_auc (claimed = size proxy) ==========")
for grp_name, grp in [("aucdef286", aucdef), ("reliable46", reliable)]:
    pr_sb, pp_sb, _ = partial_spearman(grp, "n_domains", "ens_auc", ["log_npair", "base_rate"])
    x = grp["n_domains"]; y = grp["log_npair"]
    rho_size, _ = stats.spearmanr(x, y)
    print(f"[{grp_name}]: n_domains vs log_npair rho={rho_size:+.3f} | n_domains->ens_auc partial(size,base) rho={pr_sb:+.3f}(p={pp_sb:.4f})")

# ---- Robustness: does corporate_share->delta_auc survive dropping cinder? (caveat 2) ----
print("\n========== ROBUSTNESS: corporate_share->delta_auc EXCLUDING cinder ==========")
for grp_name, grp in [("aucdef286", aucdef), ("reliable46", reliable)]:
    g = grp[grp["top_dir"] != "cinder"].copy()
    x = g["corporate_share"]; y = g["delta_auc"]
    m = (~x.isna()) & (~y.isna())
    if m.sum() >= 10:
        rho, p = stats.spearmanr(x[m], y[m])
        pr, pp, _ = partial_spearman(g, "corporate_share", "delta_auc", ["log_npair", "base_rate"])
        print(f"[{grp_name} no-cinder] n={m.sum()}: raw rho={rho:+.3f}(p={p:.4f}) | partial(size,base) rho={pr:+.3f}(p={pp:.4f})")

# Also: per-project within-project correlation (Fisher combine intuition)
print("\n========== WITHIN-PROJECT corporate_share->delta_auc (aucdef, projects with n>=15) ==========")
for proj, g in aucdef.groupby("top_dir"):
    x = g["corporate_share"]; y = g["delta_auc"]
    m = (~x.isna()) & (~y.isna())
    if m.sum() >= 15:
        rho, p = stats.spearmanr(x[m], y[m])
        print(f"  {proj:20s} n={m.sum():3d}: rho={rho:+.3f} p={p:.4f}  corp_share[median]={x.median():.3f}")

# ---- outlier check: top_corp_domain values ----
print("\n========== top_corp_domain freq (reliable46) & corporate_share dist ==========")
print(reliable["top_corp_domain"].value_counts().head(10).to_string())
print("\ncorporate_share quartiles aucdef:", aucdef["corporate_share"].quantile([0,.25,.5,.75,1]).round(3).tolist())

# Leverage: recompute corporate_share->delta_auc partial dropping the 5 most extreme corporate_share dirs
print("\n========== LEVERAGE: drop 5 highest & 5 lowest corporate_share (aucdef286) ==========")
g = aucdef.dropna(subset=["corporate_share", "delta_auc"]).copy()
g2 = g.sort_values("corporate_share").iloc[5:-5]
pr, pp, n = partial_spearman(g2, "corporate_share", "delta_auc", ["log_npair", "base_rate"])
print(f"  after trimming 10 dirs: n={n} partial(size,base) rho={pr:+.3f}(p={pp:.4f})")
