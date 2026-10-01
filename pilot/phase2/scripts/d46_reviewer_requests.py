"""D46: 査読(再査読 F1・M4・M6)への追加分析。

A. 主アウトカム(log(1+作成者以外の人間レビュー件数)、d4/d43 の主仕様)に対する
   ランダム化推論(RI)とプラセボ。
   (i) 並べ替え: 実際の 17 個の導入月を、treated ∪ never-treated(31 リポジトリ)から
       無作為に選んだ 17 リポジトリにランダムに割り当て直す(処置数と導入月の経験分布を保つ)。
       観測値が並べ替え分布のどこに落ちるか(両側・片側 RI p)と、
       H0: 効果 <= log(0.9)(10% 以上の減少)の RI p を出す。後者は2通り:
         - shift(recentre): 帰無 0 の並べ替え分布を log(0.9) だけずらして観測値と比べる
         - sharp null imputation: Y(0) = Y - τ0·post_actual を作り、Y(0) + τ0·post' を
           並べ替えごとに推定し直す(Fisher の sharp null の標準形)
       係数そのものと、CR1 の t 値(スチューデント化)の両方で出す。
   (ii) 偽の導入日: 処置リポジトリの導入月を 12 ヶ月前にずらし、実際の導入前のデータだけで推定。
B. 品質アウトカム(d35 revert PR 率、d42 修正誘発率)の「害」の片側検定:
   H0: 増加 >= 基準率の 10/20/30% を、t(G-1)、z(スクリプトと同じ)、WCR(下側)で。
   片側 95% 上限を基準率に対する % で出す(WCR は検定の反転)。
C. PR あたりのレビュー: PR 作成数 log(1+PR 数) と、作成月コホートでの
   「PR 1 件あたりの作成者以外の人間レビュー件数」(およびその log1p)などを主仕様で推定。

出力: outputs_full/d46_reviewer_requests.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PILOT_PROCESSED=processed_full \
      PHASE2_OUT=outputs_full PHASE2_FIG=figures_full \
      .venv/bin/python pilot/phase2/scripts/d46_reviewer_requests.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from common import INTERIM_DIR, OUT_DIR, build_count_panel, fit_panel, load_repos, repo_dir_name
from d4_repo_event_study import build_balanced, detect_adoption_presence
from d43_causal_robustness import WEBB, Fit, add_post, add_trends, demean, es_cols

SEED = 20260928
N_PERM = int(os.environ.get("D46_NPERM", "2000"))
N_JOBS = int(os.environ.get("D46_JOBS", "6"))
B_WCR = 9999
TAU0 = float(np.log(0.9))
Y_PRIMARY = "log_n_E_human_nonauthor"
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# WCR(d43 Fit.wcr と同じ計算。片側の下側 p と検定反転のため t* の分布を返す)
# ---------------------------------------------------------------------------

def wcr_ts(f: Fit, name: str, null: float, W: np.ndarray) -> tuple[float, np.ndarray]:
    j = f.idx(name)
    X, y, cl, G = f.X, f.y, f.cl, f.G
    keep = [i for i in range(f.k) if i != j]
    yn = y - null * X[:, j]
    if keep:
        Xr = X[:, keep]
        br = np.linalg.lstsq(Xr, yn, rcond=None)[0]
        u = yn - Xr @ br
    else:
        u = yn
    aj = f.XtXi[j]
    z = X @ aj
    A = np.column_stack([np.bincount(cl, X[:, i] * u, G) for i in range(f.k)])
    zu = np.bincount(cl, z * u, G)
    ZX = np.column_stack([np.bincount(cl, z * X[:, i], G) for i in range(f.k)])
    t_obs = (f.beta[j] - null) / np.sqrt(f.V[j, j])
    AW = W @ A
    num = AW @ aj
    score = W * zu - (AW @ f.XtXi) @ ZX.T
    V = f.c * np.sum(score ** 2, axis=1)
    return float(t_obs), num / np.sqrt(V)


def wcr_summary(f: Fit, name: str, null: float, W: np.ndarray) -> dict:
    t, ts = wcr_ts(f, name, null, W)
    return {"null": null, "t": t, "p_wcr_two": float(np.mean(np.abs(ts) >= abs(t))),
            "p_wcr_upper": float(np.mean(ts >= t)),   # H1: β > null
            "p_wcr_lower": float(np.mean(ts <= t)),   # H1: β < null
            "B": int(W.shape[0])}


def wcr_upper_bound(f: Fit, name: str, W: np.ndarray, alpha: float = 0.05) -> float:
    """片側 WCR 検定(H0: β >= b, H1: β < b)を反転した片側 (1-α) 上限。
    同じ重み W を使い回すので p(b) は b についてほぼ単調。二分法で p_lower(b) = α を探す。"""
    c = f.coef(name)
    lo, hi = c["coef"], c["coef"] + 20 * c["se"]

    def p(b):
        t, ts = wcr_ts(f, name, b, W)
        return float(np.mean(ts <= t))
    if p(hi) > alpha:
        return float("nan")
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if p(mid) <= alpha:
            hi = mid
        else:
            lo = mid
    return float(hi)


# ---------------------------------------------------------------------------
# パネル
# ---------------------------------------------------------------------------

def cohort_counts() -> pd.DataFrame:
    """PR の作成月・作成モジュール(G イベント)ごとに、その PR が全期間で受けた
    作成者以外の人間レビュー件数を集計する(legacy_bot の PR は build_count_panel と同じく G から除外)。"""
    frames = []
    for repo in load_repos():
        p = INTERIM_DIR / repo_dir_name(repo) / "events.parquet"
        if not p.exists():
            continue
        ev = pd.read_parquet(p, columns=["stage", "actor_login", "actor_category", "module_id",
                                         "month", "pr_number"])
        ev = ev[ev["stage"].isin(["G", "E"]) & (ev["actor_category"] != "legacy_bot")]
        g = ev[ev["stage"] == "G"].sort_values("month").drop_duplicates("pr_number")
        e = ev[ev["stage"] == "E"]
        author_of = g.set_index("pr_number")["actor_login"]
        ehn = e[(e["actor_category"] == "human") & (e["actor_login"] != e["pr_number"].map(author_of))]
        cnt = ehn.groupby("pr_number").size().rename("rev_nonauthor")
        gg = g[["pr_number", "module_id", "month"]].join(cnt, on="pr_number")
        gg["rev_nonauthor"] = gg["rev_nonauthor"].fillna(0.0)
        gg["reviewed_nonauthor"] = (gg["rev_nonauthor"] > 0).astype(float)
        agg = gg.groupby(["module_id", "month"]).agg(
            n_prs_cohort=("pr_number", "size"),
            n_rev_nonauthor_cohort=("rev_nonauthor", "sum"),
            n_prs_reviewed_nonauthor_cohort=("reviewed_nonauthor", "sum")).reset_index()
        agg["module_id_global"] = repo + "::" + agg["module_id"]
        frames.append(agg.drop(columns="module_id"))
    return pd.concat(frames, ignore_index=True)


def build_sample():
    df = build_count_panel()
    months = sorted(df["month"].unique())
    adoption, never, excl = detect_adoption_presence(df)
    log(f"adoption: {len(adoption)} treated / {len(never)} never / {len(excl)} excluded")
    assert (len(adoption), len(never), len(excl)) == (17, 14, 7), "full data でない可能性(PILOT_* を確認)"
    bal = build_balanced(df)
    bal = bal[bal["repo"].isin(set(adoption) | set(never))].copy()
    co = cohort_counts()
    bal = bal.merge(co, on=["module_id_global", "month"], how="left")
    for c in ["n_prs_cohort", "n_rev_nonauthor_cohort", "n_prs_reviewed_nonauthor_cohort"]:
        bal[c] = bal[c].fillna(0.0)
    midx = {m: i for i, m in enumerate(months)}
    bal["g"] = bal["repo"].map({r: midx[m] for r, m in adoption.items()})
    bal["event_time"] = bal["month_idx"] - bal["g"]
    bal[Y_PRIMARY] = np.log1p(bal["n_E_human_nonauthor"])
    return df, bal, adoption, never, excl, months


# ---------------------------------------------------------------------------
# A(i) 並べ替え推論
# ---------------------------------------------------------------------------

ES_NAMES = ["k_m3", "k_m2", "k_p0", "k_p1", "k_p2", "k_p3"]


def treat_cols(month_idx: np.ndarray, grow: np.ndarray) -> np.ndarray:
    et = month_idx - grow
    ok = ~np.isnan(et)
    cols = [ok & (et <= -3), ok & (et == -2), ok & (et == 0), ok & (et == 1), ok & (et == 2),
            ok & (et >= 3), ok & (et >= 0)]
    return np.column_stack(cols).astype(float)


def ols_cr1(X: np.ndarray, Y: np.ndarray, cl: np.ndarray, G: int, j: int):
    """X: n x k(demeaned)、Y: n x 2(ỹ, p̃)。列 j の係数を両方について、
    CR1 の分散の計算に要るクラスタスコアと一緒に返す。"""
    keep = np.where(np.abs(X).max(axis=0) > 1e-10)[0]
    if j not in keep:
        return None
    X = X[:, keep]
    jj = int(np.where(keep == j)[0][0])
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    B = XtXi @ (X.T @ Y)                       # k x 2
    E = Y - X @ B                               # n x 2
    c = G / (G - 1) * (n - 1) / (n - k)
    # a_j' Σ_g (X_g' e_g) :  z = X a_j, スコア s_g = Σ_{i∈g} z_i e_i
    z = X @ XtXi[jj]
    Sy = np.bincount(cl, z * E[:, 0], G)
    Sp = np.bincount(cl, z * E[:, 1], G)
    return B[jj, 0], B[jj, 1], Sy, Sp, c


def perm_batch(gvecs: list[np.ndarray], month_idx, repo_code, a, b, cl, G, ytil, ptil, Ttil):
    Y = np.column_stack([ytil, ptil])
    out = []
    for gv in gvecs:
        grow = gv[repo_code]
        M = demean(treat_cols(month_idx, grow), a, b)
        res = {}
        for sname, X, j in [("es_k_p3", M[:, :6], 5), ("post_did", M[:, 6:7], 0),
                            ("trend_es_k_p3", np.column_stack([M[:, :6], Ttil]), 5),
                            ("trend_post_did", np.column_stack([M[:, 6:7], Ttil]), 0)]:
            r = ols_cr1(X, Y, cl, G, j)
            if r is None:
                res[sname] = None
                continue
            by, bp, Sy, Sp, c = r
            res[sname] = (by, bp, Sy, Sp, c)
        out.append(res)
    return out


def run_permutation(bal: pd.DataFrame, adoption: dict, never: list, months: list) -> dict:
    d = bal.copy()
    repos = sorted(set(adoption) | set(never))
    rcode = {r: i for i, r in enumerate(repos)}
    repo_code = d["repo"].map(rcode).to_numpy()
    a = pd.factorize(d["module_id_global"])[0]
    b = pd.factorize(d["month_idx"])[0]
    cl = repo_code.copy()
    G = len(repos)
    month_idx = d["month_idx"].to_numpy(float)
    trend_cols = add_trends(d)
    ytil = demean(d[[Y_PRIMARY]].to_numpy(float), a, b)[:, 0]
    g_actual = np.array([np.nan if r not in adoption else months.index(adoption[r]) for r in repos], float)
    ptil = demean(treat_cols(month_idx, g_actual[repo_code])[:, 6:7], a, b)[:, 0]
    Ttil = demean(d[trend_cols].to_numpy(float), a, b)
    log(f"perm: demeaned fixed parts (n={len(d)}, G={G})")

    # 観測値(d43 Fit と一致するか確認する)
    d2 = d.copy()
    cols = es_cols(d2, 3, 3)
    pc = add_post(d2)
    check = {}
    for sname, cc, nm in [("es_k_p3", cols, "k_p3"), ("post_did", pc, "post"),
                          ("trend_es_k_p3", cols + trend_cols, "k_p3"),
                          ("trend_post_did", pc + trend_cols, "post")]:
        f = Fit(d2, Y_PRIMARY, cc)
        check[sname] = f.coef(nm)
    obs_raw = perm_batch([g_actual], month_idx, repo_code, a, b, cl, G, ytil, ptil, Ttil)[0]

    # 各リポジトリで置ける導入月: 観測開始月より後、観測終了月以前(処置定義 m > first と同じ)
    span = d.groupby("repo")["month_idx"].agg(["min", "max"])
    lo = np.array([span.loc[r, "min"] for r in repos])
    hi = np.array([span.loc[r, "max"] for r in repos])
    gpool = np.array(sorted(months.index(m) for m in adoption.values()), float)
    rng = np.random.default_rng(SEED)
    gvecs, n_reject = [], 0
    while len(gvecs) < N_PERM:
        chosen = rng.choice(G, size=len(gpool), replace=False)
        gg = rng.permutation(gpool)
        if np.any(gg <= lo[chosen]) or np.any(gg > hi[chosen]):
            n_reject += 1
            continue
        v = np.full(G, np.nan)
        v[chosen] = gg
        gvecs.append(v)
    n_same_set = int(sum(set(np.where(~np.isnan(v))[0]) == set(np.where(~np.isnan(g_actual))[0]) for v in gvecs))
    log(f"perm: {N_PERM} draws (rejected {n_reject} infeasible), running with {N_JOBS} jobs")

    chunks = [gvecs[i::N_JOBS * 4] for i in range(N_JOBS * 4)]
    res = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(perm_batch)(ch, month_idx, repo_code, a, b, cl, G, ytil, ptil, Ttil) for ch in chunks)
    draws = [r for ch in res for r in ch]
    log(f"perm: done {len(draws)} draws")

    out = {"n_perm": N_PERM, "n_rejected_infeasible": n_reject, "seed": SEED,
           "n_draws_with_actual_treated_set": n_same_set, "tau0": TAU0,
           "design": "17 actual adoption months randomly re-assigned to 17 of the 31 treated∪never repos "
                     "(without replacement); draws where a month falls outside a repo's observed span "
                     "(<= first month or > last month) are redrawn",
           "by_spec": {}}
    for sname in ["es_k_p3", "post_did", "trend_es_k_p3", "trend_post_did"]:
        by0, bp0, Sy0, Sp0, c0 = obs_raw[sname]
        se_obs = float(np.sqrt(c0 * np.sum(Sy0 ** 2)))
        t_obs = by0 / se_obs
        se_obs_tau = float(np.sqrt(c0 * np.sum((Sy0 - TAU0 * Sp0) ** 2)))
        t_obs_tau = (by0 - TAU0) / se_obs   # 観測データでの (β - τ0)/se
        B_, BT, TT, TTt = [], [], [], []
        n_none = 0
        for dr in draws:
            r = dr[sname]
            if r is None:
                n_none += 1
                continue
            by, bp, Sy, Sp, c = r
            se = np.sqrt(c * np.sum(Sy ** 2))
            bt = by - TAU0 * bp + TAU0           # sharp null τ0 の下での並べ替え推定値
            se_t = np.sqrt(c * np.sum((Sy - TAU0 * Sp) ** 2))
            B_.append(by)
            BT.append(bt)
            TT.append(by / se)
            TTt.append((bt - TAU0) / se_t)
        B_, BT, TT, TTt = map(np.array, (B_, BT, TT, TTt))
        n = len(B_)

        def share(x):
            return float(np.mean(x)), float((np.sum(x) + 1) / (n + 1))
        o = {
            "observed": {"coef": float(by0), "se_cr1": se_obs, "t": float(t_obs),
                         "check_d43_Fit": check[sname]},
            "n_valid_draws": n, "n_draws_absorbed": n_none,
            "placebo_coef_quantiles": {q: float(np.quantile(B_, q)) for q in [0.025, 0.05, 0.5, 0.95, 0.975]},
            "placebo_coef_mean": float(B_.mean()), "placebo_coef_sd": float(B_.std(ddof=1)),
            # 帰無 0(sharp null: 効果なし)
            "ri_p_two_coef": share(np.abs(B_) >= abs(by0)),
            "ri_p_upper_coef": share(B_ >= by0),
            "ri_p_lower_coef": share(B_ <= by0),
            "ri_p_two_t": share(np.abs(TT) >= abs(t_obs)),
            "ri_p_upper_t": share(TT >= t_obs),
            "ri_p_lower_t": share(TT <= t_obs),
            # H0: 効果 <= log(0.9)
            "no_decline10_shift_coef": share(B_ + TAU0 >= by0),
            "no_decline10_shift_t": share(TT >= t_obs_tau),
            "no_decline10_sharp_coef": share(BT >= by0),
            "no_decline10_sharp_t": share(TTt >= t_obs_tau),
            "placebo_coef_under_sharp_tau0_quantiles": {q: float(np.quantile(BT, q)) for q in [0.05, 0.5, 0.95]},
            "se_obs_under_tau0": se_obs_tau,
        }
        out["by_spec"][sname] = o
        log(f"  {sname}: obs={by0:+.4f} (d43 Fit {check[sname]['coef']:+.4f}) t={t_obs:.3f} | "
            f"RI two(coef)={o['ri_p_two_coef'][0]:.4f} upper={o['ri_p_upper_coef'][0]:.4f} "
            f"two(t)={o['ri_p_two_t'][0]:.4f} | no-decline10: shift coef={o['no_decline10_shift_coef'][0]:.4f} "
            f"t={o['no_decline10_shift_t'][0]:.4f} sharp coef={o['no_decline10_sharp_coef'][0]:.4f} "
            f"t={o['no_decline10_sharp_t'][0]:.4f} | placebo q2.5/50/97.5="
            f"{o['placebo_coef_quantiles'][0.025]:+.4f}/{o['placebo_coef_quantiles'][0.5]:+.4f}/"
            f"{o['placebo_coef_quantiles'][0.975]:+.4f}")
    return out


# ---------------------------------------------------------------------------
# A(ii) 偽の導入日(-12 ヶ月)
# ---------------------------------------------------------------------------

def run_fake_date(bal: pd.DataFrame, adoption: dict, never: list, months: list, W: np.ndarray) -> dict:
    out = {}
    first = bal.groupby("repo")["month_idx"].min()
    g_true = {r: months.index(m) for r, m in adoption.items()}
    g_fake = {r: g - 12 for r, g in g_true.items()}
    ok = [r for r in adoption if g_fake[r] > first[r]]
    dropped = sorted(set(adoption) - set(ok))
    for label, controls in [("with_never_treated", never), ("treated_only", [])]:
        d = bal[bal["repo"].isin(set(ok) | set(controls))].copy()
        gt = d["repo"].map(g_true)
        d = d[gt.isna() | (d["month_idx"] < gt)].copy()     # 実際の導入前だけ
        d["g"] = d["repo"].map(g_fake)
        d["event_time"] = d["month_idx"] - d["g"]
        cols = es_cols(d, 3, 3)
        pc = add_post(d)
        tr = add_trends(d)
        r = {"n_obs": int(len(d)), "treated_used": sorted(ok), "treated_dropped_no_pre": dropped,
             "n_controls": len(controls)}
        Wd = W[:, :d["repo"].nunique()]
        for sname, cc, nm in [("es_k_p3", cols, "k_p3"), ("post_did", pc, "post"),
                              ("trend_post_did", pc + tr, "post")]:
            f = Fit(d, Y_PRIMARY, cc)
            r[sname] = f.coef(nm) | {"G": f.G, "wcr": wcr_summary(f, nm, 0.0, Wd),
                                     "wcr_no_decline10": wcr_summary(f, nm, TAU0, Wd)}
            if sname == "es_k_p3":
                r["es_coefs"] = {c: f.coef(c) for c in cols if c in f.cols}
        out[label] = r
        log(f"fake-date [{label}] n={len(d)} G={r['post_did']['G']}: k+3={r['es_k_p3']['coef']:+.4f} "
            f"(se {r['es_k_p3']['se']:.4f}, p_t={r['es_k_p3']['p_t']:.4f}, WCR={r['es_k_p3']['wcr']['p_wcr_two']:.4f}) | "
            f"post={r['post_did']['coef']:+.4f} (se {r['post_did']['se']:.4f}, p_t={r['post_did']['p_t']:.4f}, "
            f"WCR={r['post_did']['wcr']['p_wcr_two']:.4f}) | trend post={r['trend_post_did']['coef']:+.4f} "
            f"(p_t={r['trend_post_did']['p_t']:.4f}, WCR={r['trend_post_did']['wcr']['p_wcr_two']:.4f})")
    return out


# ---------------------------------------------------------------------------
# B. 品質アウトカムの「害」の片側検定
# ---------------------------------------------------------------------------

def month_to_idx(s: str) -> int:
    y, m = str(s).split("-")[:2]
    return int(y) * 12 + int(m)


def quality_panels(adoption: dict, never: list) -> dict:
    import d35_revert_outcome as d35
    import d42_fix_inducing as d42
    keep = set(adoption) | set(never)
    d5 = pd.read_parquet(OUT_DIR / "d5_pr_level.parquet")[
        ["repo", "pr_number", "module_id_global", "n_ai_reviews_total", "n_human_reviews_total"]]
    panels = {}

    prs, _ = d35.build_revert_links()
    df = prs.merge(d5, on=["repo", "pr_number"], how="inner")
    df = df[df["repo"].isin(keep)]
    df["t"] = df["month"].map(month_to_idx)
    df = df[df["t"] <= df["t"].max() - d35.FOLLOWUP_MONTHS]
    df["is_revert_pr_f"] = df["is_revert_pr"].astype(float)
    mm = df.groupby(["repo", "module_id_global", "month"]).agg(
        revert_pr_rate=("is_revert_pr_f", "mean")).reset_index()
    panels["d35_revert_pr_rate"] = (mm, "revert_pr_rate")
    log(f"B: d35 panel {len(mm)} rows")

    prs, _ = d42.build()
    df = prs.merge(d5, on=["repo", "pr_number"], how="inner")
    df = df[df["repo"].isin(keep)]
    df["t"] = df["month"].map(month_to_idx)
    df = df[df["t"] <= df["t"].max() - d42.FOLLOWUP_MONTHS]
    mm = df.groupby(["repo", "module_id_global", "month"]).agg(
        fix_rate=("fix_induced", "mean")).reset_index()
    panels["d42_fix_rate"] = (mm, "fix_rate")
    log(f"B: d42 panel {len(mm)} rows")

    for k, (mm, y) in panels.items():
        mm["t"] = mm["month"].map(month_to_idx)
        mm["adopt_t"] = mm["repo"].map(lambda r: month_to_idx(adoption[r]) if r in adoption else np.nan)
        mm["post"] = (mm["adopt_t"].notna() & (mm["t"] >= mm["adopt_t"])).astype(float)
        mm["month_period"] = pd.PeriodIndex(mm["month"], freq="M").to_timestamp()
        mm["month_idx"] = mm["t"] - mm["t"].min()
    return panels


def run_harm(adoption: dict, never: list, W: np.ndarray) -> dict:
    out = {}
    for key, (mm, y) in quality_panels(adoption, never).items():
        base_script = float(mm[y].mean())
        pre_tr = mm[mm["adopt_t"].notna() & (mm["t"] < mm["adopt_t"])]
        base_pre = float(pre_tr[y].mean())
        r_pan = fit_panel(mm, y, ["post"])   # スクリプトと同じ推定(z 推論)
        f = Fit(mm, y, ["post"])
        c = f.coef("post")
        G = f.G
        Wg = W[:, :G]
        tq = float(stats.t.ppf(0.95, G - 1))
        ub_t = c["coef"] + tq * c["se"]
        ub_z_script = r_pan["params"]["post"] + 1.645 * r_pan["se_repo"]["post"]
        ub_wcr = wcr_upper_bound(f, "post", Wg)
        res = {"n_obs": f.n_obs, "G": G, "base_rate_script(all module-months)": base_script,
               "base_rate_treated_pre": base_pre,
               "script_fit_panel": {"coef": r_pan["params"]["post"], "se_repo": r_pan["se_repo"]["post"],
                                    "p_repo_z": r_pan["p_repo"]["post"]},
               "fit_cr1": c,
               "upper95_onesided": {"t(G-1)": ub_t, "z_script": ub_z_script, "wcr_inverted": ub_wcr},
               "tests": {}}
        for bname, base in [("treated_pre", base_pre), ("script_all", base_script)]:
            res["upper95_pct_of_base_" + bname] = {k: 100 * v / base for k, v in res["upper95_onesided"].items()}
            for d_rel in [0.10, 0.20, 0.30]:
                nul = d_rel * base
                tval = (c["coef"] - nul) / c["se"]
                zval = (r_pan["params"]["post"] - nul) / r_pan["se_repo"]["post"]
                w = wcr_summary(f, "post", nul, Wg)
                res["tests"][f"{bname}_increase_{int(d_rel * 100)}pct"] = {
                    "null": nul, "p_t_onesided": float(stats.t.cdf(tval, G - 1)),
                    "p_z_onesided_script": float(stats.norm.cdf(zval)),
                    "p_wcr_onesided": w["p_wcr_lower"]}
        out[key] = res
        log(f"B {key}: coef={c['coef']:+.6f} se={c['se']:.6f} (fit_panel {r_pan['params']['post']:+.6f} "
            f"se {r_pan['se_repo']['post']:.6f}) G={G} base_pre={base_pre:.5f} base_script={base_script:.5f}")
        log(f"   upper95 t={ub_t:+.6f} ({100 * ub_t / base_pre:+.1f}% of pre base; "
            f"{100 * ub_t / base_script:+.1f}% of script base) | z={ub_z_script:+.6f} "
            f"({100 * ub_z_script / base_pre:+.1f}% / {100 * ub_z_script / base_script:+.1f}%) | "
            f"WCR={ub_wcr:+.6f} ({100 * ub_wcr / base_pre:+.1f}% / {100 * ub_wcr / base_script:+.1f}%)")
        for k, v in res["tests"].items():
            log(f"   H0 {k}: p_t={v['p_t_onesided']:.4g} p_z={v['p_z_onesided_script']:.4g} "
                f"p_wcr={v['p_wcr_onesided']:.4g}")
    return out


# ---------------------------------------------------------------------------
# C. PR あたりのレビュー / PR 作成数
# ---------------------------------------------------------------------------

def run_per_pr(bal: pd.DataFrame, W: np.ndarray) -> dict:
    d = bal.copy()
    d["log_n_prs_opened"] = np.log1p(d["n_G_prs"])
    d["log_n_prs_cohort"] = np.log1p(d["n_prs_cohort"])
    d["rpp_cohort"] = np.where(d["n_prs_cohort"] > 0, d["n_rev_nonauthor_cohort"] / d["n_prs_cohort"], np.nan)
    d["log1p_rpp_cohort"] = np.log1p(d["rpp_cohort"])
    d["share_reviewed_cohort"] = np.where(d["n_prs_cohort"] > 0,
                                          d["n_prs_reviewed_nonauthor_cohort"] / d["n_prs_cohort"], np.nan)
    d["rpp_flow"] = np.where(d["n_G_prs"] > 0, d["n_E_human_nonauthor"] / d["n_G_prs"], np.nan)
    d["log1p_rpp_flow"] = np.log1p(d["rpp_flow"])
    d["rpp_reviewed"] = np.where(d["n_E_prs"] > 0, d["n_E_human_nonauthor"] / d["n_E_prs"], np.nan)
    d["log1p_rpp_reviewed"] = np.log1p(d["rpp_reviewed"])
    cols = es_cols(d, 3, 3)
    pc = add_post(d)
    tr = add_trends(d)
    desc = {
        "log_n_prs_opened": "log(1 + PRs opened in module-month; G events, legacy bots excluded)",
        "rpp_cohort": "mean non-author human reviews (lifetime) per PR opened in module-month (NaN if none)",
        "log1p_rpp_cohort": "log(1 + rpp_cohort)",
        "share_reviewed_cohort": "share of PRs opened in module-month that ever get >=1 non-author human review",
        "rpp_flow": "non-author human reviews in month / PRs opened in month (NaN if no PR opened)",
        "log1p_rpp_flow": "log(1 + rpp_flow)",
        "rpp_reviewed": "non-author human reviews in month / PRs with any review event in month",
        "log1p_rpp_reviewed": "log(1 + rpp_reviewed)",
    }
    out = {"definitions": desc,
           "check_cohort_vs_G": {"sum_n_G_prs": float(d["n_G_prs"].sum()), "sum_n_prs_cohort": float(d["n_prs_cohort"].sum())}}
    tre_pre = d[d["g"].notna() & (d["event_time"] < 0)]
    for y in desc:
        is_log = y.startswith("log")
        base = float(tre_pre[y].mean())
        nul10 = TAU0 if is_log else -0.10 * base
        r = {"n_obs": int(d[y].notna().sum()), "treated_pre_mean": base, "null_decline10": nul10}
        for sname, cc, nm in [("es_k_p3", cols, "k_p3"), ("post_did", pc, "post"),
                              ("trend_post_did", pc + tr, "post")]:
            f = Fit(d, y, cc)
            Wg = W[:, :f.G]
            c = f.coef(nm)
            w0 = wcr_summary(f, nm, 0.0, Wg)
            w10 = wcr_summary(f, nm, nul10, Wg)
            tval = (c["coef"] - nul10) / c["se"]
            r[sname] = c | {"G": f.G, "p_wcr_two": w0["p_wcr_two"],
                            "no_decline10_p_wcr_onesided": w10["p_wcr_upper"],
                            "no_decline10_p_t_onesided": float(stats.t.sf(tval, f.G - 1))}
            if sname == "es_k_p3":
                r["pretrend_coefs"] = {k: f.coef(k) for k in ["k_m3", "k_m2"] if k in f.cols}
        out[y] = r
        log(f"C {y} (n={r['n_obs']}, pre mean {base:.4f}): "
            + " | ".join(f"{s} {r[s]['coef']:+.4f} (se {r[s]['se']:.4f}, p_t {r[s]['p_t']:.4f}, "
                         f"WCR {r[s]['p_wcr_two']:.4f}, no-decl10 WCR {r[s]['no_decline10_p_wcr_onesided']:.4f} "
                         f"t {r[s]['no_decline10_p_t_onesided']:.4f})"
                         for s in ["es_k_p3", "post_did", "trend_post_did"])
            + f" | k-3 {r['pretrend_coefs'].get('k_m3', {}).get('coef', float('nan')):+.4f} "
              f"k-2 {r['pretrend_coefs'].get('k_m2', {}).get('coef', float('nan')):+.4f}")
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    for v in ["PILOT_RAW", "PILOT_INTERIM", "PILOT_PROCESSED", "PHASE2_OUT"]:
        log(f"env {v}={os.environ.get(v)}")
    df, bal, adoption, never, excl, months = build_sample()
    log(f"sample: {len(bal)} module-months, {bal['module_id_global'].nunique()} modules")
    W = np.random.default_rng(SEED + 1).choice(WEBB, size=(B_WCR, 64))   # 共通の Webb 重み(列を G 個使う)

    out: dict = {"spec": {"primary_outcome": Y_PRIMARY, "adoption_rule": "presence N=10 (d4)",
                          "n_treated": len(adoption), "n_never": len(never), "n_excluded": len(excl),
                          "fe": "module + month", "cluster": "repo", "B_wcr": B_WCR,
                          "wcr_weights": "Webb 6-point, null imposed", "seed": SEED, "tau0_log0.9": TAU0}}
    out["C_per_pr"] = run_per_pr(bal, W)
    out["A_ii_fake_date"] = run_fake_date(bal, adoption, never, months, W)
    out["B_harm"] = run_harm(adoption, never, W)
    out["A_i_permutation"] = run_permutation(bal, adoption, never, months)

    path = OUT_DIR / "d46_reviewer_requests.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"-> {path}")


if __name__ == "__main__":
    main()
