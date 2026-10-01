"""D48: 再査読 round 3(N1・N2・Top-1/2/4)への追加分析。

1. RQ2 を論文自身のルール(作成者除外・WCR)で測り直す
   (a) 作成者以外の人間にレビューされた PR の割合(カバレッジ)。module x 作成月コホート。
       6 推定量(k=+3 / post、それぞれ repo トレンド有無、repo 水準 post 有無)で
       片側 no-decline(5pp・10pp・処置前平均の 10%)を WCR で。
   (b) 最初の人間 APPROVED より前に出た作成者以外の人間レビュー数(d5/d7 の
       n_human_reviews_before_approval の作成者除外版、log1p(module-month 平均))。
   (c) AI カバレッジ・「AI だけ」の割合・承認前 AI レビュー数(N-RQ2-01/10/12/13)を WCR 付きで再推定。
       論文と同じ repo x month(repo + month FE、post)でも、module x month でも出す。
2. レビューの深さ(新アウトカム)。本文・インラインコメントは AI レビューが付いた PR にしか
   収集していない(d20)ため全 PR では測れない。全 PR で取れる review state と時刻から作る:
   (a) 「一発承認」: 作成者以外の人間の最初のレビューが APPROVED だった PR の割合
       (作成者以外の承認がある PR が分母)。「承認のみ」: 作成者以外の人間レビューが全部 APPROVED。
       増加は rubber-stamping の兆候 → 片側「5pp / 処置前平均の 10% 以上増えた」を検定
   (b) 作成者以外の人間の COMMENTED + CHANGES_REQUESTED 数 / 作成者以外にレビューされた PR(log1p)、
       作成者以外のレビューに占める CHANGES_REQUESTED の割合
   (c) PR 作成から最初の作成者以外の人間レビューまでの時間(module-month 中央値、log1p 時間)
3. 識別の確認
   (a) CODEOWNERS モジュールに対応づく人間(作成者以外)レビューの割合(repo x レビュー月)
   (b) 主アウトカムの 10% no-decline を処置リポジトリを1つずつ抜いて(6 推定量の最悪 p の最大)
   (c) 処置リポジトリごとの導入後観測月数
   (d) PPML(Poisson、module + month FE、repo クラスタ)。pyfixest が無いので
       ppmlhdfe と同じ IRLS + 重み付き交互射影を手で実装し、小さな部分標本で statsmodels GLM(ダミー)と照合。

出力: outputs_full/d48_round4.json(PR 単位の中間表は outputs_full/d48_pr_level.parquet)
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PILOT_PROCESSED=processed_full \
      PHASE2_OUT=outputs_full PHASE2_FIG=figures_full \
      .venv/bin/python pilot/phase2/scripts/d48_round4.py
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name
from d43_causal_robustness import WEBB, Fit, add_post, add_trends, demean, es_cols
from d46_reviewer_requests import build_sample, wcr_summary

SEED = 20260929
B_WCR = 9999
N_JOBS = int(os.environ.get("D48_JOBS", "6"))
Y_PRIMARY = "log_n_E_human_nonauthor"
T0 = time.time()
SPECS6 = ["es_k_p3", "post_did", "trend_es_k_p3", "trend_post_did",
          "repo_level_post_did", "repo_level_trend_post_did"]


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def pdt(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


# ---------------------------------------------------------------------------
# PR 単位の表(prs_full.jsonl から。PR 番号の重複行は最初の行だけ使う = d45 と同じ)
# ---------------------------------------------------------------------------

def pr_table_repo(repo: str) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    rd = RAW_DIR / repo_dir_name(repo)
    idir = INTERIM_DIR / repo_dir_name(repo)
    if not (rd / "prs_full.jsonl").exists() or not (idir / "events.parquet").exists():
        return None
    actors = pd.read_parquet(idir / "actors.parquet")
    cat_of = dict(zip(actors["actor_login"], actors["category"]))
    mods = pd.read_parquet(idir / "modules.parquet")
    f2m = dict(zip(mods["file"], mods["module_id"]))
    ev = pd.read_parquet(idir / "events.parquet", columns=["stage", "module_id", "month", "pr_number"])
    g = ev[ev["stage"] == "G"].sort_values("month").drop_duplicates("pr_number")
    g_mod = dict(zip(g["pr_number"], g["module_id"]))
    rows, rev_rows = [], []
    seen = set()
    n_dup = 0
    with (rd / "prs_full.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                pr = json.loads(line)
            except json.JSONDecodeError:
                continue
            num = pr["number"]
            if num in seen:
                n_dup += 1
                continue
            seen.add(num)
            if not pr.get("createdAt"):
                continue
            created = pdt(pr["createdAt"])
            author = (pr.get("author") or {}).get("login")
            mapped = any(f2m.get(f["path"]) is not None for f in ((pr.get("files") or {}).get("nodes") or []))
            revs = []
            for rv in (pr.get("reviews") or {}).get("nodes", []):
                if not rv.get("submittedAt") or not rv.get("author"):
                    continue
                lg = rv["author"]["login"]
                c = cat_of.get(lg, "unknown")
                revs.append((pdt(rv["submittedAt"]), c, rv.get("state"), c == "human" and lg != author,
                             rv["submittedAt"][:7]))
            revs.sort(key=lambda x: x[0])
            for t, c, s, hn, m in revs:
                if c == "human":
                    rev_rows.append((m, mapped, hn))
            appr = next((i for i, r in enumerate(revs) if r[1] == "human" and r[2] == "APPROVED"), None)
            hn = [r for r in revs if r[3]]
            hn_states = [r[2] for r in hn]
            n_hn_appr = sum(s == "APPROVED" for s in hn_states)
            rows.append({
                "repo": repo, "pr_number": num, "month": pr["createdAt"][:7],
                "module_id": g_mod.get(num), "mapped_files": mapped,
                "author_cat": cat_of.get(author, "unknown") if author else "none",
                "n_h": sum(r[1] == "human" for r in revs), "n_hn": len(hn),
                "n_ai": sum(r[1] == "agent" for r in revs),
                "has_h_appr": appr is not None,
                "n_ai_before": None if appr is None else sum(r[1] == "agent" for r in revs[:appr]),
                "n_h_before": None if appr is None else sum(r[1] == "human" for r in revs[:appr]),
                "n_hn_before": None if appr is None else sum(r[3] for r in revs[:appr]),
                "n_hn_approved": n_hn_appr,
                "n_hn_commented": sum(s == "COMMENTED" for s in hn_states),
                "n_hn_changes": sum(s == "CHANGES_REQUESTED" for s in hn_states),
                "hn_first_is_approval": (hn_states[0] == "APPROVED") if hn else None,
                "hn_all_approvals": (n_hn_appr == len(hn)) if hn else None,
                "hours_to_first_hn": ((hn[0][0] - created) / 3600) if hn else None,
            })
    pr = pd.DataFrame(rows)
    rv = pd.DataFrame(rev_rows, columns=["rev_month", "mapped", "nonauthor"])
    rv["repo"] = repo
    return pr, rv, {"n_dup_lines": n_dup, "n_prs": len(pr)}


def build_pr_tables():
    res = Parallel(n_jobs=N_JOBS)(delayed(pr_table_repo)(r) for r in load_repos())
    prs, rvs, meta = [], [], {}
    for r, x in zip(load_repos(), res):
        if x is None:
            continue
        prs.append(x[0])
        rvs.append(x[1])
        meta[r] = x[2]
    pr = pd.concat(prs, ignore_index=True)
    rv = pd.concat(rvs, ignore_index=True)
    pr.to_parquet(OUT_DIR / "d48_pr_level.parquet", index=False)
    return pr, rv, meta


# ---------------------------------------------------------------------------
# 推定の道具
# ---------------------------------------------------------------------------

def repo_level(d: pd.DataFrame, y: str) -> pd.DataFrame:
    R = d.dropna(subset=[y]).groupby(["repo", "month_idx"])[y].mean().reset_index()
    g = d.groupby("repo")["g"].first()
    R["g"] = R["repo"].map(g)
    R["event_time"] = R["month_idx"] - R["g"]
    return R


def fit_specs(d: pd.DataFrame, y: str, W: np.ndarray, nulls: dict[str, float], direction: str,
              specs: list[str] = SPECS6, unit: str = "module_id_global") -> dict:
    """指定の推定量ごとに係数・CR1・WCR(両側、帰無 0)と、各帰無値での片側 WCR p を返す。

    direction="decline": H0 β <= null(減少)を棄却できるか = p_wcr_upper。
    direction="increase": H0 β >= null(増加)を棄却できるか = p_wcr_lower。
    """
    d = d.dropna(subset=[y]).copy()
    cols = es_cols(d, 3, 3)
    pc = add_post(d)
    tr = add_trends(d)
    R = repo_level(d, y) if any(s.startswith("repo_level") for s in specs) else None
    if R is not None:
        rpc = add_post(R)
        rtr = add_trends(R)
    out = {"n_obs": int(len(d)), "n_repos": int(d["repo"].nunique()),
           "n_units": int(d[unit].nunique()), "specs": {}}
    table = {"es_k_p3": (d, cols, "k_p3", unit), "post_did": (d, pc, "post", unit),
             "trend_es_k_p3": (d, cols + tr, "k_p3", unit), "trend_post_did": (d, pc + tr, "post", unit)}
    if R is not None:
        table |= {"repo_level_post_did": (R, rpc, "post", "repo"),
                  "repo_level_trend_post_did": (R, rpc + rtr, "post", "repo")}
    for s in specs:
        dd, cc, nm, un = table[s]
        f = Fit(dd, y, cc, unit=un)
        Wg = W[:, :f.G]
        c = f.coef(nm)
        r = c | {"G": f.G, "n_obs": f.n_obs, "p_wcr_two": wcr_summary(f, nm, 0.0, Wg)["p_wcr_two"]}
        if s == "es_k_p3":
            r["leads"] = {k: f.coef(k) for k in ["k_m3", "k_m2"] if k in f.cols}
            r["lags"] = {k: f.coef(k) for k in ["k_p0", "k_p1", "k_p2"] if k in f.cols}
        for key, nul in nulls.items():
            w = wcr_summary(f, nm, nul, Wg)
            tval = (c["coef"] - nul) / c["se"]
            if direction == "decline":
                r[key] = {"null": nul, "p_wcr_onesided": w["p_wcr_upper"],
                          "p_t_onesided": float(stats.t.sf(tval, f.G - 1))}
            else:
                r[key] = {"null": nul, "p_wcr_onesided": w["p_wcr_lower"],
                          "p_t_onesided": float(stats.t.cdf(tval, f.G - 1))}
        out["specs"][s] = r
    for key in nulls:
        ps = {s: out["specs"][s][key]["p_wcr_onesided"] for s in specs}
        out[f"worst_p_wcr_{key}"] = max(ps.values())
        out[f"worst_spec_{key}"] = max(ps, key=ps.get)
    return out


def fmt(y: str, r: dict, keys: list[str]) -> str:
    s = r["specs"]
    parts = [f"{y} n={r['n_obs']} G={s[next(iter(s))]['G']}"]
    for sp in s:
        parts.append(f"{sp} {s[sp]['coef']:+.4f} (se {s[sp]['se']:.4f}, p_t {s[sp]['p_t']:.4f}, "
                     f"WCR {s[sp]['p_wcr_two']:.4f}; " +
                     ", ".join(f"{k} {s[sp][k]['p_wcr_onesided']:.4f}" for k in keys) + ")")
    if "es_k_p3" in s and "leads" in s["es_k_p3"]:
        L = s["es_k_p3"]["leads"]
        parts.append("leads " + " ".join(f"{k} {v['coef']:+.4f}(p_t {v['p_t']:.3f})" for k, v in L.items()))
    parts.append("WORST " + ", ".join(f"{k} {r['worst_p_wcr_' + k]:.4f} [{r['worst_spec_' + k]}]" for k in keys))
    return " | ".join(parts)


def share_nulls(base: float, direction: str) -> dict[str, float]:
    sg = -1.0 if direction == "decline" else 1.0
    return {"d5pp": sg * 0.05, "d10pp": sg * 0.10, "rel10": sg * 0.10 * base}


LOG_NULLS = {"rel10": float(np.log(0.9)), "rel5": float(np.log(0.95))}


# ---------------------------------------------------------------------------
# パネル
# ---------------------------------------------------------------------------

def cohort_panel(pr: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    p = pr[pr["module_id"].notna()].copy()
    p["module_id_global"] = p["repo"] + "::" + p["module_id"]
    p["cov_hn"] = (p["n_hn"] > 0).astype(float)
    p["cov_h"] = (p["n_h"] > 0).astype(float)
    p["cov_ai"] = (p["n_ai"] > 0).astype(float)
    p["cov_any"] = ((p["n_h"] > 0) | (p["n_ai"] > 0)).astype(float)
    p["ai_only_h"] = ((p["n_h"] == 0) & (p["n_ai"] > 0)).astype(float)
    p["ai_only_hn"] = ((p["n_hn"] == 0) & (p["n_ai"] > 0)).astype(float)
    p["hn_appr_pr"] = p["n_hn_approved"] > 0
    tf = {True: 1.0, False: 0.0}
    p["oneshot"] = np.where(p["hn_appr_pr"], p["hn_first_is_approval"].map(tf), np.nan)
    p["approval_only"] = np.where(p["hn_appr_pr"], p["hn_all_approvals"].map(tf), np.nan)
    p["subst_per_rev_pr"] = np.where(p["n_hn"] > 0, p["n_hn_commented"] + p["n_hn_changes"], np.nan)
    agg = p.groupby(["module_id_global", "month"]).agg(
        n_prs=("pr_number", "size"),
        cov_hn=("cov_hn", "mean"), cov_h=("cov_h", "mean"), cov_ai=("cov_ai", "mean"),
        cov_any=("cov_any", "mean"), ai_only_h=("ai_only_h", "mean"), ai_only_hn=("ai_only_hn", "mean"),
        avg_hn_before=("n_hn_before", "mean"), avg_h_before=("n_h_before", "mean"),
        avg_ai_before=("n_ai_before", "mean"),
        oneshot_approval=("oneshot", "mean"), approval_only=("approval_only", "mean"),
        n_hn_appr_prs=("hn_appr_pr", "sum"),
        avg_subst=("subst_per_rev_pr", "mean"),
        n_hn=("n_hn", "sum"), n_hn_changes=("n_hn_changes", "sum"),
        med_hours_first_hn=("hours_to_first_hn", "median"),
    ).reset_index()
    agg["log_hn_before_approval"] = np.log1p(agg["avg_hn_before"])
    agg["log_human_before_approval"] = np.log1p(agg["avg_h_before"])
    agg["log_ai_before_approval"] = np.log1p(agg["avg_ai_before"])
    agg["log1p_subst_per_reviewed_pr"] = np.log1p(agg["avg_subst"])
    agg["changes_share_hn"] = np.where(agg["n_hn"] > 0, agg["n_hn_changes"] / agg["n_hn"], np.nan)
    agg["log1p_med_hours_first_hn"] = np.log1p(agg["med_hours_first_hn"].clip(lower=0))
    keys = bal[["module_id_global", "month", "repo", "month_idx", "g", "event_time"]]
    return agg.merge(keys, on=["module_id_global", "month"], how="inner")


def repo_month_panel(pr: pd.DataFrame, bal: pd.DataFrame) -> pd.DataFrame:
    """論文(c_misc2)と同じ repo x 作成月の平均(CODEOWNERS モジュールに帰属した PR = d5 の母集団)。"""
    p = pr[pr["module_id"].notna() & pr["repo"].isin(set(bal["repo"]))].copy()
    for c, v in [("cov_hn", p["n_hn"] > 0), ("cov_h", p["n_h"] > 0), ("cov_ai", p["n_ai"] > 0),
                 ("cov_any", (p["n_h"] > 0) | (p["n_ai"] > 0)),
                 ("ai_only_h", (p["n_h"] == 0) & (p["n_ai"] > 0)),
                 ("ai_only_hn", (p["n_hn"] == 0) & (p["n_ai"] > 0))]:
        p[c] = v.astype(float)
    R = p.groupby(["repo", "month"])[["cov_hn", "cov_h", "cov_ai", "cov_any", "ai_only_h", "ai_only_hn"]].mean()
    R = R.join(p.groupby(["repo", "month"]).size().rename("n_prs")).reset_index()
    mi = bal[["month", "month_idx"]].drop_duplicates()
    R = R.merge(mi, on="month", how="inner")
    g = bal.groupby("repo")["g"].first()
    R["g"] = R["repo"].map(g)
    R["event_time"] = R["month_idx"] - R["g"]
    return R


# ---------------------------------------------------------------------------
# PPML(ppmlhdfe と同じ IRLS + 重み付き交互射影)
# ---------------------------------------------------------------------------

def wdemean(M: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray, it: int = 1000, tol: float = 1e-10):
    M = M.copy()
    na, nb = a.max() + 1, b.max() + 1
    wa = np.maximum(np.bincount(a, w, na), 1e-300)
    wb = np.maximum(np.bincount(b, w, nb), 1e-300)
    for _ in range(it):
        delta = 0.0
        for j in range(M.shape[1]):
            M[:, j] -= (np.bincount(a, w * M[:, j], na) / wa)[a]
            mb = (np.bincount(b, w * M[:, j], nb) / wb)[b]
            M[:, j] -= mb
            delta = max(delta, float(np.max(np.abs(mb))))
        if delta < tol:
            break
    return M


def ppml(d: pd.DataFrame, y: str, cols: list[str], unit: str = "module_id_global", maxit: int = 100) -> dict:
    d = d.copy()
    # 全期間 0 のユニット・月は FE が -inf になり尤度に寄与しないので落とす(ppmlhdfe と同じ)
    pos_u = d.groupby(unit)[y].transform("sum") > 0
    pos_m = d.groupby("month_idx")[y].transform("sum") > 0
    n0 = len(d)
    d = d[pos_u & pos_m]
    a = pd.factorize(d[unit])[0]
    b = pd.factorize(d["month_idx"])[0]
    cl, repos = pd.factorize(d["repo"])
    G = len(repos)
    Y = d[y].to_numpy(float)
    X0 = d[cols].to_numpy(float)
    mu = (Y + Y.mean()) / 2
    eta = np.log(mu)
    dev_old = np.inf
    for itn in range(maxit):
        z = eta + (Y - mu) / mu
        M = wdemean(np.column_stack([z, X0]), a, b, mu)
        zt, Xt = M[:, 0], M[:, 1:]
        WX = Xt * mu[:, None]
        beta = np.linalg.solve(Xt.T @ WX, WX.T @ zt)
        resid = zt - Xt @ beta
        eta = z - resid
        mu = np.exp(eta)
        with np.errstate(divide="ignore", invalid="ignore"):
            dev = 2 * np.sum(np.where(Y > 0, Y * np.log(Y / mu), 0.0) - (Y - mu))
        if abs(dev - dev_old) / max(abs(dev), 0.1) < 1e-9:
            break
        dev_old = dev
    # 収束点での重みで最終の X̃ を作り直す
    Xt = wdemean(X0, a, b, mu)
    H = Xt.T @ (Xt * mu[:, None])
    Hi = np.linalg.inv(H)
    sc = Xt * (Y - mu)[:, None]
    S = np.column_stack([np.bincount(cl, sc[:, j], G) for j in range(Xt.shape[1])])
    n, k = Xt.shape
    V = (G / (G - 1)) * Hi @ (S.T @ S) @ Hi
    out = {"n_obs_used": int(n), "n_dropped_allzero": int(n0 - n), "G": int(G), "iterations": itn + 1,
           "deviance": float(dev), "coefs": {}}
    for j, c in enumerate(cols):
        se = float(np.sqrt(V[j, j]))
        out["coefs"][c] = {"coef": float(beta[j]), "se": se,
                           "p_t": float(2 * stats.t.sf(abs(beta[j] / se), G - 1)),
                           "p_z": float(2 * stats.norm.sf(abs(beta[j] / se)))}
    return out


def ppml_check_vs_glm(bal: pd.DataFrame, adoption: dict, never: list) -> dict:
    """小さな部分標本(処置 3・対照 3 リポジトリ、各リポジトリで行数の多い 15 モジュール)で
    statsmodels GLM(Poisson、module・month ダミー)と係数・クラスタ SE を突き合わせる。"""
    import statsmodels.api as sm
    rs = sorted(adoption)[:3] + sorted(never)[:3]
    d = bal[bal["repo"].isin(rs)].copy()
    top = d.groupby(["repo", "module_id_global"])["n_E_human_nonauthor"].sum().reset_index()
    top = top.sort_values("n_E_human_nonauthor", ascending=False).groupby("repo").head(15)
    d = d[d["module_id_global"].isin(top["module_id_global"])].copy()
    pc = add_post(d)
    mine = ppml(d, "n_E_human_nonauthor", pc)
    dd = d[(d.groupby("module_id_global")["n_E_human_nonauthor"].transform("sum") > 0)
           & (d.groupby("month_idx")["n_E_human_nonauthor"].transform("sum") > 0)]
    X = pd.get_dummies(dd[["module_id_global"]].astype(str), drop_first=False, dtype=float)
    Xm = pd.get_dummies(dd["month_idx"].astype(str), prefix="m", drop_first=True, dtype=float)
    Xall = pd.concat([dd[["post"]].reset_index(drop=True), X.reset_index(drop=True), Xm.reset_index(drop=True)], axis=1)
    m = sm.GLM(dd["n_E_human_nonauthor"].to_numpy(float), Xall, family=sm.families.Poisson())
    r = m.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["repo"])[0], "use_correction": False})
    G = dd["repo"].nunique()
    se_glm = float(r.bse["post"]) * np.sqrt(G / (G - 1))
    return {"repos": rs, "n": int(len(dd)), "mine_coef": mine["coefs"]["post"]["coef"],
            "mine_se": mine["coefs"]["post"]["se"], "glm_coef": float(r.params["post"]),
            "glm_se_cluster_x_G/(G-1)": se_glm}


# ---------------------------------------------------------------------------
# 1 ジョブ = 1 アウトカムの 6 推定量
# ---------------------------------------------------------------------------

def job(name, d, y, W, nulls, direction, specs=SPECS6, unit="module_id_global"):
    t = time.time()
    r = fit_specs(d, y, W, nulls, direction, specs=specs, unit=unit)
    r["seconds"] = round(time.time() - t, 1)
    return name, r


def loo_job(r_out, bal, W):
    d = bal[bal["repo"] != r_out]
    t = time.time()
    r = fit_specs(d, Y_PRIMARY, W, {"rel10": float(np.log(0.9))}, "decline")
    return r_out, {"worst_p_wcr_rel10": r["worst_p_wcr_rel10"], "worst_spec": r["worst_spec_rel10"],
                   "by_spec": {s: {"coef": v["coef"], "se": v["se"], "p_wcr_rel10": v["rel10"]["p_wcr_onesided"],
                                   "p_t_rel10": v["rel10"]["p_t_onesided"]} for s, v in r["specs"].items()},
                   "seconds": round(time.time() - t, 1)}


def main() -> None:
    for v in ["PILOT_RAW", "PILOT_INTERIM", "PILOT_PROCESSED", "PHASE2_OUT", "PHASE2_FIG"]:
        log(f"env {v}={os.environ.get(v)}")
    df, bal, adoption, never, excl, months = build_sample()
    log(f"sample: {len(bal)} module-months, {bal['module_id_global'].nunique()} modules")
    assert len(bal) == 412997, "primary panel が 412,997 module-months でない"
    W = np.random.default_rng(SEED).choice(WEBB, size=(B_WCR, 64))
    out: dict = {"spec": {"adoption_rule": "presence N=10 (d4)", "n_treated": len(adoption), "n_never": len(never),
                          "n_excluded": len(excl), "fe": "unit + month", "cluster": "repo", "B_wcr": B_WCR,
                          "wcr_weights": "Webb 6-point, null imposed", "seed": SEED,
                          "specs6": SPECS6,
                          "one_sided": "decline: H0 beta <= null, p = P*(t* >= t); increase: H0 beta >= null, "
                                       "p = P*(t* <= t)",
                          "nulls_share": "d5pp/d10pp absolute share points; rel10 = 10% of treated-pre mean",
                          "nulls_log": "rel10 = log(0.9), rel5 = log(0.95)"}}

    # ---- PR 表
    pr, rv, meta = build_pr_tables()
    log(f"PR table: {len(pr)} PRs, dup lines skipped {sum(m['n_dup_lines'] for m in meta.values())}")
    out["pr_table"] = {"n_prs": int(len(pr)), "dup_lines_skipped": int(sum(m["n_dup_lines"] for m in meta.values())),
                       "path": str(OUT_DIR / "d48_pr_level.parquet")}
    C = cohort_panel(pr, bal)
    chk = float(C["n_prs"].sum())
    ref = float(bal["n_prs_cohort"].sum())
    log(f"cohort panel {len(C)} cells; PRs {chk:.0f} vs d46 cohort {ref:.0f}")
    out["pr_table"]["check_cohort_prs_vs_d46"] = [chk, ref]
    # RQ1 の件数パネル・d46 と同じく legacy_bot が作成した PR を除いた版(論文 RQ2 の c_misc2 は含む版)
    CN = cohort_panel(pr[pr["author_cat"] != "legacy_bot"], bal)
    log(f"cohort panel (legacy-bot PRs excluded) {len(CN)} cells; PRs {CN['n_prs'].sum():.0f} vs d46 cohort {ref:.0f}")
    out["pr_table"]["check_cohort_nobot_prs_vs_d46"] = [float(CN["n_prs"].sum()), ref]
    RM = repo_month_panel(pr, bal)
    log(f"repo-month panel {len(RM)} rows, {RM['repo'].nunique()} repos")

    def pre_mean(d, y):
        return float(d[d["g"].notna() & (d["event_time"] < 0)][y].mean())

    # ---- ジョブ定義
    jobs = []
    means = {"B:" + Y_PRIMARY: pre_mean(bal, Y_PRIMARY)}
    jobs.append(("T0_primary:" + Y_PRIMARY, bal, Y_PRIMARY, LOG_NULLS, "decline"))
    share_decl = ["cov_hn", "cov_h", "cov_ai", "cov_any"]
    for y in share_decl:
        means[f"C:{y}"] = pre_mean(C, y)
        jobs.append((f"T1_cohort:{y}", C, y, share_nulls(means[f"C:{y}"], "decline"), "decline"))
    for y in ["ai_only_h", "ai_only_hn"]:
        means[f"C:{y}"] = pre_mean(C, y)
        jobs.append((f"T1_cohort:{y}", C, y, share_nulls(means[f"C:{y}"], "increase"), "increase"))
    for y in ["log_hn_before_approval", "log_human_before_approval", "log_ai_before_approval"]:
        means[f"C:{y}"] = pre_mean(C, y)
        jobs.append((f"T1_cohort:{y}", C, y, LOG_NULLS, "decline"))
    # 深さ
    for y in ["oneshot_approval", "approval_only"]:
        means[f"C:{y}"] = pre_mean(C, y)
        jobs.append((f"T2_depth:{y}", C, y, {"i5pp": 0.05, "i10pp": 0.10, "rel10": 0.10 * means[f"C:{y}"]},
                     "increase"))
    means["C:changes_share_hn"] = pre_mean(C, "changes_share_hn")
    jobs.append(("T2_depth:changes_share_hn", C, "changes_share_hn",
                 share_nulls(means["C:changes_share_hn"], "decline"), "decline"))
    means["C:log1p_subst_per_reviewed_pr"] = pre_mean(C, "log1p_subst_per_reviewed_pr")
    jobs.append(("T2_depth:log1p_subst_per_reviewed_pr", C, "log1p_subst_per_reviewed_pr", LOG_NULLS, "decline"))
    means["C:log1p_med_hours_first_hn"] = pre_mean(C, "log1p_med_hours_first_hn")
    jobs.append(("T2_depth:log1p_med_hours_first_hn", C, "log1p_med_hours_first_hn",
                 {"slower10": float(np.log(1.1))}, "increase"))
    for y in ["cov_hn", "cov_h", "cov_ai"]:
        means[f"CN:{y}"] = pre_mean(CN, y)
        jobs.append((f"T1_cohort_nobot:{y}", CN, y, share_nulls(means[f"CN:{y}"], "decline"), "decline"))
    means["CN:ai_only_hn"] = pre_mean(CN, "ai_only_hn")
    jobs.append(("T1_cohort_nobot:ai_only_hn", CN, "ai_only_hn", share_nulls(means["CN:ai_only_hn"], "increase"), "increase"))
    means["CN:log_hn_before_approval"] = pre_mean(CN, "log_hn_before_approval")
    jobs.append(("T1_cohort_nobot:log_hn_before_approval", CN, "log_hn_before_approval", LOG_NULLS, "decline"))
    for y in ["oneshot_approval", "approval_only"]:
        means[f"CN:{y}"] = pre_mean(CN, y)
        jobs.append((f"T2_depth_nobot:{y}", CN, y, {"i5pp": 0.05, "i10pp": 0.10, "rel10": 0.10 * means[f"CN:{y}"]},
                     "increase"))
    means["CN:log1p_subst_per_reviewed_pr"] = pre_mean(CN, "log1p_subst_per_reviewed_pr")
    jobs.append(("T2_depth_nobot:log1p_subst_per_reviewed_pr", CN, "log1p_subst_per_reviewed_pr", LOG_NULLS, "decline"))
    # 論文と同じ repo x month(post と k=+3、トレンド無しの 2 推定量 + トレンド付き post)
    rm_specs = ["es_k_p3", "post_did", "trend_post_did"]
    for y in ["cov_hn", "cov_h", "cov_ai", "cov_any"]:
        means[f"RM:{y}"] = pre_mean(RM, y)
        jobs.append((f"T1_repomonth:{y}", RM, y, share_nulls(means[f"RM:{y}"], "decline"), "decline", rm_specs, "repo"))
    for y in ["ai_only_h", "ai_only_hn"]:
        means[f"RM:{y}"] = pre_mean(RM, y)
        jobs.append((f"T1_repomonth:{y}", RM, y, share_nulls(means[f"RM:{y}"], "increase"), "increase", rm_specs, "repo"))

    # 3a. CODEOWNERS への対応づけ率(repo x レビュー月)
    keep = set(adoption) | set(never)
    r2 = rv[rv["repo"].isin(keep)]
    mp = []
    for lab, sub in [("nonauthor", r2[r2["nonauthor"]]), ("all_human", r2)]:
        a = sub.groupby(["repo", "rev_month"]).agg(n=("mapped", "size"), n_mapped=("mapped", "sum")).reset_index()
        a[f"mapped_share_{lab}"] = a["n_mapped"] / a["n"]
        mp.append(a[["repo", "rev_month", f"mapped_share_{lab}"] + (["n"] if lab == "nonauthor" else [])])
    MP = mp[0].merge(mp[1], on=["repo", "rev_month"], how="outer").rename(columns={"rev_month": "month"})
    mi = {m: i for i, m in enumerate(months)}
    MP["month_idx"] = MP["month"].map(mi)
    span = bal.groupby("repo")["month_idx"].agg(["min", "max"])
    MP = MP.dropna(subset=["month_idx"])
    MP = MP[(MP["month_idx"] >= MP["repo"].map(span["min"])) & (MP["month_idx"] <= MP["repo"].map(span["max"]))]
    MP["g"] = MP["repo"].map({r: mi[m] for r, m in adoption.items()})
    MP["event_time"] = MP["month_idx"] - MP["g"]
    for y in ["mapped_share_nonauthor", "mapped_share_all_human"]:
        means[f"MP:{y}"] = pre_mean(MP, y)
        jobs.append((f"T3a_mapping:{y}", MP, y, share_nulls(means[f"MP:{y}"], "decline"), "decline",
                     ["es_k_p3", "post_did", "trend_post_did"], "repo"))
    log(f"{len(jobs)} outcome jobs; running with {N_JOBS} jobs")

    res = Parallel(n_jobs=N_JOBS)(delayed(job)(j[0], j[1], j[2], W, *j[3:]) for j in jobs)
    results = dict(res)
    for (name, *_rest) in jobs:
        r = results[name]
        key = name.split(":", 1)
        mkey = {"T0_primary": "B", "T1_cohort": "C", "T2_depth": "C", "T1_cohort_nobot": "CN", "T2_depth_nobot": "CN", "T1_repomonth": "RM", "T3a_mapping": "MP"}[key[0]] + ":" + key[1]
        r["treated_pre_mean"] = means[mkey]
        nk = [k for k in r if k.startswith("worst_p_wcr_")]
        log(f"[{name}] pre mean {means[mkey]:.4f} | " + fmt(key[1], r, [k[len('worst_p_wcr_'):] for k in nk]))
    out["results"] = results

    # 参照値との照合(d7 の N-RQ1-70 / N-RQ2-01、d46 の N-PPR-04)
    try:
        d7 = json.loads((OUT_DIR / "d7_burden_event_study.json").read_text())
        out["check_d7"] = {y: {"d7_k_p3": d7["models"][y]["params"]["k_p3"], "d7_n": d7["models"][y]["n_obs"],
                               "d48_k_p3": results[f"T1_cohort:{y}"]["specs"]["es_k_p3"]["coef"],
                               "d48_n": results[f"T1_cohort:{y}"]["n_obs"]}
                           for y in ["log_human_before_approval", "log_ai_before_approval"]}
        log(f"check d7: {out['check_d7']}")
    except Exception as e:  # noqa: BLE001
        log(f"check d7 failed: {e}")

    # 3c. 導入後の観測月数
    last = bal.groupby("repo")["month_idx"].max()
    post_m = {r: int(last[r] - mi[m] + 1) for r, m in adoption.items()}
    v = np.array(list(post_m.values()))
    out["T3c_post_months"] = {"by_repo": post_m, "median": float(np.median(v)), "min": int(v.min()), "max": int(v.max()),
                              "last_month": months[-1], "note": "counts adoption month through last observed month "
                                                                "(the last month, 2026-09, is partial)"}
    log(f"3c post-adoption months: median {np.median(v)} range {v.min()}..{v.max()} {post_m}")

    # 3d. PPML
    t = time.time()
    chk = ppml_check_vs_glm(bal, adoption, never)
    log(f"PPML check vs GLM: {chk}")
    d = bal.copy()
    cols = es_cols(d, 3, 3)
    pc = add_post(d)
    pp = {"check_vs_statsmodels_glm": chk}
    nul = float(np.log(0.9))
    for sname, cc, nm in [("es_k_p3", cols, "k_p3"), ("post_did", pc, "post")]:
        r = ppml(d, "n_E_human_nonauthor", cc)
        c = r["coefs"][nm]
        tval = (c["coef"] - nul) / c["se"]
        r["no_decline10_p_t_onesided"] = float(stats.t.sf(tval, r["G"] - 1))
        r["no_decline10_p_z_onesided"] = float(stats.norm.sf(tval))
        pp[sname] = r
        log(f"PPML {sname}: {nm} {c['coef']:+.4f} (se {c['se']:.4f}, p_t {c['p_t']:.4f}) n_used {r['n_obs_used']} "
            f"dropped {r['n_dropped_allzero']} it {r['iterations']} | no-decline10 p_t {r['no_decline10_p_t_onesided']:.4g}"
            + (" | leads " + " ".join(f"{k} {r['coefs'][k]['coef']:+.4f}(p_t {r['coefs'][k]['p_t']:.3f})"
                                     for k in ["k_m3", "k_m2", "k_p0", "k_p1", "k_p2"]) if sname == "es_k_p3" else ""))
    pp["inference"] = "CR1 repo-clustered sandwich, G/(G-1); one-sided t(G-1) vs log(0.9); no WCR for PPML"
    pp["seconds"] = round(time.time() - t, 1)
    out["T3d_ppml"] = pp

    # 3b. leave-one-treated-out(6 推定量の最悪 p)
    log("3b LOO start")
    lo = Parallel(n_jobs=N_JOBS)(delayed(loo_job)(r, bal, W) for r in sorted(adoption))
    loo = dict(lo)
    worst = max(loo.values(), key=lambda x: x["worst_p_wcr_rel10"])
    wr = max(loo, key=lambda r: loo[r]["worst_p_wcr_rel10"])
    out["T3b_loo_no_decline10"] = {"by_excluded_repo": loo, "max_worst_p": worst["worst_p_wcr_rel10"],
                                   "argmax_excluded_repo": wr, "argmax_spec": worst["worst_spec"],
                                   "rule": "for each excluded treated repo, worst one-sided WCR p over the 6 specs; "
                                           "then max over the 17 exclusions"}
    for r in sorted(loo):
        log(f"  LOO -{r}: worst p10 {loo[r]['worst_p_wcr_rel10']:.4f} [{loo[r]['worst_spec']}] "
            + " ".join(f"{s} {v['coef']:+.4f}/{v['p_wcr_rel10']:.4f}" for s, v in loo[r]["by_spec"].items()))
    log(f"3b LOO max worst p10 = {worst['worst_p_wcr_rel10']:.4f} (excluding {wr}, {worst['worst_spec']})")

    path = OUT_DIR / "d48_round4.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"-> {path}")


if __name__ == "__main__":
    main()
