"""D44: PR 作成者本人のコメントを除いて、重なり・委譲上限・trade-off を測り直す。

査読会議 F1(R2 の指摘): 位置付きの「人間のレビューコメント」の約 4 割は PR 作成者本人の
返信("done" "good catch" など)である。返信は AI のコメントと同じ path/line に付くので、
  - AI と人間の「重なり」は「人間が独立に同じ箇所を見つけた」ではなく「AI に返事をした」を測る
  - 人間どうしの対照も、レビュアーのスレッドへの作成者の返信で水増しされる
  - 「AI が先」は返信が必ず後に来るので機械的に高くなる
そこで人間コメントを 3 つの定義で数え、既存の結果(D21 / D26 / D40 / D41)を並べて比べる。

  (a) all        : 人間のコメントすべて(論文の現行定義)
  (b) nonauthor  : PR 作成者以外の人間(= レビュアーのコメント)
  (c) nonauthor_nonmerger : (b) からさらにその PR をマージした人も除く

再計算するもの:
  1. 重なり(D21): 同じ PR 内の AI×人間 vs 人間×人間の行 Jaccard(±3 行 = 7 行の箱)、
     同じ箇所での AI が先の割合
  2. 委譲上限(D41): 人間コメントのうち AI も同じ箇所 / suggestion 付き AI が同じ箇所、
     許容幅 0/3/10/20、月次系列とトレンド(リポジトリ FE、リポジトリクラスタ)、ベンダー別
  3. suggestion の trade-off(D40): エージェント内で suggestion の有無ごとに
     対処率(isOutdated)と「人間も同じ箇所を指摘した」率
  4. 導入後の月数に伴う重なりの収束(同じ PR 内で対にして)
  5. RQ1 の人間レビュー提出のうち PR 作成者本人の割合(処置群の導入前後)

出力: outputs_full/d44_author_excluded.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d44_author_excluded.py
"""
from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name

DEFS = ["all", "nonauthor", "nonauthor_nonmerger"]
TOLS = [0, 3, 10, 20]
BIN = 7  # D21 / D26 と同じ「±3 行」= 7 行の箱
COPILOT = "copilot-pull-request-reviewer"
BIG = 10**9


def norm(login) -> str | None:
    """login を小文字にし、末尾の [bot] を落とす(cursor と cursor[bot] を同一視する)。"""
    if login is None:
        return None
    if isinstance(login, dict):
        login = login.get("login")
    if not login:
        return None
    s = str(login).lower()
    return s[:-5] if s.endswith("[bot]") else s


def ts(s: str | None) -> float:
    if not s:
        return np.nan
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


# ------------------------------------------------------------------ 読み込み
def load_all():
    """コメント(位置付き)と、PR 単位の人間レビュー提出(RQ5 用)を読み込む。"""
    crow, rrow = [], []
    for repo in load_repos():
        d = RAW_DIR / repo_dir_name(repo)
        apath = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not apath.exists() or not (d / "prs_full.jsonl").exists():
            continue
        a = pd.read_parquet(apath)
        cat = {}
        for lg, c in zip(a["actor_login"], a["category"]):
            cat.setdefault(norm(lg), c)

        # PR 作成者(prs_full)と、RQ5 用の人間レビュー提出
        author: dict[int, str | None] = {}
        with (d / "prs_full.jsonl").open() as f:
            for line in f:
                try:
                    pr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                au = norm(pr.get("author"))
                author[pr["number"]] = au
                if (pr.get("createdAt") or "") < "2023-01":
                    continue
                nodes = (pr.get("reviews") or {}).get("nodes") or []
                has_ai = any(cat.get(norm(r.get("author"))) == "agent" for r in nodes)
                for r in nodes:
                    lg = norm(r.get("author"))
                    if cat.get(lg) != "human":
                        continue
                    rrow.append((repo, (r.get("submittedAt") or pr["createdAt"])[:7],
                                 lg == au, r.get("state"), has_ai))
        merger: dict[int, str | None] = {}
        mp = d / "merged_by.jsonl"
        if mp.exists():
            with mp.open() as f:
                for line in f:
                    try:
                        x = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    merger[x["number"]] = norm(x.get("merged_by"))

        # スレッドの状態(D26 と同じく (pr, path, line) で突き合わせる)
        threads: dict[tuple, bool] = {}
        tp = d / "thread_status.jsonl"
        if tp.exists():
            with tp.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for t in rec.get("threads") or []:
                        threads[(rec["number"], t.get("path"), t.get("line"))] = bool(t.get("isOutdated"))

        bp = d / "review_bodies.jsonl"
        if not bp.exists():
            continue
        with bp.open() as f:
            for line in f:
                try:
                    pr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                num = pr["number"]
                au, mg = author.get(num), merger.get(num)
                for ri, rv in enumerate(pr.get("reviews") or []):
                    lg = norm(rv.get("author"))
                    c = cat.get(lg)
                    if c not in ("human", "agent"):
                        continue
                    t = ts(rv.get("submittedAt"))
                    for cm in rv.get("comments") or []:
                        path = cm.get("path")
                        ln = cm.get("line") or cm.get("originalLine")
                        if not path or ln is None:
                            continue
                        # D26 の同一箇所判定は originalLine を優先する(outdated で line が消えるため)
                        ln26 = cm.get("originalLine") if cm.get("originalLine") is not None else cm.get("line")
                        body = cm.get("body") or ""
                        od = threads.get((num, path, cm.get("line"))) if c == "agent" else None
                        crow.append((repo, num, path, int(ln), int(ln26), c == "agent", lg,
                                     t, (rv.get("submittedAt") or "")[:7],
                                     "```suggestion" in body,
                                     au is not None and lg == au,
                                     mg is not None and lg == mg,
                                     np.nan if od is None else float(od)))
    df = pd.DataFrame(crow, columns=["repo", "pr", "path", "line", "line26", "ai", "login", "t", "month",
                                     "sug", "is_author", "is_merger", "outdated"])
    rv = pd.DataFrame(rrow, columns=["repo", "month", "is_author", "state", "pr_has_ai"])
    return df, rv


def human_mask(df: pd.DataFrame, dfn: str) -> np.ndarray:
    h = ~df["ai"].to_numpy()
    if dfn == "all":
        return h
    m = h & ~df["is_author"].to_numpy()
    if dfn == "nonauthor":
        return m
    return m & ~df["is_merger"].to_numpy()


# ------------------------------------------------------------------ 距離の前計算
def nearest(df: pd.DataFrame) -> pd.DataFrame:
    """各コメントについて、同じ PR・同じ path で最も近い相手との行の距離を求める。

    人間行: 最寄りの AI / suggestion 付き AI / Copilot / Copilot の suggestion / それ以外の AI、
            ±3 行以内に「自分より前の」AI があるか
    AI 行 : 最寄りの人間(3 定義それぞれ)
    許容幅ごとの判定は距離との比較だけで済むので、ここで一度だけ計算する。
    """
    n = len(df)
    out = {k: np.full(n, BIG, dtype=np.int64) for k in
           ["d_ai", "d_sug", "d_cop", "d_cop_sug", "d_oth", "d_oth_sug",
            "d_h_all", "d_h_nonauthor", "d_h_nonauthor_nonmerger"]}
    ai_before = np.zeros(n, dtype=bool)
    ai_before_exact = np.zeros(n, dtype=bool)  # 同じ行に先に AI がいた(返信である可能性が高い)
    hm = {k: human_mask(df, k) for k in DEFS}
    line = df["line"].to_numpy()
    isai = df["ai"].to_numpy()
    sug = df["sug"].to_numpy()
    cop = (df["login"] == COPILOT).to_numpy()
    tt = df["t"].to_numpy()
    for _, idx in df.groupby(["repo", "pr", "path"], sort=False).indices.items():
        a = idx[isai[idx]]
        if len(a) == 0:
            continue
        h = idx[~isai[idx]]
        if len(h) == 0:
            continue
        hl = line[h][:, None]

        def mind(sel):
            if len(sel) == 0:
                return np.full(len(h), BIG)
            return np.abs(hl - line[sel][None, :]).min(axis=1)
        out["d_ai"][h] = mind(a)
        out["d_sug"][h] = mind(a[sug[a]])
        out["d_cop"][h] = mind(a[cop[a]])
        out["d_cop_sug"][h] = mind(a[cop[a] & sug[a]])
        out["d_oth"][h] = mind(a[~cop[a]])
        out["d_oth_sug"][h] = mind(a[~cop[a] & sug[a]])
        dist = np.abs(hl - line[a][None, :])
        ai_before[h] = ((dist <= 3) & (tt[a][None, :] < tt[h][:, None])).any(axis=1)
        ai_before_exact[h] = ((dist == 0) & (tt[a][None, :] < tt[h][:, None])).any(axis=1)
        al = line[a][:, None]
        for k in DEFS:
            hk = idx[hm[k][idx]]
            if len(hk):
                out[f"d_h_{k}"][a] = np.abs(al - line[hk][None, :]).min(axis=1)
    res = df.copy()
    for k, v in out.items():
        res[k] = v
    res["ai_before"] = ai_before
    res["ai_before_exact"] = ai_before_exact
    return res


# ------------------------------------------------------------------ 1. 重なり(D21 流)
def overlap(df: pd.DataFrame) -> dict:
    """PR ごとに AI×人間、人間×人間の行 Jaccard と、同じ箱で AI が先の割合を求める。"""
    hm = {k: human_mask(df, k) for k in DEFS}
    x = df.assign(bin=df["line"] // BIN)
    rows = {k: [] for k in DEFS}
    for (repo, pr), g in x.groupby(["repo", "pr"], sort=False):
        ai = g[g["ai"]]
        if ai.empty:
            continue
        ai_locs = set(zip(ai["path"], ai["bin"]))
        ai_first_t = ai.groupby(["path", "bin"])["t"].min().to_dict()
        month = ai["month"].min()
        for k in DEFS:
            hu = g[hm[k][g.index.to_numpy()]]
            if hu.empty:
                continue
            hlocs = set(zip(hu["path"], hu["bin"]))
            both = ai_locs & hlocs
            h_first_t = hu.groupby(["path", "bin"])["t"].min().to_dict()
            timed = [b for b in both if not np.isnan(ai_first_t.get(b, np.nan)) and not np.isnan(h_first_t.get(b, np.nan))]
            af = sum(ai_first_t[b] < h_first_t[b] for b in timed)
            per = [s for s in hu.groupby("login").apply(lambda z: set(zip(z["path"], z["bin"]))) if s]
            hh = None
            if len(per) >= 2:
                v = [len(per[i] & per[j]) / len(per[i] | per[j]) for i in range(len(per)) for j in range(i + 1, len(per))]
                hh = float(np.mean(v))
            rows[k].append({"repo": repo, "pr": pr, "month": month,
                            "ah": len(both) / len(ai_locs | hlocs), "hh": hh,
                            "ai_covered": len(both) / len(ai_locs),
                            "ai_first": af / len(timed) if timed else None})
    res, frames = {}, {}
    for k in DEFS:
        p = pd.DataFrame(rows[k])
        frames[k] = p
        q = p.dropna(subset=["hh"])
        t, pv = stats.ttest_rel(q["ah"], q["hh"])
        res[k] = {"n_prs": int(len(p)), "ai_human_all": float(p["ah"].mean()),
                  "paired_n": int(len(q)), "paired_ai_human": float(q["ah"].mean()),
                  "paired_human_human": float(q["hh"].mean()),
                  "paired_diff": float((q["ah"] - q["hh"]).mean()), "t": float(t), "p": float(pv),
                  "ai_locs_also_human": float(p["ai_covered"].mean()),
                  "ai_first_share_pr_mean": float(p["ai_first"].dropna().mean()),
                  "ai_first_n_prs": int(p["ai_first"].notna().sum())}
    return res, frames


# ------------------------------------------------------------------ 回帰(リポジトリ FE、リポジトリクラスタ)
def fe_trend(d: pd.DataFrame, y: str, x: str, w: str | None = None) -> dict:
    d = d.dropna(subset=[y, x])
    if d["repo"].nunique() < 3 or len(d) < 10:
        return {"n": int(len(d))}
    X = pd.get_dummies(d["repo"], drop_first=True, dtype=float)
    X.insert(0, x, d[x].astype(float))
    X = sm.add_constant(X)
    groups = pd.factorize(d["repo"])[0]
    m = (sm.WLS(d[y].astype(float), X, weights=d[w]) if w else sm.OLS(d[y].astype(float), X))
    r = m.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return {"coef": float(r.params[x]), "se": float(r.bse[x]), "p": float(r.pvalues[x]),
            "n": int(len(d)), "n_repos": int(d["repo"].nunique())}


# ------------------------------------------------------------------ 2. 委譲上限
def ceiling(nd: pd.DataFrame) -> dict:
    res = {}
    for k in DEFS:
        h = nd[human_mask(nd, k)]
        r = {"n_human": int(len(h))}
        for tol in TOLS:
            det, dl = (h["d_ai"] <= tol).mean(), (h["d_sug"] <= tol).mean()
            r[f"tol{tol}"] = {"detected": float(det), "delegable": float(dl),
                              "expression_ratio": float(dl / det) if det else None}
        co = h[h["d_ai"] <= 3]
        r["ai_before_among_colocated"] = float(co["ai_before"].mean())
        r["author_share_among_colocated"] = float(co["is_author"].mean())
        # 同じ行に先に AI コメントがある = AI スレッドへの返信の代理指標
        r["reply_proxy_among_colocated"] = float(co["ai_before_exact"].mean())
        # 返信の代理指標を除いた「独立に同じ箇所を見つけた」上限
        ind = h[~h["ai_before_exact"]]
        r["independent_tol3"] = {"n": int(len(ind)), "detected": float((ind["d_ai"] <= 3).mean()),
                                 "delegable": float((ind["d_sug"] <= 3).mean())}
        r["by_vendor_tol3"] = {
            "copilot_detected": float((h["d_cop"] <= 3).mean()),
            "copilot_delegable": float((h["d_cop_sug"] <= 3).mean()),
            "other_detected": float((h["d_oth"] <= 3).mean()),
            "other_delegable": float((h["d_oth_sug"] <= 3).mean()),
        }
        res[k] = r
    # 定義 (a) のうち作成者が占める割合
    hall = nd[~nd["ai"]]
    res["author_share_of_located_human"] = float(hall["is_author"].mean())
    res["merger_nonauthor_share_of_located_human"] = float((hall["is_merger"] & ~hall["is_author"]).mean())
    return res


def monthly(nd: pd.DataFrame, min_n: int = 300) -> dict:
    """月次系列とトレンド。月は人間コメントの投稿月。
    対象月: 定義 (a) で AI レビューが始まって以降(2025-02〜)かつ人間コメント min_n 件以上。"""
    res = {}
    base = nd[~nd["ai"] & (nd["month"] >= "2025-02")]
    cnt = base.groupby("month").size()
    months = sorted(cnt[cnt >= min_n].index)
    m0 = pd.Period(months[0], "M")
    for k in DEFS:
        h = nd[human_mask(nd, k) & nd["month"].isin(months)].copy()
        for col, dcol in [("det", "d_ai"), ("dl", "d_sug"), ("cop_det", "d_cop"), ("cop_dl", "d_cop_sug"),
                          ("oth_det", "d_oth"), ("oth_dl", "d_oth_sug")]:
            h[col] = (h[dcol] <= 3).astype(float)
        ser = h.groupby("month")[["det", "dl", "cop_det", "cop_dl", "oth_det", "oth_dl"]].mean()
        ser["n"] = h.groupby("month").size()
        cell = h.groupby(["repo", "month"])[["det", "dl", "cop_dl", "oth_dl"]].mean().reset_index()
        cell["n"] = h.groupby(["repo", "month"]).size().to_numpy()
        cell["year"] = [(pd.Period(m, "M") - m0).n / 12 for m in cell["month"]]
        tr = {c: fe_trend(cell, c, "year", "n") for c in ["det", "dl", "cop_dl", "oth_dl"]}
        tr_uw = {c: fe_trend(cell, c, "year") for c in ["det", "dl"]}
        # 全体の月次系列(集計値)に直線を当てたもの(論文の図の読み方に近い)
        sm_ = ser.reset_index().rename(columns={"index": "month"})
        sm_["year"] = [(pd.Period(m, "M") - m0).n / 12 for m in sm_["month"]]
        agg = {}
        for c in ["det", "dl"]:
            rr = sm.OLS(sm_[c], sm.add_constant(sm_["year"])).fit(cov_type="HC1")
            agg[c] = {"coef": float(rr.params["year"]), "p": float(rr.pvalues["year"])}
        # 崩壊前後の比較
        def pick(m, c):
            return float(ser.loc[m, c]) if m in ser.index else None
        last4 = ser.index[-4:]
        res[k] = {"series": {m: {c: float(v) for c, v in r.items()} for m, r in ser.iterrows()},
                  "trend_per_year_repoFE_weighted": tr,
                  "trend_per_year_repoFE_unweighted": tr_uw,
                  "trend_per_year_aggregate_series_HC1": agg,
                  "delegable_2026_04": pick("2026-04", "dl"), "delegable_2026_05": pick("2026-05", "dl"),
                  "delegable_last4_mean": float(ser.loc[last4, "dl"].mean()), "last4": list(last4),
                  "copilot_delegable_2026_04": pick("2026-04", "cop_dl"),
                  "copilot_delegable_last4_mean": float(ser.loc[last4, "cop_dl"].mean()),
                  "other_delegable_2026_04": pick("2026-04", "oth_dl"),
                  "other_delegable_last4_mean": float(ser.loc[last4, "oth_dl"].mean())}
    res["months"] = months
    return res


# ------------------------------------------------------------------ 3. suggestion の trade-off
def tradeoff(nd: pd.DataFrame) -> dict:
    """エージェント内で suggestion あり / なしの対処率と「人間も同じ箇所」率。

    同じ箇所の判定は 2 通り:
      bin26 : D26 と同じ(originalLine 優先の 7 行箱に人間の指摘があるか) ← 論文の数値
      tol3  : 同じ path で ±3 行以内に人間の指摘があるか
    """
    ai = nd[nd["ai"]].copy()
    x = nd.assign(b26=nd["line26"] // BIN)
    for k in DEFS:
        hk = x[human_mask(x, k)]
        locs = set(zip(hk["repo"], hk["pr"], hk["path"], hk["b26"]))
        ai[f"bin26_{k}"] = [(r, p, pa, b) in locs for r, p, pa, b in
                            zip(ai["repo"], ai["pr"], ai["path"], ai["line26"] // BIN)]
        ai[f"tol3_{k}"] = ai[f"d_h_{k}"] <= 3
    res = {}
    counts = ai.dropna(subset=["outdated"]).groupby("login").size()
    agents = [a for a in counts[counts >= 300].index]
    for ag in agents:
        # 論文(D26)と同じく、スレッドと突き合わせられた(= 対処率が定義できる)コメントに限る
        g = ai[(ai["login"] == ag) & ai["outdated"].notna()]
        r = {"n": int(len(g)), "n_sug": int(g["sug"].sum())}
        for s, gs in [("sug", g[g["sug"]]), ("plain", g[~g["sug"]])]:
            e = {"n": int(len(gs)), "acted_on": float(gs["outdated"].mean()) if gs["outdated"].notna().any() else None}
            for k in DEFS:
                e[f"bin26_{k}"] = float(gs[f"bin26_{k}"].mean()) if len(gs) else None
                e[f"tol3_{k}"] = float(gs[f"tol3_{k}"].mean()) if len(gs) else None
            r[s] = e
        # リポジトリ FE で suggestion の効果(非作成者の同意、±3)
        if g["sug"].sum() >= 30 and (~g["sug"]).sum() >= 30:
            gg = g.assign(y=g["tol3_nonauthor"].astype(float), s=g["sug"].astype(float))
            r["sug_effect_nonauthor_tol3_repoFE"] = fe_trend(gg, "y", "s")
            gg2 = g.dropna(subset=["outdated"]).assign(s=lambda z: z["sug"].astype(float))
            r["sug_effect_actedon_repoFE"] = fe_trend(gg2, "outdated", "s")
        res[ag] = r
    return res


# ------------------------------------------------------------------ 4. 収束トレンド
def convergence(frames: dict) -> dict:
    ad = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())["adoption"]
    res = {"adoption": ad}
    for k, p in frames.items():
        q = p.dropna(subset=["hh"]).copy()
        q = q[q["repo"].isin(ad)]
        q["ms"] = [(pd.Period(m, "M") - pd.Period(ad[r], "M")).n for r, m in zip(q["repo"], q["month"])]
        q = q[(q["ms"] >= 0) & (q["ms"] <= 12)]
        q["gap"] = q["hh"] - q["ah"]
        e = {"n_prs": int(len(q))}
        for y in ["ah", "hh", "gap"]:
            e[y] = fe_trend(q, y, "ms")
            e[y]["mean_m0_2"] = float(q.loc[q["ms"] <= 2, y].mean())
            e[y]["mean_m10_12"] = float(q.loc[q["ms"] >= 10, y].mean())
        res[k] = e
    return res


# ------------------------------------------------------------------ 5. RQ1 の人間レビュー提出
def rq1_author(rv: pd.DataFrame) -> dict:
    ad = json.loads((OUT_DIR / "d4_repo_event_study.json").read_text())
    adop, never = ad["adoption"], set(ad["never_treated"])
    rv = rv.copy()
    rv["group"] = np.where(rv["repo"].isin(adop), "treated", np.where(rv["repo"].isin(never), "never", "other"))
    rv["post"] = [m >= adop[r] if r in adop else None for r, m in zip(rv["repo"], rv["month"])]
    res = {"all_human_reviews": int(len(rv)), "author_share_overall": float(rv["is_author"].mean()),
           "author_share_pr_has_ai": float(rv.loc[rv["pr_has_ai"], "is_author"].mean()),
           "author_share_pr_no_ai": float(rv.loc[~rv["pr_has_ai"], "is_author"].mean())}
    t = rv[rv["group"] == "treated"]
    for post in (False, True):
        s = t[t["post"] == post]
        res[f"treated_{'post' if post else 'pre'}"] = {
            "n": int(len(s)), "author_share": float(s["is_author"].mean()),
            "author_share_commented": float(s.loc[s["state"] == "COMMENTED", "is_author"].mean()),
            "nonauthor_n": int((~s["is_author"]).sum())}
    # 対照: never-treated を 2025-06(処置群の導入月の中央付近)で前後に分ける
    nv = rv[rv["group"] == "never"]
    for lab, s in [("never_pre_2025_06", nv[nv["month"] < "2025-06"]), ("never_post_2025_06", nv[nv["month"] >= "2025-06"])]:
        res[lab] = {"n": int(len(s)), "author_share": float(s["is_author"].mean())}
    per = t.groupby(["repo", "post"])["is_author"].mean().unstack()
    res["treated_per_repo_pre_post"] = {r: {"pre": float(v[False]) if pd.notna(v[False]) else None,
                                            "post": float(v[True]) if pd.notna(v[True]) else None}
                                        for r, v in per.iterrows()}
    return res


def main() -> None:
    df, rv = load_all()
    print(f"位置付きコメント {len(df):,} 件(人間 {int((~df.ai).sum()):,} / AI {int(df.ai.sum()):,})")
    print(f"人間レビュー提出 {len(rv):,} 件")
    nd = nearest(df)
    out = {"definitions": {"all": "人間すべて(論文の現行)", "nonauthor": "PR 作成者以外の人間",
                           "nonauthor_nonmerger": "PR 作成者でもマージした人でもない人間"}}
    out["n_located"] = {"human": int((~df.ai).sum()), "ai": int(df.ai.sum())}

    print("\n=== 1. 重なり ===")
    ov, frames = overlap(df)
    out["overlap"] = ov
    for k, v in ov.items():
        print(f"  {k:<22} AI×人 {v['paired_ai_human']:.3f} vs 人×人 {v['paired_human_human']:.3f} "
              f"差 {v['paired_diff']:+.3f} p={v['p']:.3g} n={v['paired_n']:,} / AI先 {v['ai_first_share_pr_mean']:.1%}")

    print("\n=== 2. 委譲上限 ===")
    ce = ceiling(nd)
    out["ceiling"] = ce
    for k in DEFS:
        v = ce[k]
        print(f"  {k:<22} n={v['n_human']:,} " + " ".join(
            f"tol{t}: {v[f'tol{t}']['detected']:.1%}/{v[f'tol{t}']['delegable']:.1%}" for t in TOLS)
            + f"  AI先(同箇所) {v['ai_before_among_colocated']:.1%}")
    mo = monthly(nd)
    out["monthly"] = mo
    for k in DEFS:
        v = mo[k]
        tr = v["trend_per_year_repoFE_weighted"]
        print(f"  {k:<22} 検出 {tr['det']['coef']*100:+.1f}pt/年 p={tr['det']['p']:.3g}  "
              f"委譲可能 {tr['dl']['coef']*100:+.1f}pt/年 p={tr['dl']['p']:.3g}  "
              f"04月 {v['delegable_2026_04']:.1%} 05月 {v['delegable_2026_05']:.1%} 直近4か月 {v['delegable_last4_mean']:.2%}")

    print("\n=== 3. trade-off ===")
    to = tradeoff(nd)
    out["tradeoff"] = to
    for ag, v in to.items():
        s, p = v["sug"], v["plain"]
        f = lambda z: "  -  " if z is None or (isinstance(z, float) and np.isnan(z)) else f"{z:.1%}"
        print(f"  {ag:<30} n={v['n']:,} sug={v['n_sug']:,} 対処 {f(s['acted_on'])}/{f(p['acted_on'])} "
              f"人間(全) {f(s['bin26_all'])}/{f(p['bin26_all'])} 非作成者 {f(s['bin26_nonauthor'])}/{f(p['bin26_nonauthor'])} "
              f"(±3 非作成者 {f(s['tol3_nonauthor'])}/{f(p['tol3_nonauthor'])})")

    print("\n=== 4. 収束 ===")
    cv = convergence(frames)
    out["convergence"] = cv
    for k in DEFS:
        v = cv[k]
        print(f"  {k:<22} n={v['n_prs']:,} AI×人 {v['ah']['coef']:+.4f}/月 p={v['ah']['p']:.3g} "
              f"人×人 {v['hh']['coef']:+.4f} p={v['hh']['p']:.3g} 差 {v['gap']['coef']:+.4f} p={v['gap']['p']:.3g}")

    print("\n=== 5. RQ1 の人間レビューのうち作成者本人 ===")
    rq = rq1_author(rv)
    out["rq1_author_reviews"] = rq
    print(f"  全体 {rq['author_share_overall']:.1%} / 処置群 前 {rq['treated_pre']['author_share']:.1%} "
          f"後 {rq['treated_post']['author_share']:.1%} / never 前 {rq['never_pre_2025_06']['author_share']:.1%} "
          f"後 {rq['never_post_2025_06']['author_share']:.1%}")

    path = OUT_DIR / "d44_author_excluded.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
