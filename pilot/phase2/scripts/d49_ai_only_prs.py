"""D49: AI だけがレビューした PR はどんな PR か(再査読 round 6 の item 1・7)。

定義は d48 と同じ(論文 RQ2 coverage 節):
  - 作成者以外の人間レビュー n_hn > 0   = cov_hn
  - AI だけ ai_only_hn = n_ai > 0 かつ n_hn == 0(レビュー時刻は問わない。d48 と同じ)
  - PR は CODEOWNERS モジュールに帰属し、主パネル(d46 build_sample)のセルに入るもの
  - 主仕様は legacy_bot 作成 PR を除く(CN)。含む版(C)も併記
PR 単位の d48 の列は d48.pr_table_repo をそのまま呼んで作り、raw から追加の列
(マージ時刻・マージした人・変更行数・ファイル数・作成者の過去 PR 数・マージ前の承認)を足す。

比較するグループ(処置リポジトリのみ):
  A  導入後(event_time >= 0)で AI だけにレビューされた PR
  B  導入後で作成者以外の人間にレビューされた PR(B_same: A が 1 件以上ある repo x 月に限定)
  C  導入前で作成者以外の人間レビューが無い PR(C12: 導入前 12 ヶ月に限定)
  D  導入後で作成者以外の人間レビューも AI レビューも無い PR(参考)

イベントスタディ(d48 と同じ module x 作成月、WCR、マージされた PR のみ):
  (a) m_ai_only_hn: マージ PR のうち AI だけ
  (b) m_no_hn: マージ PR のうち作成者以外の人間レビューが無い(AI の有無を問わない)
      m_no_hn_premerge: マージ時点までに作成者以外の人間レビューが無い
      m_no_hn_appr_premerge: マージ前に作成者以外の人間の APPROVED が無い
  照合: 全 PR の cov_hn(CN)が d48 の -0.0205 を再現するか

出力: outputs_full/d49_ai_only_prs.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PILOT_PROCESSED=processed_full \
      PHASE2_OUT=outputs_full PHASE2_FIG=figures_full \
      .venv/bin/python pilot/phase2/scripts/d49_ai_only_prs.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name
from d46_reviewer_requests import build_sample
from d43_causal_robustness import WEBB
from d48_round4 import SPECS6, fit_specs, fmt, pdt, pr_table_repo, share_nulls

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "pilot"))
from loop1_actors import classify, load_config  # noqa: E402

SEED = 20260930
B_WCR = 9999
N_JOBS = int(os.environ.get("D49_JOBS", "6"))
T0 = time.time()
FOLLOWUP_MONTHS = 3


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def extra_repo(repo: str) -> pd.DataFrame | None:
    """raw から d48 に無い列を作る(PR 番号の重複行は最初の行だけ = d48 と同じ)。"""
    rd = RAW_DIR / repo_dir_name(repo)
    idir = INTERIM_DIR / repo_dir_name(repo)
    if not (rd / "prs_full.jsonl").exists() or not (idir / "actors.parquet").exists():
        return None
    cfg = load_config()
    agent_acc = {a.lower() for a in cfg["agent"]["accounts"]}
    legacy_acc = {a.lower() for a in cfg["legacy_bot"]["accounts"]}
    legacy_pat = [re.compile(p) for p in cfg["legacy_bot"].get("patterns", [])]
    actors = pd.read_parquet(idir / "actors.parquet")
    cat_of = dict(zip(actors["actor_login"], actors["category"]))
    merged_by = {}
    if (rd / "merged_by.jsonl").exists():
        with (rd / "merged_by.jsonl").open() as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                merged_by.setdefault(d["number"], d.get("merged_by"))
    rows, seen = [], set()
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
                continue
            seen.add(num)
            if not pr.get("createdAt"):
                continue
            author = (pr.get("author") or {}).get("login")
            files = (pr.get("files") or {}).get("nodes") or []
            created = pdt(pr["createdAt"])
            merged_at = pdt(pr["mergedAt"]) if pr.get("mergedAt") else None
            hn_before = hn_appr_before = ai_before = 0
            for rv in (pr.get("reviews") or {}).get("nodes", []):
                if not rv.get("submittedAt") or not rv.get("author") or merged_at is None:
                    continue
                if pdt(rv["submittedAt"]) > merged_at:
                    continue
                lg = rv["author"]["login"]
                c = cat_of.get(lg, "unknown")
                if c == "human" and lg != author:
                    hn_before += 1
                    hn_appr_before += rv.get("state") == "APPROVED"
                if c == "agent":
                    ai_before += 1
            mb = merged_by.get(num) if merged_at is not None else None
            if merged_at is None:
                mcat = "not_merged"
            elif not mb:
                mcat = "missing"
            elif author and mb == author:
                mcat = "author_self"
            else:
                r = classify(mb, agent_acc, legacy_acc, legacy_pat)
                c = r if r != "human" else cat_of.get(mb, "unknown")
                mcat = {"human": "other_human", "legacy_bot": "machine", "agent": "machine"}.get(c, "other_unknown")
            rows.append({
                "repo": repo, "pr_number": num, "created_ts": created, "author_login": author,
                "merged": merged_at is not None,
                "hours_to_merge": (merged_at - created) / 3600 if merged_at is not None else np.nan,
                "lines": float(sum((f.get("additions") or 0) + (f.get("deletions") or 0) for f in files)),
                "n_files": len(files), "files_capped": len(files) >= 100,
                "merger": mcat, "hn_before_merge": hn_before, "hn_appr_before_merge": hn_appr_before,
                "ai_before_merge": ai_before,
            })
    df = pd.DataFrame(rows).sort_values(["created_ts", "pr_number"])
    df["author_prior_prs"] = df.groupby(df["author_login"].fillna("__none__")).cumcount()
    return df


def quality_links() -> pd.DataFrame:
    """d35(revert)・d42(fix-inducing)の PR 単位フラグを、それぞれの関数で作り直して結合する。"""
    from d35_revert_outcome import build_revert_links
    from d42_fix_inducing import build as build_fix
    (rv, _), (fx, _) = Parallel(n_jobs=2)(delayed(f)() for f in [build_revert_links, build_fix])
    rv = rv.drop_duplicates(["repo", "pr_number"])[["repo", "pr_number", "was_reverted", "is_revert_pr"]]
    fx = fx.drop_duplicates(["repo", "pr_number"])[["repo", "pr_number", "fix_induced"]]
    return rv.merge(fx, on=["repo", "pr_number"], how="outer")


def q(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) == 0:
        return {"n": 0}
    return {"n": int(len(s)), "median": float(s.median()), "p25": float(s.quantile(.25)),
            "p75": float(s.quantile(.75)), "mean": float(s.mean())}


def describe(g: pd.DataFrame, end_idx: int) -> dict:
    m = g[g["merged"]]
    out = {"n": int(len(g)), "n_repos": int(g["repo"].nunique()),
           "lines": q(g["lines"]), "files": q(g["n_files"].astype(float)),
           "share_files_capped_100": float(g["files_capped"].mean()),
           "author_cat": {k: float(v) for k, v in g["author_cat"].value_counts(normalize=True).items()},
           "author_first_pr_in_data": float((g["author_prior_prs"] == 0).mean()),
           "author_le5_prior_prs": float((g["author_prior_prs"] <= 5).mean()),
           "author_prior_prs": q(g["author_prior_prs"].astype(float)),
           "merged_share": float(g["merged"].mean()), "n_merged": int(len(m))}
    if len(m):
        out["merger_among_merged"] = {k: float(v) for k, v in m["merger"].value_counts(normalize=True).items()}
        out["hn_approval_before_merge"] = float((m["hn_appr_before_merge"] > 0).mean())
        out["hn_review_before_merge"] = float((m["hn_before_merge"] > 0).mean())
        out["ai_review_before_merge"] = float((m["ai_before_merge"] > 0).mean())
        out["hours_to_merge"] = q(m["hours_to_merge"])
        mm = m[m["month_idx"] <= end_idx - FOLLOWUP_MONTHS]
        out["followup"] = {"n_merged_created_3m_before_end": int(len(mm)),
                           "was_reverted": float(mm["was_reverted"].mean()) if len(mm) else None,
                           "fix_induced": float(mm["fix_induced"].mean()) if len(mm) else None}
    return out


def fmt_desc(name: str, d: dict) -> str:
    if d["n"] == 0:
        return f"{name}: n=0"
    s = (f"{name}: n={d['n']} repos={d['n_repos']} | lines med {d['lines']['median']:.0f} "
         f"[{d['lines']['p25']:.0f},{d['lines']['p75']:.0f}] files med {d['files']['median']:.0f} "
         f"[{d['files']['p25']:.0f},{d['files']['p75']:.0f}] cap100 {d['share_files_capped_100']:.3f} | author "
         + " ".join(f"{k} {v:.3f}" for k, v in d["author_cat"].items())
         + f" | first-PR {d['author_first_pr_in_data']:.3f} <=5prior {d['author_le5_prior_prs']:.3f} "
           f"| merged {d['merged_share']:.3f} (n {d['n_merged']})")
    if d["n_merged"]:
        s += (" | merger " + " ".join(f"{k} {v:.3f}" for k, v in d["merger_among_merged"].items())
              + f" | hn-approval-before-merge {d['hn_approval_before_merge']:.3f} hn-review-before-merge "
                f"{d['hn_review_before_merge']:.3f} ai-before-merge {d['ai_review_before_merge']:.3f} | "
                f"hours-to-merge med {d['hours_to_merge']['median']:.1f} [{d['hours_to_merge']['p25']:.1f},"
                f"{d['hours_to_merge']['p75']:.1f}]")
        f = d["followup"]
        if f["n_merged_created_3m_before_end"]:
            s += (f" | reverted {f['was_reverted']:.4f} fix-induced {f['fix_induced']:.4f} "
                  f"(n {f['n_merged_created_3m_before_end']})")
    return s


def merged_panel(p: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    m = p[p["merged"]].copy()
    m["m_ai_only_hn"] = ((m["n_hn"] == 0) & (m["n_ai"] > 0)).astype(float)
    m["m_no_hn"] = (m["n_hn"] == 0).astype(float)
    m["m_no_hn_premerge"] = (m["hn_before_merge"] == 0).astype(float)
    m["m_no_hn_appr_premerge"] = (m["hn_appr_before_merge"] == 0).astype(float)
    m["m_no_hn_no_ai"] = ((m["n_hn"] == 0) & (m["n_ai"] == 0)).astype(float)
    ys = ["m_ai_only_hn", "m_no_hn", "m_no_hn_premerge", "m_no_hn_appr_premerge", "m_no_hn_no_ai"]
    agg = m.groupby(["module_id_global", "month"])[ys].mean()
    agg = agg.join(m.groupby(["module_id_global", "month"]).size().rename("n_merged")).reset_index()
    return agg.merge(keys, on=["module_id_global", "month"], how="inner")


def main() -> None:
    for v in ["PILOT_RAW", "PILOT_INTERIM", "PILOT_PROCESSED", "PHASE2_OUT", "PHASE2_FIG"]:
        log(f"env {v}={os.environ.get(v)}")
    df, bal, adoption, never, excl, months = build_sample()
    assert len(bal) == 412997
    log(f"sample: {len(bal)} module-months; adoption {len(adoption)}/{len(never)}/{len(excl)}")
    end_idx = len(months) - 1
    repos = load_repos()
    res = Parallel(n_jobs=N_JOBS)(delayed(pr_table_repo)(r) for r in repos)
    pr = pd.concat([x[0] for x in res if x is not None], ignore_index=True)
    log(f"d48 PR table rebuilt: {len(pr)} PRs")
    old = pd.read_parquet(OUT_DIR / "d48_pr_level.parquet")
    chk = old.merge(pr, on=["repo", "pr_number"], suffixes=("_old", ""))
    same = {c: float((chk[c + "_old"].fillna(-1).astype(str) == chk[c].fillna(-1).astype(str)).mean())
            for c in ["n_hn", "n_ai", "author_cat", "module_id"]}
    log(f"vs saved d48_pr_level.parquet: n {len(old)} vs {len(pr)}; agreement {same}")
    ex = pd.concat([x for x in Parallel(n_jobs=N_JOBS)(delayed(extra_repo)(r) for r in repos) if x is not None],
                   ignore_index=True)
    log(f"extra columns: {len(ex)} PRs")
    ql = quality_links()
    log(f"revert / fix-inducing links: {len(ql)} PRs")
    pr = pr.merge(ex, on=["repo", "pr_number"], how="left").merge(ql, on=["repo", "pr_number"], how="left")
    for c in ["was_reverted", "fix_induced"]:
        pr[c] = pr[c].astype(float)

    keys = bal[["module_id_global", "month", "repo", "month_idx", "g", "event_time"]]
    p = pr[pr["module_id"].notna()].copy()
    p["module_id_global"] = p["repo"] + "::" + p["module_id"]
    p = p.merge(keys.drop(columns="repo"), on=["module_id_global", "month"], how="inner")
    log(f"PRs in primary cells: {len(p)}")

    out: dict = {"spec": {"definitions": "d48: n_hn = reviews by humans != PR author (any time); ai_only_hn = n_ai>0 & "
                                         "n_hn==0; merger: author_self / other_human / machine(legacy_bot|agent) / "
                                         "other_unknown (login not in actors) / missing (no merged_by row)",
                          "caps": "files(first:100), reviews(first:30) in collection; lines = sum over collected files",
                          "author_prior_prs": "PRs by same login earlier in this repo's collected window (left-censored)",
                          "followup": f"revert (d35 build_revert_links) and fix-induced (d42 build) among merged PRs "
                                      f"created >= {FOLLOWUP_MONTHS} months before last month",
                          "B_wcr": B_WCR, "seed": SEED, "check_vs_saved_d48_parquet": same}}

    W = np.random.default_rng(SEED).choice(WEBB, size=(B_WCR, 64))
    jobs = []
    for lab, sub in [("nobot", p[p["author_cat"] != "legacy_bot"]), ("withbot", p)]:
        t = sub[sub["g"].notna()]
        post, pre = t[t["event_time"] >= 0], t[t["event_time"] < 0]
        A = post[(post["n_hn"] == 0) & (post["n_ai"] > 0)]
        am = set(zip(A["repo"], A["month"]))
        B = post[post["n_hn"] > 0]
        B_same = B[[x in am for x in zip(B["repo"], B["month"])]]
        C = pre[pre["n_hn"] == 0]
        C12 = C[C["event_time"] >= -12]
        D = post[(post["n_hn"] == 0) & (post["n_ai"] == 0)]
        groups = {"A_post_ai_only": A, "B_post_hn_reviewed": B, "B_same_repo_month": B_same,
                  "C_pre_no_hn": C, "C12_pre12_no_hn": C12, "D_post_no_hn_no_ai": D,
                  "ALL_post": post, "ALL_pre": pre}
        desc = {k: describe(v, end_idx) for k, v in groups.items()}
        for k in groups:
            log(f"[{lab}] " + fmt_desc(k, desc[k]))
        # A の作成者 x マージした人
        if len(A):
            ct = pd.crosstab(A["author_cat"], A["merger"])
            desc["A_author_x_merger"] = {str(i): {str(c): int(ct.loc[i, c]) for c in ct.columns} for i in ct.index}
            log(f"[{lab}] A author x merger: {desc['A_author_x_merger']}")
            desc["A_by_repo"] = {r: int(n) for r, n in A["repo"].value_counts().items()}
            log(f"[{lab}] A by repo: {desc['A_by_repo']}")
        # 生の割合(PR 単位、処置リポジトリ)
        raw = {}
        for nm, s in [("pre", pre), ("post", post)]:
            sm = s[s["merged"]]
            raw[nm] = {"n": int(len(s)), "share_no_hn_all": float((s["n_hn"] == 0).mean()),
                       "share_ai_only_all": float(((s["n_hn"] == 0) & (s["n_ai"] > 0)).mean()),
                       "n_merged": int(len(sm)), "merged_share_no_hn": float((sm["n_hn"] == 0).mean()),
                       "merged_share_ai_only": float(((sm["n_hn"] == 0) & (sm["n_ai"] > 0)).mean()),
                       "merged_share_no_hn_premerge": float((sm["hn_before_merge"] == 0).mean()),
                       "merged_share_no_hn_appr_premerge": float((sm["hn_appr_before_merge"] == 0).mean())}
        desc["raw_pr_level_treated"] = raw
        log(f"[{lab}] raw PR-level treated: {raw}")
        out[f"groups_{lab}"] = desc

        M = merged_panel(sub, keys)
        log(f"[{lab}] merged panel: {len(M)} cells, {int(M['n_merged'].sum())} merged PRs")
        for y in ["m_ai_only_hn", "m_no_hn", "m_no_hn_premerge", "m_no_hn_appr_premerge", "m_no_hn_no_ai"]:
            base = float(M[M["g"].notna() & (M["event_time"] < 0)][y].mean())
            jobs.append((f"{lab}:{y}", M, y, share_nulls(base, "increase"), base))
        # 照合用: 全 PR の cov_hn / no_hn(d48 の CN / C を再現するか)
        Ca = sub.assign(cov_hn=(sub["n_hn"] > 0).astype(float), no_hn=(sub["n_hn"] == 0).astype(float))
        Ca = Ca.groupby(["module_id_global", "month"])[["cov_hn", "no_hn"]].mean().reset_index()
        Ca = Ca.merge(keys, on=["module_id_global", "month"], how="inner")
        base = float(Ca[Ca["g"].notna() & (Ca["event_time"] < 0)]["cov_hn"].mean())
        jobs.append((f"{lab}:allPR_cov_hn_check", Ca, "cov_hn", share_nulls(base, "decline"), base))
        base = float(Ca[Ca["g"].notna() & (Ca["event_time"] < 0)]["no_hn"].mean())
        jobs.append((f"{lab}:allPR_no_hn", Ca, "no_hn", share_nulls(base, "increase"), base))

    log(f"{len(jobs)} event-study jobs")

    def run(name, d, y, nulls, base):
        direction = "decline" if y == "cov_hn" else "increase"
        r = fit_specs(d, y, W, nulls, direction, specs=SPECS6)
        r["treated_pre_mean"] = base
        return name, r

    res = dict(Parallel(n_jobs=N_JOBS)(delayed(run)(*j) for j in jobs))
    for name, *_ in jobs:
        r = res[name]
        keysn = [k[len("worst_p_wcr_"):] for k in r if k.startswith("worst_p_wcr_")]
        log(f"[ES {name}] pre mean {r['treated_pre_mean']:.4f} | " + fmt(name, r, keysn))
    out["event_study"] = res

    # 派生: 論文の -0.021(CN cov_hn k=+3)を「作成者以外の人間レビューが無い PR」の割合に換算
    k3 = res["nobot:allPR_cov_hn_check"]["specs"]["es_k_p3"]["coef"]
    base_cov = res["nobot:allPR_cov_hn_check"]["treated_pre_mean"]
    out["derived"] = {"cov_hn_k3": k3, "cov_hn_pre_mean": base_cov, "no_hn_pre_mean": 1 - base_cov,
                      "relative_increase_in_no_hn_share": -k3 / (1 - base_cov),
                      "relative_decline_in_cov_hn": k3 / base_cov}
    log(f"derived: {out['derived']}")
    path = OUT_DIR / "d49_ai_only_prs.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"-> {path}")


if __name__ == "__main__":
    main()
