"""D45: 判断の境界を1つの母集団で数え直す(査読会議 M4/M5/F3/F4 への対応, B3)。

これまでの RQ3 の数値は母集団が混ざっていた。
  - D24 は「module に紐づいた PR」(383,918 件の承認あり人間作成 PR)
  - D31 は「prs_full の全 PR」(640,005 件)
  - 表 exceptions の 208 は AI のみ、表 merge の 318 は CI ボット込み
さらに査読で次の問題が見つかった。
  (i)   大半の AI 製品はそもそも APPROVED を出さない(製品の機能の問題であって制度の排除ではない可能性)
  (ii)  「640,005 分の 5」は分母が不適切。「AI が承認し人間が承認しなかった PR のうち何件がマージされたか」が正しい
  (iii) マージボット(craig, istio-testing ...)が分類漏れで、機械マージが 1 桁過小だった
  (iv)  コメントコマンド(/lgtm, bors r+)による承認は APPROVED として記録されない
  (v)   resolvedBy の自己解決判定に [bot] サフィックスのバグがあった(d38 で修正済み)

本スクリプトは分類を直したあとの生データから、**1つの母集団**で全部を数え直す。

母集団:
  主: prs_full.jsonl の全 PR のうち、作成者が human に分類されるもの(=「全人間作成 PR」)
      理由: 承認・マージは module 帰属と無関係に起きる。module に紐づかない PR を落とすと
      CODEOWNERS の無いディレクトリ(docs など)が系統的に抜ける
  副: そのうち events.parquet の G 工程で module に紐づくもの(D24 と同じ定義。整合確認用)

分類:
  login はまず yaml の規則(loop1_actors.classify)で判定し、機械と出たらそれを優先。
  規則で human のときは actors.parquet の category を使い、載っていなければ unknown。
  (D30 と同じ。マージしかしないアカウントは actors に載らないため)

出力: outputs_full/d45_verdict_boundary.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d45_verdict_boundary.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "pilot"))
from loop1_actors import classify, load_config, normalize_login  # noqa: E402

MACHINE = ("agent", "legacy_bot")
REVIEW_CAP = 30  # 収集時の reviews(first:30)
FLAG_REPOS = ["istio/istio", "cockroachdb/cockroach", "pytorch/pytorch"]
# yaml からまだ漏れている機械アカウント(本スクリプトで発見。yaml は触らず感度分析として付け替える)
#   azure-sdk: dotnet/docs で docs/azure/includes/dotnet-*.md を自動更新する Azure SDK の自動化アカウント。
#              dotnet-policy-service が1分以内に承認し、人間の承認なしでマージされる(800件中798件)
EXTRA_MACHINE = {"azure-sdk": "legacy_bot"}


def make_classifier(extra: dict[str, str] | None = None):
    extra = {k.lower(): v for k, v in (extra or {}).items()}
    cfg = load_config()
    agent_acc = {a.lower() for a in cfg["agent"]["accounts"]}
    legacy_acc = {a.lower() for a in cfg["legacy_bot"]["accounts"]}
    pats = [re.compile(p) for p in cfg["legacy_bot"].get("patterns", [])]

    def cat(login: str | None, cat_of: dict[str, str]) -> str | None:
        if not login:
            return None
        if normalize_login(login).lower() in extra:
            return extra[normalize_login(login).lower()]
        r = classify(login, agent_acc, legacy_acc, pats)
        if r != "human":
            return r
        return cat_of.get(login) or cat_of.get(normalize_login(login)) or "unknown"

    return cat


def ts(s: str | None) -> pd.Timestamp | None:
    return pd.Timestamp(s) if s else None


# ---------------------------------------------------------------------------
# 1. PR 単位の表と、機械レビュアーの集計
# ---------------------------------------------------------------------------
def build(cat) -> tuple:
    pr_rows = []
    # 機械レビュアー login -> 集計
    rv_stats: dict[str, dict] = defaultdict(lambda: {
        "category": None, "reviews": 0, "APPROVED": 0, "CHANGES_REQUESTED": 0,
        "COMMENTED": 0, "DISMISSED": 0, "appr_on_human_pr": 0, "appr_on_bot_pr": 0,
        "cr_on_human_pr": 0, "first_appr_human_pr": 0,
        "repos_reviewing": Counter(), "repos_approving": Counter()})
    monthly = []   # agent review の月別(トレンド用)
    merge_rows = []
    actor_tables = {}

    for repo in load_repos():
        d = RAW_DIR / repo_dir_name(repo)
        ip = INTERIM_DIR / repo_dir_name(repo)
        if not ((d / "prs_full.jsonl").exists() and (ip / "actors.parquet").exists()):
            continue
        a = pd.read_parquet(ip / "actors.parquet")
        actor_tables[repo] = a
        cat_of = dict(zip(a["actor_login"], a["category"]))
        ev = pd.read_parquet(ip / "events.parquet", columns=["stage", "pr_number", "module_id"])
        g = ev[(ev["stage"] == "G") & ev["module_id"].notna()]
        module_prs = set(g["pr_number"].tolist())

        seen = set()
        with (d / "prs_full.jsonl").open() as f:
            for line in f:
                try:
                    pr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                num = pr.get("number")
                if num in seen or not pr.get("createdAt"):
                    continue
                seen.add(num)
                author = (pr.get("author") or {}).get("login")
                acat = cat(author, cat_of) or "none"
                nodes = (pr.get("reviews") or {}).get("nodes", []) or []
                merged_at = ts(pr.get("mergedAt"))
                appr = {"human": [], "agent": [], "legacy_bot": [], "unknown": []}
                first_t, first_login, first_cat = None, None, None
                for rv in nodes:
                    au = rv.get("author")
                    if not au:
                        continue
                    lg, st = au["login"], rv.get("state")
                    c = cat(lg, cat_of)
                    t = ts(rv.get("submittedAt"))
                    if c in MACHINE:
                        s = rv_stats[lg]
                        s["category"] = c
                        s["reviews"] += 1
                        if st in ("APPROVED", "CHANGES_REQUESTED", "COMMENTED", "DISMISSED"):
                            s[st] += 1
                        s["repos_reviewing"][repo] += 1
                        if st == "APPROVED":
                            s["repos_approving"][repo] += 1
                            s["appr_on_human_pr" if acat == "human" else "appr_on_bot_pr"] += 1
                        if st == "CHANGES_REQUESTED" and acat == "human":
                            s["cr_on_human_pr"] += 1
                        if c == "agent" and t is not None:
                            monthly.append((t.strftime("%Y-%m"), lg, st, repo))
                    if st == "APPROVED" and t is not None:
                        appr.setdefault(c, []).append(t)
                        if first_t is None or t < first_t:
                            first_t, first_login, first_cat = t, lg, c
                files = (pr.get("files") or {}).get("nodes", []) or []
                if acat == "human" and first_cat in MACHINE:
                    rv_stats[first_login]["first_appr_human_pr"] += 1
                hum_before = (merged_at is not None
                              and any(t <= merged_at for t in appr["human"]))
                pr_rows.append({
                    "repo": repo, "pr_number": num, "author": author, "author_cat": acat,
                    "month": pr["createdAt"][:7],
                    "module_matched": num in module_prs,
                    "merged": merged_at is not None,
                    "n_reviews": len(nodes),
                    "reviews_capped": len(nodes) >= REVIEW_CAP,
                    "any_appr": first_t is not None,
                    "first_approver": first_login, "first_approver_cat": first_cat,
                    "has_human_appr": bool(appr["human"]),
                    "has_human_appr_before_merge": hum_before,
                    "has_agent_appr": bool(appr["agent"]),
                    "has_legacy_appr": bool(appr["legacy_bot"]),
                    "has_unknown_appr": bool(appr.get("unknown")),
                    "pr_lines": sum(x.get("additions", 0) + x.get("deletions", 0) for x in files),
                    "pr_files": len(files),
                })

        mp = d / "merged_by.jsonl"
        if mp.exists():
            with mp.open() as f:
                for line in f:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    mb = m.get("merged_by")
                    merge_rows.append({"repo": repo, "pr_number": m["number"], "merged_by": mb,
                                       "merger_cat": cat(mb, cat_of) or "none"})

    pr = pd.DataFrame(pr_rows)
    mb = pd.DataFrame(merge_rows).drop_duplicates(["repo", "pr_number"])
    mon = pd.DataFrame(monthly, columns=["month", "login", "state", "repo"])
    return pr, dict(rv_stats), mb, mon, actor_tables


# ---------------------------------------------------------------------------
# 2. 例外表(同じ母集団で一貫した数値)
# ---------------------------------------------------------------------------
def exceptions(hp: pd.DataFrame) -> dict:
    ap = hp[hp["any_appr"]]
    fa = ap["first_approver_cat"].value_counts()
    out = {
        "n_human_authored": int(len(hp)),
        "merge_rate_all": float(hp["merged"].mean()),
        "n_with_any_approval": int(len(ap)),
        "first_approver_cat": {k: int(v) for k, v in fa.items()},
    }
    mf = ap[ap["first_approver_cat"].isin(MACHINE)]
    rows = {}
    for lab, s in [("machine (AI+CI)", mf),
                   ("AI agent", mf[mf["first_approver_cat"] == "agent"]),
                   ("CI bot", mf[mf["first_approver_cat"] == "legacy_bot"])]:
        noh = s[~s["has_human_appr"]]
        rows[lab] = {
            "first_approver": int(len(s)),
            "share_of_approved": len(s) / len(ap) if len(ap) else 0.0,
            "human_also_approved": int(s["has_human_appr"].sum()),
            "no_human_approval": int(len(noh)),
            "no_human_approval_merged": int(noh["merged"].sum()),
            "no_human_approval_merge_rate": float(noh["merged"].mean()) if len(noh) else None,
            "median_lines": float(s["pr_lines"].median()) if len(s) else None,
            "median_files": float(s["pr_files"].median()) if len(s) else None,
            "by_login": {k: int(v) for k, v in s["first_approver"].value_counts().items()},
            "by_repo": {k: int(v) for k, v in s["repo"].value_counts().items()},
            "no_human_by_repo_merged": {r: [int(len(g)), int(g["merged"].sum())]
                                        for r, g in noh.groupby("repo")},
        }
    out["machine_first"] = rows
    hf = ap[ap["first_approver_cat"] == "human"]
    out["human_first_median_lines"] = float(hf["pr_lines"].median())
    out["human_first_median_files"] = float(hf["pr_files"].median())

    # 条件付きの比較(「承認されたら人間の承認なしで通るか」)
    def rate(s):
        return {"n": int(len(s)), "merged": int(s["merged"].sum()),
                "merge_rate": float(s["merged"].mean()) if len(s) else None}
    no_appr = hp[~hp["any_appr"]]
    out["conditional"] = {
        "no_approval_at_all": rate(no_appr),
        "no_approval_at_all_excl_pytorch": rate(no_appr[no_appr["repo"] != "pytorch/pytorch"]),
        "ai_approved_no_human": rate(hp[hp["has_agent_appr"] & ~hp["has_human_appr"]]),
        "ai_only_approval": rate(hp[hp["has_agent_appr"] & ~hp["has_human_appr"]
                                    & ~hp["has_legacy_appr"]]),
        "ci_approved_no_human": rate(hp[hp["has_legacy_appr"] & ~hp["has_human_appr"]]),
        "human_approved": rate(hp[hp["has_human_appr"]]),
        "any_ai_approval": rate(hp[hp["has_agent_appr"]]),
    }
    s = hp[hp["has_agent_appr"] & ~hp["has_human_appr"]]
    out["ai_approved_no_human_by_repo"] = {r: [int(len(g)), int(g["merged"].sum())]
                                           for r, g in s.groupby("repo")}
    # 同じリポジトリの「承認なし」との比較(リポジトリ構成の違いを除く)
    same = no_appr[no_appr["repo"].isin(s["repo"].unique())]
    out["no_approval_same_repos"] = rate(same)
    out["no_approval_by_repo_for_ai_repos"] = {
        r: rate(g) for r, g in same.groupby("repo")}
    out["reviews_capped_share"] = float(hp["reviews_capped"].mean())
    # 分類漏れの目安: 機械が最初に承認し人間が承認しなかった PR の作成者
    sus = mf[~mf["has_human_appr"]]
    out["machine_first_no_human_top_authors"] = {
        f"{r}:{a}": int(n) for (r, a), n in
        sus.groupby(["repo", "author"]).size().sort_values(ascending=False).head(10).items()}
    return out


# ---------------------------------------------------------------------------
# 3. マージ権限
# ---------------------------------------------------------------------------
def merge_authority(hp: pd.DataFrame, mb: pd.DataFrame, actor_tables: dict) -> dict:
    # ★merged_by.jsonl(D29)は module に紐づいた PR についてしか収集していない。
    #   紐づかない PR を混ぜると mergedBy 欠損が機械マージの分母だけを膨らませるので、
    #   mergedBy が取れた PR に限って数える(カバレッジは別に報告する)。
    allm = hp[hp["merged"]].merge(mb, on=["repo", "pr_number"], how="left")
    coverage = float(allm["merger_cat"].notna().mean())
    # APPROVED を一切持たないマージの割合は mergedBy 不要なので、全マージで先に出す
    per_all = {r: {"merged": int(len(g)), "no_approved_share": float((~g["any_appr"]).mean())}
               for r, g in allm.groupby("repo")}
    m = allm[allm["merger_cat"].notna()].copy()
    mach = m[m["merger_cat"].isin(MACHINE)]
    fully = mach[~mach["has_human_appr"]]
    out = {
        "merged_by_coverage": coverage,
        "n_merged_human_authored_all": int(len(allm)),
        "no_approved_share_all_merged_by_repo": per_all,
        "n_merged_human_authored": int(len(m)),
        "merger_cat": {k: int(v) for k, v in m["merger_cat"].value_counts().items()},
        "machine_merges": int(len(mach)),
        "machine_merge_share": len(mach) / len(m),
        "machine_merge_with_human_appr": int(mach["has_human_appr"].sum()),
        "machine_merge_with_human_appr_share": float(mach["has_human_appr"].mean()),
        "machine_merge_with_human_appr_before_merge_share":
            float(mach["has_human_appr_before_merge"].mean()),
        "fully_machine": int(len(fully)),
        "fully_machine_share_of_merged": len(fully) / len(m),
        "fully_machine_by_repo": {k: int(v) for k, v in fully["repo"].value_counts().items()},
        "fully_machine_any_approval": int(fully["any_appr"].sum()),
        "fully_machine_zero_reviews": int((fully["n_reviews"] == 0).sum()),
        "by_merger_login": {},
    }
    for lg, g in mach.groupby("merged_by"):
        out["by_merger_login"][lg] = {
            "n": int(len(g)), "repos": sorted(g["repo"].unique().tolist()),
            "with_human_appr": float(g["has_human_appr"].mean()),
            "no_human_appr": int((~g["has_human_appr"]).sum())}
    out["by_merger_login"] = dict(sorted(out["by_merger_login"].items(),
                                         key=lambda kv: -kv[1]["n"]))
    # 全 PR(作成者を問わない)の機械マージ(査読の「約3.5万」との照合用)
    out["machine_merges_all_authors"] = int(mb["merger_cat"].isin(MACHINE).sum())

    # リポジトリ別: APPROVED を一切持たないマージの割合(コメントコマンド承認の目安)
    per = []
    for r, g in m.groupby("repo"):
        per.append({"repo": r, "merged": int(len(g)),
                    "no_approved_share": float((~g["any_appr"]).mean()),
                    "no_human_approved_share": float((~g["has_human_appr"]).mean()),
                    "machine_merge_share": float(g["merger_cat"].isin(MACHINE).mean())})
    per = sorted(per, key=lambda x: -x["no_approved_share"])
    out["per_repo"] = per
    out["flag_repos"] = {x["repo"]: x for x in per if x["repo"] in FLAG_REPOS}
    pt = hp[hp["repo"] == "pytorch/pytorch"]
    out["pytorch_note"] = {"human_prs": int(len(pt)), "merged_state": int(pt["merged"].sum())}

    # 分類漏れの候補: レビューも PR 作成もしないのにマージの5%以上を担う human/unknown
    cand = []
    for r, g in mb.groupby("repo"):
        a = actor_tables.get(r)
        info = {} if a is None else a.set_index("actor_login")[
            ["n_reviews_submitted", "n_pr_authored"]].to_dict("index")
        vc = g[g["merger_cat"].isin(["human", "unknown"])]["merged_by"].value_counts()
        for lg, n in vc.items():
            sh = n / len(g)
            i = info.get(lg, {"n_reviews_submitted": 0, "n_pr_authored": 0})
            if sh >= 0.05 and i["n_reviews_submitted"] == 0 and i["n_pr_authored"] == 0:
                cand.append({"repo": r, "login": lg, "merges": int(n), "share": float(sh)})
    out["unlisted_merger_candidates"] = cand
    return out


# ---------------------------------------------------------------------------
# 4. スレッドの解決者
# ---------------------------------------------------------------------------
def threads(cat) -> dict:
    rows = []
    for repo in load_repos():
        p = RAW_DIR / repo_dir_name(repo) / "thread_resolvers.jsonl"
        ap = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not (p.exists() and ap.exists()):
            continue
        a = pd.read_parquet(ap)
        cat_of = dict(zip(a["actor_login"], a["category"]))
        seen = set()
        with p.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d["number"] in seen:
                    continue
                seen.add(d["number"])
                for th in d.get("threads") or []:
                    st = th.get("starter")
                    if not st:
                        continue
                    sc = cat(st, cat_of)
                    if sc not in ("human", "agent"):
                        continue
                    rb = th.get("resolvedBy")
                    rc = cat(rb, cat_of) if rb else None
                    self_res = bool(rb) and normalize_login(rb).lower() == normalize_login(st).lower()
                    res = bool(th.get("isResolved"))
                    if not res:
                        who = "unresolved"
                    elif rb is None:
                        who = "resolved_no_actor"
                    elif self_res and sc == "agent":
                        who = "self"
                    elif self_res:
                        who = "human"  # 人間が自分のスレッドを閉じたのは「人間が解決」
                    elif rc == "agent":
                        who = "other_agent"
                    elif rc == "legacy_bot":
                        who = "ci_bot"
                    elif rc == "human":
                        who = "human"
                    else:
                        who = "unknown"
                    rows.append({"repo": repo, "pr_number": d["number"], "starter": st,
                                 "starter_cat": sc, "is_ai": float(sc == "agent"),
                                 "line_live": th.get("line") is not None,
                                 "resolved": float(res), "outdated": float(bool(th.get("isOutdated"))),
                                 "who": who})
    df = pd.DataFrame(rows)
    df["human_resolved"] = (df["who"] == "human").astype(float)
    df["human_or_self_fixed"] = ((df["who"] == "human")
                                 | ((df["who"] == "self") & (df["outdated"] == 1))).astype(float)
    df["pid"] = df["repo"] + "#" + df["pr_number"].astype(str)
    out: dict = {"n_threads": int(len(df))}
    res = df[df["resolved"] == 1]
    tab = {}
    for sc, g in res.groupby("starter_cat"):
        vc = g["who"].value_counts()
        tab[sc] = {"n_resolved": int(len(g)),
                   **{k: {"n": int(v), "share": float(v / len(g))} for k, v in vc.items()}}
    out["who_resolves_all"] = tab
    ai = res[res["is_ai"] == 1]
    per = {}
    for lg, g in ai.groupby("starter"):
        vc = g["who"].value_counts(normalize=True)
        sf = g[g["who"] == "self"]
        hu = g[g["who"] == "human"]
        per[lg] = {"n_resolved": int(len(g)),
                   **{k: float(v) for k, v in vc.items()},
                   "outdated_if_self": float(sf["outdated"].mean()) if len(sf) else None,
                   "outdated_if_human": float(hu["outdated"].mean()) if len(hu) else None,
                   "repos_self": {k: int(v) for k, v in sf["repo"].value_counts().items()}}
    out["by_agent"] = dict(sorted(per.items(), key=lambda kv: -kv[1]["n_resolved"]))
    machine_closed = ai[ai["who"].isin(["self", "other_agent", "ci_bot"])]
    out["ai_threads_machine_closed"] = int(len(machine_closed))
    out["ai_threads_machine_closed_self_share"] = float(
        (machine_closed["who"] == "self").mean()) if len(machine_closed) else None

    # PR 固定効果(line がまだ有効なスレッド。D36/D38 と同じ標本条件)
    live = df[df["line_live"]]
    mixed = live.groupby("pid")["is_ai"].transform("nunique") > 1
    dm = live[mixed].copy()
    fe = {}
    for y in ["resolved", "human_resolved", "human_or_self_fixed"]:
        d2 = dm.copy()
        for c in [y, "is_ai"]:
            d2[c + "_dm"] = d2[c] - d2.groupby("pid")[c].transform("mean")
        m = smf.ols(f"{y}_dm ~ is_ai_dm - 1", data=d2).fit(
            cov_type="cluster", cov_kwds={"groups": d2["repo"]})
        base = float(live[live["is_ai"] == 0][y].mean())
        c, se = float(m.params["is_ai_dm"]), float(m.bse["is_ai_dm"])
        eq = {}
        for rel in (0.05, 0.10, 0.20):
            dl = rel * base
            eq[f"{rel:.0%}"] = {"p_upper": float(1 - norm.cdf((dl - c) / se)),
                                "p_lower": float(norm.cdf((-dl - c) / se))}
        fe[y] = {"coef": c, "se": se, "p": float(m.pvalues["is_ai_dm"]), "n": int(m.nobs),
                 "n_prs": int(d2["pid"].nunique()), "n_repos": int(d2["repo"].nunique()),
                 "human_base": base, "equivalence": eq}
    out["pr_fe_live_line"] = fe
    return out


# ---------------------------------------------------------------------------
# 5. 「境界は動いている」
# ---------------------------------------------------------------------------
def trend(mon: pd.DataFrame) -> dict:
    mon = mon.copy()
    mon["verdict"] = mon["state"].isin(["APPROVED", "CHANGES_REQUESTED"]).astype(float)
    mon["is_claude"] = (mon["login"] == "claude").astype(float)
    g = mon.groupby("month").agg(n=("verdict", "size"), verdict=("verdict", "mean"),
                                 claude_share=("is_claude", "mean")).reset_index()
    g_nc = mon[mon["login"] != "claude"].groupby("month")["verdict"].mean().rename("verdict_ex_claude")
    g = g.merge(g_nc, on="month", how="left")
    g = g[g["n"] >= 50].reset_index(drop=True)  # 極端に薄い月は除く
    g["t"] = np.arange(len(g))
    out = {"months": g.to_dict("records")}
    for y in ["verdict", "claude_share", "verdict_ex_claude"]:
        m = smf.ols(f"{y} ~ t", data=g).fit()
        out[f"slope_{y}"] = {"slope": float(m.params["t"]), "p": float(m.pvalues["t"]),
                             "n_months": int(len(g))}
    cl = mon[mon["login"] == "claude"]
    out["claude_reviews_by_repo"] = {k: int(v) for k, v in cl["repo"].value_counts().items()}
    out["claude_approvals_by_repo"] = {
        k: int(v) for k, v in cl[cl["state"] == "APPROVED"]["repo"].value_counts().items()}
    out["claude_first_month"] = cl["month"].min() if len(cl) else None
    return out


def main() -> None:
    cat = make_classifier()
    pr, rv_stats, mb, mon, actor_tables = build(cat)
    out: dict = {"n_prs_all": int(len(pr)), "n_repos": int(pr["repo"].nunique()),
                 "author_cat": {k: int(v) for k, v in pr["author_cat"].value_counts().items()}}
    print(f"=== PR {len(pr):,} / repos {pr['repo'].nunique()} ===")

    # 1. 機械レビュアー
    agents, bots = {}, {}
    for lg, s in rv_stats.items():
        rec = {k: v for k, v in s.items() if k not in ("repos_reviewing", "repos_approving")}
        rec["n_repos_reviewing"] = len(s["repos_reviewing"])
        rec["repos_approving"] = dict(s["repos_approving"].most_common())
        rec["verdict_rate"] = (s["APPROVED"] + s["CHANGES_REQUESTED"]) / s["reviews"]
        rec["capability"] = "approves" if s["APPROVED"] > 0 else "never_approves"
        (agents if s["category"] == "agent" else bots)[lg] = rec
    agents = dict(sorted(agents.items(), key=lambda kv: -kv[1]["reviews"]))
    bots = dict(sorted(bots.items(), key=lambda kv: -kv[1]["reviews"]))
    out["agents"] = agents
    out["ci_bots_top"] = dict(list(bots.items())[:40])

    def agg(d):
        t = Counter()
        for r in d.values():
            for k in ("reviews", "APPROVED", "CHANGES_REQUESTED", "appr_on_human_pr",
                      "appr_on_bot_pr", "first_appr_human_pr"):
                t[k] += r[k]
        t["n_accounts"] = len(d)
        t["n_accounts_approving"] = sum(r["APPROVED"] > 0 for r in d.values())
        return dict(t)
    out["agents_total"] = agg(agents)
    out["agents_capable_total"] = agg({k: v for k, v in agents.items() if v["APPROVED"] > 0})
    out["agents_never_total"] = agg({k: v for k, v in agents.items() if v["APPROVED"] == 0})
    out["ci_bots_total"] = agg(bots)
    for lg, r in agents.items():
        print(f"  {lg:32s} rev={r['reviews']:7,} APPR={r['APPROVED']:5,} CR={r['CHANGES_REQUESTED']:3,} "
              f"repos_appr={r['repos_approving']}")

    # 2. 例外表: 主母集団 / module 部分集合
    hp = pr[pr["author_cat"] == "human"]
    out["exceptions_all"] = exceptions(hp)
    out["exceptions_module_matched"] = exceptions(hp[hp["module_matched"]])
    for k in ("exceptions_all", "exceptions_module_matched"):
        e = out[k]
        print(f"\n[{k}] human PRs {e['n_human_authored']:,} approved {e['n_with_any_approval']:,} "
              f"first={e['first_approver_cat']}")
        for lab, r in e["machine_first"].items():
            print(f"   {lab:16s} first={r['first_approver']:5,} +human={r['human_also_approved']:5,} "
                  f"nohuman={r['no_human_approval']:4,} merged={r['no_human_approval_merged']:4,}")
        for lab, r in e["conditional"].items():
            print(f"   {lab:32s} n={r['n']:8,} merge={r['merge_rate']}")

    # 2b. 分類漏れ(EXTRA_MACHINE)を機械に付け替えた感度分析
    pr2, _, mb2, _, at2 = build(make_classifier(EXTRA_MACHINE))
    hp2 = pr2[pr2["author_cat"] == "human"]
    out["corrected_extra_machine"] = sorted(EXTRA_MACHINE)
    out["exceptions_all_corrected"] = exceptions(hp2)
    out["exceptions_module_matched_corrected"] = exceptions(hp2[hp2["module_matched"]])
    out["merge_module_matched_corrected"] = merge_authority(hp2[hp2["module_matched"]], mb2, at2)
    for k in ("exceptions_all_corrected", "exceptions_module_matched_corrected"):
        e = out[k]
        print(f"\n[{k}] human PRs {e['n_human_authored']:,} approved {e['n_with_any_approval']:,} "
              f"first={e['first_approver_cat']}")
        for lab, r in e["machine_first"].items():
            print(f"   {lab:16s} first={r['first_approver']:5,} +human={r['human_also_approved']:5,} "
                  f"nohuman={r['no_human_approval']:4,} merged={r['no_human_approval_merged']:4,}")

    # 3. マージ
    out["merge_all"] = merge_authority(hp, mb, actor_tables)
    out["merge_module_matched"] = merge_authority(hp[hp["module_matched"]], mb, actor_tables)
    ma = out["merge_all"]
    print(f"\n[merge] merged {ma['n_merged_human_authored']:,} machine {ma['machine_merges']:,} "
          f"({ma['machine_merge_share']:.2%}) w/human {ma['machine_merge_with_human_appr_share']:.2%} "
          f"fully {ma['fully_machine']:,}; candidates={ma['unlisted_merger_candidates']}")

    # 4. スレッド
    out["threads"] = threads(cat)
    print("\n[threads]", json.dumps(out["threads"]["who_resolves_all"], ensure_ascii=False)[:800])
    print("[threads FE]", {k: (round(v["coef"], 4), round(v["p"], 4))
                           for k, v in out["threads"]["pr_fe_live_line"].items()})

    # 5. トレンド
    out["trend"] = trend(mon)
    print("\n[trend]", {k: v for k, v in out["trend"].items() if k.startswith("slope")},
          out["trend"]["claude_approvals_by_repo"])

    path = OUT_DIR / "d45_verdict_boundary.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
