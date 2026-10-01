"""D31: アクター分類の妥当性監査 — human に混ざった機械を洗い出す。

D24/D30 の中心的な数値は `actors.parquet` の category に依存している。
その分類規則(`pilot/config/agent_identifiers.yaml` の明示リスト +
正規表現 `.*\\[bot\\]$`)には2つの系統的な抜けがある。

  抜け1: GraphQL の `author.login` は GitHub App でも `[bot]` を返さないため、
         App がパターンに掛からず human に落ちる(例: `coderabbitai` は human、
         `coderabbitai[bot]` は legacy_bot で別行になる)
  抜け2: agent リストが AIDev の6エージェントのみで、商用 AI レビューサービス
         (ellipsis-dev, sourcery-ai, coderabbitai 等)を含んでいない

本スクリプトは category を経由せず **login を直接数え**、両方の抜けの影響量を測る。
中心主張が生き残るかどうかは、漏れた AI レビュアーが APPROVED を出しているかで決まる。

出力: outputs_full/d31_classification_audit.json
実行: PILOT_RAW=raw_full PILOT_INTERIM=interim_full PHASE2_OUT=outputs_full \
      uv run python pilot/phase2/scripts/d31_classification_audit.py
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime

import pandas as pd

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, load_repos, repo_dir_name

# agent リストから漏れている商用 AI レビュー / AI コーディングサービス。
# 実データの actors を走査して目視確認したもの(§1.17)。
MISSING_AI = {
    "ellipsis-dev", "sourcery-ai", "SourceryAI", "coderabbitai",
    "greptile-apps", "graphite-app", "QodoAI-Agent", "openhands-agent",
}
# 現行 agent リストのうち実際にレビューを持つ login。
KNOWN_AGENT = {
    "copilot-pull-request-reviewer", "cursor", "copilot-swe-agent",
    "claude", "chatgpt-codex-connector", "devin-ai-integration",
}
# マージキュー。レビューは出さないので actors に現れず `unknown` に落ちる。
MERGE_QUEUE = {"trunk-io", "typescript-automation", "github-merge-queue"}


def scan_reviews() -> tuple[dict, Counter, Counter, int, int]:
    """全 PR を走査し、login 別の state 分布と「最初の承認者」を集める。"""
    state_by_login: dict[str, Counter] = defaultdict(Counter)
    first_appr: Counter = Counter()
    first_appr_cat: Counter = Counter()
    n_human_pr = 0
    n_with_appr = 0

    for repo in load_repos():
        prs_path = RAW_DIR / repo_dir_name(repo) / "prs_full.jsonl"
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not (prs_path.exists() and actors_path.exists()):
            continue
        a = pd.read_parquet(actors_path)
        cat_of = dict(zip(a["actor_login"], a["category"]))
        with prs_path.open() as f:
            for line in f:
                try:
                    pr = json.loads(line)
                except json.JSONDecodeError:
                    continue
                author = (pr.get("author") or {}).get("login")
                author_is_human = cat_of.get(author, "unknown") == "human"
                best_t, best_login = None, None
                for rv in (pr.get("reviews") or {}).get("nodes", []):
                    au = rv.get("author")
                    if not au:
                        continue
                    lg, st = au["login"], rv.get("state")
                    if lg in MISSING_AI or lg in KNOWN_AGENT:
                        state_by_login[lg][st] += 1
                    if st == "APPROVED" and rv.get("submittedAt"):
                        t = datetime.fromisoformat(rv["submittedAt"].replace("Z", "+00:00"))
                        if best_t is None or t < best_t:
                            best_t, best_login = t, lg
                if author_is_human:
                    n_human_pr += 1
                    if best_login is not None:
                        n_with_appr += 1
                        first_appr[best_login] += 1
                        first_appr_cat[cat_of.get(best_login, "unknown")] += 1
    return state_by_login, first_appr, first_appr_cat, n_human_pr, n_with_appr


def scan_mergers() -> pd.DataFrame:
    """mergedBy を読み、マージキューを機械として再分類した列を足す。"""
    rows = []
    for repo in load_repos():
        path = RAW_DIR / repo_dir_name(repo) / "merged_by.jsonl"
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not (path.exists() and actors_path.exists()):
            continue
        a = pd.read_parquet(actors_path)
        cat_of = dict(zip(a["actor_login"], a["category"]))
        with path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mb = d.get("merged_by")
                cat = cat_of.get(mb, "unknown") if mb else "none"
                rows.append({"repo": repo, "pr_number": d["number"],
                             "merged_by": mb, "merger_cat": cat})
    mb = pd.DataFrame(rows)
    mb["merger_cat2"] = mb["merger_cat"]
    mb.loc[mb["merged_by"].isin(MERGE_QUEUE), "merger_cat2"] = "merge_queue"
    return mb


def summarize(state_by_login: dict, first_appr: Counter, logins: set, n_with_appr: int) -> dict:
    tot = Counter()
    per = {}
    for lg in logins:
        c = state_by_login.get(lg, Counter())
        n = sum(c.values())
        if n == 0 and first_appr[lg] == 0:
            continue
        per[lg] = {"reviews": n, "approved": c["APPROVED"],
                   "changes_requested": c["CHANGES_REQUESTED"],
                   "commented": c["COMMENTED"], "first_approvals": first_appr[lg]}
        tot.update(c)
    n = sum(tot.values())
    judged = tot["APPROVED"] + tot["CHANGES_REQUESTED"]
    fa = sum(first_appr[lg] for lg in logins)
    return {
        "per_login": per,
        "reviews": n,
        "approved": tot["APPROVED"],
        "changes_requested": tot["CHANGES_REQUESTED"],
        "judgment_rate": judged / n if n else 0.0,
        "first_approvals": fa,
        "first_approver_share": fa / n_with_appr if n_with_appr else 0.0,
    }


def main() -> None:
    state_by_login, first_appr, first_appr_cat, n_human_pr, n_with_appr = scan_reviews()
    print(f"=== 人間作成 PR {n_human_pr:,} / APPROVED あり {n_with_appr:,} ===")

    print("\n【1】既存 category による最初の承認者")
    for k, v in first_appr_cat.most_common():
        print(f"  {k:12s} {v:8,}  ({v / n_with_appr:.4%})")

    missing = summarize(state_by_login, first_appr, MISSING_AI, n_with_appr)
    known = summarize(state_by_login, first_appr, KNOWN_AGENT, n_with_appr)

    print("\n【2】★agent リストから漏れていた AI レビュアー")
    for lg, d in sorted(missing["per_login"].items(), key=lambda kv: -kv[1]["reviews"]):
        print(f"  {lg:30s} reviews={d['reviews']:6,} APPROVED={d['approved']:4,} "
              f"CR={d['changes_requested']:3,} 初承認={d['first_approvals']:4,}")
    print(f"  小計 reviews={missing['reviews']:,} APPROVED={missing['approved']:,} "
          f"判断率={missing['judgment_rate']:.2%} 初承認={missing['first_approvals']:,}")

    print("\n【3】(参考)現行 agent リスト")
    for lg, d in sorted(known["per_login"].items(), key=lambda kv: -kv[1]["reviews"]):
        print(f"  {lg:30s} reviews={d['reviews']:6,} APPROVED={d['approved']:4,} "
              f"判断率={(d['approved'] + d['changes_requested']) / d['reviews']:6.2%} "
              f"初承認={d['first_approvals']:4,}")
    print(f"  小計 reviews={known['reviews']:,} 判断率={known['judgment_rate']:.2%} "
          f"初承認={known['first_approvals']:,} ({known['first_approver_share']:.4%})")

    merged_rate = ((known["approved"] + known["changes_requested"]
                    + missing["approved"] + missing["changes_requested"])
                   / (known["reviews"] + missing["reviews"]))
    print(f"\n【4】★漏れを agent に編入した場合の判断率: {merged_rate:.2%} "
          f"(現行 {known['judgment_rate']:.2%})")
    print(f"     初承認シェア: {(known['first_approvals'] + missing['first_approvals']) / n_with_appr:.4%} "
          f"(現行 {known['first_approver_share']:.4%})")

    # --- マージ側 ---
    mb = scan_mergers()
    pr = pd.read_parquet(OUT_DIR / "d24_approval.parquet")
    keep = ["repo", "pr_number", "author_cat", "has_human_appr"]
    df = mb.merge(pr[keep].drop_duplicates(["repo", "pr_number"]),
                  on=["repo", "pr_number"], how="inner")
    hum = df[df["author_cat"] == "human"]
    mach = hum[hum["merger_cat2"].isin(["merge_queue", "legacy_bot", "agent"])]
    with_appr = mach["has_human_appr"].astype(bool)
    closed = hum[(~hum["has_human_appr"].astype(bool))
                 & (hum["merger_cat2"].isin(["merge_queue", "legacy_bot", "agent"]))]

    print(f"\n【5】★マージキューを機械として数え直す(人間作成 PR {len(hum):,} 件)")
    print(f"  機械によるマージ: {len(mach):,} ({len(mach) / len(hum):.4%})")
    print(f"  うち人間の承認あり: {int(with_appr.sum()):,} ({with_appr.mean():.2%})")
    print(f"  ★人間の承認が無く機械がマージ: {len(closed):,} ({len(closed) / len(hum):.4%})")
    if len(closed):
        print(f"    リポジトリ別: {dict(closed.groupby('repo').size())}")

    out = {
        "n_human_authored_pr": n_human_pr,
        "n_with_approval": n_with_appr,
        "first_approver_by_existing_category": dict(first_appr_cat),
        "missing_ai_reviewers": missing,
        "known_agents": known,
        "judgment_rate_if_merged": merged_rate,
        "merge": {
            "n_human_authored": int(len(hum)),
            "machine_merged": int(len(mach)),
            "machine_merged_share": len(mach) / len(hum),
            "machine_merged_with_human_approval": int(with_appr.sum()),
            "machine_merged_with_human_approval_share": float(with_appr.mean()),
            "fully_machine_closed": int(len(closed)),
            "fully_machine_closed_share": len(closed) / len(hum),
            "by_merge_queue_login": {k: int(v) for k, v in
                                     hum[hum["merger_cat2"] == "merge_queue"]["merged_by"]
                                     .value_counts().items()},
        },
    }
    path = OUT_DIR / "d31_classification_audit.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
