"""Loop 0 (足場): GitHub API の試験取得。

パイロット手順.md の Loop 0 に対応:
  - issue / PR / review / commit / CODEOWNERS を試験取得し、pilot/raw/<repo>/ に jsonl で保存
  - 最優先確認: review イベントが取れるか、CODEOWNERS の履歴が取れるか

実行: uv run python scripts/pilot/loop0_scaffold.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from github_client import GitHubClient, now_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "pilot" / "raw"

# Loop 0 対象: (owner/repo, CODEOWNERSのパス)
TARGET_REPOS = [
    ("home-assistant/core", "CODEOWNERS"),
    ("grafana/grafana", ".github/CODEOWNERS"),
]

ISSUE_SAMPLE = 100
PR_SAMPLE = 100
REVIEW_SAMPLE_PRS = 20
COMMIT_SAMPLE = 100


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fetch_issues(client: GitHubClient, repo: str) -> list[dict[str, Any]]:
    fetched_at = now_iso()
    out = []
    for item in client.paginate(f"/repos/{repo}/issues", params={"state": "all", "sort": "created", "direction": "desc"}, max_pages=1):
        item["_fetched_at"] = fetched_at
        item["_is_pr"] = "pull_request" in item
        out.append(item)
        if len(out) >= ISSUE_SAMPLE:
            break
    return out


def fetch_prs(client: GitHubClient, repo: str) -> list[dict[str, Any]]:
    fetched_at = now_iso()
    out = []
    for item in client.paginate(f"/repos/{repo}/pulls", params={"state": "all", "sort": "created", "direction": "desc"}, max_pages=1):
        item["_fetched_at"] = fetched_at
        out.append(item)
        if len(out) >= PR_SAMPLE:
            break
    return out


def fetch_reviews(client: GitHubClient, repo: str, prs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fetched_at = now_iso()
    out = []
    prs_with_reviews = 0
    prs_checked = 0
    for pr in prs[:REVIEW_SAMPLE_PRS]:
        prs_checked += 1
        pr_number = pr["number"]
        resp = client.get(f"/repos/{repo}/pulls/{pr_number}/reviews")
        if resp.status_code != 200:
            continue
        reviews = resp.json()
        if reviews:
            prs_with_reviews += 1
        for rv in reviews:
            rv["_fetched_at"] = fetched_at
            rv["_pr_number"] = pr_number
            out.append(rv)
    summary = {
        "prs_checked": prs_checked,
        "prs_with_reviews": prs_with_reviews,
        "total_reviews": len(out),
    }
    return out, summary


def fetch_commits(client: GitHubClient, repo: str) -> list[dict[str, Any]]:
    fetched_at = now_iso()
    out = []
    for item in client.paginate(f"/repos/{repo}/commits", max_pages=1):
        item["_fetched_at"] = fetched_at
        out.append(item)
        if len(out) >= COMMIT_SAMPLE:
            break
    return out


def fetch_codeowners_history(client: GitHubClient, repo: str, path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fetched_at = now_iso()
    out = []
    for item in client.paginate(f"/repos/{repo}/commits", params={"path": path}):
        item["_fetched_at"] = fetched_at
        out.append(item)

    summary: dict[str, Any] = {"n_commits": len(out), "path": path}
    if out:
        summary["oldest"] = out[-1]["commit"]["committer"]["date"]
        summary["newest"] = out[0]["commit"]["committer"]["date"]

    # 現在時点の内容を1件取得しておく(月次スナップショットの本実装はLoop2)
    resp = client.get(f"/repos/{repo}/contents/{path}")
    content_ok = resp.status_code == 200
    summary["current_content_fetch_ok"] = content_ok
    return out, summary


def main() -> None:
    client = GitHubClient()
    rl = client.rate_limit()
    print("=== rate_limit (実行前) ===")
    print(json.dumps(rl["rate"], indent=2))

    overall_summary: dict[str, Any] = {}

    for repo, codeowners_path in TARGET_REPOS:
        print(f"\n=== {repo} ===")
        d = RAW_DIR / repo_dir_name(repo)

        print("issues 取得中...")
        issues = fetch_issues(client, repo)
        write_jsonl(d / "issues.jsonl", issues)
        n_pure_issue = sum(1 for i in issues if not i["_is_pr"])
        print(f"  {len(issues)}件 (うちPRでない純粋issue: {n_pure_issue}件)")

        print("PRs 取得中...")
        prs = fetch_prs(client, repo)
        write_jsonl(d / "prs.jsonl", prs)
        print(f"  {len(prs)}件")

        print("reviews 取得中 (直近PRのサブセット)...")
        reviews, review_summary = fetch_reviews(client, repo, prs)
        write_jsonl(d / "reviews.jsonl", reviews)
        print(f"  {review_summary}")

        print("commits 取得中...")
        commits = fetch_commits(client, repo)
        write_jsonl(d / "commits.jsonl", commits)
        print(f"  {len(commits)}件")

        print("CODEOWNERS 履歴取得中...")
        co_history, co_summary = fetch_codeowners_history(client, repo, codeowners_path)
        write_jsonl(d / "codeowners" / "history.jsonl", co_history)
        print(f"  {co_summary}")

        overall_summary[repo] = {
            "issues_fetched": len(issues),
            "pure_issues": n_pure_issue,
            "prs_fetched": len(prs),
            "review_summary": review_summary,
            "commits_fetched": len(commits),
            "codeowners_summary": co_summary,
        }

    rl_after = client.rate_limit()
    print("\n=== rate_limit (実行後) ===")
    print(json.dumps(rl_after["rate"], indent=2))
    overall_summary["_rate_limit_before"] = rl["rate"]
    overall_summary["_rate_limit_after"] = rl_after["rate"]

    summary_path = RAW_DIR / "_loop0_fetch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(overall_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nサマリーを書き出しました: {summary_path}")


if __name__ == "__main__":
    main()
