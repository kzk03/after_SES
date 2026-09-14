"""Loop 7 — リポジトリ候補の探索。

GitHub Search API で人気リポジトリを集め、CODEOWNERS の保有を確認する。
研究設計書§4.1: 「CODEOWNERS を持つ中〜大規模リポジトリ」「2023-01時点で活動があり、観測期間を通じて継続」。

実行: uv run python scripts/pilot/loop7_discover_repos.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_client import GitHubClient

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "pilot" / "config"

CODEOWNERS_PATHS = ["CODEOWNERS", ".github/CODEOWNERS", "docs/CODEOWNERS"]

# 既にLoop0〜6で使った2件は除外し、多様な言語・規模で候補を集める
# 1クエリ最大1000件のGitHub Search API制約があるため、star数レンジを分割して母集団を増やす
SEARCH_QUERIES = [
    "stars:>20000 pushed:>2026-08-01 created:<2023-01-01",
    "stars:5000..20000 pushed:>2026-08-01 created:<2023-01-01",
    "stars:2000..5000 pushed:>2026-08-01 created:<2023-01-01",
    "stars:1000..2000 pushed:>2026-08-01 created:<2023-01-01",
]


def find_codeowners_path(client: GitHubClient, repo: str) -> str | None:
    for path in CODEOWNERS_PATHS:
        resp = client.get(f"/repos/{repo}/contents/{path}")
        if resp.status_code == 200:
            return path
        time.sleep(0.1)
    return None


def main() -> None:
    client = GitHubClient()

    candidates: dict[str, dict] = {}
    for q in SEARCH_QUERIES:
        for page in range(1, 4):  # per_page=100 x 3 = 300件/レンジ(CODEOWNERS確認のAPI消費を抑えるため制限)
            resp = client.get(
                "/search/repositories",
                params={"q": q, "sort": "stars", "order": "desc", "per_page": 100, "page": page},
            )
            if resp.status_code != 200:
                print(f"  search error page={page}: {resp.status_code} {resp.text[:200]}")
                break
            items = resp.json().get("items", [])
            if not items:
                break
            for item in items:
                full_name = item["full_name"]
                candidates[full_name] = {
                    "full_name": full_name,
                    "stars": item["stargazers_count"],
                    "language": item.get("language"),
                    "created_at": item["created_at"],
                    "pushed_at": item["pushed_at"],
                }
            print(f"  query='{q}' page={page}: {len(items)}件 (累計候補 {len(candidates)})")
    print(f"検索で集めた候補: {len(candidates)}件")

    # 既に使用済みの8件は除外
    already_used = [
        "home-assistant/core",
        "grafana/grafana",
        "juspay/hyperswitch",
        "ray-project/ray",
        "python/cpython",
        "ghostty-org/ghostty",
        "apache/airflow",
        "godotengine/godot",
    ]
    for done in already_used:
        candidates.pop(done, None)

    out_path = CONFIG_DIR / "loop7_candidates_raw.csv"
    rows = list(candidates.values())

    print("CODEOWNERS の有無を確認中...")
    checked = 0
    for row in rows:
        path = find_codeowners_path(client, row["full_name"])
        row["codeowners_path"] = path or ""
        checked += 1
        if checked % 50 == 0:
            print(f"  ...{checked}/{len(rows)}件確認済み")

    with_codeowners = [r for r in rows if r["codeowners_path"]]
    print(f"CODEOWNERSを持つ候補: {len(with_codeowners)}/{len(rows)}件")

    fieldnames = ["full_name", "stars", "language", "created_at", "pushed_at", "codeowners_path"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: -r["stars"]):
            writer.writerow(row)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
