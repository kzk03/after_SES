"""全PR(files/reviews/commits込み)をGraphQLで取得し pilot/raw/<repo>/prs_full.jsonl に保存する。

2023-01-01 以降を対象(研究設計書の期間に合わせる)。CREATED_AT の降順でページングし、
2023-01-01 より前の PR に到達したら打ち切る。

実行: uv run python scripts/pilot/collect_prs.py <owner>/<repo>
"""
from __future__ import annotations

import sys
from pathlib import Path

from graphql_client import GraphQLClient, paginate_connection

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "pilot" / "raw"
CUTOFF = "2025-09-01T00:00:00Z"  # 全期間(2023-01〜)だと1リポジトリ4〜5時間かかる見込みのため、パイロットは直近12ヶ月に短縮

QUERY_TEMPLATE = """
query($owner:String!,$repo:String!,$cursor:String){
  repository(owner:$owner,name:$repo){
    pullRequests(first:PAGE_SIZE, after:$cursor, orderBy:{field:CREATED_AT, direction:DESC}){
      pageInfo{hasNextPage endCursor}
      nodes{
        number
        createdAt
        mergedAt
        closedAt
        author{login}
        files(first:100){nodes{path additions deletions}}
        reviews(first:30){nodes{author{login} state submittedAt}}
        commits(first:20){nodes{commit{oid message committedDate author{user{login} name email}}}}
      }
    }
  }
  rateLimit{cost remaining limit}
}
"""


def build_query(page_size: int) -> str:
    return QUERY_TEMPLATE.replace("PAGE_SIZE", str(page_size))


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("usage: collect_prs.py <owner>/<repo> [page_size=75]")
        sys.exit(1)
    repo = sys.argv[1]
    page_size = int(sys.argv[2]) if len(sys.argv) == 3 else 75
    owner, name = repo.split("/")

    client = GraphQLClient()
    d = RAW_DIR / repo_dir_name(repo)
    out_path = d / "prs_full.jsonl"
    checkpoint_path = d / ".checkpoint_prs.json"

    def stop_condition(node: dict) -> bool:
        return node["createdAt"] < CUTOFF

    count = paginate_connection(
        client,
        build_query(page_size),
        {"owner": owner, "repo": name},
        connection_path=["repository", "pullRequests"],
        checkpoint_path=checkpoint_path,
        out_path=out_path,
        stop_condition=stop_condition,
    )
    print(f"{repo}: {count}件のPRを保存しました -> {out_path}")


if __name__ == "__main__":
    main()
