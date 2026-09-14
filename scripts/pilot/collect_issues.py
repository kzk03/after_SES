"""全issue(PRでない純粋issue)をGraphQLで取得し pilot/raw/<repo>/issues_full.jsonl に保存する。

意図(I)工程の帰属に使う: closedByPullRequestsReferences でリンクされたPR番号を残す。
2023-01-01 以降が対象。CREATED_AT 降順でページングし、それより前に到達したら打ち切る。

実行: uv run python scripts/pilot/collect_issues.py <owner>/<repo>
"""
from __future__ import annotations

import sys
from pathlib import Path

from graphql_client import GraphQLClient, paginate_connection

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "pilot" / "raw"
CUTOFF = "2023-01-01T00:00:00Z"

QUERY = """
query($owner:String!,$repo:String!,$cursor:String){
  repository(owner:$owner,name:$repo){
    issues(first:100, after:$cursor, orderBy:{field:CREATED_AT, direction:DESC}){
      pageInfo{hasNextPage endCursor}
      nodes{
        number
        createdAt
        closedAt
        author{login}
        closedByPullRequestsReferences(first:5){nodes{number}}
      }
    }
  }
  rateLimit{cost remaining limit}
}
"""


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: collect_issues.py <owner>/<repo>")
        sys.exit(1)
    repo = sys.argv[1]
    owner, name = repo.split("/")

    client = GraphQLClient()
    d = RAW_DIR / repo_dir_name(repo)
    out_path = d / "issues_full.jsonl"
    checkpoint_path = d / ".checkpoint_issues.json"

    def stop_condition(node: dict) -> bool:
        return node["createdAt"] < CUTOFF

    count = paginate_connection(
        client,
        QUERY,
        {"owner": owner, "repo": name},
        connection_path=["repository", "issues"],
        checkpoint_path=checkpoint_path,
        out_path=out_path,
        stop_condition=stop_condition,
    )
    print(f"{repo}: {count}件のissueを保存しました -> {out_path}")


if __name__ == "__main__":
    main()
