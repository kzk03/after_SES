"""phase2 の最大の制約(期間13ヶ月)を外すため、2023-01 以降の全 PR を収集する。

phase1 は「全期間だと1リポジトリ4〜5時間かかる見込み」として直近12ヶ月に短縮したが、
実測すると現行クエリのコストは 75PR あたり cost=2・5.4秒で、見積もりは過大だった。
39リポジトリ合計で逐次13時間、3並列で4〜5時間程度で終わる。

期間を延ばすと何が良くなるか:
  - AI レビュー導入イベントが増える(現在は treated 7件しかない)
  - 導入効果を k=+3 より先まで追える
  - 「AI がまだ無かった頃」のベースラインが入る(生成側の検出力、A-1 の検証にも効く)

既存の pilot/raw/ を壊さないよう、出力先は pilot/raw_full/ にする。

実行: uv run python pilot/phase2/scripts/collect_full_history.py <owner>/<repo> [page_size]
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pilot"))

from graphql_client import GraphQLClient, paginate_connection  # noqa: E402

RAW_FULL_DIR = REPO_ROOT / "pilot" / "raw_full"
CUTOFF = "2023-01-01T00:00:00Z"  # 研究設計書 §4.2 が本来指定している期間

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


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("usage: collect_full_history.py <owner>/<repo> [page_size=75]")
        sys.exit(1)
    repo = sys.argv[1]
    page_size = int(sys.argv[2]) if len(sys.argv) == 3 else 75
    owner, name = repo.split("/")

    client = GraphQLClient()
    d = RAW_FULL_DIR / repo_dir_name(repo)
    out_path = d / "prs_full.jsonl"
    checkpoint_path = d / ".checkpoint_prs.json"

    count = paginate_connection(
        client,
        QUERY_TEMPLATE.replace("PAGE_SIZE", str(page_size)),
        {"owner": owner, "repo": name},
        connection_path=["repository", "pullRequests"],
        checkpoint_path=checkpoint_path,
        out_path=out_path,
        stop_condition=lambda node: node["createdAt"] < CUTOFF,
    )
    print(f"{repo}: {count}件のPRを保存しました -> {out_path}")


if __name__ == "__main__":
    main()
