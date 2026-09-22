"""D20: レビュー本文と指摘位置を収集する — メカニズムの直接証拠を取りに行く。

現在の主張「AI は判断していない(99.7% が COMMENTED)」は GitHub の review state からしか
言えていない。当然の反論は「COMMENTED でも内容が有用なら価値がある。state だけ見るのは表層的だ」である。

レビュー本文とコメントの位置(path, line)が取れれば、以下が直接測れる。
  - AI と人間が **同じファイル・同じ行** を指摘しているか(冗長性の直接証拠)
  - AI コメントの分量・粒度は人間とどう違うか
  - AI が指摘した箇所に人間も後から言及しているか

全 PR を取り直すと 64 時間かかるので、**AI レビューが付いた 41,970 PR だけ**に絞る。
PR 番号を alias で1クエリに 20 件詰めると、cost≈7/クエリ・約 2,100 クエリで済む(実測 3 時間見込み)。

出力: pilot/raw_full/<repo>/review_bodies.jsonl
      {number, reviews: [{author, state, submittedAt, body, comments: [{path, line, body}]}]}

実行: uv run python pilot/phase2/scripts/d20_collect_review_bodies.py <owner>/<repo> [batch=20]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pilot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphql_client import GraphQLClient  # noqa: E402

RAW_FULL_DIR = REPO_ROOT / "pilot" / "raw_full"
PR_LEVEL = REPO_ROOT / "pilot" / "phase2" / "outputs_full" / "d5_pr_level.parquet"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def build_query(numbers: list[int]) -> str:
    """PR 番号を alias で1クエリに詰め込む。"""
    parts = []
    for i, n in enumerate(numbers):
        parts.append(f"""
    p{i}: pullRequest(number: {n}) {{
      number
      reviews(first: 40) {{
        nodes {{
          author {{ login }}
          state
          submittedAt
          body
          comments(first: 30) {{ nodes {{ path line originalLine body }} }}
        }}
      }}
    }}""")
    return (
        "query($owner:String!,$repo:String!){\n"
        "  repository(owner:$owner,name:$repo){" + "".join(parts) + "\n  }\n"
        "  rateLimit{cost remaining}\n}"
    )


def target_prs(repo: str) -> list[int]:
    pr = pd.read_parquet(PR_LEVEL)
    sub = pr[(pr["repo"] == repo) & (pr["n_ai_reviews_total"] > 0)]
    return sorted(int(x) for x in sub["pr_number"].unique())


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("usage: d20_collect_review_bodies.py <owner>/<repo> [batch=20]")
        sys.exit(1)
    repo = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) == 3 else 20
    owner, name = repo.split("/")

    d = RAW_FULL_DIR / repo_dir_name(repo)
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / "review_bodies.jsonl"
    ckpt_path = d / ".checkpoint_bodies.json"

    numbers = target_prs(repo)
    done: set[int] = set()
    if ckpt_path.exists():
        done = set(json.loads(ckpt_path.read_text()).get("done", []))
    todo = [n for n in numbers if n not in done]
    print(f"{repo}: 対象 {len(numbers)} PR / 未取得 {len(todo)} PR")
    if not todo:
        ckpt_path.write_text(json.dumps({"done": sorted(done), "complete": True}))
        print(f"{repo}: 既に完了しています")
        return

    client = GraphQLClient()
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            data = client.query(build_query(chunk), {"owner": owner, "repo": name})
            rep = data.get("repository") or {}
            for k, v in rep.items():
                if not k.startswith("p") or v is None:
                    continue
                reviews = []
                for rv in (v.get("reviews") or {}).get("nodes", []):
                    if not rv.get("author"):
                        continue
                    reviews.append({
                        "author": rv["author"]["login"],
                        "state": rv.get("state"),
                        "submittedAt": rv.get("submittedAt"),
                        "body": rv.get("body") or "",
                        "comments": [
                            {"path": c.get("path"), "line": c.get("line"),
                             "originalLine": c.get("originalLine"), "body": c.get("body") or ""}
                            for c in (rv.get("comments") or {}).get("nodes", [])
                        ],
                    })
                f.write(json.dumps({"number": v["number"], "reviews": reviews}, ensure_ascii=False) + "\n")
                written += 1
            done.update(chunk)
            ckpt_path.write_text(json.dumps({"done": sorted(done)}))
            if (i // batch) % 20 == 0:
                rl = data.get("rateLimit", {})
                print(f"  {i + len(chunk)}/{len(todo)} 取得済み (remaining={rl.get('remaining')})", flush=True)

    ckpt_path.write_text(json.dumps({"done": sorted(done), "complete": True}))
    print(f"{repo}: {written} PR 分を保存しました -> {out_path}")


if __name__ == "__main__":
    main()
