"""D22: レビュースレッドの解決状態を収集する — AI の指摘は本当に扱われているのか。

D21 で「AI は人間より先に、人間とは違う箇所を、4.5倍の分量で指摘している」ことが分かった。
では、その指摘は実際に処理されているのか。D13 の「後続 commit 率」は粗い近似でしかない
(PR は元々活発に更新されるので、レビューと無関係な commit も拾ってしまう)。

GitHub のレビュースレッドは isResolved / isOutdated を持つ。
スレッドを開始した人(最初のコメントの著者)が AI か人間かで解決率を比べれば、
**同じ PR の中で AI の指摘と人間の指摘がどう扱われたか**を直接比較できる。

  - AI の解決率が人間より低い → 補完的な情報を出しているのに相対的に軽視されている
  - 同等                      → AI の指摘も人間の指摘と同じように処理されている

どちらでも意味がある。isResolved は「対応した」と「不要と判断した」の両方を含むので、
「未解決のまま放置された割合」として解釈するのが安全。

出力: pilot/raw_full/<repo>/thread_status.jsonl
実行: uv run python pilot/phase2/scripts/d22_collect_thread_status.py <owner>/<repo> [batch=20]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "pilot"))

from graphql_client import GraphQLClient  # noqa: E402

RAW_FULL_DIR = REPO_ROOT / "pilot" / "raw_full"
PR_LEVEL = REPO_ROOT / "pilot" / "phase2" / "outputs_full" / "d5_pr_level.parquet"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def build_query(numbers: list[int]) -> str:
    parts = []
    for i, n in enumerate(numbers):
        parts.append(f"""
    p{i}: pullRequest(number: {n}) {{
      number
      reviewThreads(first: 60) {{
        nodes {{
          isResolved
          isOutdated
          comments(first: 1) {{ nodes {{ author {{ login }} path line createdAt }} }}
        }}
      }}
    }}""")
    return (
        "query($owner:String!,$repo:String!){\n"
        "  repository(owner:$owner,name:$repo){" + "".join(parts) + "\n  }\n"
        "  rateLimit{cost remaining}\n}"
    )


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("usage: d22_collect_thread_status.py <owner>/<repo> [batch=20]")
        sys.exit(1)
    repo = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) == 3 else 20
    owner, name = repo.split("/")

    d = RAW_FULL_DIR / repo_dir_name(repo)
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / "thread_status.jsonl"
    ckpt_path = d / ".checkpoint_threads.json"

    pr = pd.read_parquet(PR_LEVEL)
    numbers = sorted(int(x) for x in pr[(pr["repo"] == repo) & (pr["n_ai_reviews_total"] > 0)]["pr_number"].unique())
    done: set[int] = set()
    if ckpt_path.exists():
        done = set(json.loads(ckpt_path.read_text()).get("done", []))
    todo = [n for n in numbers if n not in done]
    print(f"{repo}: 対象 {len(numbers)} PR / 未取得 {len(todo)} PR", flush=True)
    if not todo:
        ckpt_path.write_text(json.dumps({"done": sorted(done), "complete": True}))
        return

    client = GraphQLClient()
    written = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            data = client.query(build_query(chunk), {"owner": owner, "repo": name})
            for k, v in (data.get("repository") or {}).items():
                if not k.startswith("p") or v is None:
                    continue
                threads = []
                for th in (v.get("reviewThreads") or {}).get("nodes", []):
                    first = ((th.get("comments") or {}).get("nodes") or [None])[0]
                    if not first or not first.get("author"):
                        continue
                    threads.append({
                        "starter": first["author"]["login"],
                        "path": first.get("path"),
                        "line": first.get("line"),
                        "createdAt": first.get("createdAt"),
                        "isResolved": th.get("isResolved"),
                        "isOutdated": th.get("isOutdated"),
                    })
                f.write(json.dumps({"number": v["number"], "threads": threads}, ensure_ascii=False) + "\n")
                written += 1
            done.update(chunk)
            ckpt_path.write_text(json.dumps({"done": sorted(done)}))
            if (i // batch) % 25 == 0:
                rl = data.get("rateLimit", {})
                print(f"  {i + len(chunk)}/{len(todo)} (cost={rl.get('cost')}, remaining={rl.get('remaining')})",
                      flush=True)

    ckpt_path.write_text(json.dumps({"done": sorted(done), "complete": True}))
    print(f"{repo}: {written} PR 分を保存 -> {out_path}")


if __name__ == "__main__":
    main()
