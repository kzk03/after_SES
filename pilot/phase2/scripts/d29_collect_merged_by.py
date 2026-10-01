"""D29: マージを実行したのは誰か(mergedBy)を収集する。

本研究の中心的な発見は「判断の回路が機械全体に対して閉じている」ことだった。
根拠は**レビュー承認**である(人間作成 PR の最初の承認者は人間 99.9%、CI ボット 0.01%)。

しかしレビュー承認は判断のひとつにすぎない。より強い権限である**マージの実行**は
どうなっているのか。承認と同じく人間に閉じているのか、それとも
「承認は人間、マージは自動化」という分業があるのか。

GitHub の `mergedBy` で直接取れる。実測で 100 PR あたり cost=1、1.3秒と非常に安い。

判断の地図における位置づけ:
  レビュー承認  : 閉じている(D24 で実測)
  CI の失敗判定 : 開いている(legacy_bot の判断率 60.7%)
  依存の更新    : 開いている(dependabot → CI 承認)
  マージ実行    : ← 本スクリプトで埋める

出力: pilot/raw_full/<repo>/merged_by.jsonl
実行: uv run python pilot/phase2/scripts/d29_collect_merged_by.py <owner>/<repo> [batch=100]
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
    parts = [f"p{i}: pullRequest(number:{n}){{ number mergedBy{{login}} }}"
             for i, n in enumerate(numbers)]
    return ("query($owner:String!,$repo:String!){ repository(owner:$owner,name:$repo){"
            + " ".join(parts) + "} rateLimit{cost remaining} }")


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("usage: d29_collect_merged_by.py <owner>/<repo> [batch=100]")
        sys.exit(1)
    repo = sys.argv[1]
    batch = int(sys.argv[2]) if len(sys.argv) == 3 else 100
    owner, name = repo.split("/")

    d = RAW_FULL_DIR / repo_dir_name(repo)
    d.mkdir(parents=True, exist_ok=True)
    out_path = d / "merged_by.jsonl"
    ckpt_path = d / ".checkpoint_mergedby.json"

    pr = pd.read_parquet(PR_LEVEL)
    numbers = sorted(int(x) for x in pr[(pr["repo"] == repo) & (pr["merged"])]["pr_number"].unique())
    done: set[int] = set()
    if ckpt_path.exists():
        done = set(json.loads(ckpt_path.read_text()).get("done", []))
    todo = [n for n in numbers if n not in done]
    print(f"{repo}: マージ済み {len(numbers)} PR / 未取得 {len(todo)}", flush=True)
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
                mb = v.get("mergedBy")
                f.write(json.dumps({"number": v["number"],
                                    "merged_by": mb["login"] if mb else None},
                                   ensure_ascii=False) + "\n")
                written += 1
            done.update(chunk)
            ckpt_path.write_text(json.dumps({"done": sorted(done)}))
            if (i // batch) % 30 == 0:
                rl = data.get("rateLimit", {})
                print(f"  {i + len(chunk)}/{len(todo)} (remaining={rl.get('remaining')})", flush=True)

    ckpt_path.write_text(json.dumps({"done": sorted(done), "complete": True}))
    print(f"{repo}: {written} 件を保存 -> {out_path}")


if __name__ == "__main__":
    main()
