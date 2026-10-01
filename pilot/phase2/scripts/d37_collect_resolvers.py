"""D37: レビュースレッドを**誰が**解決したかを収集する。

D36 は「同じ PR の同じ行でも、AI が立てたスレッドの方が 13.6 ポイント多く
解決済みにされる」を示した。だが `thread_status.jsonl` は解決者を持たないため、
「AI ツールが自分のスレッドを自動解決しているだけ」という対抗説明を
間接的にしか潰せていない(エージェント間のばらつき・リポジトリ別・コメント長)。

GitHub GraphQL の `PullRequestReviewThread` には **`resolvedBy`** がある
(2026-09-24 にイントロスペクションで確認)。これを取れば直接答えが出る。

  AI のスレッドを人間が閉じている  → 人間が機械の通知を片付けている。制度的な説明が確定
  AI ツール自身が閉じている        → 自動解決。D36 の解釈は撤回が要る

対象は D36 の対分析に使った PR に絞る(全 PR だと 25,413件・約3時間)。
`resolvedBy` は取得済みノードのフィールドなので追加コストは無い。

出力: pilot/raw_full/<repo>/thread_resolvers.jsonl
実行: PILOT_RAW=raw_full ... uv run python pilot/phase2/scripts/d37_collect_resolvers.py [--all]
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
from common import RAW_DIR, repo_dir_name  # noqa: E402

BATCH = 20


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
          path
          line
          resolvedBy {{ login }}
          comments(first: 1) {{ nodes {{ author {{ login }} path line createdAt }} }}
        }}
      }}
    }}""")
    return ("query($owner:String!,$repo:String!){\n"
            "  repository(owner:$owner,name:$repo){" + "".join(parts) + "\n  }\n"
            "  rateLimit{cost remaining}\n}")


def all_prs_with_threads() -> dict[str, list[int]]:
    """thread_status.jsonl を持つ全 PR。--all で使う。"""
    out: dict[str, list[int]] = {}
    for d in sorted(RAW_DIR.glob("*/thread_status.jsonl")):
        repo_dir = d.parent.name
        repo = repo_dir.replace("__", "/", 1)
        nums = []
        with d.open() as f:
            for line in f:
                try:
                    nums.append(json.loads(line)["number"])
                except Exception:
                    pass
        if nums:
            out[repo] = sorted(set(nums))
    return out


def target_prs() -> dict[str, list[int]]:
    """D36 の対分析に使った PR(同一箇所に AI と人間の両方がスレッドを立てた PR)。"""
    sys.argv = ["x"]
    src = (Path(__file__).resolve().parent / "d36_who_said_it.py").read_text()
    ns: dict = {}
    exec(src.split("def main()")[0], ns)                      # noqa: S102
    df = ns["load_threads"]()
    dfp = ns["add_pair_flags"](df)
    pair = dfp[dfp["has_counterpart"] == 1]
    return {r: sorted(g["pr_number"].unique().tolist())
            for r, g in pair.groupby("repo")}


def main() -> None:
    use_all = "--all" in sys.argv
    targets = all_prs_with_threads() if use_all else target_prs()
    if use_all:
        print("全 PR を対象にする(--all)")
    total = sum(len(v) for v in targets.values())
    print(f"対象: {len(targets)} リポジトリ / {total:,} PR "
          f"(約 {total // BATCH + len(targets)} クエリ)\n")

    client = GraphQLClient()
    for repo, numbers in sorted(targets.items(), key=lambda kv: -len(kv[1])):
        owner, name = repo.split("/")
        out_path = RAW_DIR / repo_dir_name(repo) / "thread_resolvers.jsonl"
        done: set[int] = set()
        if out_path.exists():
            with out_path.open() as f:
                for line in f:
                    try:
                        done.add(json.loads(line)["number"])
                    except Exception:
                        pass
        todo = [n for n in numbers if n not in done]
        if not todo:
            print(f"  {repo}: 取得済み")
            continue
        print(f"  {repo}: {len(todo):,} PR")
        with out_path.open("a") as f:
            for i in range(0, len(todo), BATCH):
                chunk = todo[i:i + BATCH]
                r = client.query(build_query(chunk), {"owner": owner, "repo": name})
                repo_data = (r or {}).get("repository") or {}
                for k, v in repo_data.items():
                    if not k.startswith("p") or not v:
                        continue
                    threads = []
                    for th in ((v.get("reviewThreads") or {}).get("nodes") or []):
                        cs = ((th.get("comments") or {}).get("nodes") or [])
                        starter = (cs[0].get("author") or {}).get("login") if cs else None
                        threads.append({
                            "starter": starter,
                            "path": th.get("path") or (cs[0].get("path") if cs else None),
                            "line": th.get("line") or (cs[0].get("line") if cs else None),
                            "isResolved": th.get("isResolved"),
                            "isOutdated": th.get("isOutdated"),
                            "resolvedBy": (th.get("resolvedBy") or {}).get("login"),
                            "createdAt": cs[0].get("createdAt") if cs else None,
                        })
                    f.write(json.dumps({"number": v["number"], "threads": threads},
                                       ensure_ascii=False) + "\n")
                f.flush()
                rl = (r or {}).get("rateLimit") or {}
                if (i // BATCH) % 5 == 0:
                    print(f"    {min(i + BATCH, len(todo))}/{len(todo)} "
                          f"(cost={rl.get('cost')}, remaining={rl.get('remaining')})")
    print("\n収集完了")


if __name__ == "__main__":
    main()
