"""CODEOWNERSの変更コミット履歴を取得する(Loop0で2件用に書いたロジックの汎用版)。

実行: uv run python scripts/pilot/collect_codeowners_history.py <owner>/<repo> <path>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_client import GitHubClient, now_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "pilot" / "raw"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: collect_codeowners_history.py <owner>/<repo> <path>")
        sys.exit(1)
    repo, path = sys.argv[1], sys.argv[2]

    client = GitHubClient()
    fetched_at = now_iso()
    out = []
    for item in client.paginate(f"/repos/{repo}/commits", params={"path": path}):
        item["_fetched_at"] = fetched_at
        out.append(item)

    d = RAW_DIR / repo_dir_name(repo) / "codeowners"
    d.mkdir(parents=True, exist_ok=True)
    with (d / "history.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if out:
        print(f"{repo}: {len(out)}コミット ({out[-1]['commit']['committer']['date']} 〜 {out[0]['commit']['committer']['date']})")
    else:
        print(f"{repo}: コミット履歴なし")


if __name__ == "__main__":
    main()
