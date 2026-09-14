"""リポジトリの現在の全ファイル一覧を取得する(Loop2の網羅率算出用)。

実行: uv run python scripts/pilot/collect_file_tree.py <owner>/<repo>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_client import GitHubClient

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "pilot" / "raw"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: collect_file_tree.py <owner>/<repo>")
        sys.exit(1)
    repo = sys.argv[1]

    client = GitHubClient()
    repo_info = client.get(f"/repos/{repo}").json()
    default_branch = repo_info["default_branch"]

    resp = client.get(f"/repos/{repo}/git/trees/{default_branch}", params={"recursive": "1"})
    data = resp.json()
    if data.get("truncated"):
        print("WARNING: tree is truncated (>100,000 entries). 網羅率算出には影響し得る")

    files = [item["path"] for item in data["tree"] if item["type"] == "blob"]

    d = RAW_DIR / repo_dir_name(repo)
    out_path = d / "file_tree.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"default_branch": default_branch, "truncated": data.get("truncated", False), "files": files}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"{repo}: {len(files)}ファイル -> {out_path}")


if __name__ == "__main__":
    main()
