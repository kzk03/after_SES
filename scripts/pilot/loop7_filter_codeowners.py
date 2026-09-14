"""Loop 7 — CODEOWNERSの中身を取得し、エントリ数でフィルタする。

Loop0の教訓: contents APIで存在確認しただけでは、microsoft/vscode(12行)のような
「モジュール分割としては薄すぎる」CODEOWNERSも「あり」に混ざってしまう。
非コメント・非空行のエントリ数を数えて、十分な粒度があるものだけ残す。

実行: uv run python scripts/pilot/loop7_filter_codeowners.py
"""
from __future__ import annotations

import base64
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_client import GitHubClient

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "pilot" / "config"

MIN_ENTRIES = 50  # home-assistant(2189)/grafana(1280)よりは緩めだが、vscode(12)/langchain(3)は弾く目安


def count_entries(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def main() -> None:
    client = GitHubClient()
    in_path = CONFIG_DIR / "loop7_candidates_raw.csv"
    with in_path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["codeowners_path"]]

    print(f"CODEOWNERSを持つ候補: {len(rows)}件。中身を取得中...")
    checked = 0
    for row in rows:
        resp = client.get(f"/repos/{row['full_name']}/contents/{row['codeowners_path']}")
        if resp.status_code == 200:
            content = base64.b64decode(resp.json()["content"]).decode("utf-8", errors="replace")
            row["n_entries"] = count_entries(content)
        else:
            row["n_entries"] = 0
        checked += 1
        if checked % 50 == 0:
            print(f"  ...{checked}/{len(rows)}件")

    filtered = [r for r in rows if int(r["n_entries"]) >= MIN_ENTRIES]
    print(f"エントリ数>={MIN_ENTRIES}の候補: {len(filtered)}/{len(rows)}件")

    out_path = CONFIG_DIR / "loop7_candidates_filtered.csv"
    fieldnames = ["full_name", "stars", "language", "created_at", "pushed_at", "codeowners_path", "n_entries"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(filtered, key=lambda r: -int(r["n_entries"])):
            writer.writerow(row)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
