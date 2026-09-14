"""CODEOWNERSの月次スナップショットを取得する。

pilot/raw/<repo>/codeowners/history.jsonl (Loop 0 で取得済みのコミット履歴) を使い、
各月の月末時点で有効だった CODEOWNERS の内容を REST Contents API で取得して
pilot/raw/<repo>/codeowners/<YYYY-MM>.txt に保存する。

実行: uv run python scripts/pilot/collect_codeowners_snapshots.py <owner>/<repo> <path> [start=2023-01] [end=]
"""
from __future__ import annotations

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from github_client import GitHubClient

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "pilot" / "raw"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def month_range(start: str, end: str) -> list[str]:
    """"YYYY-MM" のリストを start から end まで(両端含む)生成する。"""
    y, m = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: collect_codeowners_snapshots.py <owner>/<repo> <path> [start=2023-01] [end=]")
        sys.exit(1)
    repo = sys.argv[1]
    path = sys.argv[2]
    start = sys.argv[3] if len(sys.argv) > 3 else "2023-01"
    end = sys.argv[4] if len(sys.argv) > 4 else datetime.now(timezone.utc).strftime("%Y-%m")

    d = RAW_DIR / repo_dir_name(repo)
    history_path = d / "codeowners" / "history.jsonl"
    if not history_path.exists():
        print(f"ERROR: {history_path} が見つかりません。先に Loop0 の loop0_scaffold.py を実行してください")
        sys.exit(1)

    commits = []
    with history_path.open() as f:
        for line in f:
            c = json.loads(line)
            commits.append({"sha": c["sha"], "date": c["commit"]["committer"]["date"]})
    commits.sort(key=lambda c: c["date"])  # 昇順

    client = GitHubClient()
    out_dir = d / "codeowners" / "snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)

    months = month_range(start, end)
    manifest = []
    for ym in months:
        year, mon = map(int, ym.split("-"))
        month_end = f"{ym}-31T23:59:59Z" if mon != 2 else f"{ym}-28T23:59:59Z"
        # その月末以前の最新コミットを選ぶ
        candidates = [c for c in commits if c["date"] <= month_end]
        if not candidates:
            manifest.append({"month": ym, "sha": None, "note": "該当コミットなし(CODEOWNERS作成前)"})
            continue
        chosen = candidates[-1]
        resp = client.get(f"/repos/{repo}/contents/{path}", params={"ref": chosen["sha"]})
        if resp.status_code != 200:
            manifest.append({"month": ym, "sha": chosen["sha"], "note": f"取得失敗 status={resp.status_code}"})
            continue
        content_b64 = resp.json()["content"]
        content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        (out_dir / f"{ym}.txt").write_text(content, encoding="utf-8")
        manifest.append({"month": ym, "sha": chosen["sha"], "commit_date": chosen["date"], "note": "ok"})

    manifest_path = d / "codeowners" / "snapshots_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    ok = sum(1 for m in manifest if m["note"] == "ok")
    print(f"{repo}: {ok}/{len(months)}ヶ月分のスナップショットを取得 -> {out_dir}")


if __name__ == "__main__":
    main()
