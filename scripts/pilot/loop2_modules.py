"""Loop 2 — モジュール展開。

CODEOWNERS の月次スナップショット(Loop 0/専用スクリプトで取得済み)をパースし、
「後勝ち」ルールでファイル集合に展開する。網羅率とモジュール数を算出する。

ファイルツリーは現在時点のもの(pilot/raw/<repo>/file_tree.json)を全月で共通に使う近似とする
(月ごとのツリー取得は重いための簡略化。Loop8で厳密化を検討)。

実行: uv run python scripts/pilot/loop2_modules.py <owner>/<repo>
"""
from __future__ import annotations

import json
import re
import sys
import warnings
import os
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from pathspec.patterns import GitWildMatchPattern

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "pilot" / os.environ.get("PILOT_RAW", "raw")
INTERIM_DIR = REPO_ROOT / "pilot" / os.environ.get("PILOT_INTERIM", "interim")
REPORTS_DIR = REPO_ROOT / "pilot" / "reports"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


class CodeownersEntry(NamedTuple):
    line_no: int
    pattern: str
    owners: tuple[str, ...]


def parse_codeowners(text: str) -> list[CodeownersEntry]:
    entries = []
    for i, line in enumerate(text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        pattern = parts[0]
        owners = tuple(parts[1:])
        entries.append(CodeownersEntry(i, pattern, owners))
    return entries


def assign_modules(entries: list[CodeownersEntry], files: list[str]) -> dict[str, CodeownersEntry]:
    """ファイル -> 担当エントリ(後勝ち)のマップを返す。"""
    compiled = [(e, GitWildMatchPattern(e.pattern)) for e in entries]
    assignment: dict[str, CodeownersEntry] = {}
    for f in files:
        # 後ろから見て最初にマッチしたものを採用(CODEOWNERSの後勝ちルール)
        for e, pat in reversed(compiled):
            if pat.match_file(f):
                assignment[f] = e
                break
    return assignment


def module_id(e: CodeownersEntry) -> str:
    return f"L{e.line_no}:{e.pattern}"


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: loop2_modules.py <owner>/<repo>")
        sys.exit(1)
    repo = sys.argv[1]
    d = RAW_DIR / repo_dir_name(repo)

    file_tree = json.loads((d / "file_tree.json").read_text())
    files = file_tree["files"]

    snapshots_dir = d / "codeowners" / "snapshots"
    manifest = json.loads((d / "codeowners" / "snapshots_manifest.json").read_text())

    per_month_stats = []
    latest_assignment = None
    latest_entries = None
    latest_month = None

    for m in manifest:
        month = m["month"]
        if m["note"] != "ok":
            per_month_stats.append({"month": month, "n_entries": 0, "n_active_modules": 0, "coverage": None})
            continue
        text = (snapshots_dir / f"{month}.txt").read_text()
        entries = parse_codeowners(text)
        assignment = assign_modules(entries, files)
        n_active_modules = len({module_id(e) for e in assignment.values()})
        coverage = len(assignment) / len(files) if files else 0.0
        per_month_stats.append(
            {
                "month": month,
                "n_entries": len(entries),
                "n_active_modules": n_active_modules,
                "coverage": coverage,
            }
        )
        latest_assignment = assignment
        latest_entries = entries
        latest_month = month

    stats_df = pd.DataFrame(per_month_stats)
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    interim_d.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(interim_d / "loop2_monthly_stats.csv", index=False)

    # 最新月のモジュール展開を modules.parquet として保存(Loop3以降で使う)
    if latest_assignment is not None:
        rows = [
            {"file": f, "module_id": module_id(e), "pattern": e.pattern, "owners": ",".join(e.owners)}
            for f, e in latest_assignment.items()
        ]
        modules_df = pd.DataFrame(rows)
        modules_df.to_parquet(interim_d / "modules.parquet", index=False)

    print(f"=== {repo} ===")
    print(f"全ファイル数: {len(files)}")
    print(f"最新月({latest_month}) 網羅率: {per_month_stats[-1]['coverage']:.1%}" if per_month_stats[-1]["coverage"] is not None else "N/A")
    print(f"最新月 有効モジュール数: {per_month_stats[-1]['n_active_modules']}")
    print(f"月次統計 -> {interim_d / 'loop2_monthly_stats.csv'}")
    print(f"モジュール展開(最新月) -> {interim_d / 'modules.parquet'}")


if __name__ == "__main__":
    main()
