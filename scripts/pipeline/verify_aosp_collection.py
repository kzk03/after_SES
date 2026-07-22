"""AOSP32収集データの健全性検証

各repoのraw_jsonについて: change数 / updated期間 / 月別の欠け(活動開始〜終了の間の空白月) /
CSV行数 を検証し、要注意repoを報告する。10k打ち切り(件数と期間の不整合)の検出が主目的。

実行: uv run python scripts/pipeline/verify_aosp_collection.py
"""
import csv as csvmod
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data/aosp/raw_json"
CSV_DIR = ROOT / "data/aosp/csv"

issues = []
rows_total = 0
changes_total = 0
print(f"{'repo':45s} {'changes':>8} {'csv行':>8} {'期間':>19} {'空白月':>4}")
for raw in sorted(RAW_DIR.glob("aosp__*.json")):
    repo = raw.stem.replace("aosp__", "").replace("__", "/")
    changes = json.load(open(raw))
    months = Counter(c["updated"][:7] for c in changes)
    lo, hi = min(months), max(months)
    # 活動開始月〜終了月の間で change が 0 の月を数える
    ys, ms = int(lo[:4]), int(lo[5:7])
    ye, me = int(hi[:4]), int(hi[5:7])
    span = [f"{y}-{m:02d}" for y in range(ys, ye + 1) for m in range(1, 13)
            if (y, m) >= (ys, ms) and (y, m) <= (ye, me)]
    gaps = [m for m in span if months.get(m, 0) == 0]
    csv_path = CSV_DIR / f"{raw.stem.replace('aosp__', '')}.csv"
    n_rows = sum(1 for _ in open(csv_path)) - 1 if csv_path.exists() else -1
    changes_total += len(changes)
    rows_total += max(n_rows, 0)
    flag = ""
    if n_rows < 0:
        flag = "CSVなし"
    elif len(gaps) > 2 and len(changes) > 5000:
        flag = f"空白月多い: {gaps[:4]}"
    elif hi < "2024-11":
        flag = f"終端が早い({hi})=打ち切り疑い"
    if flag:
        issues.append((repo, flag))
    print(f"{repo:45s} {len(changes):>8,} {n_rows:>8,} {lo}〜{hi} {len(gaps):>4} {flag}")

print(f"\n合計: {changes_total:,} changes / {rows_total:,} 依頼行")
if issues:
    print("要確認:")
    for r, f in issues:
        print(f"  {r}: {f}")
    sys.exit(1)
print("全repo健全(期間カバレッジ・CSV生成とも問題なし)")
