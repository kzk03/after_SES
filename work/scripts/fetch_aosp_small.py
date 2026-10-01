"""AOSP の主要 5 リポジトリ × 2 年を収集する (再現実験用)。

32 リポジトリ × 6 年は 429 制限下で数日かかるので規模を絞る。
_make_request に 429 待機が入ったので二分割の暴走は起きない。
既存ファイルは skip するので中断しても再開できる。
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import pandas as pd
sys.path.insert(0, "scripts/pipeline")
from build_dataset import GerritDataFetcher

OUT = Path("work/aosp_small"); OUT.mkdir(parents=True, exist_ok=True)
repos = pd.read_csv("data/aosp_candidate_repos.csv").query("selected == 1") \
          .nlargest(5, "human_per_week")["repo"].tolist()
WINDOWS = [(datetime(y, m, 1), datetime(y + (m == 7), 1 if m == 7 else 7, 1))
           for y in (2023, 2024) for m in (1, 7)]
print(f"{len(repos)} リポジトリ × {len(WINDOWS)} 期間: {repos}", flush=True)

for i, proj in enumerate(repos, 1):
    f = OUT / (proj.replace("/", "__") + ".json")
    if f.exists() and f.stat().st_size > 0:
        print(f"[{i}/{len(repos)}] {proj} 既存 skip", flush=True); continue
    allch = []
    for (s, e) in WINDOWS:
        try:
            allch.extend(GerritDataFetcher(gerrit_url="https://android-review.googlesource.com")
                         .fetch_changes(proj, s, e, limit=100))
            time.sleep(5)
        except Exception as ex:
            print(f"    {s:%Y-%m} 失敗: {str(ex)[:70]}", flush=True)
    seen, ded = set(), []
    for c in allch:
        k = c.get("id") or c.get("_number")
        if k not in seen: seen.add(k); ded.append(c)
    json.dump(ded, open(f, "w"))
    print(f"[{i}/{len(repos)}] {proj}: {len(ded)} changes", flush=True)
    time.sleep(10)
print("完了", flush=True)
