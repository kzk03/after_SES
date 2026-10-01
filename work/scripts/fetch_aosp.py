"""AOSP32 を Gerrit から取得する (割り当てイベント抽出用)。

android-review は 429 (レート制限) が厳しいので
  - 期間を半年ずつに区切る
  - リクエスト間・リポジトリ間に長めの sleep を入れる
  - 429 を踏んだら待って再試行する
既存ファイルは skip するので中断しても再開できる。
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import pandas as pd, requests
sys.path.insert(0, "scripts/pipeline")
from build_dataset import GerritDataFetcher

OUT = Path("work/aosp_json"); OUT.mkdir(parents=True, exist_ok=True)
repos = pd.read_csv("data/aosp_candidate_repos.csv").query("selected == 1")["repo"].tolist()
URL = "https://android-review.googlesource.com"
WINDOWS = [(datetime(y, m, 1), datetime(y + (m == 7), 1 if m == 7 else 7, 1))
           for y in range(2020, 2026) for m in (1, 7)]
print(f"{len(repos)} リポジトリ × {len(WINDOWS)} 期間", flush=True)

for i, proj in enumerate(repos, 1):
    f = OUT / (proj.replace("/", "__") + ".json")
    if f.exists() and f.stat().st_size > 0:
        print(f"[{i}/{len(repos)}] {proj} 既存 skip", flush=True); continue
    allch, ok = [], True
    for (s, e) in WINDOWS:
        for attempt in range(4):
            try:
                ch = GerritDataFetcher(gerrit_url=URL).fetch_changes(proj, s, e, limit=100)
                allch.extend(ch); time.sleep(8); break
            except requests.exceptions.HTTPError as ex:
                if "429" in str(ex) and attempt < 3:
                    w = 60 * (attempt + 1)
                    print(f"    429 → {w}s 待機", flush=True); time.sleep(w)
                else:
                    print(f"    {s:%Y-%m} 失敗: {str(ex)[:80]}", flush=True); ok = False; break
            except Exception as ex:
                print(f"    {s:%Y-%m} 失敗: {str(ex)[:80]}", flush=True); ok = False; break
    seen, ded = set(), []
    for c in allch:
        k = c.get("id") or c.get("_number")
        if k not in seen: seen.add(k); ded.append(c)
    json.dump(ded, open(f, "w"))
    print(f"[{i}/{len(repos)}] {proj}: {len(ded)} changes{'' if ok else ' (一部失敗)'}", flush=True)
    time.sleep(15)
print("完了", flush=True)
