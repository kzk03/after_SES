"""main32 の全期間を Gerrit から取得する (割り当てイベント抽出用)。

reviewer_updates と Code-Review 投票が要る。特徴量の履歴は既存 CSV を流用するので
ここで取るのは「誰がいつ誰を追加したか」と「実際に投票/発言したか」だけ。
保存先はリポジトリ内の work/ (git 管理外)。/tmp は再起動で消えるため。
"""
import json, sys, time
from datetime import datetime
from pathlib import Path
import pandas as pd
sys.path.insert(0, "scripts/pipeline")
from build_dataset import GerritDataFetcher

OUT = Path("work/assign_json_full"); OUT.mkdir(parents=True, exist_ok=True)
repos = pd.read_csv("data/service_teams_main_repos.csv")["main_repo"].dropna().tolist()
print(f"{len(repos)} リポジトリ", flush=True)
for i, proj in enumerate(repos, 1):
    f = OUT / (proj.replace("/", "__") + ".json")
    if f.exists() and f.stat().st_size > 0:
        print(f"[{i}/{len(repos)}] {proj} 既存 skip", flush=True); continue
    try:
        ch = GerritDataFetcher(gerrit_url="https://review.opendev.org").fetch_changes(
            proj, datetime(2020, 1, 1), datetime(2026, 1, 1), limit=300)
        json.dump(ch, open(f, "w"))
        print(f"[{i}/{len(repos)}] {proj}: {len(ch)} changes", flush=True)
    except Exception as e:
        print(f"[{i}/{len(repos)}] {proj}: 失敗 {e}", flush=True)
    time.sleep(3)
print("完了", flush=True)
