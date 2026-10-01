"""429 待機処理の動作確認と、AOSP 小規模収集の所要時間の見積もり。"""
import sys, time, json
from datetime import datetime
from pathlib import Path
sys.path.insert(0, "scripts/pipeline")
from build_dataset import GerritDataFetcher

OUT = Path("work/aosp_small"); OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
f = GerritDataFetcher(gerrit_url="https://android-review.googlesource.com")
ch = f.fetch_changes("platform/packages/modules/Bluetooth",
                     datetime(2023, 1, 1), datetime(2023, 4, 1), limit=100)
el = time.time() - t0
json.dump(ch, open(OUT / "platform__packages__modules__Bluetooth.json", "w"))
print(f"\n取得 {len(ch)} changes / {el:.0f}s", flush=True)
print(f"reviewer_updates あり: {sum(1 for c in ch if c.get('reviewer_updates'))} / {len(ch)}")
print(f"→ 1リポジトリ3ヶ月で {el:.0f}s。5リポジトリ×2年なら約 {el*5*8/3600:.1f} 時間")
