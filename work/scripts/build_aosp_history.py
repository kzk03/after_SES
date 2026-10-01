"""AOSP の生 JSON から特徴量用の履歴 CSV を作る。

OpenStack は `data/combined_raw_main32.csv` を流用できたが AOSP には無い。
`FeatureBuilder._extract_review_requests` が change → レビュー依頼行の変換を
持っているので、それをそのまま使って同じスキーマの CSV を作る。

注意: ここで作る履歴の `label` は**旧定義**(14日以内に応答したか)。
これは特徴量計算 (直近承諾率など) の入力としてのみ使い、
学習の目的変数には使わない。目的変数は extract_assign.py が別に作る。
"""
import glob, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, "scripts/pipeline")
from build_dataset import FeatureBuilder

OUT = Path("work/aosp_history.csv")
fb = FeatureBuilder()
rows = []
for f in sorted(glob.glob("work/aosp_small/*.json")):
    ch = json.load(open(f))
    rr = fb._extract_review_requests(ch)
    rows.extend(rr)
    print(f"  {Path(f).stem}: {len(ch)} changes → {len(rr)} 行", flush=True)

d = pd.DataFrame(rows)
if d.empty:
    print("行が無い"); sys.exit(1)
# combined_raw と同じ列名に揃える
d = d.rename(columns={"reviewer_email": "email", "request_time": "timestamp"})
for c in ["change_insertions", "change_deletions", "change_files_count"]:
    d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0).astype(int)
d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
d = d.dropna(subset=["timestamp"]).sort_values("timestamp")
cols = ["change_id", "project", "owner_email", "email", "timestamp", "label",
        "first_response_time", "response_latency_days",
        "change_insertions", "change_deletions", "change_files_count"]
d = d[[c for c in cols if c in d.columns]]
d.to_csv(OUT, index=False)
print(f"\n保存 {OUT}: {len(d):,} 行 / {d.project.nunique()} プロジェクト / "
      f"{d.email.nunique():,} 人")
print(f"  期間 {d.timestamp.min():%Y-%m-%d} 〜 {d.timestamp.max():%Y-%m-%d}")
print(f"  label=1 の割合 {d.label.mean():.4f} (旧定義。特徴量の入力にのみ使う)")
