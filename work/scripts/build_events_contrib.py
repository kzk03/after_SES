"""割り当てイベントに contributed 列を付けて保存する。

path 親和度の索引 (PathAffinity) は「その dir で過去に貢献したか」を数えるので
contributed が要る。生のイベント parquet には無く、学習データ側にしかないため、
change + reviewer で突き合わせて付ける。

これを付けずに索引を作ると pa_dir_contrib_rate が全て 0 になり、
学習データと食い違う (G2 検査で一致率 0.248 / 相関 0.000 として検出された)。
"""
import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--events", required=True)
ap.add_argument("--train", default=None, help="一致確認用 (任意)")
ap.add_argument("--horizon", type=float, required=True, help="貢献とみなす応答日数")
ap.add_argument("--out", required=True)
a = ap.parse_args()

ev = pd.read_parquet(a.events)
# 学習データと同じ定義で作る (build_*_episodes.py: lat_days <= horizon)。
# 学習データから merge すると、エピソード範囲外のイベント (AOSP で 21%) が
# 引けず 0 埋めになって貢献率が下がるため、イベント側で直接判定する。
out = ev.copy()
out["contributed"] = (out["lat_days"] <= a.horizon).fillna(False).astype(float)
out.to_parquet(a.out)
print(f"{a.events} {len(ev):,} 件  horizon={a.horizon} 日")
print(f"  lat_days が欠損 (無応答): {int(ev['lat_days'].isna().sum()):,} "
      f"({ev['lat_days'].isna().mean():.4f})")
print(f"  貢献率: {out['contributed'].mean():.4f}")
if a.train:
    tr = pd.read_parquet(a.train)
    m = out.merge(tr[["change", "reviewer", "contributed"]].drop_duplicates(["change", "reviewer"]),
                  on=["change", "reviewer"], how="inner", suffixes=("", "_tr"))
    agree = float((m["contributed"] == m["contributed_tr"]).mean())
    print(f"  学習データと突き合わせた {len(m):,} 行の一致率: {agree:.4f}")
print(f"保存 {a.out}")
