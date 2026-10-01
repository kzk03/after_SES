"""学習データに path 親和度 6 次元を付ける (f26..f31 として追加)。

リーク防止のため、各行の added_at より前の履歴だけから作る
(PathAffinity.features が now で切る)。
"""
import pickle, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from review_predictor.RL.env.path_affinity import PathAffinity, PATH_AFFINITY_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--train", default="work/aosp_train.parquet")
ap.add_argument("--events", default="work/aosp_events.parquet")
ap.add_argument("--dirs", default="work/aosp_change_dirs.pkl")
ap.add_argument("--out", default="work/aosp_train_path.parquet")
a = ap.parse_args()

t = pd.read_parquet(a.train)
ev = pd.read_parquet(a.events)
cd = pickle.load(open(a.dirs, "rb"))
print(f"学習データ {len(t):,} 行 / イベント {len(ev):,} 件 / dir 表 {len(cd):,} change")

# 索引は「イベント全体」から作る。時刻で切るのは features 側なのでリークしない。
ev2 = ev.rename(columns={"added_at": "added_at"}).copy()
if "contributed" not in ev2.columns:
    # contributed は学習データ側にあるので change+reviewer で引く
    key = t[["change", "reviewer", "contributed"]].drop_duplicates()
    ev2 = ev2.merge(key, on=["change", "reviewer"], how="left")
    ev2["contributed"] = ev2["contributed"].fillna(0.0)
pa = PathAffinity.build(ev2, cd)
print(f"索引: dev×dir {len(pa.dev_dir_times):,} / dir {len(pa.dir_owner_times):,} / "
      f"近傍を持つ dir {len(pa.neighbors):,}")

ts = pd.to_datetime(t["added_at"], errors="coerce")
X = np.zeros((len(t), len(PATH_AFFINITY_NAMES)), dtype=np.float32)
revs = t["reviewer"].astype(str).to_numpy(); chs = t["change"].to_numpy()
for i in range(len(t)):
    X[i] = pa.features(revs[i], chs[i], ts.iloc[i])
    if (i + 1) % 20000 == 0:
        print(f"  {i+1:,}/{len(t):,}")

for j, name in enumerate(PATH_AFFINITY_NAMES):
    t[f"f{26+j}"] = X[:, j]
t.to_parquet(a.out)
print(f"\n保存 {a.out}")
print("\n付けた特徴量の分布 (非ゼロ率 / 平均):")
for j, name in enumerate(PATH_AFFINITY_NAMES):
    v = X[:, j]
    print(f"  {name:<24} 非ゼロ {np.mean(v != 0):.4f}  平均 {v.mean():.4f}  最大 {v.max():.4f}")
print("\n貢献との相関:")
y = t["contributed"].to_numpy(np.float64)
for j, name in enumerate(PATH_AFFINITY_NAMES):
    v = X[:, j].astype(np.float64)
    r = np.corrcoef(v, y)[0, 1] if v.std() > 0 else float("nan")
    print(f"  {name:<24} {r:+.4f}")
