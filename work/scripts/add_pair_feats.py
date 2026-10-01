"""学習データにペア相互性 3 次元を足す (既存の列は消さない)。"""
import sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from review_predictor.RL.env.pair_reciprocity import PairReciprocity, PAIR_FEATURE_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--train", required=True)
ap.add_argument("--events", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--base-dim", type=int, required=True, help="既存の f 列の数")
a = ap.parse_args()

t = pd.read_parquet(a.train)
ev = pd.read_parquet(a.events)
pr = PairReciprocity.build(ev)
print(f"学習データ {len(t):,} 行 / ペア索引 {len(pr.times):,} 組 / 全体貢献率 {pr.prior:.4f}")

ts = pd.to_datetime(t["added_at"], errors="coerce")
rv = t["reviewer"].astype(str).to_numpy()
ow = t["owner"].astype(str).to_numpy() if "owner" in t.columns else None
if ow is None:
    key = ev[["change", "reviewer", "owner"]].drop_duplicates(["change", "reviewer"])
    t = t.merge(key, on=["change", "reviewer"], how="left")
    ow = t["owner"].astype(str).to_numpy()
    print(f"  owner 列を events から補完 (欠損 {t['owner'].isna().sum():,})")

X = np.zeros((len(t), len(PAIR_FEATURE_NAMES)), dtype=np.float32)
for i in range(len(t)):
    X[i] = pr.features(rv[i], ow[i], ts.iloc[i])
    if (i+1) % 20000 == 0: print(f"  {i+1:,}/{len(t):,}")
for j, n in enumerate(PAIR_FEATURE_NAMES):
    t[f"f{a.base_dim+j}"] = X[:, j]
t.to_parquet(a.out)
print(f"\n保存 {a.out}")
y = t["contributed"].to_numpy(np.float64)
print("貢献との相関:")
for j, n in enumerate(PAIR_FEATURE_NAMES):
    v = X[:, j].astype(np.float64)
    r = np.corrcoef(v, y)[0,1] if v.std() > 0 else float("nan")
    print(f"  {n:<20} 非ゼロ {np.mean(v!=0):.4f}  相関 {r:+.4f}")
