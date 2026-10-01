"""path 特徴量のどれが効いているかを 1 つずつ確かめる (ablation)。

相関では pa_dir_contrib_rate だけが突出していたが、
相関が低くても他の特徴量と組み合わせて効く可能性があるので、
実際に学習して test AUC を比べる。
"""
import re, sys, pickle, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from review_predictor.RL.env.acceptance_model import AcceptanceModel, binary_metrics
from review_predictor.RL.env.path_affinity import PATH_AFFINITY_NAMES

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--train", default="work/aosp_train_path.parquet")
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb"))
n_test = d["meta"]["n_test"]; test_start = pd.Timestamp(d["episodes"][-n_test]["start"])
t = pd.read_parquet(a.train).sort_values("added_at").reset_index(drop=True)
base = [c for c in t.columns if re.fullmatch(r"f\d{2}", c) and int(c[1:]) < 26]
path = [f"f{26+i}" for i in range(len(PATH_AFFINITY_NAMES))]
tr, te = t[t.added_at < test_start], t[t.added_at >= test_start]
n_cal = int(len(tr) * 0.15); fit, cal = tr.iloc[:-n_cal], tr.iloc[-n_cal:]
y_te = te.contributed.to_numpy()

def run(cols, label):
    m = AcceptanceModel(obs_dim=len(cols))
    m.fit(fit[cols].to_numpy(np.float32), fit.contributed.to_numpy(),
          cal[cols].to_numpy(np.float32), cal.contributed.to_numpy())
    p = m.predict_proba(te[cols].to_numpy(np.float32))
    mm = binary_metrics(y_te, p)
    print(f"  {label:<38} AUC={mm['auc_roc']:.4f}  PR={mm['auc_pr']:.4f}  ECE={mm['ece']:.4f}")
    return mm["auc_roc"]

print(f"{a.train}  学習 {len(tr):,} / 評価 {len(te):,}\n")
print("基準:")
a0 = run(base, "26次元のみ")
a1 = run(base + path, "26 + path 6次元")
print("\npath を 1 つずつ足す:")
for i, n in enumerate(PATH_AFFINITY_NAMES):
    run(base + [f"f{26+i}"], f"26 + {n}")
print("\npath を 1 つずつ抜く:")
for i, n in enumerate(PATH_AFFINITY_NAMES):
    cols = base + [c for j, c in enumerate(path) if j != i]
    run(cols, f"全部 - {n}")
print(f"\n26次元 {a0:.4f} → 全部入り {a1:.4f} (+{a1-a0:.4f})")

print("\n最小構成の探索:")
combos = [
    ([0], "review_count"), ([1], "recency"), ([2], "contrib_rate"),
    ([1, 2], "recency + contrib_rate"),
    ([1, 2, 3], "recency + contrib_rate + share"),
    ([1, 2, 0], "recency + contrib_rate + review_count"),
    ([0, 1, 2, 3], "review_count + recency + contrib_rate + share"),
]
best = (a1, "全部入り")
for idx, lbl in combos:
    v = run(base + [f"f{26+i}" for i in idx], f"26 + {lbl}")
    if v > best[0]:
        best = (v, lbl)
print(f"\n最良: {best[1]} (AUC {best[0]:.4f})   全部入り {a1:.4f}   26次元のみ {a0:.4f}")
