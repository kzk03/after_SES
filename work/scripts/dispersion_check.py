"""承諾モデルに依存しない集中度の比較。

logged 方策を force_logged_outcome で再生した「実際に起きた状態列」の各状態で、
PPO が各候補を選ぶ確率から期待割り当て件数を積み上げてジニを計算する。
貢献判定を一切通さないので、貢献モデルが間違っていてもこの比較は汚染されない。
"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="work/ope_new4.npz")
a = ap.parse_args()
d = np.load(a.npz)
PE, CID, C, M, EP = d["pe"], d["cand_ids"], d["chosen"], d["masks"], d["episode"]
print(f"{len(C):,} リクエスト × 候補 {PE.shape[1]} 人")

def gini(x):
    x = np.asarray(x, dtype=np.float64); x = x[x >= 0]
    if x.size == 0 or x.sum() <= 0: return 0.0
    xs = np.sort(x); i = np.arange(1, xs.size + 1)
    return float((2*(i*xs).sum())/(xs.size*xs.sum()) - (xs.size+1)/xs.size)

rows = []
for ep in np.unique(EP):
    m = EP == ep; ids = CID[m]; nd = int(ids.max()) + 1
    ep_, er_, al_ = np.zeros(nd), np.zeros(nd), np.zeros(nd)
    PEm, Mm, Cm = PE[m], M[m], C[m]
    for r in range(m.sum()):
        row, p, msk = ids[r], PEm[r], Mm[r]
        v = (row >= 0) & (msk == 1)
        if not v.any(): continue
        np.add.at(ep_, row[v], p[v]); np.add.at(er_, row[v], 1.0/v.sum())
        al_[row[Cm[r]]] += 1
    rows.append({"episode": int(ep), "n": int(m.sum()),
                 "ppo": gini(ep_), "logged": gini(al_), "random": gini(er_)})
r = pd.DataFrame(rows); w = r["n"]
print("\n割り当てのジニ (件数加重、低いほど分散)")
for c in ("ppo", "logged", "random"):
    print(f"  {c:<8} {np.average(r[c], weights=w):.4f}")
diff = np.average(r["ppo"] - r["logged"], weights=w)
rs = np.random.default_rng(0)
boot = np.array([np.average((r["ppo"]-r["logged"]).values[i], weights=w.values[i])
                 for i in rs.integers(0, len(r), (2000, len(r)))])
print(f"\nPPO − logged: {diff:+.4f}  95%CI [{np.percentile(boot,2.5):+.4f}, {np.percentile(boot,97.5):+.4f}]")
print(f"  PPO の方が分散した割合: {(boot<0).mean():.1%}")
print(f"  エピソード単位: {(r['ppo']<r['logged']).sum()} / {len(r)}")
