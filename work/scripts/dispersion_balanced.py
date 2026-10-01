"""balanced の集中緩和を承諾モデル非依存の経路で測る。

シミュレータ内の gini は「貢献判定を通した後」の量なので、
貢献モデルが間違っていれば汚染される。
ここでは logged 方策を force_logged_outcome で再生した
「実際に起きた状態列」の各ステップで balanced が誰を選ぶかだけを見て、
期待割り当て件数のジニを計算する。**貢献判定を一切通さない。**

balanced は決定的方策なので、選んだ 1 人に確率 1 を置く。
"""
import pickle, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--label", default="AOSP")
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb")); eps = d["episodes"]; n_test = d["meta"]["n_test"]
m = AcceptanceModel.load(a.model)
env = AssignmentSimEnv(episodes=eps, history=d["history"], acceptance_model=m,
    state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
    n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=0.05, w_gini=0.25,
    load_cap=5, background_history=True, stuck_when_no_response=True,
    force_logged_outcome=True)

def gini(x):
    x = np.asarray(x, float); x = x[x >= 0]
    if x.size == 0 or x.sum() <= 0: return 0.0
    xs = np.sort(x); i = np.arange(1, xs.size + 1)
    return float((2*(i*xs).sum())/(xs.size*xs.sum()) - (xs.size+1)/xs.size)

rows = []
for ep in range(len(eps) - n_test, len(eps)):
    obs, _ = env.reset(seed=42+ep, options={"ep_idx": ep})
    ids = {}                       # email -> index
    bal = {}; cap = {}; lg = {}; rnd = {}
    done = False; n = 0
    while not done:
        cands = env._current_candidates()
        for e in cands: ids.setdefault(e, len(ids))
        p = env.candidate_p_accept().astype(np.float64)
        load = env.candidate_loads().astype(np.float64)
        msk = env.action_masks()
        nv = int(msk.sum())
        # balanced
        v = np.where(msk, p/(1.0+np.maximum(load,0.0)), -np.inf)
        bal[ids[cands[int(np.argmax(v))]]] = bal.get(ids[cands[int(np.argmax(v))]], 0)+1
        # balanced_cap
        free = msk & (load <= 5)
        w = np.where(free if free.any() else msk, p if free.any() else -load, -np.inf)
        cap[ids[cands[int(np.argmax(w))]]] = cap.get(ids[cands[int(np.argmax(w))]], 0)+1
        # random (期待値)
        for i, e in enumerate(cands):
            if msk[i]: rnd[ids[e]] = rnd.get(ids[e], 0.0) + 1.0/nv
        act = env.logged_action()
        if act is not None:
            lg[ids[cands[act]]] = lg.get(ids[cands[act]], 0)+1
        obs, _, done, _, _ = env.step(act if act is not None else int(np.flatnonzero(msk)[0]))
        n += 1
    nd = len(ids)
    def vec(dd):
        z = np.zeros(nd)
        for k, v in dd.items(): z[k] = v
        return z
    rows.append({"episode": ep, "n": n, "balanced": gini(vec(bal)),
                 "balanced_cap": gini(vec(cap)), "logged": gini(vec(lg)), "random": gini(vec(rnd))})

r = pd.DataFrame(rows); w = r["n"]
print(f"{a.label}: 割り当てのジニ (貢献判定を通さない、test {len(r)} エピソード)")
for c in ("balanced", "balanced_cap", "logged", "random"):
    print(f"  {c:<14} {np.average(r[c], weights=w):.4f}")
rs = np.random.default_rng(0)
for c in ("balanced", "balanced_cap"):
    dv = (r[c] - r["logged"]).values
    bs = np.array([np.average(dv[i], weights=w.values[i]) for i in rs.integers(0, len(r), (4000, len(r)))])
    print(f"\n{c} − logged: {np.average(dv, weights=w):+.4f} "
          f"[{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  "
          f"分散した割合 {(bs<0).mean():.1%}  エピソード {(r[c]<r['logged']).sum()}/{len(r)}")
