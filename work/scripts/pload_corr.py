"""説明の要を実測する: 候補の p_accept と負荷は本当に正相関するのか。

「PPO が負荷の重い人を選ぶのは、p_accept と負荷が元から正相関するから」
と説明してきたが、この相関自体を測っていなかった。
崩れると説明全体が崩れるので確かめる。

方策に依存しない量なので、logged を走らせて各ステップの候補集合で測る。
"""
import argparse, pickle, sys
import numpy as np
sys.path.insert(0, "src")
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--w-load", type=float, default=0.05)
ap.add_argument("--n-ep", type=int, default=15)
ap.add_argument("--label", default="AOSP")
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb"))
eps, nt = d["episodes"], d["meta"]["n_test"]
env = AssignmentSimEnv(
    episodes=eps, history=d["history"], acceptance_model=AcceptanceModel.load(a.model),
    state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
    n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=a.w_load, w_gini=0.25,
    load_cap=5, background_history=True, stuck_when_no_response=True,
)
corr, hi_p, lo_p = [], [], []
for ep in range(len(eps) - nt, len(eps) - nt + a.n_ep):
    obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep}); done = False
    while not done:
        m = env.action_masks(); v = m.astype(bool)
        load = env.candidate_loads().astype(np.float64)[v]
        p = env.candidate_p_accept().astype(np.float64)[v]
        if v.sum() > 2 and load.std() > 1e-9 and p.std() > 1e-9:
            corr.append(np.corrcoef(p, load)[0, 1])
            med = np.median(load)
            if (load > med).any() and (load <= med).any():
                hi_p.append(p[load > med].mean())   # 負荷が重い側の p_accept
                lo_p.append(p[load <= med].mean())  # 軽い側
        # logged を選ぶ (方策に依存しない量を測るため)
        act = None
        for i, e in enumerate(env._current_candidates()):
            if e == str(env._requests[env._ptr].get("logged_reviewer", "")): act = i; break
        if act is None or not v[act]: act = int(np.flatnonzero(v)[0])
        obs, r, term, trunc, _ = env.step(act); done = term or trunc

print(f"=== {a.label} ({len(corr):,} ステップ) ===")
print(f"  候補内 p_accept と負荷の相関: {np.mean(corr):+.4f}  (中央値 {np.median(corr):+.4f})")
print(f"  正の相関を示すステップ: {np.mean(np.array(corr) > 0):.1%}")
print(f"  負荷が重い側の p_accept 平均: {np.mean(hi_p):.4f}")
print(f"  負荷が軽い側の p_accept 平均: {np.mean(lo_p):.4f}")
print(f"  差: {np.mean(hi_p) - np.mean(lo_p):+.4f}")
