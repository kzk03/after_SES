"""滞留モデルでの負荷分布を測り、load_cap の妥当な値を決める。

candidate_loads() はマスク外の候補に番兵値が入るので、有効候補だけを見る。
"""
import pickle, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder
from train_assignment_rl import policy_logged, policy_random, policy_greedy_accept

ap = argparse.ArgumentParser(); ap.add_argument("--n-ep", type=int, default=25); a = ap.parse_args()
d = pickle.load(open("work/assign_ep.pkl", "rb")); eps = d["episodes"]; n_test = d["meta"]["n_test"]
m = AcceptanceModel.load("outputs/rl/contrib_model_v3/contrib_model.joblib")
for stuck in (False, True):
    env = AssignmentSimEnv(episodes=eps, history=d["history"], acceptance_model=m,
        state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
        n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=0.2, w_gini=0.25,
        background_history=True, stuck_when_no_response=stuck)
    print(f"\n=== stuck_when_no_response={stuck} ===")
    for name, fn in [("logged", policy_logged), ("random", policy_random), ("greedy_accept", policy_greedy_accept)]:
        chosen_load, pool_max = [], []
        for ep in range(len(eps) - n_test, len(eps) - n_test + a.n_ep):
            rng = np.random.default_rng(42 + ep)
            obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep}); done = False
            while not done:
                loads = env.candidate_loads(); mask = env.action_masks().astype(bool)
                v = loads[mask]
                if v.size: pool_max.append(float(v.max()))
                act = fn(env, obs, rng)
                chosen_load.append(float(loads[act]))
                obs, _, done, _, _ = env.step(act)
        c = np.array(chosen_load); pm = np.array(pool_max)
        print(f"  {name:<14} 選んだ人の負荷: 平均 {c.mean():.2f} 中央値 {np.median(c):.0f} "
              f"90% {np.percentile(c,90):.0f} 最大 {c.max():.0f} / 候補内最大: 平均 {pm.mean():.2f}")
        for cap in (1, 2, 3, 5):
            print(f"      cap={cap}: 超過するステップ {np.mean(c>cap):.3f}  平均超過率 {np.mean(np.maximum(0,c-cap)/cap):.4f}")
