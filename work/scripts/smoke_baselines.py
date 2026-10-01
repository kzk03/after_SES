"""追加したベースライン方策が動くかの確認 (2 エピソードだけ回す)。"""
import pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder
import train_assignment_rl as T

d = pickle.load(open("work/aosp_ep_7d.pkl", "rb")); eps = d["episodes"]; n_test = d["meta"]["n_test"]
m = AcceptanceModel.load("outputs/rl/contrib_model_aosp/contrib_model.joblib")
env = AssignmentSimEnv(episodes=eps, history=d["history"], acceptance_model=m,
    state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
    n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=0.05, w_gini=0.25,
    load_cap=5, background_history=True, stuck_when_no_response=True)
pols = [("logged",T.policy_logged),("least_load",T.policy_least_load),
        ("greedy_accept",T.policy_greedy_accept),
        ("most_experienced",T.policy_most_experienced),("most_recent",T.policy_most_recent),
        ("core_reviewer",T.policy_core_reviewer),("round_robin",T.policy_round_robin),
        ("softmax_accept",T.policy_softmax_accept),("balanced",T.policy_balanced)]
print(f"{'方策':<18}{'貢献率':>8}{'滞留':>8}{'報酬/step':>10}")
for name, fn in pols:
    acc=[]; ld=[]; rw=[]; n=0
    for ep in range(len(eps)-n_test, len(eps)-n_test+2):
        rng=np.random.default_rng(42+ep)
        obs,_=env.reset(seed=42+ep, options={"ep_idx":ep}); done=False
        while not done:
            loads=env.candidate_loads(); a=fn(env,obs,rng)
            ld.append(float(loads[a]))
            obs,r,done,_,info=env.step(a); rw.append(r); n+=1
            acc.append(1.0 if info.get("accepted") else 0.0)
    print(f"{name:<18}{np.mean(acc):>8.4f}{np.mean(ld):>8.2f}{np.sum(rw)/n:>10.4f}")
