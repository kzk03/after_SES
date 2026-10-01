"""PPO が「重い人を選ぶ」のは原因か結果か。

仮説: 自己強化。方策が同じ人を選ぶ → その人の負荷が上がる →
      また選ぶ → さらに上がる。つまり**集中の結果として**負荷相関が出る。

確かめ方:
  1. random を対照に置く。負荷と無関係に選ぶので順位は 0.5 付近になるはず。
     PPO が 0.80 ならそれは本当に「重い人を選んでいる」
  2. エピソード内の**前半と後半**で順位を比べる。
     自己強化なら前半は 0.5 付近で、後半ほど高くなるはず。
     最初から高いなら、候補プール側の性質 (別の原因)。
"""
import argparse, pickle, sys
import numpy as np
sys.path.insert(0, "src")
import torch
from sb3_contrib import MaskablePPO
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--ppo", default="outputs/rl/assignment/aosp_bl10/ppo_assignment.zip")
ap.add_argument("--w-load", type=float, default=0.05)
ap.add_argument("--n-ep", type=int, default=12)
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
model = MaskablePPO.load(a.ppo, device="cpu")

def rank(load, v, chosen):
    x = load[v]
    return float((x < load[chosen]).sum() + 0.5 * (x == load[chosen]).sum()) / len(x)

print(f"=== {a.label} ({a.n_ep} エピソード) ===")
print(f"{'方策':<12}{'全体':>9}{'前半':>9}{'後半':>9}{'差(後-前)':>11}")
for name in ("random", "ppo", "greedy_accept"):
    early, late = [], []
    for ep in range(len(eps) - nt, len(eps) - nt + a.n_ep):
        rng = np.random.default_rng(7 + ep)
        obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep})
        done, n = False, len(env._requests)
        while not done:
            m = env.action_masks(); v = m.astype(bool)
            load = env.candidate_loads().astype(np.float64)
            if name == "random":
                act = int(rng.choice(np.flatnonzero(v)))
            elif name == "greedy_accept":
                act = int(np.argmax(np.where(v, env.candidate_p_accept(), -np.inf)))
            else:
                act, _ = model.predict(obs, action_masks=m, deterministic=False)
                act = int(act)
            if v.sum() > 1:
                (early if env._ptr < n / 2 else late).append(rank(load, v, act))
            obs, r, term, trunc, _ = env.step(act); done = term or trunc
    e, l = np.mean(early), np.mean(late)
    print(f"{name:<12}{np.mean(early+late):>9.4f}{e:>9.4f}{l:>9.4f}{l-e:>+11.4f}", flush=True)
print("\n順位: 0=候補内で最も軽い人 / 0.5=無作為 / 1=最も重い人")
print("random が 0.5 付近でなければ、候補プール側に偏りがある")
print("後半ほど高いなら自己強化 (選ぶ→負荷が上がる→また選ぶ)")
