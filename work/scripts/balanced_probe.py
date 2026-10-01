"""balanced が本当に負荷と負の相関を持つかを実測する。

balanced は決定的なので「確率と負荷の相関」は直接は取れない。
そこで 2 つの見方で測る:
  (1) スコア p/(1+load) と負荷の相関 (PPO の確率に対応するもの)
  (2) 選んだ人の負荷が候補プール内で何パーセンタイルか
      → 決定的・確率的を問わず比較できる
"""
import pickle, sys
import numpy as np
sys.path.insert(0, "src")
from sb3_contrib import MaskablePPO
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.state_builder import RLStateBuilder
import torch

d = pickle.load(open("work/aosp_ep_7d.pkl", "rb"))
eps, nt = d["episodes"], d["meta"]["n_test"]
test_idx = list(range(len(eps) - nt, len(eps)))[:12]
env = AssignmentSimEnv(
    episodes=eps, history=d["history"],
    acceptance_model=AcceptanceModel.load("outputs/rl/contrib_model_aosp/contrib_model.joblib"),
    state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
    n_candidates=20, seed=42, w_accept=1.0, w_reject=0.3, w_load=0.05, w_gini=0.25,
    load_cap=5, background_history=True, stuck_when_no_response=True,
)
model = MaskablePPO.load("outputs/rl/assignment/aosp_fair200k/ppo_assignment.zip", device="cpu")

def pct(load, v, chosen):
    """選んだ人の負荷が候補内で何パーセンタイルか (0=最軽、1=最重)"""
    x = load[v]
    return float((x < load[chosen]).sum() + 0.5 * (x == load[chosen]).sum()) / len(x)

res = {}
for name in ("balanced", "greedy_accept", "least_load", "ppo", "logged"):
    corr, pcts = [], []
    for ep in test_idx:
        obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep}); done = False
        while not done:
            m = env.action_masks(); v = m.astype(bool)
            load = env.candidate_loads().astype(np.float64)
            p = env.candidate_p_accept().astype(np.float64)
            if name == "balanced":      score = np.where(v, p / (1.0 + load), -np.inf)
            elif name == "greedy_accept": score = np.where(v, p, -np.inf)
            elif name == "least_load":  score = np.where(v, -load, -np.inf)
            elif name == "logged":      score = None
            else:
                dist = model.policy.get_distribution(
                    {k: torch.as_tensor(x).unsqueeze(0) for k, x in obs.items()},
                    action_masks=m[None, :])
                score = np.where(v, dist.distribution.probs.detach().numpy()[0], -np.inf)
            if score is None:
                act = None
                for i, e in enumerate(env._current_candidates()):
                    if e == str(env._requests[env._ptr].get("logged_reviewer", "")): act = i; break
                if act is None or not v[act]: act = int(np.flatnonzero(v)[0])
            else:
                act = int(np.argmax(score))
                s = score[v]
                if v.sum() > 2 and s.std() > 1e-9 and load[v].std() > 1e-9:
                    corr.append(np.corrcoef(s, load[v])[0, 1])
            if v.sum() > 1: pcts.append(pct(load, v, act))
            obs, r, term, trunc, _ = env.step(act); done = term or trunc
    res[name] = (np.mean(corr) if corr else np.nan, np.mean(pcts))

print(f"{'方策':<16}{'スコアと負荷の相関':>20}{'選んだ人の負荷の順位':>22}")
for k, (c, pc) in res.items():
    print(f"{k:<16}{c:>20.4f}{pc:>22.4f}")
print("\n順位: 0=候補内で最も軽い人を選ぶ / 0.5=無作為 / 1=最も重い人を選ぶ")
