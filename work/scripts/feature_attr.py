"""方策は何を見て「重い人」を選んでいるのか。

26 次元の各列について 2 つの相関を測る:
  A: その列と**シミュレータの負荷**の相関 (負荷の代理になっている列はどれか)
  B: その列と**方策が置く確率**の相関 (方策が実際に見ている列はどれか)

A も B も高い列があれば、
「方策はその列を見ており、それが負荷の代理になっている」と言える。
"""
import argparse, pickle, sys
import numpy as np
sys.path.insert(0, "src")
import torch
from sb3_contrib import MaskablePPO
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.state_builder import RLStateBuilder, STATE_FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--ppo", default="outputs/rl/assignment/aosp_bl10/ppo_assignment.zip")
ap.add_argument("--w-load", type=float, default=0.05)
ap.add_argument("--n-ep", type=int, default=10)
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
A = [[] for _ in STATE_FEATURES]   # 列 vs 負荷
B = [[] for _ in STATE_FEATURES]   # 列 vs 方策確率

for ep in range(len(eps) - nt, len(eps) - nt + a.n_ep):
    obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep}); done = False
    while not done:
        m = env.action_masks(); v = m.astype(bool)
        load = env.candidate_loads().astype(np.float64)[v]
        dist = model.policy.get_distribution(
            {k: torch.as_tensor(x).unsqueeze(0) for k, x in obs.items()}, action_masks=m[None, :])
        pr = dist.distribution.probs.detach().numpy()[0][v]
        C = obs["candidates"][v]           # [n_valid, 26]
        if v.sum() > 2 and load.std() > 1e-9 and pr.std() > 1e-9:
            for j in range(len(STATE_FEATURES)):
                col = C[:, j].astype(np.float64)
                if col.std() > 1e-9:
                    A[j].append(np.corrcoef(col, load)[0, 1])
                    B[j].append(np.corrcoef(col, pr)[0, 1])
        act, _ = model.predict(obs, action_masks=m, deterministic=False)
        obs, r, term, trunc, _ = env.step(int(act)); done = term or trunc

rows = []
for j, name in enumerate(STATE_FEATURES):
    if A[j]:
        rows.append((name, np.mean(A[j]), np.mean(B[j]), np.mean(A[j]) * np.mean(B[j])))
rows.sort(key=lambda r: -abs(r[3]))
print(f"=== {a.label} ===")
print(f"{'特徴量':<28}{'A:負荷との相関':>15}{'B:方策との相関':>15}{'A×B':>9}")
for name, x, y, z in rows[:10]:
    print(f"{name:<28}{x:>15.4f}{y:>15.4f}{z:>+9.4f}")
print("\nA×B が正で大きい列 = 方策がその列を見ており、それが負荷の代理になっている")
