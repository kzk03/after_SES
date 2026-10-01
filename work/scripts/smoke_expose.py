"""expose_policy_inputs の動作確認: 次元・値域・承諾モデルへの影響がないこと。"""
import pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder

d = pickle.load(open("work/aosp_ep_7d.pkl", "rb")); eps = d["episodes"]; n_test = d["meta"]["n_test"]
m = AcceptanceModel.load("outputs/rl/contrib_model_aosp/contrib_model.joblib")
def mk(expose):
    return AssignmentSimEnv(episodes=eps, history=d["history"], acceptance_model=m,
        state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
        n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=0.05, w_gini=0.25,
        load_cap=5, background_history=True, stuck_when_no_response=True,
        expose_policy_inputs=expose)

ep = len(eps) - n_test
a, b = mk(False), mk(True)
oa, _ = a.reset(seed=42+ep, options={"ep_idx": ep})
ob, _ = b.reset(seed=42+ep, options={"ep_idx": ep})
print(f"次元      : off {oa['candidates'].shape}  on {ob['candidates'].shape}")
print(f"先頭26列が一致: {np.allclose(oa['candidates'][:, :26], ob['candidates'][:, :26])}")
print(f"承諾確率が一致: {np.allclose(a.candidate_p_accept(), b.candidate_p_accept())}")
n = int(a.action_masks().sum())
ex = ob["candidates"][:n, 26:]
print(f"\n追加3列 (有効候補 {n} 人)")
for j, nm in enumerate(["p_accept", "load_norm", "over_cap"]):
    print(f"  {nm:<10} 最小 {ex[:,j].min():.4f}  最大 {ex[:,j].max():.4f}  平均 {ex[:,j].mean():.4f}")
print(f"  p_accept が env と一致: {np.allclose(ex[:,0], b.candidate_p_accept()[:n], atol=1e-6)}")
print(f"  無効候補は 0: {np.allclose(ob['candidates'][n:, 26:], 0)}")
# 20 ステップ回して値域を確認
mx = np.zeros(3)
for _ in range(20):
    o, _, done, _, _ = b.step(int(np.flatnonzero(b.action_masks())[0]))
    k = int(b.action_masks().sum())
    if k: mx = np.maximum(mx, o["candidates"][:k, 26:].max(axis=0))
    if done: break
print(f"\n20ステップ後の最大値: p {mx[0]:.3f}  load_norm {mx[1]:.3f}  over_cap {mx[2]:.3f}  (Box は ±10)")
