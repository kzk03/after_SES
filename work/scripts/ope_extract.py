"""off-policy 評価と「承諾モデル非依存の集中度」のための元データを抽出する。

logged 方策を force_logged_outcome で再生し、各リクエストについて
候補の特徴量・実際に選ばれた人・実際に貢献したか・PPO の選択確率・候補IDを保存する。
貢献はログの実測を使うので、これを使った評価は貢献モデルに依存しない。
"""
import pickle, sys, argparse
import numpy as np, torch
sys.path.insert(0, "src")
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder
from sb3_contrib import MaskablePPO

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/assign_ep.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_v3/contrib_model.joblib")
ap.add_argument("--ppo", default="outputs/rl/assignment/new4_stuck/ppo_assignment.zip")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="work/ope_new3.npz")
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb"))
eps = d["episodes"]; n_test = d["meta"]["n_test"]
test_idx = list(range(len(eps) - n_test, len(eps)))
model = AcceptanceModel.load(a.model)
env = AssignmentSimEnv(episodes=eps, history=d["history"], acceptance_model=model,
                       state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
                       n_candidates=20, seed=a.seed, w_accept=1.0, w_reject=0.0,
                       w_load=0.2, w_gini=0.25, load_cap=5,
                       force_logged_outcome=True, background_history=True, stuck_when_no_response=True)
ppo = MaskablePPO.load(a.ppo)

def probs(obs, mask):
    o, _ = ppo.policy.obs_to_tensor(obs)
    with torch.no_grad():
        dist = ppo.policy.get_distribution(
            o, action_masks=torch.as_tensor(mask, device=ppo.device).reshape(1, -1))
        return dist.distribution.probs.cpu().numpy().ravel()

feats, chosen, contributed, pe, masks, epi, cand_ids = [], [], [], [], [], [], []
vocab: dict = {}
for k, ep in enumerate(test_idx):
    obs, _ = env.reset(seed=a.seed + ep, options={"ep_idx": ep})
    done = False
    while not done:
        req = env._requests[env._ptr]
        act = env.logged_action(); m = env.action_masks()
        if act is not None:
            feats.append(obs["candidates"].astype(np.float32))
            masks.append(m.astype(np.int8)); chosen.append(act)
            contributed.append(int(req.get("logged_label", 0)))
            pe.append(probs(obs, m).astype(np.float32)); epi.append(ep)
            row = np.full(env.n_candidates, -1, dtype=np.int32)
            for i, e in enumerate(env._current_candidates()):
                row[i] = vocab.setdefault(e, len(vocab))
            cand_ids.append(row)
        obs, _, done, _, _ = env.step(act if act is not None else 0)
    if (k + 1) % 20 == 0:
        print(f"  {k+1}/{len(test_idx)} エピソード", flush=True)

np.savez_compressed(a.out, feats=np.stack(feats), masks=np.stack(masks),
                    chosen=np.array(chosen), accepted=np.array(contributed),
                    pe=np.stack(pe), episode=np.array(epi), cand_ids=np.stack(cand_ids))
print(f"\n保存 {a.out}: {len(chosen):,} リクエスト  実測貢献率 {np.mean(contributed):.4f}")
print(f"  PPO がログと同じ人を選ぶ確率の平均: {np.mean([p[c] for p,c in zip(pe,chosen)]):.4f}")
