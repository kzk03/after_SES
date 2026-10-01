"""保存済みの最良チェックポイント (ppo_best.zip) を test で評価する。

学習スクリプトは最終モデルしか評価しないので、別途これで測る。
評価条件は学習 run と完全に同じ (同じ env 設定・同じ seed)。
"""
import argparse, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from sb3_contrib import MaskablePPO
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--ppo", required=True)
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--w-load", type=float, default=5.0)
ap.add_argument("--label", default="ppo_best")
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

for det in (False, True):
    rows = []
    for ep in range(len(eps) - nt, len(eps)):
        obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep})
        done, total = False, 0.0
        while not done:
            act, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=det)
            obs, r, term, trunc, info = env.step(int(act))
            total += float(info["reward_eval"])
            done = term or trunc
        s = env.episode_summary()
        rows.append({"reward": total, "n": s["n_requests"], "accept": s["accept_rate"],
                     "gini": s["gini_assigned"], "bus": s["bus_factor_50"]})
    df = pd.DataFrame(rows)
    tag = f"{a.label}{'_deterministic' if det else ''}"
    print(f"{tag:<26} 報酬/step {df.reward.sum()/df.n.sum():>8.4f}  "
          f"貢献率 {df.accept.mean():.4f}  集中度 {df['gini'].mean():.4f}  "
          f"busF {df.bus.mean():.2f}", flush=True)
