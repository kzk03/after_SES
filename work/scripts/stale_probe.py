"""無応答の依頼を N 日で自動解除したら、余裕のある候補が戻るかを測る。

「黙殺が多いコミュニティでは、賢く割り当てる前に滞留を解消する仕組みが要る」
という含意をシミュレータで直接確かめる。方策は least_load で固定し、
stale_timeout_days だけを変えて候補プールの余裕を比較する。

注意: 解除日数がエピソード長を超えるとエピソード内で一度も発火しない。
AOSP は 7 日エピソードなので、意味のある値は 1〜3 日程度。
(14/30 日で試して 3 設定とも完全に同じ数値になったのがこれのため)
"""
import pickle, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder
from train_assignment_rl import policy_logged, policy_least_load

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--n-ep", type=int, default=8)
ap.add_argument("--timeouts", default="0,30,14,7")
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb")); eps = d["episodes"]; n_test = d["meta"]["n_test"]
am = AcceptanceModel.load(a.model)

print(f"{'解除':>8} {'方策':<12} {'選んだ人の滞留':>14} {'cap以下の候補数':>15} {'超過率':>8}")
for t in [float(x) for x in a.timeouts.split(",")]:
    env = AssignmentSimEnv(episodes=eps, history=d["history"], acceptance_model=am,
        state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
        n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=0.2, w_gini=0.25,
        background_history=True, stuck_when_no_response=True, stale_timeout_days=t)
    for name, fn in [("least_load", policy_least_load), ("logged", policy_logged)]:
        chosen, free = [], []
        for ep in range(len(eps) - n_test, len(eps) - n_test + a.n_ep):
            rng = np.random.default_rng(42 + ep)
            obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep}); done = False
            while not done:
                loads = env.candidate_loads(); mask = env.action_masks().astype(bool)
                free.append(int(((loads <= 5) & mask).sum()))
                act = fn(env, obs, rng)
                chosen.append(float(loads[act]))
                obs, _, done, _, _ = env.step(act)
        c = np.array(chosen)
        lbl = "解除なし" if t == 0 else f"{t:.0f}日"
        print(f"{lbl:>8} {name:<12} {c.mean():>14.2f} {np.mean(free):>15.2f} {np.mean(c>5):>8.3f}")
