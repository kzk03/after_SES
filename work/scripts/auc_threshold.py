"""貢献確率モデルの AUC をどこまで落とすと balanced が PPO に負けるかを測る。

N22 ではノイズを 0.5 固定で混ぜたが、その比率は恣意的だった。
ここでは混合比 alpha を振り、各点で
  (1) 劣化させた p_accept の AUC
  (2) balanced の報酬/step
を両方測って、AUC と性能の対応を出す。

  p' = (1-alpha)*p + alpha*rand    alpha=0 が元のモデル、1 が完全なランダム
"""
import pickle, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from sklearn.metrics import roc_auc_score
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--n-ep", type=int, default=25)
ap.add_argument("--w-load", type=float, default=0.05)
ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0")
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb")); eps = d["episodes"]; n_test = d["meta"]["n_test"]
m = AcceptanceModel.load(a.model)

def run(alpha):
    env = AssignmentSimEnv(episodes=eps, history=d["history"], acceptance_model=m,
        state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
        n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=a.w_load, w_gini=0.25,
        load_cap=5, background_history=True, stuck_when_no_response=True)
    rw=[]; n=0; ps=[]; ys=[]
    for ep in range(len(eps)-n_test, len(eps)-n_test+a.n_ep):
        rng = np.random.default_rng(42+ep)
        obs,_ = env.reset(seed=42+ep, options={"ep_idx": ep}); done=False
        while not done:
            p0 = env.candidate_p_accept().astype(np.float64)
            p = (1-alpha)*p0 + alpha*rng.random(len(p0))
            load = env.candidate_loads().astype(np.float64)
            msk = env.action_masks()
            v = np.where(msk, p/(1.0+np.maximum(load,0.0)), -np.inf)
            hi = v.max(); act = int(rng.choice(np.flatnonzero(v >= hi-1e-12)))
            ps.append(float(p[act]))
            obs, r, done, _, info = env.step(act); rw.append(r); n += 1
            ys.append(1 if info.get("accepted") else 0)
    # 選ばれた候補の劣化後 p と実際の貢献で AUC を測る
    auc = roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan")
    return np.sum(rw)/n, auc, np.mean(ys)

print(f"AUC と balanced の性能の対応 (AOSP、test {a.n_ep} エピソード、w_load={a.w_load})")
print("p' = (1-alpha)*p + alpha*rand\n")
print(f"{'alpha':>7}{'劣化後AUC':>11}{'報酬/step':>11}{'貢献率':>9}  PPO(+0.3941)と比較")
for al in [float(x) for x in a.alphas.split(",")]:
    r, auc, ac = run(al)
    mark = "勝ち" if r > 0.3941 else "**負け**"
    print(f"{al:>7.2f}{auc:>11.4f}{r:>11.4f}{ac:>9.4f}  {mark}")
