"""balanced が貢献確率モデルにどれだけ依存しているかを測る。

balanced = p_accept / (1 + load) は貢献確率モデルの出力を使う。
モデルが無い運用でも使えるのか、モデルの質にどれだけ左右されるのかを、
p_accept を段階的に劣化させて確かめる。

  full   : そのまま (AUC 0.82)
  shuffle: p_accept を候補間でシャッフル (情報を完全に壊す)
  const  : p_accept を全員同じ値に (= least_load と同じ)
  noisy  : p_accept に一様ノイズを混ぜる (質を落とす)
"""
import pickle, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--n-ep", type=int, default=20)
ap.add_argument("--w-load", type=float, default=0.05)
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb")); eps = d["episodes"]; n_test = d["meta"]["n_test"]
m = AcceptanceModel.load(a.model)

def make_env():
    return AssignmentSimEnv(episodes=eps, history=d["history"], acceptance_model=m,
        state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
        n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=a.w_load, w_gini=0.25,
        load_cap=5, background_history=True, stuck_when_no_response=True)

def run(mode):
    env = make_env(); rw=[]; n=0; acc=[]
    for ep in range(len(eps)-n_test, len(eps)-n_test+a.n_ep):
        rng = np.random.default_rng(42+ep)
        obs,_ = env.reset(seed=42+ep, options={"ep_idx": ep}); done=False
        while not done:
            p = env.candidate_p_accept().astype(np.float64)
            if mode == "shuffle":   p = rng.permutation(p)
            elif mode == "const":   p = np.full_like(p, 0.5)
            elif mode == "noisy":   p = np.clip(0.5*p + 0.5*rng.random(len(p)), 0, 1)
            load = env.candidate_loads().astype(np.float64)
            msk = env.action_masks()
            v = np.where(msk, p/(1.0+np.maximum(load,0.0)), -np.inf)
            hi = v.max(); act = int(rng.choice(np.flatnonzero(v >= hi-1e-12)))
            obs, r, done, _, info = env.step(act); rw.append(r); n += 1
            acc.append(1.0 if info.get("accepted") else 0.0)
    return np.sum(rw)/n, np.mean(acc)

print(f"balanced の貢献確率モデルへの依存 (AOSP、test {a.n_ep} エピソード、w_load={a.w_load})\n")
print(f"{'p_accept の扱い':<28}{'報酬/step':>10}{'貢献率':>9}")
for mode, lbl in [("full","そのまま (AUC 0.82)"), ("noisy","ノイズ半分混ぜる"),
                  ("shuffle","候補間でシャッフル"), ("const","全員 0.5 (= least_load)")]:
    r, ac = run(mode)
    print(f"{lbl:<28}{r:>10.4f}{ac:>9.4f}")
