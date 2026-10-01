"""学習済み方策が「負荷」をどう扱っているかを直接調べる。

評価指標(集中度)は結果でしかないので、方策そのものを見る。
候補の負荷と、方策がその候補に置く確率の関係を測る。

  balanced なら p/(1+load) なので負荷と確率は強い負の相関を持つはず。
  貪欲寄りなら相関はほぼ 0 になる。
"""
import argparse, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from sb3_contrib import MaskablePPO
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--ppo", nargs="+", required=True, help="比較する ppo_assignment.zip")
ap.add_argument("--labels", nargs="+", required=True)
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--n-ep", type=int, default=12)
ap.add_argument("--w-load", type=float, default=0.05)
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb"))
eps, nt = d["episodes"], d["meta"]["n_test"]
test_idx = list(range(len(eps) - nt, len(eps)))[: a.n_ep]
env = AssignmentSimEnv(
    episodes=eps, history=d["history"], acceptance_model=AcceptanceModel.load(a.model),
    state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
    n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=a.w_load, w_gini=0.25,
    load_cap=5, background_history=True, stuck_when_no_response=True,
)

print(f"test {len(test_idx)} エピソードで方策の中身を調べる\n")
print(f"{'方策':<22}{'負荷との相関':>13}{'p_acceptとの相関':>17}{'エントロピー':>12}")

for path, lbl in zip(a.ppo, a.labels):
    model = MaskablePPO.load(path, device="cpu")
    c_load, c_p, ent = [], [], []
    for ep in test_idx:
        obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep}); done = False
        while not done:
            m = env.action_masks()
            dist = model.policy.get_distribution(
                {k: __import__("torch").as_tensor(v).unsqueeze(0) for k, v in obs.items()},
                action_masks=m[None, :],
            )
            pr = dist.distribution.probs.detach().numpy()[0]
            load = env.candidate_loads(); pa = env.candidate_p_accept()
            v = m.astype(bool)
            if v.sum() > 2 and pr[v].std() > 1e-9:
                if load[v].std() > 1e-9: c_load.append(np.corrcoef(pr[v], load[v])[0, 1])
                if pa[v].std() > 1e-9: c_p.append(np.corrcoef(pr[v], pa[v])[0, 1])
                q = pr[v] / pr[v].sum()
                ent.append(float(-(q * np.log(q + 1e-12)).sum() / np.log(v.sum())))
            act, _ = model.predict(obs, action_masks=m, deterministic=False)
            obs, r, term, trunc, _ = env.step(int(act)); done = term or trunc
    print(f"{lbl:<22}{np.mean(c_load):>13.4f}{np.mean(c_p):>17.4f}{np.mean(ent):>12.4f}", flush=True)

print("\n負荷との相関: 負なら「重い人を避ける」を学べている (balanced は強い負)")
print("エントロピー: 1.0 に近いほど一様乱択に近い")
