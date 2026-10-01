"""ベースラインが持つ「神託 p_accept」の価値を測る。

環境は accepted = random() < p_accept で結果を決める (assignment_sim_env.py:694)。
ベースラインはその p_accept をそのまま読む。PPO は 26 次元から推論する。

そこで **p_accept を意図的に劣化させた balanced** を作り、
劣化の度合いと性能の関係を見る。PPO の性能がこの曲線のどこに乗るかで、
「PPO は実質どれくらいの精度で p_accept を推定できているか」が読める。
"""
import argparse, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--n-ep", type=int, default=25)
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb"))
eps, nt = d["episodes"], d["meta"]["n_test"]
test_idx = list(range(len(eps) - nt, len(eps)))[: a.n_ep]
env = AssignmentSimEnv(
    episodes=eps, history=d["history"], acceptance_model=AcceptanceModel.load(a.model),
    state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
    n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=0.05, w_gini=0.25,
    load_cap=5, background_history=True, stuck_when_no_response=True,
)

def run(noise: float, seed: int = 42) -> dict:
    """p_accept を noise の割合だけ一様乱数に置き換えた balanced を評価する。"""
    rows = []
    for ep in test_idx:
        rng = np.random.default_rng(seed + ep)
        obs, _ = env.reset(seed=seed + ep, options={"ep_idx": ep})
        done, total = False, 0.0
        while not done:
            p = env.candidate_p_accept().astype(np.float64)
            m = env.action_masks()
            if noise > 0:
                p = (1 - noise) * p + noise * rng.random(len(p))
            load = env.candidate_loads().astype(np.float64)
            score = np.where(m, p / (1.0 + load), -np.inf)
            obs, r, term, trunc, info = env.step(int(np.argmax(score)))
            total += float(r)
            done = term or trunc
        s_ = env.episode_summary()
        rows.append({"reward": total, "n": s_["n_requests"],
                     "accept": s_["accept_rate"], "gini": s_["gini_assigned"]})
    df = pd.DataFrame(rows)
    return {"報酬/step": df.reward.sum() / df.n.sum(),
            "貢献率": df.accept.mean(), "集中度": df["gini"].mean()}

print(f"test {len(test_idx)} エピソードで測定\n")
print(f"{'p_accept の劣化':>16}{'報酬/step':>11}{'貢献率':>9}{'集中度':>9}")
for noise in (0.0, 0.25, 0.5, 0.75, 1.0):
    r = run(noise)
    lbl = "そのまま(神託)" if noise == 0 else ("完全ランダム" if noise == 1 else f"{noise:.0%} 置換")
    print(f"{lbl:>16}{r['報酬/step']:>11.4f}{r['貢献率']:>9.4f}{r['集中度']:>9.4f}", flush=True)
