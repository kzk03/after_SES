"""報酬の各項が実際にどれだけの大きさかを測る。

「負荷を罰しているのに効かない」のは、項の大きさが釣り合っていないからかもしれない。
方策を変えたときに各項がどれだけ動くかを見る (動かない項は学習信号にならない)。
"""
import pickle, sys
import numpy as np
sys.path.insert(0, "src")
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.state_builder import RLStateBuilder

d = pickle.load(open("work/aosp_ep_7d.pkl", "rb"))
eps, nt = d["episodes"], d["meta"]["n_test"]
test_idx = list(range(len(eps) - nt, len(eps)))[:10]
env = AssignmentSimEnv(
    episodes=eps, history=d["history"],
    acceptance_model=AcceptanceModel.load("outputs/rl/contrib_model_aosp/contrib_model.joblib"),
    state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
    n_candidates=20, seed=42, w_accept=1.0, w_reject=0.3, w_load=0.05, w_gini=0.25,
    load_cap=5, background_history=True, stuck_when_no_response=True,
)
W = dict(accept=1.0, reject=0.3, load=0.05, gini=0.25)

def run(name):
    terms = {"貢献": [], "拒否": [], "滞留超過": [], "集中度": []}
    for ep in test_idx:
        obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep}); done = False
        while not done:
            m = env.action_masks(); v = m.astype(bool)
            load = env.candidate_loads().astype(np.float64)
            p = env.candidate_p_accept().astype(np.float64)
            score = (np.where(v, p/(1.0+load), -np.inf) if name == "balanced"
                     else np.where(v, p, -np.inf))
            obs, r, term, trunc, info = env.step(int(np.argmax(score)))
            # 環境が各項をそのまま info に出しているので、それを読む
            terms["貢献"].append(float(info["r_accept"]))
            terms["拒否"].append(float(info["r_reject"]))
            terms["滞留超過"].append(float(info["r_load"]))
            terms["集中度"].append(float(info["r_gini"]))
            done = term or trunc
    return {k: (np.mean(x), np.std(x)) for k, x in terms.items()}

a, b = run("balanced"), run("greedy_accept")
print(f"{'項':<10}{'balanced 平均':>15}{'greedy 平均':>14}{'差':>10}{'balanced のばらつき':>20}")
for k in a:
    print(f"{k:<10}{a[k][0]:>15.4f}{b[k][0]:>14.4f}{a[k][0]-b[k][0]:>10.4f}{a[k][1]:>20.4f}")
print("\n差 = 方策を変えたときにその項がどれだけ動くか (学習信号の大きさ)")
print("ばらつき = ステップ間の変動 (小さいほど「いつも同じ値」で信号にならない)")
