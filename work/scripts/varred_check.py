"""分散低減の実装が意図どおり効いているかを単体で確かめる。

確認すること:
  1. 学習報酬の分散が下がっているか (これが狙い)
  2. **評価報酬 (reward_eval) は変わっていないか** (比較の公平性が保たれる条件)
  3. 環境の遷移 (承諾の成否・集中度) が変わっていないか

同じ seed で 2 つの env を走らせ、同じ行動列を与えて比べる。
"""
import pickle, sys
import numpy as np
sys.path.insert(0, "src")
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.state_builder import RLStateBuilder

d = pickle.load(open("work/aosp_ep_7d.pkl", "rb"))
eps, nt = d["episodes"], d["meta"]["n_test"]
m = AcceptanceModel.load("outputs/rl/contrib_model_aosp/contrib_model.joblib")

def mk(flag):
    return AssignmentSimEnv(
        episodes=eps, history=d["history"], acceptance_model=m,
        state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
        n_candidates=20, seed=42, w_accept=1.0, w_reject=0.3, w_load=0.05, w_gini=0.25,
        load_cap=5, background_history=True, stuck_when_no_response=True,
        expected_accept_reward=flag,
    )

e_off, e_on = mk(False), mk(True)
r_off, r_on, ev_off, ev_on, acc_off, acc_on, g_off, g_on = [], [], [], [], [], [], [], []
for ep in range(len(eps) - nt, len(eps) - nt + 6):
    o1, _ = e_off.reset(seed=42 + ep, options={"ep_idx": ep})
    o2, _ = e_on.reset(seed=42 + ep, options={"ep_idx": ep})
    rng = np.random.default_rng(1000 + ep)
    done = False
    while not done:
        v = np.flatnonzero(e_off.action_masks())
        a = int(rng.choice(v))                       # 両者に同じ行動を与える
        _, r1, t1, tr1, i1 = e_off.step(a)
        _, r2, t2, tr2, i2 = e_on.step(a)
        r_off.append(r1); r_on.append(r2)
        ev_off.append(i1["reward_eval"]); ev_on.append(i2["reward_eval"])
        acc_off.append(i1["accepted"]); acc_on.append(i2["accepted"])
        g_off.append(i1["gini"]); g_on.append(i2["gini"])
        done = t1 or tr1

print(f"ステップ数 {len(r_off):,}\n")
print(f"{'':<22}{'OFF':>12}{'ON':>12}{'判定':>10}")
print(f"{'学習報酬の SD':<22}{np.std(r_off):>12.4f}{np.std(r_on):>12.4f}"
      f"{'下がった' if np.std(r_on)<np.std(r_off) else '⚠️下がらず':>10}")
print(f"{'学習報酬の平均':<22}{np.mean(r_off):>12.4f}{np.mean(r_on):>12.4f}"
      f"{'ほぼ同じ' if abs(np.mean(r_on)-np.mean(r_off))<0.02 else '⚠️ずれた':>10}")
same_ev = np.allclose(ev_off, ev_on)
same_acc = np.array_equal(acc_off, acc_on)
same_g = np.allclose(g_off, g_on)
print(f"\n{'評価報酬が同一':<22}{'':>24}{'OK' if same_ev else '⚠️ NG':>10}")
print(f"{'承諾の成否が同一':<22}{'':>24}{'OK' if same_acc else '⚠️ NG':>10}")
print(f"{'集中度が同一':<22}{'':>24}{'OK' if same_g else '⚠️ NG':>10}")
print("\n評価報酬・承諾・集中度が同一 = 環境の遷移も評価も変えていない (公平性が保たれる)")
