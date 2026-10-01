"""G2: シミュレータが計算する特徴量と貢献モデルの学習データが一致するか。

一致していないと、モデルは学習時と違う入力に対して予測を出すことになり、
その予測を PPO が最大化する構図になって結果が意味を失う。
"""
import pickle, re, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder
from review_predictor.IRL.features.common_features import STATE_FEATURES

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/assign_ep.pkl")
ap.add_argument("--train", default="work/contrib_train.parquet")
ap.add_argument("--model", default="outputs/rl/contrib_model_v3/contrib_model.joblib")
ap.add_argument("--n-rows", type=int, default=400)
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb"))
eps, hist = d["episodes"], d["history"]; n_test = d["meta"]["n_test"]
model = AcceptanceModel.load(a.model)
env = AssignmentSimEnv(episodes=eps, history=hist, acceptance_model=model,
                       state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
                       n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=0.2, w_gini=0.25,
                       force_logged_outcome=True, background_history=True)
tr = pd.read_parquet(a.train)
F = [c for c in tr.columns if re.fullmatch(r"f\d{2}", c)]
key = {(r.reviewer, pd.Timestamp(r.added_at)): i for i, r in enumerate(tr.itertuples())}
A, B = [], []
for ep in range(len(eps) - n_test, len(eps)):
    obs, _ = env.reset(seed=42 + ep, options={"ep_idx": ep})
    done = False
    while not done:
        req = env._requests[env._ptr]; cs = env._current_candidates()
        lg = req.get("logged_reviewer", "")
        if lg in cs:
            k = key.get((lg, pd.Timestamp(req["timestamp"])))
            if k is not None:
                A.append(tr.iloc[k][F].to_numpy(dtype=np.float64))
                B.append(obs["candidates"][cs.index(lg)].astype(np.float64))
        act = env.logged_action()
        obs, _, done, _, _ = env.step(act if act is not None else 0)
    if len(A) >= a.n_rows:
        break

A = np.stack(A); B = np.stack(B)
print(f"突き合わせた行: {len(A):,}\n")
print(f"{'次元':<32}{'一致率':>10}{'平均差':>12}{'相関':>10}")
ok = 0
for j, n in enumerate(STATE_FEATURES[:26]):
    x, y = A[:, j], B[:, j]
    same = np.mean(np.abs(x - y) < 1e-4)
    c = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else (1.0 if np.allclose(x, y) else 0.0)
    ok += same > 0.8
    print(f"{n:<32}{same:>10.4f}{np.mean(x-y):>+12.5f}{c:>10.4f}{'' if same>0.8 else '  <<<'}")
print(f"\n一致率0.8超の次元: {ok} / 26")
pa = model.predict_proba(A.astype(np.float32)); pb = model.predict_proba(B.astype(np.float32))
print(f"予測確率: 平均差 {np.mean(pa-pb):+.5f}  絶対平均差 {np.mean(np.abs(pa-pb)):.5f}  相関 {np.corrcoef(pa,pb)[0,1]:.4f}")
