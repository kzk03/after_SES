"""学習済み PPO の評価だけを回す (学習はやり直さない)。

セッション終了などで評価フェーズが中断した run の復旧用。
train_assignment_rl.py と同じ手順でベースラインと PPO を評価し、
per_episode.csv / summary.csv を書き出す。
"""
import pickle, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from sb3_contrib import MaskablePPO
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from train_assignment_rl import (make_env, parse_args, evaluate_policy, evaluate_ppo,
                                 policy_logged, policy_random, policy_greedy_accept,
                                 policy_least_load)

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/assign_ep.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_v3/contrib_model.joblib")
ap.add_argument("--run-dir", required=True)
ap.add_argument("--seed", type=int, required=True)
a = ap.parse_args()

sys.argv = ["x", "--episodes", a.episodes, "--acceptance-model", a.model,
            "--output-dir", a.run_dir, "--state-dim", "26", "--seed", str(a.seed),
            "--n-test-episodes", "109", "--w-accept", "1.0", "--w-reject", "0.0",
            "--w-load", "0.2", "--w-gini", "0.25", "--load-cap", "5",
            "--background-history", "--stuck-when-no-response"]
base = parse_args()
d = pickle.load(open(a.episodes, "rb"))
eps = d["episodes"]; n_test = d["meta"]["n_test"]
test_idx = list(range(len(eps) - n_test, len(eps)))
env = make_env(eps, d["history"], AcceptanceModel.load(a.model), base, a.seed, None)
res = []
for name, fn in [("logged", policy_logged), ("random", policy_random),
                 ("greedy_accept", policy_greedy_accept), ("least_load", policy_least_load)]:
    res.append(evaluate_policy(env, fn, test_idx, a.seed, name))
    print(f"  {name} 完了", flush=True)
ppo = MaskablePPO.load(f"{a.run_dir}/ppo_assignment.zip")
res.append(evaluate_ppo(env, ppo, test_idx, a.seed))
res.append(evaluate_ppo(env, ppo, test_idx, a.seed, deterministic=True))
out = pd.concat(res, ignore_index=True)
out.to_csv(f"{a.run_dir}/per_episode.csv", index=False)
g = out.groupby("policy").agg(["mean", "std"])
g.to_csv(f"{a.run_dir}/summary.csv")
print(f"\n保存 {a.run_dir}/per_episode.csv")
print(out.groupby("policy")[["accept_rate","gini_assigned","n_active_reviewers","total_reward"]].mean().round(4).to_string())
