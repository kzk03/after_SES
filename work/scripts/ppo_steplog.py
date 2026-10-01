"""学習済み PPO のステップログを取る (重み感度の再採点用)。

train_assignment_rl.py の evaluate_ppo にステップログ対応を入れたが、
既存の run は取れていないので評価だけ回して取得する。
"""
import pickle, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from sb3_contrib import MaskablePPO
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from train_assignment_rl import make_env, parse_args, evaluate_ppo

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/assign_ep.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_v3/contrib_model.joblib")
ap.add_argument("--ppo", default="outputs/rl/assignment/new4_stuck/ppo_assignment.zip")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default="work/ppo_steps_seed42.csv")
a = ap.parse_args()

sys.argv = ["x", "--episodes", a.episodes, "--acceptance-model", a.model,
            "--output-dir", "/tmp/_x", "--state-dim", "26", "--seed", str(a.seed),
            "--n-test-episodes", "109", "--w-accept", "1.0", "--w-reject", "0.0",
            "--w-load", "0.2", "--w-gini", "0.25", "--load-cap", "5",
            "--background-history", "--stuck-when-no-response"]
base = parse_args()
d = pickle.load(open(a.episodes, "rb"))
eps = d["episodes"]; n_test = d["meta"]["n_test"]
env = make_env(eps, d["history"], AcceptanceModel.load(a.model), base, a.seed, None)
ppo = MaskablePPO.load(a.ppo)
logs = []
evaluate_ppo(env, ppo, list(range(len(eps)-n_test, len(eps))), a.seed, step_logs=logs)
out = pd.concat(logs, ignore_index=True)
out.to_csv(a.out, index=False)
print(f"保存 {a.out}: {len(out):,} ステップ")
print(out.agg({"accepted":"mean","load_excess":"mean","gini":"mean"}).round(4).to_string())
