"""1 run の結果をまとめる: 方策比較・対応差・報酬内訳・重み感度。"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="outputs/rl/assignment/new4_stuck")
ap.add_argument("--w-load", type=float, default=0.2)
ap.add_argument("--w-gini", type=float, default=0.25)
a = ap.parse_args()

p = pd.read_csv(f"{a.dir}/per_episode.csv")
cols = [c for c in ["accept_rate","gini_accepted","gini_assigned","n_active_reviewers",
                    "bus_factor_50","max_reviewer_share","total_reward"] if c in p.columns]
order = ["greedy_accept","ppo","ppo_deterministic","least_load","random","logged"]
g = p.groupby("policy")[cols].mean()
print("=== 方策比較 (test エピソード平均) ===")
print(g.reindex([o for o in order if o in g.index]).round(4).to_string())

rng = np.random.default_rng(0)
def paired(x, y, col, good):
    e = p[p.policy.isin([x, y])].pivot(index="episode", columns="policy", values=col).dropna()
    d = (e[x] - e[y]).to_numpy()
    bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
    f = np.mean(bs > 0) if good == "up" else np.mean(bs < 0)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), f, len(d)

print("\n=== PPO − 各ベースライン (エピソード対応, bootstrap 95%CI) ===")
for col, lbl, good in [("accept_rate","貢献率","up"), ("gini_assigned","集中度(割当)","down"),
                       ("gini_accepted","集中度(貢献)","down"), ("n_active_reviewers","有効レビュアー","up"),
                       ("bus_factor_50","bus factor","up")]:
    print(f"  [{lbl}]")
    for b in ["logged","random","least_load","greedy_accept"]:
        if b not in p.policy.unique(): continue
        m, lo, hi, f, n = paired("ppo", b, col, good)
        print(f"    vs {b:<14} {m:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]  改善方向 {f:>6.1%}")

try:
    s = pd.read_csv(f"{a.dir}/step_log.csv")
    print("\n=== ステップ単位 (ベースラインのみ) ===")
    gg = s.groupby("policy").agg(貢献=("accepted","mean"), 負荷=("load_before","mean"),
                                 負荷超過=("load_excess","mean"), 集中度=("gini","mean"))
    print(gg.round(4).to_string())
    print("\n報酬の内訳 (1ステップあたり)")
    for pol, x in s.groupby("policy"):
        acc, ld, gi = x.accepted.mean(), x.load_excess.mean(), x.gini.mean()
        print(f"  {pol:<14} 貢献 {acc:+.4f}  −{a.w_load}*負荷 {-a.w_load*ld:+.4f}  "
              f"−{a.w_gini}*集中 {-a.w_gini*gi:+.4f}  = {acc-a.w_load*ld-a.w_gini*gi:+.4f}")
except FileNotFoundError:
    print("\n(step_log.csv なし)")
