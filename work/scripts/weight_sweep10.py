"""10 方策の順位が重みでどう変わるかを step_log から採点し直して調べる。

balanced が PPO を上回るのが、採用した重み 1 点だけの現象かを確かめる。
⚠️ 学習済み PPO を別の重みで採点し直すだけで、各重みでの再学習はしていない
(PPO には不利な比較。再学習すれば PPO はもう少し良くなりうる)。
"""
import argparse, itertools
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--label", default="")
a = ap.parse_args()

s = pd.read_csv(f"{a.run_dir}/step_log.csv")
g = s.groupby("policy").agg(acc=("accepted","mean"), ld=("load_excess","mean"), gi=("gini","mean"))
pols = [p for p in g.index if p != "ppo_deterministic"]
loads = [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]; ginis = [0.0, 0.1, 0.25, 0.5]

win = {p: 0 for p in pols}; best = {p: 0 for p in pols}
bal_gt_ppo = 0; tot = 0
for wl, wg in itertools.product(loads, ginis):
    r = {p: g.loc[p,"acc"] - wl*g.loc[p,"ld"] - wg*g.loc[p,"gi"] for p in pols}
    b = max(r, key=r.get); best[b] += 1
    tot += 1
    if r.get("balanced", -9) > r.get("ppo", -9): bal_gt_ppo += 1
    for p in pols:
        if r[p] > r["ppo"]: win[p] += 1

print(f"{a.label or a.run_dir}  ({tot} 通りの重み)")
print(f"\n  最良になった回数:")
for p, c in sorted(best.items(), key=lambda x: -x[1]):
    if c: print(f"    {p:<18} {c:>3}/{tot}")
print(f"\n  PPO を上回った回数:")
for p, c in sorted(win.items(), key=lambda x: -x[1]):
    if p != "ppo": print(f"    {p:<18} {c:>3}/{tot}")
print(f"\n  **balanced > ppo: {bal_gt_ppo}/{tot}**")
