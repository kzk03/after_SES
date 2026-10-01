"""step_log からベースラインの指標を集計する (PPO の完了を待たずに見る用)。

⚠️ 実際には step_log.csv も per_episode.csv と同じタイミング (全評価の完了後) に
書かれるので、PPO 学習中には使えない。完了後に重みを変えて採点し直す用途で使う。
"""
import argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--w-load", type=float, required=True)
ap.add_argument("--w-gini", type=float, default=0.25)
a = ap.parse_args()

s = pd.read_csv(f"{a.run_dir}/step_log.csv")
g = s.groupby("policy").agg(貢献=("accepted", "mean"), 滞留=("load_before", "mean"),
                            滞留超過=("load_excess", "mean"), 集中度=("gini", "mean"))
g["報酬/step"] = g["貢献"] - a.w_load * g["滞留超過"] - a.w_gini * g["集中度"]
print(f"{a.run_dir}  w_load={a.w_load} w_gini={a.w_gini}  {len(s):,} ステップ")
print(g.sort_values("報酬/step", ascending=False).round(4).to_string())
