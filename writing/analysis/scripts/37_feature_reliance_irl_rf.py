# -*- coding: utf-8 -*-
"""各モデルのスコアが state 特徴量とどれだけ連動するか (rank-based) を IRL vs RF で比較.
未校正スコアでも使える Spearman |ρ| で「どの特徴量にスコアが依存するか」を測り、
累積/経験系 vs 直近/変動系 で IRL/RF の依存度を対比する (経験軸の特徴量レベルの裏付け)。
入力: data/feature_values.csv(t0=2022-01-01), data/pair_trajectories.csv
出力: results/feature_reliance_irl_rf.csv, results/fig_feature_reliance.pdf
"""
import pathlib
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa
ROOT=pathlib.Path(__file__).resolve().parents[1]; D,OUT=ROOT/"data",ROOT/"results"
H=["0-3m","3-6m","6-9m","9-12m"]
CUMUL={"total_reviews","total_changes","core_reviewer_ratio","unique_collaborator_count",
 "window_tenure_days","repeat_collaboration_rate","avg_review_size","avg_change_lines",
 "avg_action_intensity","reciprocity_score","active_months_ratio"}
fv=pd.read_csv(D/"feature_values.csv"); feats=[c for c in fv.columns if not c.startswith("__")]
t0=fv[fv["__snapshot"]=="2022-01-01"].drop_duplicates("__email").set_index("__email")
pt=pd.read_csv(D/"pair_trajectories.csv")
rows=[pt[["developer",f"irl_{h}",f"rf_{h}"]].rename(columns={f"irl_{h}":"irl",f"rf_{h}":"rf"}) for h in H]
lg=pd.concat(rows,ignore_index=True)
fj=t0.reindex(lg["developer"].values)[feats].reset_index(drop=True)
lg=pd.concat([lg.reset_index(drop=True),fj],axis=1)
res=[]
for f in feats:
    m=lg[[f,"irl","rf"]].dropna()
    if len(m)<100: continue
    res.append((f,"累積/経験系" if f in CUMUL else "直近/変動系",
                abs(spearmanr(m[f],m["irl"]).correlation),abs(spearmanr(m[f],m["rf"]).correlation)))
r=pd.DataFrame(res,columns=["feature","group","absrho_IRL","absrho_RF"])
r.to_csv(OUT/"feature_reliance_irl_rf.csv",index=False)
gm=r.groupby("group")[["absrho_IRL","absrho_RF"]].mean().reindex(["累積/経験系","直近/変動系"])
fig,ax=plt.subplots(figsize=(6.2,4.0)); x=np.arange(2); w=0.36
b1=ax.bar(x-w/2,gm["absrho_IRL"],w,label="IRL",color="#4C72B0")
b2=ax.bar(x+w/2,gm["absrho_RF"],w,label="RF",color="#DD8452")
for b in list(b1)+list(b2):
    ax.annotate(f"{b.get_height():.2f}",(b.get_x()+b.get_width()/2,b.get_height()),
                ha="center",va="bottom",fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(["累積/経験系\n(total_reviews 等)","直近/変動系\n(recency 等)"])
ax.set_ylabel("特徴量との依存度  平均 |Spearman ρ|"); ax.set_ylim(0,0.36); ax.legend()
ax.set_title("各モデルのスコアは何に依存するか\nRFは累積/経験系に強く依存・直近系はIRLと互角(=経験軸)")
fig.tight_layout(); fig.savefig(OUT/"fig_feature_reliance.pdf",format="pdf")
print(gm.round(3).to_string()); print("saved fig_feature_reliance.pdf")
