"""貢献モデルの学習データを、シミュレータと同じ計算方法で作る。

環境 (background_history=True) は
  - エピソード内の change の生 CSV 行を除外 (エージェントが割り当てを変えうるため)
  - エピソード内の過去の割り当てをシミュレート行として追加
  - ログ再生時の応答時刻は実測値
なので、学習データも同じにしないと G2 が合わない。
"""
import sys, pickle, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--out", default="work/aosp_train.parquet")
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb"))
eps, history = d["episodes"], d["history"]
print(f"エピソード {len(eps)}  履歴 {len(history):,} 行")

per = {}
for em, g in history.groupby("email"):
    per.setdefault(str(em), []).append(g)
for ow, g in history[history.owner_email.astype(str) != ""].groupby("owner_email"):
    per.setdefault(str(ow), []).append(g)
hist = {k: (v[0] if len(v) == 1 else pd.concat(v, ignore_index=True)).sort_values("timestamp", kind="mergesort")
        for k, v in per.items()}
ts_of = {k: v["timestamp"].to_numpy() for k, v in hist.items()}
chg_of = {k: pd.to_numeric(v["chg"], errors="coerce").to_numpy() for k, v in hist.items()}
all_ts = history["timestamp"].to_numpy()

sb = RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False)
rows, X = [], []
for ei, ep in enumerate(eps):
    ep_ch = list({int(r["change_id"]) for r in ep["requests"]})
    prior: dict = {}
    for r in ep["requests"]:
        em = r["logged_reviewer"]; now = pd.Timestamp(r["timestamp"])
        h = hist.get(em)
        if h is None: continue
        k = int(np.searchsorted(ts_of[em], np.datetime64(now), side="left"))
        if k == 0: continue
        df = h.iloc[:k][~np.isin(chg_of[em][:k], ep_ch)]
        extra = prior.get(em)
        if extra:
            df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
        tot = int(np.searchsorted(all_ts, np.datetime64(now), side="left"))
        X.append(sb.build_obs(email=em, activity_df=df, feature_end=now.to_pydatetime(),
                              episode_assign_count=0, total_project_reviews=max(1, tot),
                              feature_window_days=180).astype(np.float32))
        rows.append({"ep": ei, "project": ep.get("project",""), "reviewer": em,
                     "added_at": now, "change": int(r["change_id"]),
                     "contributed": int(r["logged_label"])})
        rd = r.get("logged_resp_days")
        prior.setdefault(em, []).append({
            "email": em, "timestamp": now, "label": int(r["logged_label"]),
            "owner_email": str(r.get("owner_email","")), "change_insertions": 0,
            "change_deletions": 0, "change_files_count": 0,
            "first_response_time": (now + pd.Timedelta(days=float(rd))) if rd is not None else None,
            "project": ep.get("project",""), "chg": -1})
    if (ei+1) % 100 == 0:
        print(f"  {ei+1}/{len(eps)}  {len(rows):,} 行", flush=True)

X = np.stack(X); out = pd.DataFrame(rows)
for j in range(X.shape[1]): out[f"f{j:02d}"] = X[:, j]
out.to_parquet(a.out)
print(f"保存 {a.out}: {len(out):,} 行  貢献率 {out.contributed.mean():.4f}")
