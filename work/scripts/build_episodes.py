"""割り当てイベントからエピソードを作る。

リクエスト = 他人がレビュアーを追加した瞬間、時刻 = 実際の追加時刻、
ラベル = 貢献したか。履歴 (特徴量・負荷用) は既存 CSV を流用する。
分割は cutoff で行い、train は cutoff より前に終わるもの、test は cutoff 以降に
始まるものだけを残す (期間が重ならないようにする)。
"""
import argparse, pickle
from typing import Any, Dict, List
import numpy as np, pandas as pd

ACT = ["email","timestamp","label","owner_email","project",
       "change_insertions","change_deletions","change_files_count","first_response_time","chg"]
ap = argparse.ArgumentParser()
ap.add_argument("--events", default="work/assign_events.parquet")
ap.add_argument("--raw-csv", default="data/combined_raw_main32.csv")
ap.add_argument("--start", default="2021-01-01"); ap.add_argument("--end", default="2026-01-01")
ap.add_argument("--episode-days", type=int, default=60)
ap.add_argument("--n-candidates", type=int, default=20)
ap.add_argument("--active-window-days", type=int, default=90)
ap.add_argument("--min-requests", type=int, default=8)
ap.add_argument("--min-episodes", type=int, default=3)
ap.add_argument("--horizon", type=float, default=14.0)
ap.add_argument("--cutoff", default="2024-10-01")
ap.add_argument("--out", default="work/assign_ep.pkl")
a = ap.parse_args()

ev = pd.read_parquet(a.events)
ev = ev[ev.added_at.notna()].copy()
ev["contributed"] = (ev.lat_days <= a.horizon).fillna(False).astype(int)
ev = ev[(ev.added_at >= a.start) & (ev.added_at < a.end)].sort_values("added_at")
print(f"割り当てイベント {len(ev):,}  貢献率 {ev.contributed.mean():.4f}")

raw = pd.read_csv(a.raw_csv)
raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
raw = raw.dropna(subset=["timestamp"])
for c in ["change_insertions","change_deletions","change_files_count"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0).astype(int)
raw["owner_email"] = raw["owner_email"].fillna("").astype(str)
raw["chg"] = pd.to_numeric(raw["change_id"].astype(str).str.rsplit("~", n=1).str[-1], errors="coerce")
history = raw[ACT].sort_values("timestamp", kind="mergesort").reset_index(drop=True)

def cands(obs, ts_, logged):
    lo = ts_ - pd.Timedelta(days=a.active_window_days)
    w = obs[(obs["timestamp"] >= lo) & (obs["timestamp"] < ts_)]
    if w.empty: return [logged] if logged else []
    last = w.groupby("email")["timestamp"].max().sort_values(ascending=False)
    c = list(last.index[:a.n_candidates])
    if logged and logged not in c:
        if len(c) >= a.n_candidates: c[-1] = logged
        else: c.append(logged)
    return c

start = pd.Timestamp(a.start); span = pd.Timedelta(days=a.episode_days)
episodes: List[Dict[str, Any]] = []; per_project = []; skipped = 0
for proj, g in ev.groupby("project"):
    obs = history[history.project == proj]
    if obs.empty: continue
    eps_p: List[Dict[str, Any]] = []; cur: List[Dict[str, Any]] = []; ep_start = start
    def flush():
        global cur
        if len(cur) >= a.min_requests:
            eps_p.append({"project": proj, "start": ep_start, "end": ep_start + span, "requests": cur})
        cur = []
    for r in g.itertuples(index=False):
        t = pd.Timestamp(r.added_at)
        while t >= ep_start + span:
            flush(); ep_start = ep_start + span
        cs = cands(obs, t, str(r.reviewer))
        if len(cs) < 2: skipped += 1; continue
        cur.append({"change_id": int(r.change), "timestamp": t, "owner_email": str(r.owner),
                    "change_insertions": 0, "change_deletions": 0, "change_files_count": 0,
                    "logged_reviewer": str(r.reviewer), "logged_label": int(r.contributed),
                    "logged_resp_days": (float(r.lat_days) if r.lat_days == r.lat_days else None),
                    "candidates": cs})
    flush()
    if len(eps_p) >= a.min_episodes:
        episodes.extend(eps_p)
        per_project.append({"project": proj, "n_episodes": len(eps_p),
                            "n_requests": sum(len(e["requests"]) for e in eps_p)})

c = pd.Timestamp(a.cutoff)
tr = sorted([e for e in episodes if pd.Timestamp(e["end"]) <= c], key=lambda e: e["start"])
te = sorted([e for e in episodes if pd.Timestamp(e["start"]) >= c], key=lambda e: e["start"])
eps = tr + te
n_req = sum(len(e["requests"]) for e in eps)
lab = float(np.mean([q["logged_label"] for e in eps for q in e["requests"]]))
print(f"\nエピソード {len(eps)} (train {len(tr)} / test {len(te)}, またぎ除外 {len(episodes)-len(eps)})")
print(f"リクエスト {n_req:,}  1エピソード {n_req/max(1,len(eps)):.1f} 件  貢献率 {lab:.4f}")
print(f"train 〜{max(e['end'] for e in tr):%Y-%m-%d} / test {min(e['start'] for e in te):%Y-%m-%d}〜")
with open(a.out, "wb") as f:
    pickle.dump({"meta": {"design":"assignment_events","per_project":per_project,
                          "episode_days":a.episode_days,"n_candidates":a.n_candidates,
                          "active_window_days":a.active_window_days,"horizon":a.horizon,
                          "cutoff":str(c),"n_test":len(te),"n_episodes":len(eps),
                          "n_requests":n_req,"logged_contrib_rate":lab},
                 "history": history, "episodes": eps}, f)
print(f"保存 {a.out}  (--n-test-episodes {len(te)})")
