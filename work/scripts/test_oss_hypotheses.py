"""OSS の定説がこのデータで成り立つかを検証する。

**「特徴量重要度が高いから」ではなく「定説がそう言うから」で仮説を立て、
実データで確かめてから環境に入れる。**成り立たない仮説は入れない。

H1 相互性 (Bosu & Carver 2014 ほか):
    コードレビューは互恵的な交換であり、過去に関わった相手からの依頼には
    応じやすい。→ (reviewer, owner) の過去の接触回数が貢献率を上げるはず。

H2 過負荷による離脱 (Mockus 2010, Steinmacher et al. 2015):
    抱えきれない依頼を積まれた貢献者は関与を減らし、やがて離れる。
    → 未応答を溜めた人はその後の活動が落ちるはず。

H3 新規参加者の早期経験 (Steinmacher et al. 2015):
    最初の依頼に応じた新規参加者は定着しやすい。
    → 初回に貢献した人は、しなかった人より後の活動が多いはず。
"""
import sys
import numpy as np, pandas as pd

ev = pd.read_parquet("work/aosp_events_c.parquet")
ev["ts"] = pd.to_datetime(ev.added_at, errors="coerce", utc=True).dt.tz_localize(None)
ev = ev.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
print(f"AOSP 割り当てイベント {len(ev):,} 件  貢献率 {ev.contributed.mean():.4f}\n")

# ── H1 相互性 ────────────────────────────────────────────
print("H1 相互性: 過去に関わった相手からの依頼には応じやすいか")
seen = {}
prior = np.zeros(len(ev), dtype=int)
rv = ev.reviewer.astype(str).to_numpy(); ow = ev.owner.astype(str).to_numpy()
for i in range(len(ev)):
    k = (rv[i], ow[i])
    prior[i] = seen.get(k, 0)
    seen[k] = prior[i] + 1
ev["prior_pairs"] = prior
b = pd.cut(ev.prior_pairs, [-1, 0, 1, 3, 10, 10**9],
           labels=["初対面", "1回", "2-3回", "4-10回", "11回以上"])
t = ev.groupby(b, observed=True).agg(件数=("contributed", "size"), 貢献率=("contributed", "mean"))
print(t.round(4).to_string())
lo = ev[ev.prior_pairs == 0].contributed.mean(); hi = ev[ev.prior_pairs >= 4].contributed.mean()
print(f"  → 初対面 {lo:.4f} vs 4回以上 {hi:.4f}  差 {hi-lo:+.4f}")

# ── H2 過負荷による離脱 ──────────────────────────────────
print("\nH2 過負荷: 未応答を溜めた人はその後の活動が落ちるか")
ev["month"] = ev.ts.dt.to_period("M")
g = ev.groupby(["reviewer", "month"]).agg(依頼=("contributed", "size"),
                                          未応答=("contributed", lambda x: (1-x).sum())).reset_index()
g = g.sort_values(["reviewer", "month"])
g["翌月の依頼"] = g.groupby("reviewer")["依頼"].shift(-1)
g = g.dropna(subset=["翌月の依頼"])
g["未応答率"] = g["未応答"] / g["依頼"]
g["継続率"] = g["翌月の依頼"] / g["依頼"]
bb = pd.cut(g["未応答"], [-1, 0, 2, 5, 10, 10**9],
            labels=["0件", "1-2件", "3-5件", "6-10件", "11件以上"])
t2 = g.groupby(bb, observed=True).agg(人月=("継続率", "size"), 翌月継続率=("継続率", "mean"),
                                      翌月依頼数=("翌月の依頼", "mean"))
print(t2.round(4).to_string())

# ── H3 新規参加者の早期経験 ──────────────────────────────
print("\nH3 新規参加者: 初回に貢献した人は後の活動が多いか")
first = ev.groupby("reviewer").head(1)[["reviewer", "contributed", "ts"]]
first.columns = ["reviewer", "初回貢献", "初回時刻"]
cnt = ev.groupby("reviewer").size().rename("総依頼数")
m = first.merge(cnt, on="reviewer")
m["その後の依頼数"] = m["総依頼数"] - 1
t3 = m.groupby("初回貢献").agg(人数=("その後の依頼数", "size"),
                              その後の依頼数_平均=("その後の依頼数", "mean"),
                              中央値=("その後の依頼数", "median"))
print(t3.round(2).to_string())
a = m[m.初回貢献 == 1]["その後の依頼数"].mean(); c = m[m.初回貢献 == 0]["その後の依頼数"].mean()
print(f"  → 初回貢献あり {a:.2f} vs なし {c:.2f}  比 {a/max(c,1e-9):.2f}倍")
