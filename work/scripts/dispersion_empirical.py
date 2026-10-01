"""集中緩和が貢献モデルに依存しないかを、負荷フィードバックを保ったまま確かめる。

注意: nohup で回すと stdout がバッファされ、完了まで何も見えない。
進捗を見たいときは `uv run python -u` で起動する。

N24 の問題: logged 再生では負荷が logged の選択で決まるので、
balanced 自身の負荷フィードバックが働かず、決定的方策が集中してしまう。

ここでは方策自身の選択で状態を進めるが、**貢献判定を貢献モデルではなく
「その開発者の期間内の実績貢献率」から引く**。
モデルの予測を一切使わないので、モデルが間違っていてもこの比較は汚染されない。

  貢献判定: rng.random() < emp_rate[email]     (emp_rate は履歴から作る)
"""
import pickle, sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0, "src"); sys.path.insert(0, "scripts/rl")
from review_predictor.RL.env.acceptance_model import AcceptanceModel
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv
from review_predictor.RL.env.state_builder import RLStateBuilder

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--events", default="work/aosp_events_c.parquet")
ap.add_argument("--label", default="AOSP")
ap.add_argument("--k", type=float, default=10.0, help="ベイズ平滑化の強さ")
a = ap.parse_args()

d = pickle.load(open(a.episodes, "rb")); eps = d["episodes"]; n_test = d["meta"]["n_test"]
m = AcceptanceModel.load(a.model)

# 実績貢献率 (テスト開始より前のイベントだけから作る = リークしない)
ev = pd.read_parquet(a.events)
ev["ts"] = pd.to_datetime(ev.added_at, errors="coerce", utc=True).dt.tz_localize(None)
cut = pd.Timestamp(eps[-n_test]["start"])
h = ev[ev.ts < cut]
g = h.groupby("reviewer").contributed.agg(["mean", "size"])
prior = float(h.contributed.mean())
# 件数が少ない人は全体平均に寄せる (ベイズ平滑化、k=10)
emp = ((g["mean"]*g["size"] + prior*a.k) / (g["size"] + a.k)).to_dict()
print(f"実績貢献率: {len(emp):,} 人  全体 {prior:.4f}  (テスト開始 {cut:%Y-%m-%d} より前のみ)")

def gini(x):
    x = np.asarray(x, float); x = x[x >= 0]
    if x.size == 0 or x.sum() <= 0: return 0.0
    xs = np.sort(x); i = np.arange(1, xs.size+1)
    return float((2*(i*xs).sum())/(xs.size*xs.sum()) - (xs.size+1)/xs.size)

def run(kind):
    env = AssignmentSimEnv(episodes=eps, history=d["history"], acceptance_model=m,
        state_builder=RLStateBuilder(state_dim=26, action_dim=5, include_assign_count=False),
        n_candidates=20, seed=42, w_accept=1.0, w_reject=0.0, w_load=0.05, w_gini=0.25,
        load_cap=5, background_history=True, stuck_when_no_response=True)
    out = []
    for ep in range(len(eps)-n_test, len(eps)):
        rng = np.random.default_rng(42+ep)
        obs, _ = env.reset(seed=42+ep, options={"ep_idx": ep})
        cnt = {}; done = False
        while not done:
            cands = env._current_candidates()
            load = env.candidate_loads().astype(np.float64)
            msk = env.action_masks()
            # **貢献確率は実績から引く (モデルを使わない)**
            pe = np.array([emp.get(e, prior) for e in cands], dtype=np.float64)
            if kind == "balanced":
                v = np.where(msk, pe/(1.0+np.maximum(load,0.0)), -np.inf)
            elif kind == "balanced_cap":
                free = msk & (load <= 5)
                v = np.where(free if free.any() else msk, pe if free.any() else -load, -np.inf)
            elif kind == "least_load":
                v = np.where(msk, -load, -np.inf)
            else:  # logged
                act = env.logged_action()
                v = None
            if v is not None:
                hi = v.max(); act = int(rng.choice(np.flatnonzero(v >= hi-1e-12)))
            if act is None: act = int(np.flatnonzero(msk)[0])
            cnt[cands[act]] = cnt.get(cands[act], 0) + 1
            obs, _, done, _, _ = env.step(act)
        out.append({"episode": ep, "n": sum(cnt.values()), "gini": gini(list(cnt.values()))})
    return pd.DataFrame(out)

res = {}
for k in ["balanced", "balanced_cap", "least_load", "logged"]:
    r = run(k); res[k] = r
    print(f"  {k:<14} ジニ {np.average(r.gini, weights=r.n):.4f}")

rs = np.random.default_rng(0)
lg = res["logged"]
print()
for k in ["balanced", "balanced_cap", "least_load"]:
    dv = (res[k].gini - lg.gini).values; w = lg.n.values
    bs = np.array([np.average(dv[i], weights=w[i]) for i in rs.integers(0, len(dv), (4000, len(dv)))])
    print(f"{k:<14} − logged: {np.average(dv, weights=w):+.4f} "
          f"[{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  "
          f"分散した割合 {(bs<0).mean():>6.1%}  エピソード {(dv<0).sum()}/{len(dv)}")
