"""balanced を行動クローンしてから PPO を続ける (warm start)。

**なぜやるか**: これまでの 6 通りの検証は「RL が単純な規則を**発見**できるか」
しか見ていなかった。「RL が単純な規則を**改善**できるか」は別の問い。
`balanced` の重みから出発すれば後者を直接測れる。

手順:
  1. train エピソードで `balanced` を走らせ (obs, 選んだ候補) を集める
  2. MaskablePPO の方策を**教師あり**でその選択に合わせる (行動クローン)
  3. BC 直後の方策を test で評価 (= balanced をどれだけ模倣できたか)
  4. そこから PPO を継続し、再度 test で評価 (= RL が改善できるか)

判定:
  - BC 直後が balanced に並ぶ → 表現力は足りている
  - PPO 継続で**上がる** → RL は単純な規則を改善できる
  - PPO 継続で**下がる** → この報酬設計では改善方向に動けない

⚠️ 評価は既存 run と同じ env 設定で行う。BC も PPO も**学習用 env のみ**を使う。
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from review_predictor.RL.agent.set_policy import SetMaskablePolicy  # noqa: E402
from review_predictor.RL.env.acceptance_model import AcceptanceModel  # noqa: E402
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv  # noqa: E402
from review_predictor.RL.env.state_builder import RLStateBuilder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", default="work/aosp_ep_7d.pkl")
ap.add_argument("--model", default="outputs/rl/contrib_model_aosp/contrib_model.joblib")
ap.add_argument("--out-dir", default="outputs/rl/assignment/aosp_warm")
ap.add_argument("--w-load", type=float, default=0.05)
ap.add_argument("--bc-episodes", type=int, default=60, help="BC 用に回す train エピソード数")
ap.add_argument("--bc-epochs", type=int, default=8)
ap.add_argument("--ppo-steps", type=int, default=60000, help="BC 後に回す PPO のステップ数")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

out = ROOT / a.out_dir
out.mkdir(parents=True, exist_ok=True)

d = pickle.load(open(ROOT / a.episodes, "rb"))
eps, nt = d["episodes"], d["meta"]["n_test"]
n_train = len(eps) - nt
acc = AcceptanceModel.load(str(ROOT / a.model))


def make_env(subset):
    return AssignmentSimEnv(
        episodes=subset, history=d["history"], acceptance_model=acc,
        state_builder=RLStateBuilder(state_dim=26, action_dim=5,
                                     include_assign_count=False),
        n_candidates=20, seed=a.seed, w_accept=1.0, w_reject=0.0,
        w_load=a.w_load, w_gini=0.25, load_cap=5,
        background_history=True, stuck_when_no_response=True,
    )


# ── 1. balanced の行動を集める ──────────────────────────────
collect_env = make_env(eps)
obs_c, obs_t, obs_m, acts = [], [], [], []
rng = np.random.default_rng(a.seed)
picked = rng.choice(n_train, size=min(a.bc_episodes, n_train), replace=False)
t0 = time.time()
for k, ep in enumerate(picked):
    obs, _ = collect_env.reset(seed=a.seed + int(ep), options={"ep_idx": int(ep)})
    done = False
    while not done:
        mask = collect_env.action_masks()
        v = mask.astype(bool)
        p = collect_env.candidate_p_accept().astype(np.float64)
        load = collect_env.candidate_loads().astype(np.float64)
        score = np.where(v, p / (1.0 + load), -np.inf)
        act = int(np.argmax(score))
        obs_c.append(obs["candidates"].copy())
        obs_t.append(obs["task"].copy())
        obs_m.append(mask.copy())
        acts.append(act)
        obs, _r, term, trunc, _i = collect_env.step(act)
        done = term or trunc
    if (k + 1) % 20 == 0:
        print(f"  収集 {k + 1}/{len(picked)} エピソード  {len(acts):,} 標本", flush=True)
print(f"balanced の行動を {len(acts):,} 件収集 ({time.time() - t0:.0f}s)", flush=True)

X_c = th.as_tensor(np.asarray(obs_c, dtype=np.float32))
X_t = th.as_tensor(np.asarray(obs_t, dtype=np.float32))
X_m = th.as_tensor(np.asarray(obs_m, dtype=np.bool_))
Y = th.as_tensor(np.asarray(acts, dtype=np.int64))

# ── 2. 行動クローン ────────────────────────────────────────
train_env = DummyVecEnv([lambda: Monitor(make_env(eps[:n_train]))])
model = MaskablePPO(
    SetMaskablePolicy, train_env, policy_kwargs={"hidden": 64},
    n_steps=512, batch_size=128, learning_rate=3e-4, ent_coef=0.01,
    seed=a.seed, verbose=0,
)
policy = model.policy
opt = th.optim.Adam(policy.parameters(), lr=1e-3)
n = len(Y)
idx = np.arange(n)
print(f"行動クローン開始 ({n:,} 標本 × {a.bc_epochs} epoch)", flush=True)
for epoch in range(a.bc_epochs):
    rng.shuffle(idx)
    tot, correct, loss_sum = 0, 0, 0.0
    for s in range(0, n, 256):
        b = idx[s:s + 256]
        obs_b = {"candidates": X_c[b], "task": X_t[b], "mask": X_m[b].to(th.int8)}
        dist = policy.get_distribution(obs_b, action_masks=X_m[b].numpy())
        logp = dist.log_prob(Y[b])
        loss = -logp.mean()
        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        opt.step()
        with th.no_grad():
            pred = dist.distribution.probs.argmax(dim=1)
            correct += int((pred == Y[b]).sum())
        tot += len(b)
        loss_sum += float(loss.detach()) * len(b)
    print(f"  epoch {epoch + 1}: loss {loss_sum / tot:.4f}  一致率 {correct / tot:.4f}",
          flush=True)

model.save(str(out / "ppo_bc"))
print(f"BC 済みモデルを保存: {out / 'ppo_bc.zip'}", flush=True)

# ── 3. BC 直後を test で評価 ───────────────────────────────
def evaluate(m, tag):
    env = make_env(eps)
    rows = []
    for ep in range(n_train, len(eps)):
        obs, _ = env.reset(seed=a.seed + ep, options={"ep_idx": ep})
        done, total = False, 0.0
        while not done:
            act, _ = m.predict(obs, action_masks=env.action_masks(),
                               deterministic=False)
            obs, _r, term, trunc, info = env.step(int(act))
            total += float(info["reward_eval"])
            done = term or trunc
        s = env.episode_summary()
        rows.append((total, s["n_requests"], s["accept_rate"],
                     s["gini_assigned"], s["bus_factor_50"]))
    arr = np.array(rows, dtype=float)
    r = arr[:, 0].sum() / arr[:, 1].sum()
    print(f"{tag:<22} 報酬/step {r:>8.4f}  貢献率 {arr[:, 2].mean():.4f}  "
          f"集中度 {arr[:, 3].mean():.4f}  busF {arr[:, 4].mean():.2f}", flush=True)
    return r


print("\n=== test 評価 ===", flush=True)
r_bc = evaluate(model, "BC直後 (balanced模倣)")

# ── 4. PPO を継続 ─────────────────────────────────────────
print(f"\nPPO 継続 ({a.ppo_steps:,} steps)", flush=True)
t0 = time.time()
model.learn(total_timesteps=a.ppo_steps, progress_bar=False, reset_num_timesteps=False)
print(f"PPO 完了 ({time.time() - t0:.0f}s)", flush=True)
model.save(str(out / "ppo_warm"))
r_ppo = evaluate(model, "warm start 後")

print(f"\n判定: BC {r_bc:.4f} → PPO 継続 {r_ppo:.4f}  ({r_ppo - r_bc:+.4f})")
print("balanced の実測値と比べること (同一 env 設定の run から取る)")
