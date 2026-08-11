#!/usr/bin/env python3
"""柱2: レビュアー割り当て RL の学習と評価

エピソードを時系列で train/test に分け、test 上で以下を比較する:

  logged        : 実際に振られたレビュアーを再現 (ログ方策)
  random        : 有効候補から一様ランダム
  greedy_accept : 承諾確率が最大の候補 (「受けてくれる人に振る」素朴な方策)
  least_load    : 未完了レビュー件数が最小の候補 (負荷分散のみを狙う方策)
  ppo           : MaskablePPO + 順序不変方策 (SetMaskablePolicy)

比較の要点:
  greedy_accept は承諾率では強いが、承諾しやすい少数の人に集中するので
  ジニ係数とバス係数が悪化するはず。RL は報酬に集中度ペナルティが入っているため、
  承諾率をある程度保ちながら偏りを抑えられるかを見る。

公平性のため全方策で同じ乱数列を使う (common random numbers):
  エピソードごとに同一 seed で reset するので、承諾判定のコイン列が揃う。

使い方:
    uv run python scripts/rl/train_assignment_rl.py \
        --episodes outputs/rl/assignment/nova_2021_2022.pkl \
        --acceptance-model outputs/rl/acceptance_model_s26/acceptance_model.joblib \
        --output-dir outputs/rl/assignment/run1 \
        --state-dim 26 --n-test-episodes 6 --total-timesteps 30000
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.RL.env.acceptance_model import AcceptanceModel  # noqa: E402
from review_predictor.RL.env.assignment_sim_env import AssignmentSimEnv  # noqa: E402
from review_predictor.RL.env.state_builder import RLStateBuilder  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="柱2 レビュアー割り当て RL")
    p.add_argument("--episodes", required=True, help="build_assignment_episodes.py の出力 pickle")
    p.add_argument("--acceptance-model", required=True, help="承諾モデル (.joblib)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--state-dim", type=int, default=26, help="特徴量次元 (承諾モデルと一致させる)")
    p.add_argument("--n-candidates", type=int, default=20)
    p.add_argument("--n-test-episodes", type=int, default=6, help="末尾のこの本数を test にする")
    p.add_argument("--total-timesteps", type=int, default=30000, help="PPO の総ステップ数")
    p.add_argument("--n-steps", type=int, default=512, help="PPO の rollout 長")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--service-days", type=float, default=6.0, help="承諾レビューを抱える日数")
    p.add_argument("--load-cap", type=int, default=5)
    p.add_argument("--w-accept", type=float, default=1.0)
    p.add_argument("--w-reject", type=float, default=0.3)
    p.add_argument("--w-load", type=float, default=0.5)
    p.add_argument("--w-gini", type=float, default=2.0)
    p.add_argument(
        "--load-clip", type=float, default=None,
        help="負荷超過の頭打ち。既定 (None) はクリップなし。旧挙動の再現は 2.0",
    )
    p.add_argument(
        "--w-retention", type=float, default=0.0,
        help="継続スコアの係数。0 (既定) で継続性の項を使わない",
    )
    p.add_argument(
        "--retention-parquet", nargs="*", default=None,
        help="R_θ 参照表を作る event transitions parquet。--w-retention と併用",
    )
    p.add_argument(
        "--retention-transform", choices=["rank", "identity"], default="rank",
        help="R_θ の変換。rank=訓練分布の百分位 (既定) / identity=生の値",
    )
    p.add_argument(
        "--retention-missing", choices=["median", "low", "zero"], default="median",
        help="参照表に無い (開発者,月) の埋め方",
    )
    p.add_argument("--skip-ppo", action="store_true", help="ベースラインのみ評価する")
    p.add_argument(
        "--stuck-when-no-response", action="store_true",
        help="貢献しなかった依頼は解消されず滞留する負荷モデル "
             "(割り当てイベント設計向け。既定は固定 service_days)",
    )
    p.add_argument(
        "--background-history", action="store_true",
        help="エピソード開始で履歴を切らず、now より前の実績をそのまま使う "
             "(割り当てイベント設計で必要。既定は従来どおり切る)",
    )
    p.add_argument(
        "--dump-step-log", action="store_true",
        help="ステップ単位のログを step_log.csv に保存する (重みの感度分析用)",
    )
    return p.parse_args()


def build_retention_fn(args: argparse.Namespace):
    """継続スコア関数を組み立てる。--w-retention が 0 なら None。"""
    if args.w_retention == 0.0:
        return None
    if not args.retention_parquet:
        raise SystemExit("--w-retention を使うなら --retention-parquet が要ります")
    from review_predictor.RL.env.retention_score import RetentionTable

    tbl = RetentionTable.from_parquet(
        args.retention_parquet,
        transform=args.retention_transform,
        missing=args.retention_missing,
    )
    logger.info("継続スコア参照表: %s", tbl.describe())
    return tbl


def make_env(
    episodes: List[Dict[str, Any]],
    history: pd.DataFrame,
    model: AcceptanceModel,
    args: argparse.Namespace,
    seed: int,
    retention_fn=None,
) -> AssignmentSimEnv:
    sb = RLStateBuilder(
        state_dim=args.state_dim, action_dim=5, include_assign_count=False
    )
    return AssignmentSimEnv(
        episodes=episodes,
        history=history,
        acceptance_model=model,
        state_builder=sb,
        n_candidates=args.n_candidates,
        service_days=args.service_days,
        load_cap=args.load_cap,
        w_accept=args.w_accept,
        w_reject=args.w_reject,
        w_load=args.w_load,
        w_gini=args.w_gini,
        w_retention=args.w_retention,
        background_history=getattr(args, "background_history", False),
        stuck_when_no_response=getattr(args, "stuck_when_no_response", False),
        load_clip=args.load_clip,
        retention_fn=retention_fn,
        seed=seed,
    )


# ─────────────────────────────────────────────
# 方策 (行動を選ぶ関数)
# ─────────────────────────────────────────────

def policy_random(env: AssignmentSimEnv, obs, rng: np.random.Generator) -> int:
    valid = np.flatnonzero(env.action_masks())
    return int(rng.choice(valid))


def policy_greedy_accept(env: AssignmentSimEnv, obs, rng: np.random.Generator) -> int:
    p = env.candidate_p_accept()
    p = np.where(env.action_masks(), p, -np.inf)
    return int(np.argmax(p))


def policy_least_load(env: AssignmentSimEnv, obs, rng: np.random.Generator) -> int:
    """負荷が最も軽い人を選ぶ。**同点は乱択で割る。**

    以前は同点を承諾確率で割っていたが (`load - 1e-3 * p`)、割り当てイベント設計では
    負荷がほとんど 0 で同点だらけになるため、実質「承諾確率が最大の人」= 貪欲法の
    変種になってしまい、負荷分散ベースラインとして機能しなかった。
    """
    load = env.candidate_loads().astype(np.float64)
    load = np.where(env.action_masks(), load, np.inf)
    lo = load.min()
    tied = np.flatnonzero(load <= lo + 1e-9)
    return int(rng.choice(tied))


def policy_logged(env: AssignmentSimEnv, obs, rng: np.random.Generator) -> int:
    a = env.logged_action()
    if a is None:
        return policy_random(env, obs, rng)
    return a


def evaluate_policy(
    env: AssignmentSimEnv,
    policy: Callable,
    ep_indices: List[int],
    seed: int,
    name: str,
    step_logs: Optional[List[pd.DataFrame]] = None,
) -> pd.DataFrame:
    """各テストエピソードで方策を走らせ、エピソード単位の指標を返す。

    step_logs を渡すとステップ単位のログも溜める。ログには accepted /
    load_excess / gini / retention が入っているので、**再実行せずに**
    任意の重みで報酬を計算し直せる (1 run 1.6 時間かかるため、重みの
    感度分析はオフラインで回す)。
    """
    rows = []
    t0 = time.time()
    for ep in ep_indices:
        rng = np.random.default_rng(seed + ep)
        obs, _ = env.reset(seed=seed + ep, options={"ep_idx": ep})
        done = False
        while not done:
            a = policy(env, obs, rng)
            obs, _, done, _, info = env.step(a)
        if step_logs is not None:
            d = env.episode_log()
            d["episode"] = ep
            d["policy"] = name
            step_logs.append(d)
        s = dict(info["episode_summary"])
        s["episode"] = ep
        s["policy"] = name
        rows.append(s)
    logger.info("[%s] %d エピソード評価 (%.0fs)", name, len(ep_indices), time.time() - t0)
    return pd.DataFrame(rows)


def evaluate_ppo(
    env: AssignmentSimEnv, model, ep_indices: List[int], seed: int,
    deterministic: bool = False,
    step_logs: Optional[List[pd.DataFrame]] = None,
) -> pd.DataFrame:
    """学習済み PPO をテストエピソードで評価する。

    **既定は確率的 (deterministic=False)。** 負荷分散を目的とする本課題では
    決定的方策が構造的に不利になる: 候補集合と状態がほとんど変わらないまま
    argmax を取ると同じ人が選ばれ続け、集中を罰する報酬なのに集中してしまう。
    実測でも同一モデルで報酬/step が決定的 +0.0665 に対し確率的 +0.3480、
    有効レビュアーが 2.87 対 9.53 と大きく変わった。
    """
    from sb3_contrib.common.maskable.utils import get_action_masks

    rows = []
    t0 = time.time()
    for ep in ep_indices:
        obs, _ = env.reset(seed=seed + ep, options={"ep_idx": ep})
        done = False
        while not done:
            masks = get_action_masks(env)
            a, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
            obs, _, done, _, info = env.step(int(a))
        name = "ppo" if not deterministic else "ppo_deterministic"
        if step_logs is not None:
            d = env.episode_log()
            d["episode"] = ep
            d["policy"] = name
            step_logs.append(d)
        s = dict(info["episode_summary"])
        s["episode"] = ep
        s["policy"] = name
        rows.append(s)
    logger.info("[ppo] %d エピソード評価 (%.0fs)", len(ep_indices), time.time() - t0)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.episodes, "rb") as f:
        data = pickle.load(f)
    episodes = data["episodes"]
    history = data["history"]
    logger.info("エピソード %d 本, meta=%s", len(episodes), data["meta"])

    model = AcceptanceModel.load(args.acceptance_model)
    # プロジェクト事前分布を使うモデルは、環境が作る状態次元より 1 だけ大きい
    # (その列は predict_proba 側で付ける)
    expected = model.obs_dim - (1 if model.uses_project_prior else 0)
    if expected != args.state_dim:
        raise SystemExit(
            f"承諾モデルの入力次元 {model.obs_dim}"
            f"{' (うちプロジェクト事前分布 1)' if model.uses_project_prior else ''}"
            f" と --state-dim {args.state_dim} が不一致"
        )
    if model.uses_project_prior:
        logger.info(
            "承諾モデルはプロジェクト事前分布を使用 (%d プロジェクト, 既定 %.4f)",
            len(model.project_prior), model.project_prior_default,
        )

    n_test = args.n_test_episodes
    train_eps = episodes[:-n_test]
    test_eps = episodes[-n_test:]
    logger.info(
        "分割: train %d 本 (%s~%s) / test %d 本 (%s~%s)",
        len(train_eps), train_eps[0]["start"].date(), train_eps[-1]["end"].date(),
        len(test_eps), test_eps[0]["start"].date(), test_eps[-1]["end"].date(),
    )

    retention_fn = build_retention_fn(args)

    # 評価用 env は全エピソードを持たせ、ep_idx で test を指す
    eval_env = make_env(episodes, history, model, args, args.seed, retention_fn)
    test_idx = list(range(len(episodes) - n_test, len(episodes)))

    # ── ベースライン ─────────────────────────────────────────
    results = []
    step_logs: Optional[List[pd.DataFrame]] = [] if args.dump_step_log else None
    for name, fn in [
        ("logged", policy_logged),
        ("random", policy_random),
        ("greedy_accept", policy_greedy_accept),
        ("least_load", policy_least_load),
    ]:
        results.append(
            evaluate_policy(eval_env, fn, test_idx, args.seed, name, step_logs)
        )

    # ── PPO ──────────────────────────────────────────────────
    ppo_model = None
    if not args.skip_ppo:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv

        from review_predictor.RL.agent.set_policy import SetMaskablePolicy

        train_env = DummyVecEnv([
            lambda: Monitor(
                make_env(train_eps, history, model, args, args.seed, retention_fn)
            )
        ])
        ppo_model = MaskablePPO(
            SetMaskablePolicy,
            train_env,
            policy_kwargs={"hidden": args.hidden},
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            seed=args.seed,
            verbose=1,
        )
        logger.info("PPO 学習開始 (total_timesteps=%d)", args.total_timesteps)
        t0 = time.time()
        ppo_model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
        logger.info("PPO 学習完了 (%.0fs)", time.time() - t0)
        ppo_model.save(str(out_dir / "ppo_assignment"))
        # 確率的・決定的の両方を出す (既定の比較対象は確率的な方)
        results.append(evaluate_ppo(eval_env, ppo_model, test_idx, args.seed,
                                    step_logs=step_logs))
        results.append(
            evaluate_ppo(eval_env, ppo_model, test_idx, args.seed, deterministic=True,
                         step_logs=step_logs)
        )

    # ── 集約 ─────────────────────────────────────────────────
    df = pd.concat(results, ignore_index=True)
    df.to_csv(out_dir / "per_episode.csv", index=False)

    metrics = [
        "accept_rate", "n_accepted", "gini_accepted", "bus_factor_50",
        "n_active_reviewers", "max_reviewer_share", "match_logged_rate",
        "total_reward",
    ]
    summary = df.groupby("policy")[metrics].agg(["mean", "std"]).round(4)
    summary.to_csv(out_dir / "summary.csv")

    logger.info(
        "\n=== test %d エピソードの平均 ===\n%s",
        n_test, df.groupby("policy")[metrics].mean().round(4).to_string(),
    )

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "meta": data["meta"]}, f, indent=2,
                  ensure_ascii=False, default=str)
    if step_logs:
        pd.concat(step_logs, ignore_index=True).to_csv(out_dir / "step_log.csv", index=False)
        logger.info("ステップログを保存: %s", out_dir / "step_log.csv")
    logger.info("結果を保存: %s", out_dir)


if __name__ == "__main__":
    main()
