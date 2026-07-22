#!/usr/bin/env python3
"""
Phase2: オンライン RL 訓練スクリプト (TQC with gymnasium env)

sb3-contrib の TQC を使い、ReviewAssignmentEnv 上でオンライン RL を実行する。
フェーズ1の IQL 政策（または BC 政策）を warm-start として使う。

注意:
  - ReviewAssignmentEnv は episodes データを必要とするため、
    --episodes オプションで前処理済みエピソード JSON を渡すこと
  - TQC の actor は BCPolicy と互換性がないため、warm-start は
    replay buffer 事前充填（offline-to-online）で代替する

使い方:
    python scripts/rl/train_online_rl.py \
        --episodes data/rl/episodes.json \
        --irl-model outputs/main32_mce_volat_ep150/mce_irl_model.pt \
        --irl-config outputs/main32_mce_volat_ep150/config.json \
        --offline-dataset outputs/rl/offline/transitions.parquet \
        --output-dir outputs/rl/online \
        --state-dim 32 \
        --n-candidates 10 \
        --total-timesteps 100000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Online RL training (TQC)")
    p.add_argument("--episodes", required=True, help="エピソード JSON ファイルパス")
    p.add_argument("--irl-model", required=True, help="IRL チェックポイントパス")
    p.add_argument("--irl-config", required=True, help="IRL 設定 JSON ファイルパス")
    p.add_argument("--offline-dataset", help="オフラインデータセット (.parquet) — replay buffer 事前充填用")
    p.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    p.add_argument("--state-dim", type=int, default=32)
    p.add_argument("--action-dim", type=int, default=5)
    p.add_argument("--n-candidates", type=int, default=10, help="固定候補開発者数")
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--learning-starts", type=int, default=1000, help="学習開始ステップ数")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from sb3_contrib import TQC
        from stable_baselines3.common.env_checker import check_env
    except ImportError:
        logger.error("sb3-contrib がインストールされていません: pip install sb3-contrib")
        sys.exit(1)

    from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem
    from review_predictor.RL.env.reward_fn import IRLRewardFn
    from review_predictor.RL.env.state_builder import RLStateBuilder
    from review_predictor.RL.env.review_assignment_env import ReviewAssignmentEnv

    # IRL システムのロード
    with open(args.irl_config) as f:
        irl_config = json.load(f)
    irl_config.update({"state_dim": args.state_dim, "action_dim": args.action_dim})

    system = MCEIRLSystem(irl_config)
    ckpt = torch.load(args.irl_model, map_location=args.device, weights_only=True)
    state_dict = ckpt.get("network_state_dict", ckpt)
    system.network.load_state_dict(state_dict)
    system.network.to(system.device)
    system.network.eval()

    reward_fn = IRLRewardFn(system)
    state_builder = RLStateBuilder(state_dim=args.state_dim, action_dim=args.action_dim)
    obs_dim = state_builder.obs_dim  # state_dim + 1

    # エピソードデータのロード
    logger.info("エピソードをロード: %s", args.episodes)
    with open(args.episodes) as f:
        episodes = json.load(f)

    # 環境の構築
    env = ReviewAssignmentEnv(
        n_candidates=args.n_candidates,
        obs_dim=obs_dim,
        reward_fn=reward_fn,
        state_builder=state_builder,
        episodes=episodes,
    )

    # gymnasium 環境の妥当性チェック
    logger.info("gymnasium env チェック...")
    try:
        from gymnasium.utils.env_checker import check_env as gym_check
        gym_check(env, warn=True)
        logger.info("環境チェック: OK")
    except Exception as e:
        logger.warning("環境チェック警告: %s", e)

    # TQC の学習
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("TQC 訓練開始 (total_timesteps=%d)", args.total_timesteps)
    model = TQC(
        "MlpPolicy",
        env,
        verbose=1,
        device=args.device,
        learning_starts=args.learning_starts,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    # オフラインデータセットで replay buffer を事前充填（offline-to-online）
    if args.offline_dataset:
        logger.info("replay buffer 事前充填: %s", args.offline_dataset)
        p = Path(args.offline_dataset)
        df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_json(p)

        def _to_arr(val, dim: int) -> np.ndarray:
            if isinstance(val, list):
                v = val[:dim]
                v += [0.0] * max(0, dim - len(v))
                return np.array(v, dtype=np.float32)
            return np.zeros(dim, dtype=np.float32)

        flat_dim = args.n_candidates * obs_dim
        n_prefill = min(len(df), model.buffer_size // 2)
        logger.info("  %d サンプルを replay buffer に追加", n_prefill)

        for i, (_, row) in enumerate(df.head(n_prefill).iterrows()):
            s = _to_arr(row["state"], obs_dim)
            s_full = np.zeros(flat_dim, dtype=np.float32)
            s_full[:len(s)] = s

            s_next = _to_arr(row["next_state"], obs_dim)
            s_next_full = np.zeros(flat_dim, dtype=np.float32)
            s_next_full[:len(s_next)] = s_next

            a = int(row["action"])
            r = float(row["reward"])
            done = bool(row["done"])

            model.replay_buffer.add(
                obs=s_full,
                next_obs=s_next_full,
                action=np.array([a]),
                reward=np.array([r]),
                done=np.array([done]),
                infos=[{}],
            )

    model.learn(
        total_timesteps=args.total_timesteps,
        tb_log_name="tqc_review",
    )

    model_path = output_dir / "tqc_model"
    model.save(str(model_path))
    logger.info("TQC モデルを保存: %s", model_path)


if __name__ == "__main__":
    main()
