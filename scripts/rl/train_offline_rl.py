#!/usr/bin/env python3
"""
Phase1: オフライン RL 訓練スクリプト (BC → IQL)

Step 1: Behavioral Cloning で warm-start 政策を学習
Step 2: IQL (Implicit Q-Learning) で政策を精緻化

使い方:
    python scripts/rl/train_offline_rl.py \
        --dataset outputs/rl/offline/transitions.parquet \
        --output-dir outputs/rl/offline \
        --obs-dim 33 \
        --n-actions 2 \
        --bc-epochs 50 \
        --iql-epochs 100 \
        --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.RL.agent.bc_policy import BCPolicy, BCTrainer
from review_predictor.RL.agent.iql import IQLAgent
from review_predictor.RL.data.offline_dataset import OfflineRLDataset
from review_predictor.RL.data.loading import (
    load_transitions_df,
    recompute_reward,
    states_matrix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline RL training (BC → IQL)")
    p.add_argument("--dataset", required=True, help="トランジションデータ (.parquet/.json)")
    p.add_argument("--output-dir", required=True, help="チェックポイント出力ディレクトリ")
    p.add_argument("--obs-dim", type=int, required=True, help="観測次元 (state_dim + 1)")
    p.add_argument("--n-actions", type=int, default=2, help="行動数")
    p.add_argument("--hidden-dim", type=int, default=256, help="MLP 隠れ層次元")
    p.add_argument("--bc-epochs", type=int, default=50, help="BC エポック数 (0 でスキップ)")
    p.add_argument("--iql-epochs", type=int, default=100, help="IQL エポック数")
    p.add_argument("--batch-size", type=int, default=256, help="ミニバッチサイズ")
    p.add_argument("--lr", type=float, default=3e-4, help="学習率")
    p.add_argument("--gamma", type=float, default=0.99, help="IQL 割引率")
    p.add_argument("--tau", type=float, default=0.9, help="IQL expectile パラメータ")
    p.add_argument("--beta", type=float, default=3.0, help="IQL advantage-weight 温度")
    p.add_argument(
        "--polyak", type=float, default=0.005,
        help="target Q の soft update 係数。0 で旧挙動 (target_update_freq ごとの hard copy)",
    )
    p.add_argument(
        "--adv-clip", type=float, default=10.0, help="advantage のクリップ上限",
    )
    p.add_argument(
        "--exp-clip-log", type=float, default=6.0,
        help="advantage weight exp(β·adv) の指数部の上限 (π の NaN 防止)",
    )
    p.add_argument(
        "--divergence-threshold", type=float, default=1e4,
        help="この epoch 平均 V loss を超えたら発散とみなし学習を打ち切る",
    )
    p.add_argument(
        "--reward-shaping",
        type=float,
        default=None,
        help=(
            "λ を指定すると保存済みの reward 列を無視して "
            "r = step_label + λ·R_θ を列から再計算する (λ sweep 用、"
            "データセット再ビルド不要)。未指定なら保存済み reward を使う。"
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="乱数 seed (BC split / torch 初期化)")
    p.add_argument("--device", default="cpu", help="torch デバイス")
    p.add_argument("--skip-bc", action="store_true", help="BC をスキップして IQL のみ実行")
    return p.parse_args()


def load_dataset(
    path: str,
    obs_dim: int,
    device: str,
    reward_shaping: Optional[float] = None,
) -> OfflineRLDataset:
    """parquet / json からデータセットをロードする。

    reward_shaping (λ) 指定時は保存済み reward を使わず
    step_label + λ·R_θ を列から再計算する (loading.recompute_reward)。
    """
    df = load_transitions_df(path)
    rewards = (
        recompute_reward(df, reward_shaping)
        if reward_shaping is not None
        else df["reward"].astype(float).to_numpy(dtype=np.float32)
    )
    return OfflineRLDataset.from_arrays(
        states=states_matrix(df["state"], obs_dim),
        actions=df["action"].astype(int).to_numpy(),
        rewards=rewards,
        next_states=states_matrix(df["next_state"], obs_dim),
        dones=df["done"].astype(float).to_numpy(),
        device=device,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("データセットをロード: %s", args.dataset)
    dataset = load_dataset(
        args.dataset, args.obs_dim, args.device, reward_shaping=args.reward_shaping,
    )
    logger.info("サンプル数: %d, obs_dim=%d", len(dataset), dataset.obs_dim)

    bc_policy = None

    # ── Phase 1a: Behavioral Cloning ─────────────────────────────
    if not args.skip_bc and args.bc_epochs > 0:
        logger.info("=== Phase 1a: Behavioral Cloning ===")
        bc_policy = BCPolicy(
            obs_dim=args.obs_dim,
            n_actions=args.n_actions,
            hidden_dim=args.hidden_dim,
        )
        trainer = BCTrainer(bc_policy, learning_rate=args.lr, device=args.device)
        bc_stats = trainer.train(
            dataset, epochs=args.bc_epochs, batch_size=args.batch_size
        )
        bc_path = output_dir / "bc_policy.pt"
        trainer.save(str(bc_path))

        # BC 統計を保存
        with open(output_dir / "bc_stats.json", "w") as f:
            json.dump(bc_stats, f, indent=2)
        logger.info("BC 完了: best_val=%.4f", bc_stats["best_val_loss"])

    # ── Phase 1b: IQL ─────────────────────────────────────────────
    logger.info("=== Phase 1b: IQL ===")
    agent = IQLAgent(
        obs_dim=args.obs_dim,
        n_actions=args.n_actions,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        tau=args.tau,
        beta=args.beta,
        adv_clip=args.adv_clip,
        lr=args.lr,
        polyak=args.polyak,
        exp_clip_log=args.exp_clip_log,
        divergence_threshold=args.divergence_threshold,
        device=args.device,
    )

    # BC 重みを IQL 政策の warm-start に転写
    if bc_policy is not None:
        agent.load_bc_weights(bc_policy)

    iql_stats = agent.train(
        dataset, epochs=args.iql_epochs, batch_size=args.batch_size
    )
    iql_path = output_dir / "iql_agent.pt"
    agent.save(str(iql_path))

    with open(output_dir / "iql_stats.json", "w") as f:
        json.dump(iql_stats, f, indent=2)

    # 実行条件を記録 (summarize_rl_results.py が run パラメータとして集約する)
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    logger.info(
        "IQL 完了: 最終 V=%.4f Q=%.4f π=%.4f",
        iql_stats["v_losses"][-1],
        iql_stats["q_losses"][-1],
        iql_stats["pi_losses"][-1],
    )
    if iql_stats.get("diverged"):
        logger.error(
            "IQL は発散して epoch %d で打ち切られた。この run の V/Q/π は評価に使えない",
            iql_stats.get("epochs_run", -1),
        )
    logger.info("チェックポイント保存先: %s", output_dir)


if __name__ == "__main__":
    main()
