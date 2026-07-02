#!/usr/bin/env python3
"""
RL オフラインデータセット構築スクリプト (event 単位版)

build_rl_dataset.py (月単位版) と並行運用する独立スクリプト。
月単位版は変更せず、event 単位の MDP 用データセットをこちらで構築する。

MDP 定義 (event 単位):
  state  : event (1件のレビュー依頼) 到着時点での dev 特徴量
  action : そのイベント自体の承諾有無 (0/1, 集計不要)
  reward : そのイベントが属する月の step_label + λ·R_θ[月] (月内 broadcast)

使い方:
    python scripts/rl/build_rl_dataset_event.py \
        --trajectories data/trajectories/mce_irl_trajectories.pkl \
        --irl-model outputs/main32_mce_volat_ep150/mce_irl_model.pt \
        --irl-config outputs/main32_mce_volat_ep150/config.json \
        --output outputs/rl/offline/event_transitions.parquet \
        --state-dim 32 \
        --action-dim 5 \
        --device cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem
from review_predictor.RL.env.reward_fn import IRLRewardFn
from review_predictor.RL.env.state_builder import RLStateBuilder
from review_predictor.RL.data.offline_dataset_event import build_event_transitions
from review_predictor.RL.data.dataset_io import save_transitions_with_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL offline dataset builder (event-level MDP)")
    p.add_argument("--trajectories", required=True, help="軌跡ファイルパス (.pkl または .json)")
    p.add_argument("--irl-model", required=True, help="MCE-IRL チェックポイントパス (.pt)")
    p.add_argument("--irl-config", required=True, help="IRL 設定 JSON / model_metadata.json パス")
    p.add_argument("--output", required=True, help="出力ファイルパス (.parquet または .json)")
    p.add_argument("--state-dim", type=int, default=32, help="IRL state 次元")
    p.add_argument("--action-dim", type=int, default=5, help="IRL action 次元")
    p.add_argument("--feature-window", type=int, default=180, help="特徴量抽出窓（日数）")
    p.add_argument(
        "--reward-shaping",
        type=float,
        default=0.3,
        help="R_θ シェイピング係数 λ (r = step_labels + λ·R_θ, 推奨 0.3)",
    )
    p.add_argument(
        "--test-split-date",
        type=str,
        default=None,
        help=(
            "test 分割の基準日 (YYYY-MM-DD)。feature_end (event timestamp) >= "
            "この日付の transition は test parquet に分離保存される。未指定なら分割しない。"
        ),
    )
    p.add_argument(
        "--event-action-types",
        type=str,
        default="review",
        help="行動判定の対象とする event の action_type (comma-separated, 既定: review)",
    )
    p.add_argument("--device", default="cpu", help="torch デバイス (cpu/cuda)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("軌跡をロード: %s", args.trajectories)
    traj_path = Path(args.trajectories)
    if traj_path.suffix in (".pkl", ".pickle"):
        import pickle
        with open(traj_path, "rb") as f:
            trajectories = pickle.load(f)
    else:
        with open(traj_path, encoding="utf-8") as f:
            trajectories = json.load(f)
    logger.info("軌跡数: %d", len(trajectories))

    with open(args.irl_config, encoding="utf-8") as f:
        irl_config = json.load(f)
    irl_config["state_dim"] = args.state_dim
    irl_config["action_dim"] = args.action_dim

    logger.info("MCEIRLSystem をロード: %s", args.irl_model)
    system = MCEIRLSystem(irl_config)
    ckpt = torch.load(args.irl_model, map_location=args.device, weights_only=True)
    if "network_state_dict" in ckpt:
        system.network.load_state_dict(ckpt["network_state_dict"])
    else:
        system.network.load_state_dict(ckpt)
    system.network.to(system.device)
    system.network.eval()

    reward_fn = IRLRewardFn(system)
    state_builder = RLStateBuilder(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )

    event_action_types = tuple(
        t.strip() for t in args.event_action_types.split(",") if t.strip()
    )

    logger.info("event 単位トランジションを構築中...")
    transitions = build_event_transitions(
        trajectories=trajectories,
        reward_fn=reward_fn,
        state_builder=state_builder,
        feature_window_days=args.feature_window,
        reward_shaping=args.reward_shaping,
        event_action_types=event_action_types,
    )
    logger.info("トランジション数: %d", len(transitions))

    if not transitions:
        logger.error("トランジションが空です。軌跡データを確認してください。")
        sys.exit(1)

    # 保存 (時間ベース train/test split 込み — 共通処理は dataset_io.py 参照)
    df = save_transitions_with_split(
        transitions, Path(args.output), test_split_date=args.test_split_date,
    )

    rewards = df["reward"].astype(float)
    logger.info(
        "報酬分布: mean=%.4f std=%.4f min=%.4f max=%.4f",
        rewards.mean(), rewards.std(), rewards.min(), rewards.max(),
    )
    logger.info("action=0 割合: %.2f%%", (df["action"] == 0).mean() * 100)
    logger.info("action=1 割合: %.2f%%", (df["action"] == 1).mean() * 100)
    if "step_label" in df.columns:
        logger.info(
            "step_label=1 割合 (所属月の3ヶ月先継続): %.2f%%",
            (df["step_label"] == 1).mean() * 100,
        )
    if "month_idx" in df.columns:
        events_per_month = df.groupby("email")["month_idx"].apply(lambda s: s.value_counts().mean())
        logger.info(
            "dev あたり月平均 event 数: %.2f",
            float(events_per_month.mean()) if len(events_per_month) else float("nan"),
        )
    logger.info("reward shaping λ = %.3f", args.reward_shaping)


if __name__ == "__main__":
    main()
