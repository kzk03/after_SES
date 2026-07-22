#!/usr/bin/env python3
"""
RL オフラインデータセット構築スクリプト

IRL で学習済みの R_θ を使って歴史データ軌跡にラベル付けを行い、
(s, a, r, s', done) タプルを Parquet / JSON 形式で保存する。

使い方:
    python scripts/rl/build_rl_dataset.py \
        --trajectories data/trajectories/mce_irl_trajectories.json \
        --irl-model outputs/main32_mce_volat_ep150/mce_irl_model.pt \
        --irl-config outputs/main32_mce_volat_ep150/config.json \
        --output outputs/rl/offline/transitions.parquet \
        --state-dim 32 \
        --action-dim 5 \
        --device cpu
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.RL.env.state_builder import RLStateBuilder
from review_predictor.RL.data.offline_dataset import build_transitions
from review_predictor.RL.data.dataset_io import save_transitions_with_split
from review_predictor.RL.data.loading import load_irl_reward_fn, load_trajectories

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL offline dataset builder")
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
            "test 分割の基準日 (YYYY-MM-DD)。feature_end >= この日付の transition は "
            "transitions_test.parquet に分離保存される。未指定なら分割しない。"
        ),
    )
    p.add_argument(
        "--dev-label-window",
        type=str,
        default="0,3",
        help=(
            "step_label_dev (dev 単位継続ラベル) の窓オフセット '開始月,終了月'。"
            "軌跡抽出時の future window (0,3 / 3,6 / ...) と揃えること。"
        ),
    )
    p.add_argument(
        "--with-assign-count",
        action="store_true",
        help=(
            "episode_assign_count 次元を観測に含める (Phase 2 用)。"
            "既定はオフ = obs_dim は state_dim と等しい。"
        ),
    )
    p.add_argument(
        "--holdout-devs-ratio",
        type=float,
        default=0.0,
        help=(
            "dev 単位 holdout の割合 (0.0 で無効)。email ハッシュで決定的に選んだ "
            "dev の全 transition を <stem>_devholdout に分離する (未知 dev 汎化評価用)。"
        ),
    )
    p.add_argument("--holdout-seed", type=int, default=42, help="dev holdout のハッシュ seed")
    p.add_argument(
        "--done-mode",
        choices=["episode_end", "dropout"],
        default="episode_end",
        help=(
            "エピソード終端の扱い。episode_end=最終遷移 done=1 (既定)、"
            "dropout=打ち切りを bootstrap (月次は全 done=0)"
        ),
    )
    p.add_argument("--device", default="cpu", help="torch デバイス (cpu/cuda)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 軌跡・凍結 IRL のロード (共通処理は loading.py 参照)
    logger.info("軌跡をロード: %s", args.trajectories)
    trajectories = load_trajectories(args.trajectories)
    logger.info("軌跡数: %d", len(trajectories))

    logger.info("MCEIRLSystem をロード: %s", args.irl_model)
    reward_fn = load_irl_reward_fn(
        args.irl_model, args.irl_config, args.state_dim, args.action_dim, args.device,
    )
    state_builder = RLStateBuilder(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        include_assign_count=args.with_assign_count,
    )
    logger.info("obs_dim = %d (include_assign_count=%s)", state_builder.obs_dim, args.with_assign_count)

    dev_label_window = tuple(int(x) for x in args.dev_label_window.split(","))
    if len(dev_label_window) != 2:
        logger.error("--dev-label-window は '開始月,終了月' 形式で指定してください: %s", args.dev_label_window)
        sys.exit(1)

    # (s, a, r, s', done) 構築
    logger.info("トランジションを構築中...")
    transitions = build_transitions(
        trajectories=trajectories,
        reward_fn=reward_fn,
        state_builder=state_builder,
        feature_window_days=args.feature_window,
        reward_shaping=args.reward_shaping,
        dev_label_window_months=dev_label_window,
        done_mode=args.done_mode,
    )
    logger.info("トランジション数: %d", len(transitions))

    if not transitions:
        logger.error("トランジションが空です。軌跡データを確認してください。")
        sys.exit(1)

    # 保存 (時間分割 + dev holdout — 共通処理は dataset_io.py 参照)
    df = save_transitions_with_split(
        transitions,
        Path(args.output),
        test_split_date=args.test_split_date,
        holdout_devs_ratio=args.holdout_devs_ratio,
        holdout_seed=args.holdout_seed,
    )

    # 報酬分布の簡易統計
    rewards = df["reward"].astype(float)
    logger.info(
        "報酬分布: mean=%.4f std=%.4f min=%.4f max=%.4f",
        rewards.mean(), rewards.std(), rewards.min(), rewards.max(),
    )
    logger.info("action=0 割合: %.2f%%", (df["action"] == 0).mean() * 100)
    logger.info("action=1 割合: %.2f%%", (df["action"] == 1).mean() * 100)
    if "step_label" in df.columns:
        logger.info(
            "step_label=1 割合 (3ヶ月先継続): %.2f%%",
            (df["step_label"] == 1).mean() * 100,
        )
    if "r_theta" in df.columns:
        r_th = df["r_theta"].astype(float)
        logger.info(
            "R_θ 単体分布: mean=%.4f std=%.4f min=%.4f max=%.4f",
            r_th.mean(), r_th.std(), r_th.min(), r_th.max(),
        )
    logger.info("reward shaping λ = %.3f", args.reward_shaping)


if __name__ == "__main__":
    main()
