"""RL スクリプト共通のロード処理

build_rl_dataset*.py / train_offline_rl.py / eval_assignment.py に
逐語的に重複していた以下の処理を一元化する:
  - 軌跡ファイル (pkl / json) のロード
  - 凍結 IRL (MCEIRLSystem) のロードと IRLRewardFn 化
  - transitions parquet/json のロードと state 列の行列化
  - 報酬の再計算 (λ sweep 用 — データセットを再ビルドせずに
    step_label + λ·R_θ を列から復元する)
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_trajectories(path: str | Path) -> List[dict]:
    """extract_mce_trajectories.py の出力 (pkl / json) をロードする。"""
    path = Path(path)
    if path.suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_irl_reward_fn(
    irl_model: str | Path,
    irl_config: str | Path,
    state_dim: int,
    action_dim: int,
    device: str = "cpu",
):
    """学習済み MCE-IRL をロードして IRLRewardFn を返す。

    irl_config (model_metadata.json) の state_dim / action_dim は
    CLI 引数の値で上書きする (呼び出し側とネットワーク形状を揃えるため)。
    """
    import torch

    from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem
    from review_predictor.RL.env.reward_fn import IRLRewardFn

    with open(irl_config, encoding="utf-8") as f:
        config = json.load(f)
    config["state_dim"] = state_dim
    config["action_dim"] = action_dim

    system = MCEIRLSystem(config)
    ckpt = torch.load(irl_model, map_location=device, weights_only=True)
    if "network_state_dict" in ckpt:
        system.network.load_state_dict(ckpt["network_state_dict"])
    else:
        system.network.load_state_dict(ckpt)
    system.network.to(system.device)
    system.network.eval()
    return IRLRewardFn(system)


def load_transitions_df(path: str | Path) -> pd.DataFrame:
    """transitions ファイル (.parquet / .json) を DataFrame でロードする。"""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_json(p)


def to_vec(val: Any, dim: int) -> List[float]:
    """state セルをパディング/トリミング付きで固定長リスト化する。

    parquet は list を numpy.ndarray として返すため両方受け付ける。
    """
    if isinstance(val, (list, np.ndarray)):
        v = list(val[:dim])
        v += [0.0] * max(0, dim - len(v))
        return v
    return [float(val)] + [0.0] * (dim - 1)


def states_matrix(values: Iterable[Any], dim: int) -> np.ndarray:
    """state 列を [N, dim] の float32 行列に変換する。"""
    return np.array([to_vec(v, dim) for v in values], dtype=np.float32)


def recompute_reward(df: pd.DataFrame, reward_shaping: float) -> np.ndarray:
    """保存済み列から報酬を再計算する: r = step_label + λ·R_θ。

    λ (reward_shaping) の sweep でデータセットの再ビルド (重い特徴量抽出)
    を不要にするための関数。シェイピングに使う R_θ は、月次版は
    r_theta_next 列 (fe_{t+1} 時点)、event 版は r_theta 列 (所属月) —
    どちらもビルド時に reward の計算へ使われた値そのもの。
    """
    shaping_col = "r_theta_next" if "r_theta_next" in df.columns else "r_theta"
    if "step_label" not in df.columns or shaping_col not in df.columns:
        raise ValueError(
            "報酬の再計算には step_label / r_theta(_next) 列が必要です。"
            "旧フォーマットのデータセットは再ビルドしてください。"
        )
    reward = (
        df["step_label"].astype(float)
        + reward_shaping * df[shaping_col].astype(float)
    ).to_numpy(dtype=np.float32)
    logger.info(
        "報酬を再計算: r = step_label + %.3f·%s (mean=%.4f)",
        reward_shaping, shaping_col, float(reward.mean()),
    )
    return reward
