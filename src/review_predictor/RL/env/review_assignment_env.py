"""レビュー依頼割り当て gymnasium 環境

IRL で学習済みの報酬関数 R_θ と継続確率 π(a=1|s) を使い、
「どの開発者にレビュー依頼を割り当てるか」を決める政策を強化学習で学ぶ。

観測空間: 各候補開発者の観測ベクトルを結合したフラットベクトル
          shape = [n_candidates × obs_dim] (obs_dim = state_dim + 1)
行動空間: Discrete(n_candidates) — 候補の中から1人を選ぶ
報酬:     α * π(a=1|s_chosen) + β * R_θ(s_chosen, a=1) - γ * load_penalty
エピソード: 最大 max_steps 件のレビュー依頼を順次割り当てる

注意:
  - 候補開発者数は固定（n_candidates）とする（可変候補はフェーズ3で対応）
  - 候補数が不足する場合はゼロパディングで補完する
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gymnasium
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReviewAssignmentEnv(gymnasium.Env):
    """レビュー依頼割り当て環境 (単一エージェント, 固定候補数)。

    使い方:
        env = ReviewAssignmentEnv(config, reward_fn, state_builder, episodes)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_candidates: int,
        obs_dim: int,
        reward_fn,
        state_builder,
        episodes: List[Dict[str, Any]],
        max_steps: int = 200,
        acceptance_weight: float = 0.6,
        reward_weight: float = 0.3,
        load_penalty_weight: float = 0.1,
        load_penalty_threshold: int = 5,
    ) -> None:
        """
        Args:
            n_candidates: 候補開発者数（固定）
            obs_dim: 1開発者あたりの観測次元（state_dim + 1）
            reward_fn: IRLRewardFn インスタンス
            state_builder: RLStateBuilder インスタンス
            episodes: エピソードリスト。各要素は以下のキーを持つ dict:
                - 'requests': レビュー依頼リスト（各要素に候補情報）
                - 'activity_df': 開発者活動データ
                - 'feature_end': 予測基準日
            max_steps: 1エピソードの最大ステップ数
            acceptance_weight: π(a=1|s) の報酬係数 α
            reward_weight: R_θ の報酬係数 β
            load_penalty_weight: 負荷ペナルティの係数 γ
            load_penalty_threshold: 負荷ペナルティが発動する累積割り当て件数
        """
        super().__init__()
        self.n_candidates = n_candidates
        self.obs_dim = obs_dim
        self.reward_fn = reward_fn
        self.state_builder = state_builder
        self.episodes = episodes
        self.max_steps = max_steps
        self.alpha = acceptance_weight
        self.beta = reward_weight
        self.gamma = load_penalty_weight
        self.load_threshold = load_penalty_threshold

        # gymnasium spaces
        flat_obs_dim = n_candidates * obs_dim
        self.observation_space = gymnasium.spaces.Box(
            low=-1.0, high=2.0, shape=(flat_obs_dim,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(n_candidates)

        # エピソード状態
        self._ep_idx: int = 0
        self._step_count: int = 0
        self._current_episode: Optional[Dict[str, Any]] = None
        self._assign_counts: Dict[str, int] = {}
        self._current_requests: List[Dict[str, Any]] = []
        self._request_ptr: int = 0

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if options and "ep_idx" in options:
            self._ep_idx = int(options["ep_idx"]) % len(self.episodes)
        else:
            self._ep_idx = self.np_random.integers(0, len(self.episodes))

        self._current_episode = self.episodes[self._ep_idx]
        self._current_requests = self._current_episode.get("requests", [])
        self._request_ptr = 0
        self._step_count = 0
        self._assign_counts = {}

        obs = self._build_obs()
        return obs, {"ep_idx": self._ep_idx}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self._current_episode is not None, "reset() を先に呼んでください"

        candidates = self._get_candidates()
        pi_a1, r_theta = 0.0, 0.0
        if action >= len(candidates):
            # パディング領域を選択した場合はペナルティ
            reward = -1.0
            chosen_email = ""
        else:
            chosen = candidates[action]
            chosen_email = chosen.get("email", "")
            reward, pi_a1, r_theta = self._compute_reward(chosen, action)

        # 割り当て件数を更新
        if chosen_email:
            self._assign_counts[chosen_email] = (
                self._assign_counts.get(chosen_email, 0) + 1
            )

        self._request_ptr += 1
        self._step_count += 1

        terminated = self._request_ptr >= len(self._current_requests)
        truncated = self._step_count >= self.max_steps

        obs = self._build_obs()
        info = {
            "pi_a1": pi_a1 if action < len(candidates) else 0.0,
            "r_theta": r_theta if action < len(candidates) else 0.0,
            "chosen_email": chosen_email,
            "assign_counts": dict(self._assign_counts),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _get_candidates(self) -> List[Dict[str, Any]]:
        """現在ステップの候補開発者リストを返す。"""
        if self._request_ptr >= len(self._current_requests):
            return []
        req = self._current_requests[self._request_ptr]
        return req.get("candidates", [])

    def _build_obs(self) -> np.ndarray:
        """現在ステップの観測ベクトル [n_candidates * obs_dim] を構築する。"""
        if self._request_ptr >= len(self._current_requests) or not self._current_episode:
            return np.zeros(self.n_candidates * self.obs_dim, dtype=np.float32)

        candidates = self._get_candidates()
        activity_df = self._current_episode.get("activity_df")
        feature_end = self._current_episode.get("feature_end")

        if activity_df is None or feature_end is None or not candidates:
            return np.zeros(self.n_candidates * self.obs_dim, dtype=np.float32)

        obs_matrix = self.state_builder.build_batch_obs(
            candidates=candidates,
            activity_df=activity_df,
            feature_end=feature_end,
            episode_assign_counts=self._assign_counts,
        )  # [len(candidates), obs_dim]

        # n_candidates に合わせてパディングまたはトリミング
        n = len(candidates)
        if n < self.n_candidates:
            pad = np.zeros(
                (self.n_candidates - n, self.obs_dim), dtype=np.float32
            )
            obs_matrix = np.vstack([obs_matrix, pad])
        elif n > self.n_candidates:
            obs_matrix = obs_matrix[: self.n_candidates]

        return obs_matrix.flatten()

    def _compute_reward(
        self, chosen: Dict[str, Any], action: int
    ) -> Tuple[float, float, float]:
        """選択された開発者の報酬を計算する。

        Returns:
            total_reward, pi_a1, r_theta
        """
        if not self._current_episode:
            return 0.0, 0.5, 0.0

        activity_df = self._current_episode.get("activity_df")
        feature_end = self._current_episode.get("feature_end")
        if activity_df is None or feature_end is None:
            return 0.0, 0.5, 0.0

        email = chosen.get("email", "")
        path_vec = chosen.get("path_features_vec")
        profile_vec = chosen.get("profile_features_vec")
        assign_count = self._assign_counts.get(email, 0)

        # 状態ベクトルを構築（1ステップ系列として扱う）
        state_vec = self.state_builder.build_obs(
            email=email,
            activity_df=activity_df,
            feature_end=feature_end,
            episode_assign_count=assign_count,
            path_features_vec=path_vec,
            profile_features_vec=profile_vec,
        )
        action_vec = self.state_builder.build_action_features(
            email=email,
            activity_df=activity_df,
            feature_end=feature_end,
        )

        # IRL の state_dim に対応するため、episode_assign_count を除いた部分だけ渡す
        state_for_irl = torch.tensor(
            state_vec[: self.state_builder.state_dim], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, state_dim]
        action_for_irl = torch.tensor(
            action_vec, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, action_dim]
        length = torch.tensor([1], dtype=torch.long)

        r_theta, pi_a1 = self.reward_fn.batch_reward(
            state_for_irl, action_for_irl, length
        )
        r_theta = float(r_theta[0].item())
        pi_a1 = float(pi_a1[0].item())

        # 負荷ペナルティ: 割り当て件数が threshold を超えると線形ペナルティ
        excess = max(0, assign_count - self.load_threshold)
        load_penalty = min(excess / self.load_threshold, 1.0)

        total_reward = (
            self.alpha * pi_a1
            + self.beta * max(r_theta, -1.0)  # R_θ をクリップ
            - self.gamma * load_penalty
        )
        return total_reward, pi_a1, r_theta
