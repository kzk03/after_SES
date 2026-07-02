"""オフライン RL データセット

歴史データの軌跡（trajectory）から (s, a, r, s', done) タプルを生成し、
PyTorch Dataset として返す。

報酬 r は学習済み R_θ(s, a) で計算される（事後ラベル付け）。
状態遷移は月次ステップ: s_t → s_{t+1} は翌月の特徴量ベクトル。

データリーク防止:
  既存の trajectory が持つ feature_start / feature_end の管理を踏襲し、
  予測対象月の label を訓練特徴量として使わない。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class OfflineRLDataset(Dataset):
    """オフライン RL 用 PyTorch Dataset。

    各サンプル: (s, a, r, s', done)
      s:    状態ベクトル [state_dim + 1]（IRL state + episode_assign_count）
      a:    行動（0=割り当て拒否/非継続, 1=割り当て承諾/継続）
      r:    R_θ(s, a) — IRL ラベル付き報酬
      s':   次状態ベクトル [state_dim + 1]
      done: エピソード終端フラグ
    """

    def __init__(
        self,
        transitions: List[Dict[str, Any]],
        device: str = "cpu",
    ) -> None:
        """
        Args:
            transitions: build_transitions() の出力リスト
            device: torch デバイス文字列
        """
        self.device = torch.device(device)
        if not transitions:
            raise ValueError("transitions リストが空です")

        obs_dim = len(transitions[0]["state"])
        n = len(transitions)

        self.states = torch.zeros(n, obs_dim, dtype=torch.float32)
        self.actions = torch.zeros(n, dtype=torch.long)
        self.rewards = torch.zeros(n, dtype=torch.float32)
        self.next_states = torch.zeros(n, obs_dim, dtype=torch.float32)
        self.dones = torch.zeros(n, dtype=torch.float32)

        for i, tr in enumerate(transitions):
            self.states[i] = torch.tensor(tr["state"], dtype=torch.float32)
            self.actions[i] = int(tr["action"])
            self.rewards[i] = float(tr["reward"])
            self.next_states[i] = torch.tensor(tr["next_state"], dtype=torch.float32)
            self.dones[i] = float(tr["done"])

        logger.info(
            "OfflineRLDataset: %d サンプル, obs_dim=%d, reward_mean=%.4f±%.4f",
            n,
            obs_dim,
            float(self.rewards.mean()),
            float(self.rewards.std()),
        )

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def to(self, device: str) -> "OfflineRLDataset":
        """デバイスを変更してインプレース更新する。"""
        self.device = torch.device(device)
        self.states = self.states.to(self.device)
        self.actions = self.actions.to(self.device)
        self.rewards = self.rewards.to(self.device)
        self.next_states = self.next_states.to(self.device)
        self.dones = self.dones.to(self.device)
        return self

    @property
    def obs_dim(self) -> int:
        return self.states.shape[1]

    @property
    def n_actions(self) -> int:
        return int(self.actions.max().item()) + 1


def _monthly_accept(
    month_history: List[Dict[str, Any]],
    window_start: Optional[pd.Timestamp],
    window_end: Optional[pd.Timestamp],
) -> int:
    """区間 (window_start, window_end] 内に承諾イベントが 1件以上あれば 1、無ければ 0。

    action = 「その月 dev が承諾レビューを行ったか」(承諾/非承諾の binary 判定)。

    注意: monthly_activity_histories[t] は train_model.py 側で
    `reviewer_history[date_col < month_end]` として構築される **累積履歴**
    （下限フィルタなし）であり、「月 t だけの新規イベント」ではない。
    そのため window_start/window_end で区間を絞らずに月 t の履歴全体を
    スキャンすると、一度でも承諾歴のある dev は以降ずっと action=1 に
    固定されてしまう（単調増加バグ）。ここでは直前ステップの feature_end
    より後、今回の feature_end 以下のイベントだけを見ることで、
    「その月固有の」承諾判定にする。
    window_start=None の場合は最初のステップとみなし下限なしで判定する
    （軌跡の最初の観測ステップは前区間の情報を持たないため）。
    """
    if not month_history or window_end is None:
        return 0
    for act in month_history:
        ts = act.get("timestamp")
        if ts is None:
            continue
        ts = pd.Timestamp(ts)
        if window_start is not None and ts <= window_start:
            continue
        if ts > window_end:
            continue
        if act.get("accepted", False) or act.get("label") == 1:
            return 1
    return 0


def build_transitions(
    trajectories: List[Dict[str, Any]],
    reward_fn,
    state_builder,
    feature_window_days: int = 180,
    reward_shaping: float = 0.3,
) -> List[Dict[str, Any]]:
    """軌跡リストから (s, a, r, s', done) タプルリストを構築する。

    MDP 定義:
      s_t  : 月 t の dev 特徴量
      a_t  : 月 t (新規区間) に dev が承諾レビューを行ったか (Discrete(2))
             ※ monthly_activity_histories[t] は累積履歴のため、
                前ステップの feature_end より後のイベントだけを見て判定する
                (_monthly_accept 参照)
      r_t  : step_labels[t] + λ · R_θ(s_t, action_features_t)
             - step_labels[t]: 3ヶ月先継続の binary ラベル (IRL ラベル)
             - R_θ:            IRL の logits 差分 (連続値、ノイズ除去シェイピング)
             - λ (reward_shaping): R_θ の重み (推奨 0.3)
      s_{t+1}: 月 t+1 の dev 特徴量

    Args:
        trajectories: extract_mce_trajectories.py の出力形式の軌跡リスト
        reward_fn: IRLRewardFn インスタンス
        state_builder: RLStateBuilder インスタンス
        feature_window_days: 特徴量抽出窓（日数）
        reward_shaping: R_θ のシェイピング係数 λ (0.0 で binary のみ)

    Returns:
        transition dict のリスト
    """
    import torch
    from review_predictor.IRL.features.common_features import ACTION_FEATURES

    transitions: List[Dict[str, Any]] = []
    skipped = 0

    for traj in trajectories:
        developer = traj.get("developer", traj.get("developer_info", {}))
        email = developer.get(
            "email", developer.get("developer_id", developer.get("reviewer", ""))
        )
        monthly_histories = traj.get("monthly_activity_histories", [])
        step_labels = traj.get("step_labels", [])
        step_context_dates = traj.get("step_context_dates", [])
        path_features_per_step = traj.get("path_features_per_step", [])
        profile_features_static = traj.get("profile_features", None)

        min_len = min(len(monthly_histories), len(step_labels))
        if min_len < 2:
            skipped += 1
            continue

        # 月次ステップを (s_t, a_t, s_{t+1}) の系列に変換
        state_vecs: List[np.ndarray] = []
        action_vecs: List[np.ndarray] = []
        step_feature_ends: List[Optional[pd.Timestamp]] = []

        for i in range(min_len):
            month_history = monthly_histories[i]
            feature_end_ts = (
                step_context_dates[i]
                if i < len(step_context_dates)
                else None
            )
            if feature_end_ts is None:
                if month_history:
                    feature_end_ts = month_history[-1].get("timestamp")
                if feature_end_ts is None:
                    state_vecs.append(np.zeros(state_builder.state_dim, dtype=np.float32))
                    action_vecs.append(np.zeros(state_builder.action_dim, dtype=np.float32))
                    step_feature_ends.append(None)
                    continue

            feature_end = pd.Timestamp(feature_end_ts)
            from datetime import timedelta
            feature_start = feature_end - timedelta(days=feature_window_days)

            # activity_df を月次履歴から構築
            rows = []
            for act in month_history:
                ts = act.get("timestamp")
                if ts is None:
                    continue
                rows.append({
                    "email": email if act.get("action_type") != "authored" else act.get("reviewer_email", ""),
                    "timestamp": pd.Timestamp(ts),
                    "label": 1 if act.get("accepted", False) else 0,
                    "owner_email": act.get("owner_email", email),
                    "change_insertions": act.get("lines_added", act.get("change_insertions", 0)) or 0,
                    "change_deletions": act.get("lines_deleted", act.get("change_deletions", 0)) or 0,
                    "change_files_count": act.get("files_changed", act.get("change_files_count", 0)) or 0,
                    "first_response_time": act.get("response_time", act.get("first_response_time")),
                })

            if not rows:
                state_vecs.append(np.zeros(state_builder.state_dim, dtype=np.float32))
                action_vecs.append(np.zeros(state_builder.action_dim, dtype=np.float32))
                step_feature_ends.append(feature_end)
                continue

            activity_df = pd.DataFrame(rows)
            path_vec = path_features_per_step[i] if i < len(path_features_per_step) else None

            sv = state_builder.build_obs(
                email=email,
                activity_df=activity_df,
                feature_end=feature_end.to_pydatetime(),
                episode_assign_count=0,
                path_features_vec=path_vec,
                profile_features_vec=profile_features_static,
                feature_window_days=feature_window_days,
            )
            av = state_builder.build_action_features(
                email=email,
                activity_df=activity_df,
                feature_end=feature_end.to_pydatetime(),
                feature_window_days=feature_window_days,
            )
            state_vecs.append(sv)
            action_vecs.append(av)
            step_feature_ends.append(feature_end)

        if len(state_vecs) < 2:
            skipped += 1
            continue

        # 各ステップの R_θ をバッチ計算
        # state_dim 部分だけを IRL に渡す（+1 の episode_assign_count を除く）
        irl_states = torch.tensor(
            np.stack([sv[:state_builder.state_dim] for sv in state_vecs]),
            dtype=torch.float32,
        ).unsqueeze(0)  # [1, L, state_dim]
        irl_actions = torch.tensor(
            np.stack(action_vecs), dtype=torch.float32
        ).unsqueeze(0)  # [1, L, action_dim]
        length = torch.tensor([len(state_vecs)], dtype=torch.long)

        # forward_all_steps で全ステップの R_θ を取得
        try:
            reward_fn.system.network.eval()
            import torch.nn.functional as F
            with torch.no_grad():
                logits_all = reward_fn.system.network.forward_all_steps(
                    irl_states.to(reward_fn.device),
                    irl_actions.to(reward_fn.device),
                    length.to(reward_fn.device),
                )  # [1, L, 2]
            logits_all = logits_all[0]  # [L, 2]
            rewards_all = (logits_all[:, 1] - logits_all[:, 0]).cpu().numpy()
        except Exception as e:
            logger.warning("R_θ 計算エラー (email=%s): %s", email, e)
            skipped += 1
            continue

        # (s_t, a_t, r_t, s_{t+1}, done) タプルを作成
        # action = 月 t (新規区間) に dev が承諾を行ったか (累積 monthly_histories[t] から差分抽出)
        # reward = step_labels[t] (3ヶ月先継続 binary) + λ · R_θ (ノイズ除去シェイピング)
        for t in range(len(state_vecs) - 1):
            month_history = (
                monthly_histories[t] if t < len(monthly_histories) else []
            )
            window_end = step_feature_ends[t] if t < len(step_feature_ends) else None
            # 直前ステップの feature_end が None (解決不能) な場合、そのまま None を
            # 使うと下限フィルタが外れて累積スキャンバグが再発するため、
            # 直近の非 None な feature_end まで遡って探す。
            window_start = None
            for prev_t in range(t - 1, -1, -1):
                if prev_t < len(step_feature_ends) and step_feature_ends[prev_t] is not None:
                    window_start = step_feature_ends[prev_t]
                    break
            action_label = _monthly_accept(month_history, window_start, window_end)
            label_t = float(bool(step_labels[t])) if t < len(step_labels) else 0.0
            shaped_reward = label_t + reward_shaping * float(rewards_all[t])
            done = t == len(state_vecs) - 2
            feature_end_t = step_feature_ends[t] if t < len(step_feature_ends) else None
            transitions.append({
                "state": state_vecs[t].tolist(),
                "action": action_label,
                "reward": shaped_reward,
                "next_state": state_vecs[t + 1].tolist(),
                "done": float(done),
                "email": email,
                "step_label": label_t,
                "r_theta": float(rewards_all[t]),
                "feature_end": (
                    feature_end_t.isoformat() if feature_end_t is not None else None
                ),
            })

    logger.info(
        "build_transitions 完了: %d transitions, %d 軌跡をスキップ",
        len(transitions),
        skipped,
    )
    return transitions
