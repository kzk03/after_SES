"""オフライン RL データセット (月次 MDP)

歴史データの軌跡（trajectory）から (s, a, r, s', done) タプルを生成し、
PyTorch Dataset として返す。

時間順序 (リーク防止の要):
  s_t (fe_t 時点) → a_t (月 t+1 の承諾) → r_t (fe_{t+1} 起点の3ヶ月継続)
  - 行動は状態の特徴窓より未来 → 状態から行動を読み取れない
  - 報酬窓は行動区間の後ろ → 行動が報酬に機械的に含まれない
  詳細は build_transitions の docstring を参照。

特徴量は IRL 学習時 (mce_irl_predictor._extract_one) と同じ定義
(全履歴窓 + step_total_project_reviews) で再構築する。
凍結 R_θ に学習時と同じ分布の入力を渡すため。
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
        # MDP 定義上は Discrete(2)。action=0 のみのサブセットでも
        # 出力層のサイズが 1 に潰れないよう下限 2 を保証する。
        return max(int(self.actions.max().item()) + 1, 2)


def _monthly_accept(
    month_history: List[Dict[str, Any]],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> int:
    """区間 (window_start, window_end] 内に承諾イベントが 1件以上あれば 1、無ければ 0。

    action = 「その区間 (1ヶ月) に dev が承諾レビューを行ったか」の binary 判定。

    注意: monthly_activity_histories[t] は train_model.py 側で
    `reviewer_history[date_col < month_end]` として構築される **累積履歴**
    （下限フィルタなし）であり、「月 t だけの新規イベント」ではない。
    そのため window_start/window_end で区間を絞らずに履歴全体を
    スキャンすると、一度でも承諾歴のある dev は以降ずっと action=1 に
    固定されてしまう（単調増加バグ）。区間の両端は呼び出し側が
    step_feature_ends (月末境界) から渡す。境界が解決できないステップは
    呼び出し側で transition 自体を除外する。
    """
    if not month_history:
        return 0
    for act in month_history:
        ts = act.get("timestamp")
        if ts is None:
            continue
        ts = pd.Timestamp(ts)
        if ts <= window_start or ts > window_end:
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

    MDP 定義 (状態 → 行動 → 結果の時間順序を保証する):
      s_t  : 月 t の月末 (fe_t) 時点の dev 特徴量
      a_t  : **月 t+1** (区間 (fe_t, fe_{t+1}]) に dev が承諾レビューを行ったか
             (Discrete(2))。s_t の特徴窓は fe_t までなので、行動は状態より
             未来にあり「状態から行動を読み取れる」リークがない。
             ※ monthly_activity_histories[t+1] は累積履歴のため、
                区間フィルタで月 t+1 固有のイベントだけを見る (_monthly_accept)
      r_t  : step_labels[t+1] + λ · R_θ[t+1]
             - step_labels[t+1]: fe_{t+1} 起点の3ヶ月継続 binary ラベル。
               窓は (fe_{t+1}, fe_{t+1}+3m] で行動区間 (fe_t, fe_{t+1}] の
               **後ろ**から始まるため、行動が報酬に機械的に含まれない
               (a=1 ⇒ r=1 の自明な相関を防ぐ)
             - R_θ[t+1]:         同じ fe_{t+1} 時点の IRL logits 差分
               (label と同じ窓を指す soft 信号、ノイズ除去シェイピング)
             - λ (reward_shaping): R_θ の重み (推奨 0.3)
      s_{t+1}: 月 t+1 の月末 (fe_{t+1}) 時点の dev 特徴量 (行動の結果を反映)

    備考: transition の "r_theta" 列には評価ベースライン用に
    **意思決定時点の R_θ[t]** (s_t から計算した IRL スコア) を格納する。
    報酬シェイピングに使う R_θ[t+1] は "r_theta_next" 列に別途保存する。
    (R_θ[t+1] は行動月の情報を含むため、これを IRL ベースラインの
    スコアとして評価に使うと行動情報がリークして不公平になる)

    Args:
        trajectories: extract_mce_trajectories.py の出力形式の軌跡リスト
        reward_fn: IRLRewardFn インスタンス
        state_builder: RLStateBuilder インスタンス
        feature_window_days: 特徴量抽出窓（日数）
        reward_shaping: R_θ のシェイピング係数 λ (0.0 で binary のみ)

    Returns:
        transition dict のリスト
    """
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
        step_total_project_reviews = traj.get("step_total_project_reviews", [])
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
                    # month_history はソート保証がないため、末尾要素ではなく
                    # 最大 timestamp を採用する (未ソート CSV 由来でも正しい)
                    feature_end_ts = max(
                        (
                            pd.Timestamp(a["timestamp"])
                            for a in month_history
                            if a.get("timestamp") is not None
                        ),
                        default=None,
                    )
                if feature_end_ts is None:
                    state_vecs.append(np.zeros(state_builder.state_dim, dtype=np.float32))
                    action_vecs.append(np.zeros(state_builder.action_dim, dtype=np.float32))
                    step_feature_ends.append(None)
                    continue

            feature_end = pd.Timestamp(feature_end_ts)

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
            # IRL 学習時 (_extract_one) と同じく step_total_project_reviews を渡す。
            # 0 のままだと core_reviewer_ratio 等が学習時と別分布になる。
            total_proj = (
                int(step_total_project_reviews[i])
                if i < len(step_total_project_reviews)
                else 0
            )

            sv = state_builder.build_obs(
                email=email,
                activity_df=activity_df,
                feature_end=feature_end.to_pydatetime(),
                episode_assign_count=0,
                path_features_vec=path_vec,
                profile_features_vec=profile_features_static,
                total_project_reviews=total_proj,
                feature_window_days=feature_window_days,
            )
            av = state_builder.build_action_features(
                email=email,
                activity_df=activity_df,
                feature_end=feature_end.to_pydatetime(),
                total_project_reviews=total_proj,
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
        # state  = fe_t 時点の特徴量 (行動より過去の情報のみ)
        # action = 月 t+1 (区間 (fe_t, fe_{t+1}]) に dev が承諾を行ったか
        # reward = step_labels[t+1] (fe_{t+1} 起点の3ヶ月継続) + λ · R_θ[t+1]
        #          → 報酬窓は行動区間の後ろから始まり、行動と重ならない
        traj_transitions: List[Dict[str, Any]] = []
        for t in range(len(state_vecs) - 1):
            window_start = step_feature_ends[t]
            window_end = step_feature_ends[t + 1]
            # 行動区間の境界が解決できないステップは、区間フィルタが壊れて
            # 累積スキャンバグが再発するため transition 化しない
            if window_start is None or window_end is None:
                continue
            month_history = (
                monthly_histories[t + 1] if t + 1 < len(monthly_histories) else []
            )
            action_label = _monthly_accept(month_history, window_start, window_end)
            label_next = (
                float(bool(step_labels[t + 1])) if t + 1 < len(step_labels) else 0.0
            )
            shaped_reward = label_next + reward_shaping * float(rewards_all[t + 1])
            traj_transitions.append({
                "state": state_vecs[t].tolist(),
                "action": action_label,
                "reward": shaped_reward,
                "next_state": state_vecs[t + 1].tolist(),
                "done": 0.0,
                "email": email,
                "step_label": label_next,
                # r_theta: 意思決定時点 (s_t) の IRL スコア。評価ベースライン用
                "r_theta": float(rewards_all[t]),
                # r_theta_next: 報酬シェイピングに使った fe_{t+1} 時点の R_θ
                "r_theta_next": float(rewards_all[t + 1]),
                "feature_end": window_start.isoformat(),
            })
        if traj_transitions:
            traj_transitions[-1]["done"] = 1.0
        transitions.extend(traj_transitions)

    logger.info(
        "build_transitions 完了: %d transitions, %d 軌跡をスキップ",
        len(transitions),
        skipped,
    )
    return transitions
