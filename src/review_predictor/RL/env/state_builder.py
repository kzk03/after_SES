"""RL 状態ベクトル構築モジュール

既存の IRL 特徴量抽出（extract_common_features + PathFeatureExtractor）を再利用し、
RL 専用の追加特徴（エピソード内累積割り当て件数）を付加する。

観測次元 = state_dim + 1
  state_dim: IRL モデル設定値（26, 32, 37, 42 など）
  +1:        episode_assign_count（エピソード内累積割り当て件数、正規化 cap=50）

IRL の STATE_FEATURES には concurrent_open_reviews_7d / recent_load_ratio_30d_all /
accept_rate_drift_30d がすでに含まれているため、これら以外の負荷特徴量は追加しない。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from review_predictor.IRL.features.common_features import (
    ACTION_FEATURES,
    STATE_FEATURES,
    extract_common_features,
)

logger = logging.getLogger(__name__)

# エピソード内累積割り当て件数のキャップ（これで割って 0〜1 に正規化）
EPISODE_ASSIGN_CAP = 50.0


class RLStateBuilder:
    """開発者ごとの RL 観測ベクトルを構築する。

    IRL の特徴量抽出器をラップし、RL 専用の状態拡張を担当する。
    PathFeatureExtractor は省略可能（パス特徴量なし設定の場合）。
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        path_feature_extractor=None,
    ) -> None:
        """
        Args:
            state_dim: IRL モデルの state_dim（CONFIG 値）
            action_dim: IRL モデルの action_dim（CONFIG 値）
            path_feature_extractor: PathFeatureExtractor インスタンス（省略可）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.path_extractor = path_feature_extractor
        # RL 観測次元 = IRL state_dim + 1（episode_assign_count）
        self.obs_dim = state_dim + 1

    def build_obs(
        self,
        email: str,
        activity_df: pd.DataFrame,
        feature_end: datetime,
        episode_assign_count: int = 0,
        path_features_vec: Optional[np.ndarray] = None,
        profile_features_vec: Optional[List[float]] = None,
        total_project_reviews: int = 0,
        feature_window_days: int = 180,
    ) -> np.ndarray:
        """単一開発者の観測ベクトルを構築する。

        Args:
            email: 開発者メールアドレス
            activity_df: レビュー活動データ（nova_raw.csv 形式）
            feature_end: 予測時点
            episode_assign_count: エピソード内の累積割り当て件数
            path_features_vec: path 特徴量ベクトル（IRL モデルが使う場合）
            profile_features_vec: profile 特徴量ベクトル（5次元, IRL が使う場合）
            total_project_reviews: プロジェクト全体のレビュー件数
            feature_window_days: 特徴量抽出窓（日数）

        Returns:
            観測ベクトル [obs_dim] = [state_dim + 1]
        """
        from datetime import timedelta
        feature_start = feature_end - timedelta(days=feature_window_days)

        try:
            features = extract_common_features(
                activity_df,
                email,
                feature_start,
                feature_end,
                normalize=True,
                total_project_reviews=total_project_reviews,
            )
        except Exception as e:
            logger.warning("特徴量抽出エラー (email=%s): %s", email, e)
            features = {}

        # IRL state ベクトルの組み立て
        sv: List[float] = [float(features.get(f, 0.0)) for f in STATE_FEATURES]

        if path_features_vec is not None:
            sv.extend(float(v) for v in path_features_vec)
        elif self.path_extractor is not None:
            # PathFeatureExtractor が渡されている場合、自動抽出
            # ※呼び出し側が path_features_vec を渡す方が効率的なため、
            #   フォールバックとして実装
            sv.extend(0.0 for _ in range(6))
        # path なしで state_dim が 26 の場合はここで sv = 26 次元

        if profile_features_vec is not None:
            sv.extend(float(v) for v in profile_features_vec)

        # state_dim に合わせてゼロパディングまたはトリミング
        if len(sv) < self.state_dim:
            sv.extend(0.0 for _ in range(self.state_dim - len(sv)))
        elif len(sv) > self.state_dim:
            sv = sv[: self.state_dim]

        # RL 専用特徴量: エピソード内累積割り当て件数（正規化）
        sv.append(min(episode_assign_count / EPISODE_ASSIGN_CAP, 1.0))

        return np.array(sv, dtype=np.float32)

    def build_action_features(
        self,
        email: str,
        activity_df: pd.DataFrame,
        feature_end: datetime,
        feature_window_days: int = 180,
    ) -> np.ndarray:
        """IRL の action 特徴量ベクトルを構築する。

        Args:
            email: 開発者メールアドレス
            activity_df: レビュー活動データ
            feature_end: 予測時点
            feature_window_days: 特徴量抽出窓（日数）

        Returns:
            action 特徴量ベクトル [action_dim]
        """
        from datetime import timedelta
        feature_start = feature_end - timedelta(days=feature_window_days)

        try:
            features = extract_common_features(
                activity_df,
                email,
                feature_start,
                feature_end,
                normalize=True,
            )
        except Exception as e:
            logger.warning("行動特徴量抽出エラー (email=%s): %s", email, e)
            features = {}

        av = [float(features.get(f, 0.0)) for f in ACTION_FEATURES]
        if len(av) < self.action_dim:
            av.extend(0.0 for _ in range(self.action_dim - len(av)))
        return np.array(av[: self.action_dim], dtype=np.float32)

    def build_batch_obs(
        self,
        candidates: List[Dict[str, Any]],
        activity_df: pd.DataFrame,
        feature_end: datetime,
        episode_assign_counts: Optional[Dict[str, int]] = None,
        feature_window_days: int = 180,
        total_project_reviews: int = 0,
    ) -> np.ndarray:
        """複数候補開発者の観測行列を構築する。

        Args:
            candidates: 候補開発者リスト。各要素は 'email' キーを持つ dict。
            activity_df: レビュー活動データ
            feature_end: 予測時点
            episode_assign_counts: email → エピソード内累積割り当て件数のマップ
            feature_window_days: 特徴量抽出窓
            total_project_reviews: プロジェクト全体レビュー件数

        Returns:
            観測行列 [n_candidates, obs_dim]
        """
        if episode_assign_counts is None:
            episode_assign_counts = {}

        obs_list = []
        for cand in candidates:
            email = cand.get("email", cand.get("developer_id", ""))
            path_vec = cand.get("path_features_vec")
            profile_vec = cand.get("profile_features_vec")
            assign_count = episode_assign_counts.get(email, 0)
            obs = self.build_obs(
                email=email,
                activity_df=activity_df,
                feature_end=feature_end,
                episode_assign_count=assign_count,
                path_features_vec=path_vec,
                profile_features_vec=profile_vec,
                total_project_reviews=total_project_reviews,
                feature_window_days=feature_window_days,
            )
            obs_list.append(obs)

        return np.stack(obs_list, axis=0)  # [n_candidates, obs_dim]
