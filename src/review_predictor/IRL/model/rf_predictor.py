"""
Random Forest baseline for review continuation prediction.

Uses the same 31-dimensional features as IRL (via common_features.py)
to train a scikit-learn RandomForestClassifier.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from review_predictor.IRL.features.common_features import (
    FEATURE_NAMES,
    extract_common_features,
)

logger = logging.getLogger(__name__)


def extract_features_for_window(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    project_type: str = "multi",
) -> pd.DataFrame:
    """Extract 31-dim features per developer.

    Feature period: train_start ~ train_end
    Label period: eval_start ~ eval_end
      - At least 1 acceptance -> label=1
      - All rejections -> label=0
      - No requests -> excluded

    Args:
        df: DataFrame with 'email', 'timestamp', 'label' columns.
        train_start: Feature computation start.
        train_end: Feature computation end.
        eval_start: Label window start.
        eval_end: Label window end.
        project_type: Project type identifier (unused, kept for API compat).

    Returns:
        DataFrame with feature columns + 'label' + 'email'.
    """
    # Ensure required columns
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have a 'timestamp' column")
    if "email" not in df.columns:
        raise ValueError("DataFrame must have an 'email' column")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Developers active in feature period
    feature_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
    feature_df = df[feature_mask]
    active_emails = feature_df["email"].unique()

    # Label period data
    label_mask = (df["timestamp"] >= eval_start) & (df["timestamp"] < eval_end)
    label_df = df[label_mask]

    rows: List[Dict] = []
    for email in active_emails:
        dev_label = label_df[label_df["email"] == email]
        if len(dev_label) == 0:
            continue  # No requests in eval period -> exclude

        has_acceptance = (dev_label["label"] == 1).any()
        label = 1 if has_acceptance else 0

        features = extract_common_features(
            df, email, train_start, train_end, normalize=False
        )
        features["label"] = label
        features["email"] = email
        rows.append(features)

    if not rows:
        return pd.DataFrame(columns=FEATURE_NAMES + ["label", "email"])

    return pd.DataFrame(rows)


def extract_features_for_window_directory(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    eval_start: pd.Timestamp,
    eval_end: pd.Timestamp,
    path_extractor=None,
    dirs_column: str = "dirs",
) -> pd.DataFrame:
    """Extract 37-dim features per (developer, directory) pair.

    Feature period: train_start ~ train_end
    Label: developer contributed to directory D in eval period.

    Returns:
        DataFrame with 37 feature columns + 'label' + 'email' + 'directory'.
    """
    from review_predictor.IRL.features.common_features import FEATURE_NAMES_WITH_PATH
    from review_predictor.IRL.features.path_features import PATH_FEATURE_NAMES

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    feature_mask = (df["timestamp"] >= train_start) & (df["timestamp"] < train_end)
    feature_df = df[feature_mask]
    active_emails = feature_df["email"].unique()

    label_mask = (df["timestamp"] >= eval_start) & (df["timestamp"] < eval_end)
    label_df = df[label_mask]

    # devごとのフルDF再スキャンを避けるため事前にグループ化する
    # (train_model.py._process_one_reviewer と同じ最適化パターン)。
    # extract_common_features は email(reviewer行)/owner_email(author行) の
    # 両方を email/owner_email + timestamp でしか参照しないため、該当 dev の
    # 行だけを渡しても結果は変わらない(渡された df をそのまま毎回 email/owner_email
    # でフィルタしていたのが、devあたり学習窓ぶんのフルスキャンになっていた)。
    reviewer_by_email = dict(tuple(feature_df.groupby("email")))
    empty_feature = feature_df.iloc[0:0]
    owner_by_email = (
        dict(tuple(feature_df.groupby("owner_email")))
        if "owner_email" in feature_df.columns else {}
    )
    label_by_email = dict(tuple(label_df.groupby("email")))
    empty_label = label_df.iloc[0:0]

    rows: List[Dict] = []
    for email in active_emails:
        # このレビュアーが触ったディレクトリを列挙
        dev_feature = reviewer_by_email.get(email, empty_feature)
        dev_dirs = set()
        for dirs in dev_feature[dirs_column]:
            if dirs:
                dev_dirs.update(d for d in dirs if d != ".")

        # 共通特徴量（25次元）を1回だけ計算。学習窓内の該当 dev 分
        # (reviewer行+author行)だけ渡し、フルDF再スキャンを避ける。
        dev_df = pd.concat([dev_feature, owner_by_email.get(email, empty_feature)])
        common_feats = extract_common_features(
            dev_df, email, train_start, train_end, normalize=False
        )

        dev_label = label_by_email.get(email, empty_label)
        for directory in dev_dirs:
            # ラベル: eval期間にこのディレクトリで承諾があったか
            dev_eval = dev_label[
                dev_label[dirs_column].map(
                    lambda ds: directory in ds if ds else False
                )
            ]
            if len(dev_eval) == 0:
                label = 0
                weak_neg = 1  # 依頼なし＝弱負例（IRL 側の sample_weight=0.5 と同じ区分）
            else:
                label = 1 if (dev_eval["label"] == 1).any() else 0
                weak_neg = 0

            # パス特徴量（3次元）
            if path_extractor is not None:
                pf = path_extractor.compute(
                    email, frozenset({directory}), train_end.to_pydatetime()
                )
                path_feats = {
                    name: float(val)
                    for name, val in zip(PATH_FEATURE_NAMES, pf)
                }
            else:
                path_feats = {name: 0.0 for name in PATH_FEATURE_NAMES}

            row = {**common_feats, **path_feats}
            row["label"] = label
            row["weak_neg"] = weak_neg
            row["email"] = email
            row["directory"] = directory
            rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=FEATURE_NAMES_WITH_PATH + ["label", "weak_neg", "email", "directory"]
        )

    result = pd.DataFrame(rows)
    logger.info(
        f"RF directory features: {len(result)} pairs, "
        f"positive={result['label'].sum()}, negative={(1 - result['label']).sum()}"
    )
    return result


def prepare_rf_features_directory(
    features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert directory-level features DataFrame to (X, y) arrays (37-dim)."""
    from review_predictor.IRL.features.common_features import FEATURE_NAMES_WITH_PATH

    X = features_df[FEATURE_NAMES_WITH_PATH].values.astype(float)
    y = features_df["label"].values.astype(int)
    return X, y


def prepare_rf_features(
    features_df: pd.DataFrame,
    project_type: str = "multi",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert features DataFrame to (X, y) arrays.

    Args:
        features_df: Output of extract_features_for_window.
        project_type: Unused, kept for API compat.

    Returns:
        (X, y) tuple of numpy arrays.
    """
    X = features_df[FEATURE_NAMES].values.astype(float)
    y = features_df["label"].values.astype(int)
    return X, y


