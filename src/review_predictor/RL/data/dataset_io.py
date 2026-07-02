"""RL transitions の保存・時間ベース train/test split の共通処理

build_rl_dataset.py (月次) と build_rl_dataset_event.py (event 単位) で
逐語的に重複していた split ロジックの一元化。分割は feature_end 列に
対する純粋な時間比較のため粒度に依存しない。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _save_df(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_json(path, orient="records", force_ascii=False, indent=2)


def save_transitions_with_split(
    transitions: List[Dict[str, Any]],
    output_path: Path,
    test_split_date: Optional[str] = None,
) -> pd.DataFrame:
    """transitions を DataFrame 化して保存する。

    test_split_date が指定され feature_end 列が存在する場合、
    feature_end >= test_split_date の行を `<stem>_test<suffix>` に分離保存し、
    それ以外を output_path に保存する。

    Args:
        transitions: build_transitions() / build_event_transitions() の出力
        output_path: train 側の保存先 (.parquet または .json)
        test_split_date: test 分割の基準日 (YYYY-MM-DD)。None なら分割しない

    Returns:
        train 側の DataFrame (統計ログ出力用)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(transitions)

    if test_split_date and "feature_end" in df.columns:
        split_ts = pd.Timestamp(test_split_date)
        feature_end_ts = pd.to_datetime(df["feature_end"], errors="coerce")
        is_test = feature_end_ts >= split_ts
        n_test = int(is_test.sum())
        n_train = int((~is_test).sum())
        if n_test > 0:
            test_df = df[is_test].copy()
            df = df[~is_test].copy()
            test_path = output_path.with_name(
                output_path.stem + "_test" + output_path.suffix
            )
            _save_df(test_df, test_path)
            logger.info(
                "train/test split (基準日=%s): train=%d, test=%d → test を %s に保存",
                test_split_date, n_train, n_test, test_path,
            )
        else:
            logger.warning(
                "test_split_date=%s 以降の transition が0件のため split 無効",
                test_split_date,
            )

    _save_df(df, output_path)
    logger.info("保存完了 (train): %s", output_path)
    return df
