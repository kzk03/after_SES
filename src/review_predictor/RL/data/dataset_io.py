"""RL transitions の保存・train/test split の共通処理

build_rl_dataset.py (月次) と build_rl_dataset_event.py (event 単位) で
逐語的に重複していた split ロジックの一元化。

分割は2軸を提供する (併用可):
  - 時間分割     : feature_end >= 基準日 → `<stem>_test`
    (同一 dev が train/test 両方に登場する。IRL eval と同じ流派)
  - dev 分割     : email ハッシュで決定的に選んだ dev の全 transition →
    `<stem>_devholdout` (未知 dev への汎化を測る。時間分割より先に除外
    するため train / test のどちらにも漏れない)
"""

from __future__ import annotations

import hashlib
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


def _holdout_dev_mask(
    emails: pd.Series, ratio: float, seed: int
) -> pd.Series:
    """email の決定的ハッシュで holdout 対象 dev を選ぶ。

    乱数状態に依存しないため、月次版と event 版で同じ dev 集合が
    holdout される (粒度間の比較が同じ dev 分割で行える)。
    """
    def _in_holdout(email: str) -> bool:
        h = hashlib.sha256(f"{seed}:{email}".encode("utf-8")).hexdigest()
        return (int(h[:8], 16) / 0xFFFFFFFF) < ratio

    unique = {e: _in_holdout(str(e)) for e in emails.unique()}
    return emails.map(unique)


def save_transitions_with_split(
    transitions: List[Dict[str, Any]],
    output_path: Path,
    test_split_date: Optional[str] = None,
    holdout_devs_ratio: float = 0.0,
    holdout_seed: int = 42,
) -> pd.DataFrame:
    """transitions を DataFrame 化して保存する。

    - holdout_devs_ratio > 0 のとき: email ハッシュで選んだ dev の全行を
      `<stem>_devholdout<suffix>` に分離 (時間分割より先に除外)
    - test_split_date 指定時: 残りから feature_end >= 基準日の行を
      `<stem>_test<suffix>` に分離
    - それ以外を output_path (train) に保存

    Args:
        transitions: build_transitions() / build_event_transitions() の出力
        output_path: train 側の保存先 (.parquet または .json)
        test_split_date: test 分割の基準日 (YYYY-MM-DD)。None なら分割しない
        holdout_devs_ratio: dev 単位 holdout の割合 (0.0 で無効)
        holdout_seed: dev 選択ハッシュの seed

    Returns:
        train 側の DataFrame (統計ログ出力用)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(transitions)

    if holdout_devs_ratio > 0 and "email" in df.columns:
        is_holdout = _holdout_dev_mask(df["email"], holdout_devs_ratio, holdout_seed)
        n_hold = int(is_holdout.sum())
        if n_hold > 0:
            holdout_df = df[is_holdout].copy()
            df = df[~is_holdout].copy()
            holdout_path = output_path.with_name(
                output_path.stem + "_devholdout" + output_path.suffix
            )
            _save_df(holdout_df, holdout_path)
            logger.info(
                "dev holdout (ratio=%.2f, seed=%d): %d devs / %d transitions → %s",
                holdout_devs_ratio, holdout_seed,
                holdout_df["email"].nunique(), n_hold, holdout_path,
            )
        else:
            logger.warning(
                "holdout_devs_ratio=%.2f だが対象 dev が0のため devholdout 無効",
                holdout_devs_ratio,
            )

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
