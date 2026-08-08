"""継続スコア — IRL の R_θ を割り当て報酬に使える形へ変換する。

なぜ表引きなのか
----------------
シミュレータの中で R_θ を計算し直すことはできない。理由は 2 つある。

1. IRL の入力は 32 次元 (状態 26 + path 特徴 6) だが、path 特徴は co-change
   グラフから作るもので、シミュレータはグラフを持てない
2. IRL は月次系列で学習した LSTM なので、系列長 1 で呼ぶと出力が別物になる

実測 (test 2 万件) では、系列長 1 で呼ぶと正規の R_θ との順位相関が 0.2445、
さらに path 特徴をゼロ埋めすると 0.0585 まで落ち、値域も [-5.19, -4.18] に
潰れる。報酬に足せばノイズを足すのと同じになる。

そこで、オフラインで正しく計算済みの R_θ (event transitions parquet の
``r_theta`` 列) を ``(email, 年月)`` で引く。LSTM は一切呼ばない。

何を測れて何を測れないか
------------------------
値のキーが (開発者, 年月) なので、1 エピソード (30 日 ≒ 1 ヶ月) の中では
開発者ごとに定数である。したがって

* 「継続しそうな人を選ぶ」— 表現できる。行動が選ぶ人を変えれば値が変わる
* 「酷使して離脱させない」— 表現できない。エージェントの割り当てが値に反映されない

候補プール全員の平均で報酬に入れてはいけない。母集団もキーもエピソード中
固定なので平均が定数になり、勾配が完全に消える。

変換方式
--------
``rank``     訓練分布上の百分位 [0,1]。ラベルを使わないので「IRL の報酬を
             そのまま使った (スケール調整のみ)」と書ける。既定
``identity`` R_θ をそのまま返す。値域が無制限なので割引和のスケールが暴れる。
             アブレーション用
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 参照表に無い (開発者, 月) の埋め方
MISSING_POLICIES = ("median", "low", "zero")


class RetentionTable:
    """``(email, 年月) -> 継続スコア`` の参照表。

    Args:
        raw: MultiIndex (email, Period[M]) の R_θ 実測値
        transform: "rank" (百分位) か "identity" (生の R_θ)
        missing: 参照表に無いキーの扱い
            "median" 訓練分布の中央値 (rank なら 0.5)
            "low"    最小値 (rank なら 0.0)。実績が薄い＝継続しにくいとみなす
            "zero"   0.0。identity のとき「情報なし」を 0 で表したい場合
    """

    def __init__(
        self,
        raw: pd.Series,
        transform: str = "rank",
        missing: str = "median",
    ) -> None:
        if transform not in ("rank", "identity"):
            raise ValueError(f"transform は rank / identity のみ: {transform}")
        if missing not in MISSING_POLICIES:
            raise ValueError(f"missing は {MISSING_POLICIES} のみ: {missing}")

        self.transform = transform
        self.missing = missing

        vals = raw.to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            raise ValueError("参照表に有限の R_θ が 1 件も無い")
        self._ref = np.sort(finite)

        scores = self._to_score(vals)
        self._table: Dict[Tuple[str, pd.Period], float] = {
            (str(e), m): float(s) for (e, m), s in zip(raw.index, scores)
        }

        if missing == "median":
            self._default = float(np.median(scores))
        elif missing == "low":
            self._default = float(np.min(scores))
        else:
            self._default = 0.0

        self._n_hit = 0
        self._n_miss = 0

    # ------------------------------------------------------------------

    def _to_score(self, values: np.ndarray) -> np.ndarray:
        """R_θ をスコアへ単調変換する。順位は変えない。"""
        if self.transform == "identity":
            return values
        # 百分位: 参照分布の何割がこの値以下か
        idx = np.searchsorted(self._ref, values, side="right")
        return idx / float(len(self._ref))

    @classmethod
    def from_parquet(
        cls,
        paths: Sequence[str],
        transform: str = "rank",
        missing: str = "median",
    ) -> "RetentionTable":
        """event transitions parquet から参照表を組み立てる。

        同じ (email, 年月) に複数行あるのが普通なので平均に潰す。
        """
        frames = []
        for p in paths:
            df = pd.read_parquet(p, columns=["email", "feature_end", "r_theta"])
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)
        df["feature_end"] = pd.to_datetime(df["feature_end"])
        df["ym"] = df["feature_end"].dt.to_period("M")
        raw = df.groupby(["email", "ym"])["r_theta"].mean()
        logger.info(
            "継続スコア参照表: %d 行 → (開発者,月) %d 件 / 開発者 %d 人 / %s〜%s",
            len(df), len(raw), df["email"].nunique(),
            df["ym"].min(), df["ym"].max(),
        )
        return cls(raw, transform=transform, missing=missing)

    # ------------------------------------------------------------------

    def __call__(self, email: str, now: pd.Timestamp) -> float:
        """選んだ開発者のその月の継続スコア。"""
        key = (str(email), pd.Timestamp(now).to_period("M"))
        v = self._table.get(key)
        if v is None:
            self._n_miss += 1
            return self._default
        self._n_hit += 1
        return v

    def coverage(self) -> Dict[str, float]:
        """呼び出しのうち参照表に当たった割合。エピソード実行後に確認する。"""
        total = self._n_hit + self._n_miss
        return {
            "n_hit": float(self._n_hit),
            "n_miss": float(self._n_miss),
            "hit_rate": float(self._n_hit / total) if total else 0.0,
        }

    def describe(self) -> Dict[str, float]:
        """参照表そのものの分布。報酬に足す前に必ず確認する。

        ばらつきが小さいスコアを報酬に入れても定数を足すのと変わらない。
        """
        scores = np.array(list(self._table.values()), dtype=np.float64)
        return {
            "n_entries": float(len(scores)),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_min": float(scores.min()),
            "score_max": float(scores.max()),
            "raw_mean": float(self._ref.mean()),
            "raw_std": float(self._ref.std()),
            "default": self._default,
        }
