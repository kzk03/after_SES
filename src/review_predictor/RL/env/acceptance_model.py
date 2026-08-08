"""承諾モデル P(accept | dev state)

柱2 (レビュアー割り当て RL) のシミュレータが使う**環境側の応答モデル**。
「この開発者にこの依頼を振ったら受けてくれるか」を確率で返す。

位置づけ (2026-08-03 の設計変更):
  IRL で推定した R_θ は「開発者の効用」であってエージェント (推薦システム) の
  報酬ではない。開発者は環境の一部なので、R_θ 由来の量は報酬ではなく
  **環境の応答モデル**として使う。本モジュールはその応答モデルを、IRL とは別に
  event 単位の承諾実績から直接学習する。

なぜ IRL の π をそのまま使わないか:
  IRL は step_labels (0-3ヶ月の継続ラベル) を二値 action として学習しており、
  π(a=1|s) は「この人は継続するか」であって「この依頼を承諾するか」ではない。
  別の事象なので、承諾は承諾実績 (event の accepted フラグ) から学習する。

較正 (calibration) について:
  シミュレータはこの確率でベルヌーイ試行を回すため、順位付け性能 (AUC) より
  **確率が実測承諾率と一致しているか**が重要。予測 0.8 の群が実際に 8 割承諾して
  いないと、シミュレータの承諾/拒否の頻度そのものが歪む。そのため学習後に
  isotonic 回帰で較正し、ECE / Brier で検証する。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 評価指標
# ─────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20
) -> float:
    """ECE (Expected Calibration Error) を等幅ビンで計算する。

    各ビンで |平均予測確率 - 実測陽性率| をサンプル数で重み付き平均したもの。
    0 に近いほど「予測確率がそのまま実測頻度になっている」ことを意味する。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # 右端 1.0 を最終ビンに含めるため clip する
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        ece += (n / len(y_true)) * abs(y_prob[m].mean() - y_true[m].mean())
    return float(ece)


def reliability_table(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list[Dict[str, float]]:
    """信頼性図の元になるビン別集計を返す (予測確率 vs 実測承諾率)。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, n_bins - 1)

    rows: list[Dict[str, float]] = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        rows.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "n": n,
            "mean_pred": float(y_prob[m].mean()) if n else float("nan"),
            "actual_rate": float(y_true[m].mean()) if n else float("nan"),
        })
    return rows


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """承諾モデルの評価指標一式。

    AUC/PR-AUC は順位付け性能、Brier/ECE は確率としての正しさを見る。
    シミュレータ用途では後者の方が重要。
    """
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    out: Dict[str, float] = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "mean_pred": float(y_prob.mean()) if len(y_prob) else float("nan"),
    }
    # 単一クラスだと AUC が定義できないため個別にガードする
    if len(np.unique(y_true)) < 2:
        out.update({"auc_roc": float("nan"), "auc_pr": float("nan")})
    else:
        out["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        out["auc_pr"] = float(average_precision_score(y_true, y_prob))
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["ece"] = expected_calibration_error(y_true, y_prob)
    return out


# ─────────────────────────────────────────────
# モデル本体
# ─────────────────────────────────────────────

@dataclass
class AcceptanceModel:
    """P(accept | state) を返す較正済み分類器。

    使い方:
        model = AcceptanceModel(obs_dim=32)
        model.fit(X_fit, y_fit, X_cal, y_cal)
        p = model.predict_proba(X_test)          # [N] の確率
        accepted = model.sample(X_test, rng)     # ベルヌーイ試行
    """

    obs_dim: int
    max_iter: int = 300
    learning_rate: float = 0.06
    max_leaf_nodes: int = 63
    min_samples_leaf: int = 200
    l2_regularization: float = 1.0
    early_stopping: bool = True
    seed: int = 42

    clf: Any = field(default=None, repr=False)
    calibrator: Any = field(default=None, repr=False)
    metrics: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------

    def _new_clf(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        # 400万行 x 32次元を CPU で回すため勾配ブースティング (ヒストグラム版) を使う。
        # 同規模で MLP を回すより速く、特徴量スケールにも不感。
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            l2_regularization=self.l2_regularization,
            early_stopping=self.early_stopping,
            validation_fraction=0.1 if self.early_stopping else None,
            random_state=self.seed,
        )

    def fit(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ) -> "AcceptanceModel":
        """分類器を学習し、別スライスで isotonic 較正を当てる。

        X_cal を渡さない場合は較正なし (生の予測確率をそのまま使う)。
        較正データは学習データと重ねてはいけない (自分の訓練誤差で較正すると
        過信を補正できない) ので、呼び出し側が時系列で分けて渡すこと。
        """
        self.clf = self._new_clf()
        logger.info(
            "承諾モデルを学習: n=%d, dim=%d, 承諾率=%.4f",
            len(X_fit), X_fit.shape[1], float(np.mean(y_fit)),
        )
        self.clf.fit(X_fit, y_fit)
        logger.info("学習完了 (n_iter=%d)", getattr(self.clf, "n_iter_", -1))

        if X_cal is not None and y_cal is not None and len(X_cal) > 0:
            from sklearn.isotonic import IsotonicRegression

            raw = self.clf.predict_proba(X_cal)[:, 1]
            before = binary_metrics(y_cal, raw)
            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self.calibrator.fit(raw, y_cal)
            after = binary_metrics(y_cal, self.calibrator.predict(raw))
            logger.info(
                "較正 (cal split): ECE %.4f -> %.4f, Brier %.4f -> %.4f",
                before["ece"], after["ece"], before["brier"], after["brier"],
            )
            self.metrics["calibration_split"] = {"before": before, "after": after}
        else:
            logger.warning("較正データが無いため未較正のまま使う")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """承諾確率 [N] を返す (較正済みなら較正後の値)。"""
        if self.clf is None:
            raise RuntimeError("fit() または load() を先に呼んでください")
        p = self.clf.predict_proba(np.asarray(X, dtype=np.float32))[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)

    def sample(
        self, X: np.ndarray, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """承諾確率でベルヌーイ試行し 0/1 を返す (シミュレータ用)。"""
        rng = rng or np.random.default_rng()
        p = self.predict_proba(X)
        return (rng.random(len(p)) < p).astype(np.int8)

    def evaluate(self, X: np.ndarray, y: np.ndarray, tag: str = "test") -> Dict[str, Any]:
        """指標一式と信頼性テーブルを計算して self.metrics に記録する。"""
        p = self.predict_proba(X)
        m = binary_metrics(y, p)
        m["reliability"] = reliability_table(y, p)
        self.metrics[tag] = m
        logger.info(
            "[%s] AUC=%.4f PR-AUC=%.4f Brier=%.4f ECE=%.4f (n=%d, 実測承諾率=%.4f, 平均予測=%.4f)",
            tag, m["auc_roc"], m["auc_pr"], m["brier"], m["ece"],
            m["n"], m["positive_rate"], m["mean_pred"],
        )
        return m

    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "clf": self.clf,
                "calibrator": self.calibrator,
                "obs_dim": self.obs_dim,
                "metrics": self.metrics,
                "params": {
                    "max_iter": self.max_iter,
                    "learning_rate": self.learning_rate,
                    "max_leaf_nodes": self.max_leaf_nodes,
                    "min_samples_leaf": self.min_samples_leaf,
                    "l2_regularization": self.l2_regularization,
                    "seed": self.seed,
                },
            },
            path,
        )
        logger.info("承諾モデルを保存: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "AcceptanceModel":
        import joblib

        obj = joblib.load(Path(path))
        params = obj.get("params", {})
        model = cls(obs_dim=obj["obs_dim"], **params)
        model.clf = obj["clf"]
        model.calibrator = obj.get("calibrator")
        model.metrics = obj.get("metrics", {})
        return model
