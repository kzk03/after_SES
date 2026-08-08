#!/usr/bin/env python3
"""柱2 用の承諾モデル P(accept | dev state) を event transitions から学習する。

環境 (シミュレータ) の応答モデルを作るスクリプト。IRL の π(a=1|s) は
「継続するか」であって「この依頼を承諾するか」ではないため、承諾は
event の accepted フラグ (transitions の `action` 列) から直接学習する。

データ分割:
  - fit  : train parquet を feature_end 昇順に並べた先頭 (1 - cal_frac)
  - cal  : train parquet の末尾 cal_frac (isotonic 較正専用、fit と重ねない)
  - test : test parquet (時系列で train より後、既にビルド時に分割済み)

比較対象 (ベースライン):
  - prior       : 訓練期間の承諾率を全件に出す定数予測
  - irl_sigmoid : sigmoid(R_θ) を承諾確率とみなしたもの
                  (継続モデルを承諾モデルとして流用した場合の性能を示す)

使い方:
    uv run python scripts/rl/train_acceptance_model.py \
        --train-dataset outputs/rl/sweep/lstm_baseline_0-3m_event/dataset/event_transitions.parquet \
        --test-dataset  outputs/rl/sweep/lstm_baseline_0-3m_event/dataset/event_transitions_test.parquet \
        --output-dir    outputs/rl/acceptance_model \
        --obs-dim 32
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.RL.env.acceptance_model import (  # noqa: E402
    AcceptanceModel,
    binary_metrics,
    reliability_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="承諾モデル P(accept|state) の学習")
    p.add_argument("--train-dataset", required=True, help="train transitions (.parquet)")
    p.add_argument("--test-dataset", help="test transitions (.parquet)")
    p.add_argument("--output-dir", required=True, help="出力ディレクトリ")
    p.add_argument("--obs-dim", type=int, default=32, help="state の次元")
    p.add_argument(
        "--use-first-dims", type=int, default=0,
        help=(
            "state の先頭 N 次元だけを使う (0 で全次元)。シミュレータは path 特徴 "
            "(co-change グラフ由来, 26-31 次元目) を計算できないため、"
            "26 を指定して STATE_FEATURES だけのモデルを作る用途。"
        ),
    )
    p.add_argument(
        "--cal-frac", type=float, default=0.15,
        help="train の末尾を較正用に取り分ける割合 (時系列で分ける)",
    )
    p.add_argument(
        "--subsample", type=int, default=0,
        help="動作確認用に train をこの件数までランダム抽出する (0 で全件)",
    )
    p.add_argument("--max-iter", type=int, default=300, help="ブースティング反復数")
    p.add_argument("--learning-rate", type=float, default=0.06)
    p.add_argument("--max-leaf-nodes", type=int, default=63)
    p.add_argument("--min-samples-leaf", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def stack_states(col: pd.Series, dim: int) -> np.ndarray:
    """state 列を [N, dim] float32 に変換する。

    parquet からは各セルが ndarray で返るため np.stack が使える。
    長さが揃わない (旧フォーマット) 場合のみ行単位のパディングに落とす。
    """
    values = col.to_numpy()
    try:
        arr = np.stack(values).astype(np.float32, copy=False)
    except ValueError:
        logger.warning("state の長さが不揃いのためパディングして整形する")
        arr = np.zeros((len(values), dim), dtype=np.float32)
        for i, v in enumerate(values):
            v = np.asarray(v, dtype=np.float32).ravel()[:dim]
            arr[i, : len(v)] = v
        return arr

    if arr.shape[1] < dim:
        pad = np.zeros((arr.shape[0], dim - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > dim:
        arr = arr[:, :dim]
    return arr


def load_split(
    path: str, dim: int, subsample: int = 0, seed: int = 42, use_first_dims: int = 0
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """parquet から (states, actions, メタ列) を読み出す。"""
    cols = ["state", "action", "feature_end", "email", "r_theta"]
    df = pd.read_parquet(path, columns=cols)
    if subsample and subsample < len(df):
        df = df.sample(n=subsample, random_state=seed)
        logger.info("subsample: %d 件に絞る", len(df))
    # feature_end 昇順にして時系列で fit/cal を分けられるようにする
    df = df.sort_values("feature_end", kind="mergesort").reset_index(drop=True)
    X = stack_states(df["state"], dim)
    if use_first_dims:
        X = X[:, :use_first_dims]
    y = df["action"].astype(np.int8).to_numpy()
    meta = df[["feature_end", "email", "r_theta"]]
    logger.info("%s: n=%d, dim=%d, 承諾率=%.4f", Path(path).name, len(y), X.shape[1], y.mean())
    return X, y, meta


def baseline_metrics(
    y: np.ndarray, r_theta: np.ndarray, prior: float
) -> dict:
    """ベースラインの指標。承諾モデルが本当に必要かを判定するために出す。"""
    out = {}

    const = np.full(len(y), prior, dtype=np.float64)
    m = binary_metrics(y, const)
    m["reliability"] = reliability_table(y, const)
    out["prior"] = m

    # sigmoid(R_θ): 継続モデルの出力を承諾確率として流用した場合
    sig = 1.0 / (1.0 + np.exp(-np.asarray(r_theta, dtype=np.float64)))
    m = binary_metrics(y, sig)
    m["reliability"] = reliability_table(y, sig)
    out["irl_sigmoid"] = m

    for name, mm in out.items():
        logger.info(
            "[baseline:%s] AUC=%.4f Brier=%.4f ECE=%.4f (平均予測=%.4f)",
            name, mm["auc_roc"], mm["brier"], mm["ece"], mm["mean_pred"],
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── データ読み込み ────────────────────────────────────────
    X, y, meta = load_split(
        args.train_dataset, args.obs_dim, args.subsample, args.seed, args.use_first_dims
    )
    feat_dim = X.shape[1]

    n_cal = int(len(y) * args.cal_frac)
    if n_cal < 1000:
        logger.warning("較正データが %d 件と少ない (cal_frac=%.3f)", n_cal, args.cal_frac)
    split = len(y) - n_cal
    X_fit, y_fit = X[:split], y[:split]
    X_cal, y_cal = X[split:], y[split:]
    logger.info(
        "分割: fit=%d (~%s) / cal=%d (%s~)",
        len(y_fit), meta["feature_end"].iloc[split - 1] if split else "-",
        len(y_cal), meta["feature_end"].iloc[split] if n_cal else "-",
    )

    # ── 学習 ─────────────────────────────────────────────────
    model = AcceptanceModel(
        obs_dim=feat_dim,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )
    model.fit(X_fit, y_fit, X_cal, y_cal)

    results: dict = {
        "config": vars(args),
        "train": {
            "n_fit": int(len(y_fit)),
            "n_cal": int(len(y_cal)),
            "accept_rate_fit": float(y_fit.mean()),
            "accept_rate_cal": float(y_cal.mean()),
        },
    }
    results["calibration_split"] = model.metrics.get("calibration_split")

    # cal split 上での最終性能 (較正後)
    model.evaluate(X_cal, y_cal, tag="cal")

    # ── テスト評価 ───────────────────────────────────────────
    prior = float(y_fit.mean())
    if args.test_dataset:
        Xt, yt, meta_t = load_split(
            args.test_dataset, args.obs_dim, 0, args.seed, args.use_first_dims
        )
        model.evaluate(Xt, yt, tag="test")
        results["baselines_test"] = baseline_metrics(
            yt, meta_t["r_theta"].to_numpy(), prior
        )
        results["test_period"] = {
            "from": str(meta_t["feature_end"].iloc[0]),
            "to": str(meta_t["feature_end"].iloc[-1]),
            "n_dev": int(meta_t["email"].nunique()),
        }

    results["metrics"] = model.metrics

    # ── 保存 ─────────────────────────────────────────────────
    model.save(out_dir / "acceptance_model.joblib")
    with open(out_dir / "acceptance_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info("結果を保存: %s", out_dir / "acceptance_metrics.json")


if __name__ == "__main__":
    main()
