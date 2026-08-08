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
    p.add_argument(
        "--project-prior-days", type=int, default=90,
        help=(
            "プロジェクト事前分布を直近何日の実績から計算するか。0 で静的な"
            "(fit split 全体の) 値。プロジェクトの承諾率は年跨ぎでも年内でも"
            "動くため既定は 90 日"
        ),
    )
    p.add_argument(
        "--raw-csv", default=None,
        help=(
            "プロジェクト単位の事前承諾率を特徴量に足す場合に指定する生 CSV。"
            "開発者状態 26 次元にはプロジェクトを識別する情報が無いため、"
            "指定しないと neutron (実測 0.82) と keystone (0.14) を区別できない"
        ),
    )
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
    df["feature_end"] = pd.to_datetime(df["feature_end"])
    if subsample and subsample < len(df):
        df = df.sample(n=subsample, random_state=seed)
        logger.info("subsample: %d 件に絞る", len(df))
    # feature_end 昇順にして時系列で fit/cal を分けられるようにする
    df = df.sort_values("feature_end", kind="mergesort").reset_index(drop=True)
    X = stack_states(df["state"], dim)
    if use_first_dims:
        X = X[:, :use_first_dims]
    y = df["action"].astype(np.int8).to_numpy()
    meta = df[["feature_end", "email", "r_theta"]].copy()
    logger.info("%s: n=%d, dim=%d, 承諾率=%.4f", Path(path).name, len(y), X.shape[1], y.mean())
    return X, y, meta


def attach_project(meta: pd.DataFrame, raw_csv: str) -> pd.Series:
    """(email, feature_end) で生 CSV を突き合わせてプロジェクト名を得る。

    parquet にプロジェクト列が無いため生データから引く。時刻は parquet が
    datetime64[us]、生 CSV のパースは ns になるので秒精度の文字列で揃える。
    """
    raw = pd.read_csv(raw_csv, usecols=["email", "timestamp", "project"], low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"])
    raw["key"] = (
        raw["email"].astype(str) + "|" + raw["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    m = raw.drop_duplicates("key").set_index("key")["project"]
    key = (
        meta["email"].astype(str) + "|"
        + pd.to_datetime(meta["feature_end"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    proj = key.map(m)
    logger.info(
        "プロジェクト付与: %d / %d 行が一致 (%.1f%%)",
        int(proj.notna().sum()), len(proj), 100 * proj.notna().mean(),
    )
    return proj.fillna("")


def trailing_project_rate(
    proj: np.ndarray, when: np.ndarray, raw_csv: str, window_days: int
) -> np.ndarray:
    """各行の時点から遡った window_days のプロジェクト承諾率を返す。

    その時点で既に観測済みのイベントだけを使うためリークにならない。
    プロジェクトの承諾率は年をまたいでも年内でも動く (実測: neutron は
    2019-2021 の 0.6803 から 2022 に 0.7706、nova は 0.5732 から 0.5036、
    さらに nova は 2022 年内でも通年 0.5036 に対し後半 0.3244) ため、
    固定値では追従できない。
    """
    raw = pd.read_csv(raw_csv, usecols=["project", "timestamp", "label"], low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")

    out = np.full(len(proj), np.nan, dtype=np.float64)
    win = np.timedelta64(window_days, "D")
    when = pd.to_datetime(pd.Series(when)).to_numpy()

    for p, g in raw.groupby("project"):
        idx = np.flatnonzero(proj == p)
        if idx.size == 0:
            continue
        ts = g["timestamp"].to_numpy()
        lab = g["label"].fillna(0).to_numpy(dtype=np.float64)
        csum = np.concatenate([[0.0], np.cumsum(lab)])
        q = when[idx]
        hi = np.searchsorted(ts, q, side="left")        # 自分自身は含めない
        lo = np.searchsorted(ts, q - win, side="left")
        n = hi - lo
        s = csum[hi] - csum[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


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

    # ── プロジェクト事前分布 ─────────────────────────────────
    # fit split だけから計算する。cal を含めると較正が、test を含めると評価が
    # 甘くなる (答えを一部教えることになる)。
    project_prior: dict = {}
    prior_default = 0.0
    prior_days = args.project_prior_days
    if args.raw_csv:
        proj = attach_project(meta, args.raw_csv).to_numpy()
        prior_default = float(y_fit.mean())
        # 未知プロジェクトや履歴が足りない時点のフォールバックとして
        # 静的な値も残す (fit split だけから計算する)
        s = pd.DataFrame({"p": proj[:split], "y": y_fit})
        s = s[s["p"] != ""]
        g = s.groupby("p")["y"].agg(["mean", "size"])
        g = g[g["size"] >= 50]
        project_prior = {str(k): float(v) for k, v in g["mean"].items()}

        if prior_days > 0:
            col_all = trailing_project_rate(
                proj, meta["feature_end"].to_numpy(), args.raw_csv, prior_days
            )
            n_nan = int(np.isnan(col_all).sum())
            fb = np.array(
                [project_prior.get(str(p), prior_default) for p in proj], dtype=np.float64
            )
            col_all = np.where(np.isnan(col_all), fb, col_all)
            logger.info(
                "プロジェクト事前分布: 直近%d日の実績 (静的値へのフォールバック %d 行 = %.2f%%), "
                "範囲 %.4f〜%.4f",
                prior_days, n_nan, 100 * n_nan / len(col_all),
                float(col_all.min()), float(col_all.max()),
            )
        else:
            col_all = np.array(
                [project_prior.get(str(p), prior_default) for p in proj], dtype=np.float64
            )
            logger.info(
                "プロジェクト事前分布: 静的 %d プロジェクト (既定 %.4f), 範囲 %.4f〜%.4f",
                len(project_prior), prior_default,
                float(col_all.min()), float(col_all.max()),
            )

        X_fit = np.hstack([X_fit, col_all[:split].astype(np.float32).reshape(-1, 1)])
        X_cal = np.hstack([X_cal, col_all[split:].astype(np.float32).reshape(-1, 1)])
        feat_dim = X_fit.shape[1]

    # ── 学習 ─────────────────────────────────────────────────
    model = AcceptanceModel(
        obs_dim=feat_dim,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        seed=args.seed,
    )
    model.project_prior = project_prior
    model.project_prior_default = prior_default
    model.project_prior_days = prior_days if args.raw_csv else 0
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
        if args.raw_csv:
            proj_t = attach_project(meta_t, args.raw_csv).to_numpy()
            fb_t = np.array(
                [project_prior.get(str(p), prior_default) for p in proj_t], dtype=np.float64
            )
            if prior_days > 0:
                col_t = trailing_project_rate(
                    proj_t, meta_t["feature_end"].to_numpy(), args.raw_csv, prior_days
                )
                col_t = np.where(np.isnan(col_t), fb_t, col_t)
            else:
                col_t = fb_t
            Xt = np.hstack([Xt, col_t.astype(np.float32).reshape(-1, 1)])
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
