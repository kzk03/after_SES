"""KD pretrain: RF_Dir (OOB 予測) を teacher にして MCE-IRL ネットワークの body を事前学習する.

RF を教師あり代表とし、各 (developer, directory) ペアでの RF の OOB 予測確率を
LSTM IRL ネットワークが再現するように pretrain する。出力 state_dict は
そのまま `train_mce_irl.py --init-from` で読み込める形式で保存される。

使い方:
    uv run python scripts/train/kd_pretrain.py \
        --reviews data/combined_raw_main32.csv \
        --raw-json data/raw_json/openstack__*.json \
        --train-start 2019-01-01 \
        --train-end 2022-01-01 \
        --future-window-start 0 \
        --future-window-end 3 \
        --trajectories-cache outputs/mce_irl_trajectory_cache/main32/mce_traj_0-3.pkl \
        --hub-scores experiments/dependency_analysis/results/hub_scores_main32.csv \
        --cochange-neighbors experiments/dependency_analysis/results/cochange_neighbors_main32.csv \
        --output outputs/main32_mce_kd_warmstart/kd_ckpts/train_0-3m
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch

RANDOM_SEED = 777
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.IRL.features.common_features import FEATURE_NAMES_WITH_PATH
from review_predictor.IRL.features.path_features import (
    PathFeatureExtractor,
    attach_dirs_to_df,
    load_change_dir_map_multi,
)
from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem
from review_predictor.IRL.model.rf_predictor import (
    extract_features_for_window_directory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_reviews(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "email" in df.columns and "reviewer_email" not in df.columns:
        df = df.rename(columns={"email": "reviewer_email"})
    if "timestamp" in df.columns and "request_time" not in df.columns:
        df = df.rename(columns={"timestamp": "request_time"})
    df["request_time"] = pd.to_datetime(df["request_time"])
    logger.info("レビューデータ %d 件を読み込み", len(df))
    return df


def build_rf_kd_targets(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    future_window_start: int,
    future_window_end: int,
    window_days: int,
    path_extractor: PathFeatureExtractor,
    n_estimators: int = 200,
) -> Tuple[Dict[Tuple[str, str], float], dict]:
    """RF_Dir を OOB で学習し、(email, directory) → oob_prob のマップを返す.

    Returns:
        kd_targets, stats
    """
    from sklearn.ensemble import RandomForestClassifier

    rf_train_end = train_end
    rf_train_start = rf_train_end - pd.Timedelta(days=window_days)
    rf_label_start = rf_train_end + pd.DateOffset(months=future_window_start)
    rf_label_end = rf_train_end + pd.DateOffset(months=future_window_end)
    logger.info(
        "RF window: train [%s, %s) label [%s, %s)",
        rf_train_start.date(),
        rf_train_end.date(),
        rf_label_start.date(),
        rf_label_end.date(),
    )

    df_for_path = df.rename(
        columns={"reviewer_email": "email", "request_time": "timestamp"}
    )
    feature_df = extract_features_for_window_directory(
        df_for_path,
        rf_train_start,
        rf_train_end,
        rf_label_start,
        rf_label_end,
        path_extractor=path_extractor,
    )
    n_rows = len(feature_df)
    if n_rows < 10:
        raise ValueError(f"RF 特徴量 DF 行数 {n_rows} < 10、KD pretrain 不能")
    if feature_df["label"].nunique() < 2:
        raise ValueError("RF 特徴量 DF にクラスが 1 つしかない")

    X = feature_df[FEATURE_NAMES_WITH_PATH].to_numpy(dtype=float)
    y = feature_df["label"].to_numpy(dtype=int)

    logger.info(
        "RF を学習 (n=%d, n_pos=%d, n_estimators=%d, oob=True)",
        n_rows,
        int(y.sum()),
        n_estimators,
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    clf.fit(X, y)
    oob_prob = clf.oob_decision_function_[:, 1]
    # ごく稀に OOB が NaN になる (どの tree からも除外されなかった行) ので
    # 学習データ予測でフォールバック
    nan_mask = np.isnan(oob_prob)
    if nan_mask.any():
        logger.warning(
            "OOB が NaN の行が %d 件あるため、in-bag predict_proba でフォールバック",
            int(nan_mask.sum()),
        )
        fallback = clf.predict_proba(X[nan_mask])[:, 1]
        oob_prob = oob_prob.copy()
        oob_prob[nan_mask] = fallback

    kd_targets: Dict[Tuple[str, str], float] = {}
    emails = feature_df["email"].to_numpy()
    dirs = feature_df["directory"].to_numpy()
    for i in range(n_rows):
        kd_targets[(emails[i], dirs[i])] = float(oob_prob[i])

    try:
        from sklearn.metrics import roc_auc_score

        oob_auc = float(roc_auc_score(y, oob_prob))
    except Exception:
        oob_auc = float("nan")
    stats = {
        "n_rows": int(n_rows),
        "n_positive": int(y.sum()),
        "oob_auc": oob_auc,
        "rf_train_start": str(rf_train_start.date()),
        "rf_train_end": str(rf_train_end.date()),
        "rf_label_start": str(rf_label_start.date()),
        "rf_label_end": str(rf_label_end.date()),
        "n_estimators": n_estimators,
        "window_days": window_days,
    }
    logger.info("RF OOB AUC = %.4f", oob_auc)
    return kd_targets, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviews", type=str, required=True)
    parser.add_argument("--raw-json", type=str, nargs="+", required=True)
    parser.add_argument("--train-start", type=str, required=True)
    parser.add_argument("--train-end", type=str, required=True)
    parser.add_argument("--future-window-start", type=int, required=True)
    parser.add_argument("--future-window-end", type=int, required=True)
    parser.add_argument("--trajectories-cache", type=str, required=True)
    parser.add_argument("--hub-scores", type=str, default=None)
    parser.add_argument("--cochange-neighbors", type=str, default=None)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--model-type", type=int, default=0, choices=[0, 1, 2, 3])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--rf-n-estimators", type=int, default=200)
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="出力ディレクトリ。irl_model.pt と model_metadata.json を書く。",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_end = pd.Timestamp(args.train_end)

    # ── 1. レビューデータ + ディレクトリマッピング + path_extractor ──
    df = load_reviews(args.reviews)
    cdm = load_change_dir_map_multi(args.raw_json, depth=2)
    df = attach_dirs_to_df(df, cdm, column="dirs")
    df_for_path = df.rename(
        columns={"reviewer_email": "email", "request_time": "timestamp"}
    )
    path_extractor = PathFeatureExtractor(
        df_for_path,
        window_days=args.window_days,
        hub_scores_path=args.hub_scores,
        cochange_neighbors_path=args.cochange_neighbors,
    )

    # ── 2. RF OOB 予測で KD ターゲット辞書を構築 ──
    kd_targets, rf_stats = build_rf_kd_targets(
        df,
        train_end=train_end,
        future_window_start=args.future_window_start,
        future_window_end=args.future_window_end,
        window_days=args.window_days,
        path_extractor=path_extractor,
        n_estimators=args.rf_n_estimators,
    )
    logger.info("KD targets: %d 件", len(kd_targets))

    # ── 3. 軌跡キャッシュを読み込み ──
    cache_path = Path(args.trajectories_cache)
    if not cache_path.exists():
        raise FileNotFoundError(f"軌跡キャッシュが見つからない: {cache_path}")
    logger.info("軌跡キャッシュを読み込み: %s", cache_path)
    with open(cache_path, "rb") as f:
        trajectories = pickle.load(f)
    if not trajectories:
        raise ValueError("軌跡キャッシュが空")
    sample = trajectories[0]
    if "step_actions" not in sample:
        raise ValueError(
            f"軌跡キャッシュ {cache_path} に step_actions が無い。"
            f" extract_mce_trajectories.py で生成したものを指定してください。"
        )

    # state_dim を推定
    has_path = bool(sample.get("path_features_per_step"))
    from review_predictor.IRL.features.common_features import (
        STATE_FEATURES,
        STATE_FEATURES_WITH_PATH,
    )

    state_dim = len(STATE_FEATURES_WITH_PATH) if has_path else len(STATE_FEATURES)
    logger.info("state_dim=%d (path 込み=%s)", state_dim, has_path)

    # ── 4. MCEIRLSystem を構築 ──
    config = {
        "state_dim": state_dim,
        "action_dim": 5,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "model_type": args.model_type,
    }
    system = MCEIRLSystem(config)

    # ── 5. KD pretrain ──
    stats = system.train_kd_pretrain(
        trajectories,
        kd_targets,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    # ── 6. state_dict と metadata を保存 ──
    model_path = output_dir / "irl_model.pt"
    torch.save(system.network.state_dict(), model_path)
    logger.info("KD pretrain ckpt 保存: %s", model_path)

    metadata = {
        "model_class": "mce_irl_kd_pretrain",
        "model_type": args.model_type,
        "state_dim": state_dim,
        "action_dim": 5,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "num_actions": 2,
        "loss": "bce_softmax_a1_vs_rf_oob",
        "kd_teacher": "rf_dir_oob",
        "rf_stats": rf_stats,
        "kd_stats": stats,
        "train_end": str(train_end.date()),
        "future_window_start_months": args.future_window_start,
        "future_window_end_months": args.future_window_end,
        "path_dim": int(getattr(system, "path_dim", 0)),
    }
    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    logger.info("metadata 保存: %s", metadata_path)


if __name__ == "__main__":
    main()
