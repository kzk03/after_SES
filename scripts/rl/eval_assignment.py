#!/usr/bin/env python3
"""
Phase 1 オフライン RL 評価スクリプト

粒度非依存 (月次 MDP / event 単位 MDP のどちらの transitions parquet でも動く):
  state/action/reward/step_label/r_theta/feature_end/email の列さえあれば評価できるため、
  build_rl_dataset.py (月次) と build_rl_dataset_event.py (event単位) の
  どちらの出力に対しても同じスクリプトをそのまま使う。
  "monthly_accept" という列名は月次版由来の名残だが、event 単位データでは
  「そのイベント自体の承諾有無」を表す値がそのまま入るだけで、処理は変わらない。

IRL eval (`scripts/analyze/eval/eval_mce_irl_path_prediction.py`) と互換のスキーマで
3ファイルを出力する:
  - summary_metrics.json : 手法別の集計指標 (clf_auc_roc, clf_auc_pr, clf_f1, ...)
  - pair_predictions.csv : (developer, feature_end, step_label, monthly_accept, 各スコア)
  - calibration_curve.csv: IsotonicRegression による校正後確率

評価対象:
  - 継続予測 (directory 単位): step_label (行動月の後の3ヶ月継続) ↔ score の AUC
  - 継続予測 (dev 単位)      : step_label_dev ↔ score の AUC (§3-1 の設計判断用比較軸)
  - 月次/event 承諾予測      : monthly_accept ↔ score の AUC
  - 長期価値推定 (RL のみ)   : V(s) ↔ 軌跡内残承諾数 Spearman ((email, directory) 単位)

リーク防止の規律: F1 閾値・Ensemble の β と V 正規化範囲はすべて train 上で
fit し、test には固定適用する (test 統計を評価時に使わない)。

比較手法:
  - random, r_theta_baseline (IRL raw), irl_pi (sigmoid), bc_pi, iql_pi,
    iql_q_advantage, iql_v_long_term, ensemble (IRL + V) — β grid search

使い方:
    python scripts/rl/eval_assignment.py \\
        --train-dataset outputs/rl/lstm_baseline_0-3m/transitions.parquet \\
        --test-dataset  outputs/rl/lstm_baseline_0-3m/transitions_test.parquet \\
        --iql-model     outputs/rl/lstm_baseline_0-3m/iql_agent.pt \\
        --bc-model      outputs/rl/lstm_baseline_0-3m/bc_policy.pt \\
        --obs-dim 33 \\
        --output-dir outputs/rl/lstm_baseline_0-3m/eval
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.RL.agent.iql import IQLAgent
from review_predictor.RL.data.loading import load_transitions_df, states_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 指標計算ヘルパー
# ─────────────────────────────────────────────

def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_f1_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5,
) -> Tuple[float, float, float]:
    from sklearn.metrics import f1_score, precision_score, recall_score
    y_pred = (y_score >= threshold).astype(int)
    try:
        return (
            float(f1_score(y_true, y_pred, zero_division=0)),
            float(precision_score(y_true, y_pred, zero_division=0)),
            float(recall_score(y_true, y_pred, zero_division=0)),
        )
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(x, y)
        return float(rho) if rho == rho else float("nan")
    except Exception:
        return float("nan")


def _minmax_fit(x: np.ndarray) -> Tuple[float, float]:
    """min-max 正規化の範囲 (lo, hi) を fit する (train 統計)。

    定数配列・非有限値を含む配列は fit 不能として (nan, nan) を返す。
    """
    if not np.isfinite(x).all() or x.max() <= x.min():
        return float("nan"), float("nan")
    return float(x.min()), float(x.max())


def _minmax_apply(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """fit 済みの範囲で min-max 正規化を適用する。

    test 側には必ず train で fit した範囲を渡すこと (test 統計の混入防止)。
    範囲が fit 不能だった場合はそのまま返し、範囲外の値は [0, 1] にクリップする。
    """
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return x
    return np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0)


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """train 上で F1 を最大化する閾値を返す (test に固定適用する)。

    旧実装 (スコアの自己 min-max + 0.5 閾値) は test 側の分布に依存し
    実質 AUC の言い換えになるため、train-fit 閾値の固定適用に置き換えた。
    """
    from sklearn.metrics import precision_recall_curve
    try:
        finite = np.isfinite(y_score)
        y_t, y_s = np.asarray(y_true)[finite], np.asarray(y_score)[finite]
        if len(y_s) == 0 or len(np.unique(y_t)) < 2:
            return float(np.nanmedian(y_score))
        prec, rec, thr = precision_recall_curve(y_t, y_s)
        if len(thr) == 0:
            return float(np.nanmedian(y_s))
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        return float(thr[int(np.nanargmax(f1))])
    except Exception:
        return float(np.nanmedian(y_score))


def classify_metrics(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """IRL eval 互換の分類指標を返す。

    Args:
        threshold: F1/precision/recall 用の判定閾値。train 上で
            _best_f1_threshold により fit した値を渡す。None のとき
            (train が無い場合のフォールバック) はスコア中央値を使う。
    """
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    auc_roc = _safe_auc(y_true, y_score)
    auc_pr = _safe_auc_pr(y_true, y_score)
    if threshold is None or not np.isfinite(threshold):
        threshold = float(np.nanmedian(y_score))
    f1, prec, rec = _safe_f1_at_threshold(y_true, y_score, threshold=threshold)
    return {
        "name": name,
        "clf_auc_roc": auc_roc,
        "clf_auc_pr": auc_pr,
        "clf_f1": f1,
        "clf_precision": prec,
        "clf_recall": rec,
        "clf_f1_threshold": float(threshold),
        "clf_n_pairs": int(len(y_true)),
        "clf_n_pos": n_pos,
        "clf_n_neg": n_neg,
        "score_mean": float(np.nanmean(y_score)),
        "score_std": float(np.nanstd(y_score)),
    }


def compute_remaining_acceptance(df: pd.DataFrame) -> np.ndarray:
    """各行について「同一軌跡内、現在含めて末尾までの a=1 件数」を返す。

    軌跡 (エピソード) は (dev, directory) 単位なので、directory 列があれば
    (email, directory) でグループ化する。email のみだと同一 dev の別
    directory 軌跡ブロックを跨いで累積してしまい、エピソード単位で学習した
    V(s) と比較する量として不正確になる (旧実装の既知バグ)。
    """
    remaining = np.zeros(len(df), dtype=np.float32)
    group_cols = [c for c in ("email", "directory") if c in df.columns]
    if not group_cols:
        cumacc = df["action"].astype(int).to_numpy()[::-1].cumsum()[::-1]
        return cumacc.astype(np.float32)
    for _, idx in df.groupby(group_cols).groups.items():
        idx_arr = np.array(sorted(idx))
        acc = df.loc[idx_arr, "action"].astype(int).to_numpy()
        remaining[idx_arr] = acc[::-1].cumsum()[::-1]
    return remaining


# ─────────────────────────────────────────────
# データ準備
# ─────────────────────────────────────────────

def load_transitions(path: Path, obs_dim: int) -> Dict[str, np.ndarray]:
    df = load_transitions_df(path)
    states = states_matrix(df["state"], obs_dim)
    actions = df["action"].astype(int).to_numpy()
    step_labels = (
        df["step_label"].astype(int).to_numpy()
        if "step_label" in df.columns
        else actions.copy()
    )
    r_theta = (
        df["r_theta"].astype(float).to_numpy()
        if "r_theta" in df.columns
        else df["reward"].astype(float).to_numpy()
    )
    # dev 単位継続ラベル (観測不能な行は NaN。旧データセットには列自体が無い)
    step_labels_dev = (
        pd.to_numeric(df["step_label_dev"], errors="coerce").to_numpy(dtype=np.float64)
        if "step_label_dev" in df.columns
        else np.full(len(df), np.nan)
    )
    remaining = compute_remaining_acceptance(df)
    return {
        "df": df,
        "states": states,
        "actions": actions,
        "step_labels": step_labels,
        "step_labels_dev": step_labels_dev,
        "r_theta": r_theta,
        "remaining": remaining,
    }


def _batched_forward(
    net, states: np.ndarray, device: str, batch_size: int = 8192,
) -> np.ndarray:
    """MLP をバッチ分割で forward する (event 級の大規模データでの OOM 防止)。"""
    outs = []
    dev = torch.device(device)
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            t = torch.tensor(states[i:i + batch_size], dtype=torch.float32, device=dev)
            outs.append(net(t).cpu().numpy())
    if not outs:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(outs)


def compute_scores(
    states: np.ndarray,
    bc_model_path: Optional[str],
    iql_model_path: Optional[str],
    device: str,
) -> Dict[str, np.ndarray]:
    """各手法のスコアを計算して dict で返す。"""
    scores: Dict[str, np.ndarray] = {}

    def _softmax1(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=-1, keepdims=True))[:, 1].astype(np.float32)

    if bc_model_path:
        from review_predictor.RL.agent.bc_policy import BCTrainer
        bc_policy = BCTrainer.load_policy(bc_model_path, device=device)
        bc_policy.eval()
        scores["bc_pi"] = _softmax1(_batched_forward(bc_policy, states, device))

    if iql_model_path:
        agent = IQLAgent.load(iql_model_path, device=device)
        agent.policy.eval()
        agent.v_net.eval()
        agent.q_net.eval()
        scores["iql_pi"] = _softmax1(_batched_forward(agent.policy, states, device))
        v_vals = _batched_forward(agent.v_net, states, device)
        q_vals = _batched_forward(agent.q_net, states, device)
        scores["iql_q_advantage"] = (q_vals[:, 1] - q_vals[:, 0]).astype(np.float32)
        scores["iql_v"] = v_vals.astype(np.float32)
    return scores


def compute_supervised_scores(
    train_states: np.ndarray,
    y_train: np.ndarray,
    test_states: np.ndarray,
    seed: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """教師ありベースライン (RF / Logistic Regression) を train で学習し、
    (train スコア, test スコア) の組を返す。

    「同じ特徴・同じ分割で教師あり分類器と比べて RL は何が良いのか」への
    直接回答となるベースライン。IRL 論文の比較手法 (RF) とも接続する。
    ターゲットは step_label (train 側)。fit は train のみ、test には適用のみ。
    """
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if len(np.unique(y_train)) < 2:
        logger.warning("train の step_label が単一クラスのため教師ありベースラインをスキップ")
        return out
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(train_states, y_train)
    out["rf_supervised"] = (
        rf.predict_proba(train_states)[:, 1].astype(np.float32),
        rf.predict_proba(test_states)[:, 1].astype(np.float32),
    )

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(train_states, y_train)
    out["logreg_supervised"] = (
        lr.predict_proba(train_states)[:, 1].astype(np.float32),
        lr.predict_proba(test_states)[:, 1].astype(np.float32),
    )
    logger.info("教師ありベースライン fit 完了: %s", ", ".join(out))
    return out


def fit_ensemble_beta(
    irl_pi_train: np.ndarray,
    v_train: np.ndarray,
    y_train: np.ndarray,
    betas: Tuple[float, ...] = (0.0, 0.2, 0.5, 0.7, 0.9, 1.0),
) -> Tuple[float, Tuple[float, float]]:
    """β と V の正規化範囲を train 上で fit する。

    Returns:
        (best_beta, v_range): v_range は train の V min/max。
        test への適用時も必ずこの範囲を使う (test 統計の混入防止)。
    """
    v_range = _minmax_fit(v_train)
    v_norm = _minmax_apply(v_train, *v_range)
    best_beta, best_auc = 0.5, -np.inf
    for beta in betas:
        combined = beta * irl_pi_train + (1 - beta) * v_norm
        auc = _safe_auc(y_train, combined)
        if np.isfinite(auc) and auc > best_auc:
            best_beta, best_auc = beta, auc
    logger.info("Ensemble grid search: best β = %.2f (train AUC=%.4f)", best_beta, best_auc)
    return float(best_beta), v_range


def apply_ensemble(
    irl_pi: np.ndarray, v: np.ndarray, beta: float, v_range: Tuple[float, float],
) -> np.ndarray:
    # V の正規化は train で fit した範囲を固定適用する
    return beta * irl_pi + (1 - beta) * _minmax_apply(v, *v_range)


def calibrate_isotonic(
    y_train: np.ndarray, score_train: np.ndarray, score_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IsotonicRegression による校正。

    学習が発散した手法 (iql_pi など) はスコアが NaN になりうる。その場合は
    校正だけをスキップして素のスコアを返す。ここで例外を投げると掃引スクリプト
    (set -e) が止まり、残りの λ 条件が巻き添えで実行されなくなるため。

    Returns:
        calibrated_test_scores, breakpoints_x, breakpoints_y
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return score_test, np.array([]), np.array([])

    score_train = np.asarray(score_train, dtype=float)
    score_test = np.asarray(score_test, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    train_ok = np.isfinite(score_train) & np.isfinite(y_train)
    if int(train_ok.sum()) < 2 or len(np.unique(y_train[train_ok])) < 2:
        logger.warning(
            "校正をスキップ: 有効な train スコアが不足 (finite=%d/%d, ラベル種類=%d)",
            int(train_ok.sum()), len(score_train),
            len(np.unique(y_train[train_ok])) if train_ok.any() else 0,
        )
        return score_test.astype(np.float32), np.array([]), np.array([])

    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(score_train[train_ok], y_train[train_ok])
    # test 側の NaN は変換せず NaN のまま返す (下流の指標計算は NaN 耐性がある)
    test_ok = np.isfinite(score_test)
    calibrated = np.full(len(score_test), np.nan, dtype=float)
    if test_ok.any():
        calibrated[test_ok] = ir.transform(score_test[test_ok])
    else:
        logger.warning("校正をスキップ: test スコアが全て非有限")
    breakpoints_x = ir.X_thresholds_ if hasattr(ir, "X_thresholds_") else np.array([])
    breakpoints_y = ir.y_thresholds_ if hasattr(ir, "y_thresholds_") else np.array([])
    return calibrated.astype(np.float32), breakpoints_x, breakpoints_y


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 RL prediction evaluation (IRL eval 互換)")
    p.add_argument("--train-dataset", required=True, help="train transitions (.parquet/.json)")
    p.add_argument("--test-dataset", help="test transitions (.parquet/.json); 未指定なら train で in-sample 評価")
    p.add_argument("--iql-model", help="IQL チェックポイント (.pt)")
    p.add_argument("--bc-model", help="BC policy チェックポイント (.pt)")
    p.add_argument("--obs-dim", type=int, required=True, help="観測次元")
    p.add_argument("--output-dir", required=True, help="評価結果出力ディレクトリ")
    p.add_argument("--device", default="cpu", help="torch デバイス")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ensemble-betas",
        type=str,
        default="0.0,0.2,0.5,0.7,0.9,1.0",
        help="Ensemble の β grid (comma-separated)",
    )
    p.add_argument(
        "--skip-supervised-baselines",
        action="store_true",
        help="教師ありベースライン (RF / LogReg) の学習・評価をスキップする",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # train データロード (Ensemble の β grid search + 校正の fit に使う)
    train_data = load_transitions(Path(args.train_dataset), args.obs_dim)
    logger.info(
        "train: N=%d, monthly_accept=%.2f%%, step_label=%.2f%%",
        len(train_data["states"]),
        train_data["actions"].mean() * 100,
        train_data["step_labels"].mean() * 100,
    )

    # test データロード (未指定なら in-sample)
    if args.test_dataset and Path(args.test_dataset).exists():
        test_data = load_transitions(Path(args.test_dataset), args.obs_dim)
        logger.info(
            "test:  N=%d, monthly_accept=%.2f%%, step_label=%.2f%%",
            len(test_data["states"]),
            test_data["actions"].mean() * 100,
            test_data["step_labels"].mean() * 100,
        )
        is_held_out = True
    else:
        logger.warning(
            "test_dataset 未指定または不在 — in-sample 評価にフォールバック"
        )
        test_data = train_data
        is_held_out = False

    # 各手法のスコア計算 (train と test の両方)
    train_scores = compute_scores(
        train_data["states"], args.bc_model, args.iql_model, args.device,
    )
    test_scores = compute_scores(
        test_data["states"], args.bc_model, args.iql_model, args.device,
    )

    # IRL π = sigmoid(R_θ) として近似 (実際には R_θ = logits[1]-logits[0])
    # softmax([0, R_θ])[1] = sigmoid(R_θ) と等価
    train_scores["random"] = np.random.RandomState(args.seed).uniform(size=len(train_data["states"]))
    test_scores["random"] = np.random.RandomState(args.seed).uniform(size=len(test_data["states"]))
    train_scores["r_theta_baseline"] = train_data["r_theta"]
    test_scores["r_theta_baseline"] = test_data["r_theta"]
    train_scores["irl_pi"] = 1.0 / (1.0 + np.exp(-train_data["r_theta"]))
    test_scores["irl_pi"] = 1.0 / (1.0 + np.exp(-test_data["r_theta"]))

    # 教師ありベースライン (train で fit → test に適用)
    if not args.skip_supervised_baselines:
        supervised = compute_supervised_scores(
            train_data["states"], train_data["step_labels"],
            test_data["states"], args.seed,
        )
        for name, (tr_sc, te_sc) in supervised.items():
            train_scores[name] = tr_sc
            test_scores[name] = te_sc

    # Ensemble (IRL + V) の β と V 正規化範囲を train 上で fit → test に固定適用
    if "iql_v" in train_scores:
        betas = tuple(float(b) for b in args.ensemble_betas.split(","))
        best_beta, v_range = fit_ensemble_beta(
            train_scores["irl_pi"],
            train_scores["iql_v"],
            train_data["step_labels"],
            betas=betas,
        )
        test_scores["ensemble_irl_v"] = apply_ensemble(
            test_scores["irl_pi"], test_scores["iql_v"], best_beta, v_range,
        )
        train_scores["ensemble_irl_v"] = apply_ensemble(
            train_scores["irl_pi"], train_scores["iql_v"], best_beta, v_range,
        )
    else:
        best_beta = None

    # ─────────────────────────────────────────────
    # summary_metrics.json (IRL eval 互換)
    # ─────────────────────────────────────────────
    summary: Dict[str, Dict[str, float]] = {}

    # F1 閾値は train 上で fit して test に固定適用する
    def _train_threshold(name: str, y_train: np.ndarray, valid: Optional[np.ndarray] = None) -> Optional[float]:
        if name not in train_scores:
            return None
        sc_train = train_scores[name]
        if valid is not None:
            y_train, sc_train = y_train[valid], sc_train[valid]
        if len(y_train) == 0:
            return None
        return _best_f1_threshold(y_train, sc_train)

    # 継続予測 (step_label, directory 単位) に対する各手法の指標
    for name, sc in test_scores.items():
        thr = _train_threshold(name, train_data["step_labels"])
        entry = classify_metrics(f"RL_{name}_step_label", test_data["step_labels"], sc, threshold=thr)
        summary[f"RL_{name}_step_label"] = entry

    # 継続予測 (step_label_dev, dev 単位) — 観測不能 (NaN) の行は除外
    dev_valid_test = np.isfinite(test_data["step_labels_dev"])
    if dev_valid_test.sum() >= 2 and len(np.unique(test_data["step_labels_dev"][dev_valid_test])) >= 2:
        dev_valid_train = np.isfinite(train_data["step_labels_dev"])
        y_dev_test = test_data["step_labels_dev"][dev_valid_test]
        for name, sc in test_scores.items():
            thr = _train_threshold(
                name, train_data["step_labels_dev"],
                valid=dev_valid_train if dev_valid_train.any() else None,
            )
            entry = classify_metrics(
                f"RL_{name}_step_label_dev", y_dev_test, sc[dev_valid_test], threshold=thr,
            )
            summary[f"RL_{name}_step_label_dev"] = entry

    # 月次承諾予測 (monthly_accept) に対する各手法の指標
    for name, sc in test_scores.items():
        thr = _train_threshold(name, train_data["actions"])
        entry = classify_metrics(f"RL_{name}_monthly", test_data["actions"], sc, threshold=thr)
        summary[f"RL_{name}_monthly"] = entry

    # V(s) の長期価値 (Spearman)
    if "iql_v" in test_scores:
        rho = _safe_spearman(test_scores["iql_v"], test_data["remaining"])
        summary["RL_iql_v_long_term"] = {
            "name": "RL_iql_v_long_term",
            "spearman_v_vs_remaining_accept": rho,
            "v_mean": float(test_scores["iql_v"].mean()),
            "v_std": float(test_scores["iql_v"].std()),
            "remaining_mean": float(test_data["remaining"].mean()),
            "clf_n_pairs": int(len(test_scores["iql_v"])),
        }

    # メタ情報
    summary["_meta"] = {
        "is_held_out": is_held_out,
        "train_n": int(len(train_data["states"])),
        "test_n": int(len(test_data["states"])),
        "ensemble_best_beta": best_beta,
        "seed": args.seed,
    }

    summary_path = output_dir / "summary_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("summary_metrics.json: %s", summary_path)

    # ─────────────────────────────────────────────
    # pair_predictions.csv
    # ─────────────────────────────────────────────
    pair_df = pd.DataFrame({
        "developer": test_data["df"].get("email", pd.Series([""] * len(test_data["df"]))),
        "directory": test_data["df"].get("directory", pd.Series([""] * len(test_data["df"]))),
        "feature_end": test_data["df"].get("feature_end", pd.Series([None] * len(test_data["df"]))),
        "step_label": test_data["step_labels"],
        "step_label_dev": test_data["step_labels_dev"],
        "monthly_accept": test_data["actions"],
        "remaining_accept": test_data["remaining"],
    })
    for name, sc in test_scores.items():
        pair_df[f"score_{name}"] = sc
    pair_path = output_dir / "pair_predictions.csv"
    pair_df.to_csv(pair_path, index=False)
    logger.info("pair_predictions.csv: %s (rows=%d)", pair_path, len(pair_df))

    # ─────────────────────────────────────────────
    # calibration_curve.csv (IsotonicRegression)
    # iql_pi と irl_pi を例として校正
    # ─────────────────────────────────────────────
    cal_records = []
    for target_name, scores_dict in (
        ("iql_pi", test_scores),
        ("irl_pi", test_scores),
        ("ensemble_irl_v", test_scores),
    ):
        if target_name not in scores_dict:
            continue
        cal_scores, bx, by = calibrate_isotonic(
            train_data["step_labels"].astype(float),
            train_scores[target_name].astype(float),
            scores_dict[target_name].astype(float),
        )
        # 校正後の AUC / Brier
        from sklearn.metrics import brier_score_loss
        try:
            brier_raw = float(
                brier_score_loss(test_data["step_labels"], scores_dict[target_name])
            )
            brier_cal = float(
                brier_score_loss(test_data["step_labels"], cal_scores)
            )
        except Exception:
            brier_raw, brier_cal = float("nan"), float("nan")
        summary[f"RL_{target_name}_calibrated_step_label"] = {
            "name": f"RL_{target_name}_calibrated_step_label",
            "brier_raw": brier_raw,
            "brier_calibrated": brier_cal,
            "clf_auc_roc": _safe_auc(test_data["step_labels"], cal_scores),
        }
        # 校正曲線 (x=raw, y=calibrated)
        for xv, yv in zip(bx, by):
            cal_records.append({"target": target_name, "raw_score": float(xv), "calibrated": float(yv)})

    if cal_records:
        cal_df = pd.DataFrame(cal_records)
        cal_path = output_dir / "calibration_curve.csv"
        cal_df.to_csv(cal_path, index=False)
        logger.info("calibration_curve.csv: %s", cal_path)

    # 校正情報を summary に上書き保存
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ─────────────────────────────────────────────
    # サマリ表示
    # ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("評価結果サマリ (test set, held_out=%s, N=%d)", is_held_out, len(test_data["states"]))
    logger.info("=" * 60)
    display_methods = [
        "random", "r_theta_baseline", "irl_pi", "rf_supervised", "logreg_supervised",
        "bc_pi", "iql_pi", "iql_q_advantage", "ensemble_irl_v",
    ]
    for axis, label in (
        ("step_label", "継続予測 (step_label, directory 単位)"),
        ("step_label_dev", "継続予測 (step_label_dev, dev 単位)"),
        ("monthly", "月次/event 承諾予測"),
    ):
        keys = [f"RL_{name}_{axis}" for name in display_methods]
        if not any(k in summary for k in keys):
            continue
        logger.info("[%s AUC-ROC]", label)
        for name, key in zip(display_methods, keys):
            if key in summary:
                logger.info("  %-25s %.4f", name, summary[key].get("clf_auc_roc", float("nan")))
    if "RL_iql_v_long_term" in summary:
        rho = summary["RL_iql_v_long_term"]["spearman_v_vs_remaining_accept"]
        logger.info("[長期価値] V(s) ↔ remaining_accept Spearman = %.4f", rho)
    if best_beta is not None:
        logger.info("[Ensemble] best β = %.2f", best_beta)


if __name__ == "__main__":
    main()
