#!/usr/bin/env python3
"""
Phase 1 オフライン RL 評価スクリプト

IRL eval (`scripts/analyze/eval/eval_mce_irl_path_prediction.py`) と互換のスキーマで
3ファイルを出力する:
  - summary_metrics.json : 手法別の集計指標 (clf_auc_roc, clf_auc_pr, clf_f1, ...)
  - pair_predictions.csv : (developer, feature_end, step_label, monthly_accept, 各スコア)
  - calibration_curve.csv: IsotonicRegression による校正後確率

評価対象:
  - 継続予測 (IRL タスク)   : step_label (3ヶ月先継続) ↔ score の AUC
  - 月次承諾予測             : monthly_accept ↔ score の AUC
  - 長期価値推定 (RL のみ)   : V(s) ↔ 軌跡内残承諾数 Spearman

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


def classify_metrics(
    name: str, y_true: np.ndarray, y_score: np.ndarray,
) -> Dict[str, float]:
    """IRL eval 互換の分類指標を返す。"""
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    auc_roc = _safe_auc(y_true, y_score)
    auc_pr = _safe_auc_pr(y_true, y_score)
    # スコアを 0-1 に正規化して f1 threshold=0.5 で計算
    if np.isfinite(y_score).all() and y_score.max() > y_score.min():
        norm = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-12)
    else:
        norm = y_score
    f1, prec, rec = _safe_f1_at_threshold(y_true, norm, threshold=0.5)
    return {
        "name": name,
        "clf_auc_roc": auc_roc,
        "clf_auc_pr": auc_pr,
        "clf_f1": f1,
        "clf_precision": prec,
        "clf_recall": rec,
        "clf_n_pairs": int(len(y_true)),
        "clf_n_pos": n_pos,
        "clf_n_neg": n_neg,
        "score_mean": float(np.nanmean(y_score)),
        "score_std": float(np.nanstd(y_score)),
    }


def compute_remaining_acceptance(df: pd.DataFrame) -> np.ndarray:
    """各行について「同一 email の軌跡内、現在含めて末尾までの a=1 件数」を返す。"""
    remaining = np.zeros(len(df), dtype=np.float32)
    if "email" not in df.columns:
        cumacc = df["action"].astype(int).to_numpy()[::-1].cumsum()[::-1]
        return cumacc.astype(np.float32)
    for email, idx in df.groupby("email").groups.items():
        idx_arr = np.array(sorted(idx))
        acc = df.loc[idx_arr, "action"].astype(int).to_numpy()
        remaining[idx_arr] = acc[::-1].cumsum()[::-1]
    return remaining


# ─────────────────────────────────────────────
# データ準備
# ─────────────────────────────────────────────

def load_transitions(path: Path, obs_dim: int) -> Dict[str, np.ndarray]:
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_json(path)

    def _to_vec(val, dim: int) -> list:
        if isinstance(val, list):
            v = val[:dim]
            v += [0.0] * max(0, dim - len(v))
            return v
        return [float(val)] + [0.0] * (dim - 1)

    states = np.array(
        [_to_vec(r, obs_dim) for r in df["state"]], dtype=np.float32,
    )
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
    remaining = compute_remaining_acceptance(df)
    return {
        "df": df,
        "states": states,
        "actions": actions,
        "step_labels": step_labels,
        "r_theta": r_theta,
        "remaining": remaining,
    }


def compute_scores(
    states: np.ndarray,
    bc_model_path: Optional[str],
    iql_model_path: Optional[str],
    device: str,
) -> Dict[str, np.ndarray]:
    """各手法のスコアを計算して dict で返す。"""
    scores: Dict[str, np.ndarray] = {}
    states_t = torch.tensor(states, dtype=torch.float32, device=torch.device(device))

    if bc_model_path:
        from review_predictor.RL.agent.bc_policy import BCTrainer
        bc_policy = BCTrainer.load_policy(bc_model_path, device=device)
        bc_policy.eval()
        with torch.no_grad():
            logits = bc_policy(states_t)
            scores["bc_pi"] = (
                torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            )

    if iql_model_path:
        agent = IQLAgent.load(iql_model_path, device=device)
        agent.policy.eval()
        agent.v_net.eval()
        agent.q_net.eval()
        with torch.no_grad():
            pi_logits = agent.policy(states_t)
            scores["iql_pi"] = torch.softmax(pi_logits, dim=-1)[:, 1].cpu().numpy()
            v_vals = agent.v_net(states_t).cpu().numpy()
            q_vals = agent.q_net(states_t).cpu().numpy()
        scores["iql_q_advantage"] = (q_vals[:, 1] - q_vals[:, 0]).astype(np.float32)
        scores["iql_v"] = v_vals.astype(np.float32)
    return scores


def fit_ensemble_beta(
    irl_pi_train: np.ndarray,
    v_train: np.ndarray,
    y_train: np.ndarray,
    betas: Tuple[float, ...] = (0.0, 0.2, 0.5, 0.7, 0.9, 1.0),
) -> Tuple[float, np.ndarray]:
    """β を train 上で grid search し、AUC 最大の β を返す。"""
    # V(s) を 0-1 に正規化
    v_norm = (v_train - v_train.min()) / (v_train.max() - v_train.min() + 1e-12)
    best_beta, best_auc = 0.5, -np.inf
    for beta in betas:
        combined = beta * irl_pi_train + (1 - beta) * v_norm
        auc = _safe_auc(y_train, combined)
        if np.isfinite(auc) and auc > best_auc:
            best_beta, best_auc = beta, auc
    logger.info("Ensemble grid search: best β = %.2f (train AUC=%.4f)", best_beta, best_auc)
    return float(best_beta), np.array([best_auc])


def apply_ensemble(
    irl_pi: np.ndarray, v: np.ndarray, beta: float,
) -> np.ndarray:
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-12)
    return beta * irl_pi + (1 - beta) * v_norm


def calibrate_isotonic(
    y_train: np.ndarray, score_train: np.ndarray, score_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """IsotonicRegression による校正。

    Returns:
        calibrated_test_scores, breakpoints_x, breakpoints_y
    """
    try:
        from sklearn.isotonic import IsotonicRegression
    except ImportError:
        return score_test, np.array([]), np.array([])
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(score_train, y_train)
    calibrated = ir.transform(score_test)
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

    # Ensemble (IRL + V) の β を train 上で grid search → test に適用
    if "iql_v" in train_scores:
        betas = tuple(float(b) for b in args.ensemble_betas.split(","))
        best_beta, _ = fit_ensemble_beta(
            train_scores["irl_pi"],
            train_scores["iql_v"],
            train_data["step_labels"],
            betas=betas,
        )
        test_scores["ensemble_irl_v"] = apply_ensemble(
            test_scores["irl_pi"], test_scores["iql_v"], best_beta,
        )
        train_scores["ensemble_irl_v"] = apply_ensemble(
            train_scores["irl_pi"], train_scores["iql_v"], best_beta,
        )
    else:
        best_beta = None

    # ─────────────────────────────────────────────
    # summary_metrics.json (IRL eval 互換)
    # ─────────────────────────────────────────────
    summary: Dict[str, Dict[str, float]] = {}

    # 継続予測 (step_label) に対する各手法の指標
    for name, sc in test_scores.items():
        entry = classify_metrics(f"RL_{name}_step_label", test_data["step_labels"], sc)
        summary[f"RL_{name}_step_label"] = entry

    # 月次承諾予測 (monthly_accept) に対する各手法の指標
    for name, sc in test_scores.items():
        entry = classify_metrics(f"RL_{name}_monthly", test_data["actions"], sc)
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
        "feature_end": test_data["df"].get("feature_end", pd.Series([None] * len(test_data["df"]))),
        "step_label": test_data["step_labels"],
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
    logger.info("[継続予測 (step_label) AUC-ROC]")
    for name in ["random", "r_theta_baseline", "irl_pi", "bc_pi", "iql_pi", "iql_q_advantage", "ensemble_irl_v"]:
        key = f"RL_{name}_step_label"
        if key in summary:
            logger.info("  %-25s %.4f", name, summary[key].get("clf_auc_roc", float("nan")))
    logger.info("[月次承諾予測 (monthly) AUC-ROC]")
    for name in ["random", "r_theta_baseline", "irl_pi", "bc_pi", "iql_pi", "iql_q_advantage"]:
        key = f"RL_{name}_monthly"
        if key in summary:
            logger.info("  %-25s %.4f", name, summary[key].get("clf_auc_roc", float("nan")))
    if "RL_iql_v_long_term" in summary:
        rho = summary["RL_iql_v_long_term"]["spearman_v_vs_remaining_accept"]
        logger.info("[長期価値] V(s) ↔ remaining_accept Spearman = %.4f", rho)
    if best_beta is not None:
        logger.info("[Ensemble] best β = %.2f", best_beta)


if __name__ == "__main__":
    main()
