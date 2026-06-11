"""IRL_Dir (cold MCE-IRL) と RF_Dir の予測集合の相補性を定量化する.

入力: outputs/main32_mce_v2_phase2/lstm_baseline/train_*/eval_*/pair_predictions.csv
出力: outputs/analysis/irl_rf_complementarity/
  - cells_pair_level.csv     セル別 pair-level metrics
  - cells_dev_level.csv      セル別 dev-level metrics (dev ごとに max prob 集約)
  - summary.txt              人間可読サマリ
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def _topk_hit(y: np.ndarray, s: np.ndarray, k: int) -> np.ndarray:
    """スコア上位 k に入っていれば 1, それ以外 0 の bool 配列を返す."""
    if k <= 0:
        return np.zeros_like(y, dtype=bool)
    # 同順位の扱い: argsort は安定でないので, k 個ちょうど取れる threshold で切る
    idx = np.argsort(-s, kind="stable")[:k]
    hit = np.zeros(len(s), dtype=bool)
    hit[idx] = True
    return hit


def _mcnemar_p(b: int, c: int) -> float:
    """McNemar 検定 (連続補正なし, 二項分布の両側 p)."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    p = 2.0 * stats.binom.cdf(k, n, 0.5)
    return float(min(p, 1.0))


def _cell_metrics(df: pd.DataFrame, score_irl: str, score_rf: str, prefix: str) -> dict:
    y = df["label"].to_numpy().astype(int)
    s_irl = df[score_irl].to_numpy().astype(float)
    s_rf = df[score_rf].to_numpy().astype(float)
    n = len(y)
    n_pos = int(y.sum())
    k = n_pos

    out = {
        f"{prefix}_n": n,
        f"{prefix}_n_pos": n_pos,
        f"{prefix}_auc_irl": _safe_auc(y, s_irl),
        f"{prefix}_auc_rf": _safe_auc(y, s_rf),
        f"{prefix}_ap_irl": float(average_precision_score(y, s_irl)) if n_pos > 0 else float("nan"),
        f"{prefix}_ap_rf": float(average_precision_score(y, s_rf)) if n_pos > 0 else float("nan"),
        f"{prefix}_brier_irl": float(brier_score_loss(y, np.clip(s_irl, 0, 1))),
        f"{prefix}_brier_rf": float(brier_score_loss(y, np.clip(s_rf, 0, 1))),
        f"{prefix}_spearman": float(stats.spearmanr(s_irl, s_rf).statistic) if n > 1 else float("nan"),
    }

    hit_irl = _topk_hit(y, s_irl, k)
    hit_rf = _topk_hit(y, s_rf, k)
    tp_irl = int((hit_irl & (y == 1)).sum())
    tp_rf = int((hit_rf & (y == 1)).sum())
    both_tp = int((hit_irl & hit_rf & (y == 1)).sum())
    only_irl_tp = int((hit_irl & ~hit_rf & (y == 1)).sum())
    only_rf_tp = int((~hit_irl & hit_rf & (y == 1)).sum())

    out.update(
        {
            f"{prefix}_topk": k,
            f"{prefix}_tp_irl": tp_irl,
            f"{prefix}_tp_rf": tp_rf,
            f"{prefix}_tp_both": both_tp,
            f"{prefix}_tp_only_irl": only_irl_tp,
            f"{prefix}_tp_only_rf": only_rf_tp,
            f"{prefix}_topk_jaccard_pos": (
                both_tp / (tp_irl + tp_rf - both_tp) if (tp_irl + tp_rf - both_tp) > 0 else float("nan")
            ),
        }
    )

    # McNemar: 正解 (top-K hit かつ y=1) を「当たり」と定義し,
    # IRL only 当たり / RF only 当たり の差が偶然かを検定
    b = int(((hit_irl & (y == 1)) & ~(hit_rf & (y == 1))).sum())  # IRL only
    c = int((~(hit_irl & (y == 1)) & (hit_rf & (y == 1))).sum())  # RF only
    out[f"{prefix}_mcnemar_b_irl_only"] = b
    out[f"{prefix}_mcnemar_c_rf_only"] = c
    out[f"{prefix}_mcnemar_p"] = _mcnemar_p(b, c)

    return out


def _aggregate_to_dev(df: pd.DataFrame) -> pd.DataFrame:
    """(dev, dir) → dev に集約. score は max, label は any (1 つでも正例なら 1)."""
    g = df.groupby("developer", as_index=False).agg(
        label=("label", "max"),
        irl_dir_prob_calibrated=("irl_dir_prob_calibrated", "max"),
        rf_dir_prob=("rf_dir_prob", "max"),
    )
    return g


def _parse_cell_name(path: Path) -> tuple[str, str]:
    eval_dir = path.parent.name  # eval_X-Ym
    train_dir = path.parent.parent.name  # train_X-Ym
    train = train_dir.replace("train_", "").replace("m", "")
    eval_ = eval_dir.replace("eval_", "").replace("m", "")
    return train, eval_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-root",
        type=Path,
        default=Path("outputs/main32_mce_v2_phase2/lstm_baseline"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis/irl_rf_complementarity"),
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pair_files = sorted(args.input_root.glob("train_*/eval_*/pair_predictions.csv"))
    if not pair_files:
        raise FileNotFoundError(f"no pair_predictions.csv under {args.input_root}")

    pair_rows: list[dict] = []
    dev_rows: list[dict] = []

    for path in pair_files:
        train, eval_ = _parse_cell_name(path)
        df = pd.read_csv(path)
        needed = {"developer", "directory", "label", "irl_dir_prob_calibrated", "rf_dir_prob"}
        missing = needed - set(df.columns)
        if missing:
            print(f"[skip] {path}: missing {missing}")
            continue

        # pair-level
        row = {"train": train, "eval": eval_}
        row.update(_cell_metrics(df, "irl_dir_prob_calibrated", "rf_dir_prob", prefix="pair"))
        pair_rows.append(row)

        # dev-level
        dev_df = _aggregate_to_dev(df)
        row2 = {"train": train, "eval": eval_}
        row2.update(_cell_metrics(dev_df, "irl_dir_prob_calibrated", "rf_dir_prob", prefix="dev"))
        dev_rows.append(row2)

    pair_csv = args.output_dir / "cells_pair_level.csv"
    dev_csv = args.output_dir / "cells_dev_level.csv"
    pd.DataFrame(pair_rows).to_csv(pair_csv, index=False)
    pd.DataFrame(dev_rows).to_csv(dev_csv, index=False)

    summary_lines = []
    summary_lines.append(f"# IRL_Dir (cold MCE-IRL) vs RF_Dir complementarity")
    summary_lines.append(f"# input: {args.input_root}")
    summary_lines.append(f"# cells: {len(pair_rows)}")
    summary_lines.append("")
    summary_lines.append("## Pair-level (per (dev, dir) row)")
    summary_lines.append(
        f"{'train':>6} {'eval':>6} {'n':>7} {'pos':>5} {'AUC_irl':>8} {'AUC_rf':>7} "
        f"{'ρ':>6} {'both':>5} {'oIRL':>5} {'oRF':>5} {'mcP':>7}"
    )
    for r in pair_rows:
        summary_lines.append(
            f"{r['train']:>6} {r['eval']:>6} {r['pair_n']:>7} {r['pair_n_pos']:>5} "
            f"{r['pair_auc_irl']:>8.3f} {r['pair_auc_rf']:>7.3f} "
            f"{r['pair_spearman']:>6.3f} {r['pair_tp_both']:>5} "
            f"{r['pair_tp_only_irl']:>5} {r['pair_tp_only_rf']:>5} "
            f"{r['pair_mcnemar_p']:>7.1e}"
        )
    summary_lines.append("")
    summary_lines.append("## Dev-level (max score per developer)")
    summary_lines.append(
        f"{'train':>6} {'eval':>6} {'n':>6} {'pos':>4} {'AUC_irl':>8} {'AUC_rf':>7} "
        f"{'ρ':>6} {'both':>5} {'oIRL':>5} {'oRF':>5} {'mcP':>7}"
    )
    for r in dev_rows:
        summary_lines.append(
            f"{r['train']:>6} {r['eval']:>6} {r['dev_n']:>6} {r['dev_n_pos']:>4} "
            f"{r['dev_auc_irl']:>8.3f} {r['dev_auc_rf']:>7.3f} "
            f"{r['dev_spearman']:>6.3f} {r['dev_tp_both']:>5} "
            f"{r['dev_tp_only_irl']:>5} {r['dev_tp_only_rf']:>5} "
            f"{r['dev_mcnemar_p']:>7.1e}"
        )

    summary_path = args.output_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[ok] wrote {pair_csv}")
    print(f"[ok] wrote {dev_csv}")
    print(f"[ok] wrote {summary_path}")
    print()
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
