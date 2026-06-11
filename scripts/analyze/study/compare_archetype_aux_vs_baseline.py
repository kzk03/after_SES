"""baseline (model_type=0) vs archetype_aux (model_type=7) の AUC 比較レポート。

入力:
    --base OUTBASE/<baseline_vname>/train_<w>m/eval_<w>m/summary_metrics.json (4 win)
    --aux-roots OUTBASE/<aux_vname1>/... OUTBASE/<aux_vname2>/... (複数の λ 別 root)

ロジック:
    1. baseline と各 aux variant から IRL_Dir_calibrated.clf_auc_roc を 4 horizon 抽出
    2. pair_predictions.csv も読込み、archetype α CSV と join して
       archetype 別 pair-level AUC を計算 (option)
    3. ペア単位 paired bootstrap で Δ AUC の 95% CI を計算
    4. 出力: overall_compare.csv, per_archetype_compare.csv, fig_compare_auc.pdf

使い方:
    uv run python scripts/analyze/study/compare_archetype_aux_vs_baseline.py \\
        --base outputs/archetype_aux_compare/legacy_baseline \\
        --aux-roots outputs/archetype_aux_compare/legacy_aux_l0.1 \\
                    outputs/archetype_aux_compare/legacy_aux_l0.3 \\
                    outputs/archetype_aux_compare/legacy_aux_l1.0 \\
        --archetype-csv outputs/analysis/archetype_irl_rf_coldonly/k4_dev_alpha_with_metrics.csv \\
        --out outputs/analysis/archetype_aux_compare
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


HORIZONS = ["0-3", "3-6", "6-9", "9-12"]
ARCHETYPE_INT = {"A1": 0, "A2": 1, "A3": 2, "A4": 3}


def read_summary_auc(root: pathlib.Path, horizon: str) -> Optional[float]:
    """root/train_<w>m/eval_<w>m/summary_metrics.json から IRL_Dir_calibrated AUC を返す。"""
    p = root / f"train_{horizon}m" / f"eval_{horizon}m" / "summary_metrics.json"
    if not p.exists():
        logger.warning("missing: %s", p)
        return None
    with open(p) as f:
        d = json.load(f)
    irl = d.get("IRL_Dir_calibrated") or d.get("IRL_Dir")
    if irl is None:
        return None
    return float(irl.get("clf_auc_roc", float("nan")))


def read_pair_predictions(root: pathlib.Path, horizon: str) -> Optional[pd.DataFrame]:
    p = root / f"train_{horizon}m" / f"eval_{horizon}m" / "pair_predictions.csv"
    if not p.exists():
        logger.warning("missing pair_predictions: %s", p)
        return None
    return pd.read_csv(p)


def bootstrap_auc_diff(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 777,
) -> Tuple[float, float, float]:
    """Δ AUC = AUC(b) - AUC(a) の bootstrap CI。"""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            d = roc_auc_score(y_true[idx], p_b[idx]) - roc_auc_score(y_true[idx], p_a[idx])
            deltas.append(d)
        except ValueError:
            continue
    if not deltas:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(deltas)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def label_to_int(s: pd.Series) -> Optional[pd.Series]:
    """dominant 列 'A1'..'A4' を int に。"""
    return s.map(ARCHETYPE_INT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True, type=pathlib.Path,
                    help="baseline variant の出力ルート (train_<w>m/eval_<w>m を含む)")
    ap.add_argument("--aux-roots", required=True, nargs="+", type=pathlib.Path,
                    help="archetype_aux 各 λ 設定の出力ルート (複数指定可)")
    ap.add_argument("--archetype-csv", type=pathlib.Path, default=None,
                    help="archetype α CSV (省略時は archetype 別集計をスキップ)")
    ap.add_argument("--out", required=True, type=pathlib.Path,
                    help="出力ディレクトリ")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # ── 1. overall AUC 比較 (summary_metrics から) ──
    rows = []
    for h in HORIZONS:
        base_auc = read_summary_auc(args.base, h)
        row = {"horizon": h, "baseline_auc": base_auc}
        for aux_root in args.aux_roots:
            tag = aux_root.name  # e.g. "legacy_aux_l0.3"
            aux_auc = read_summary_auc(aux_root, h)
            row[f"{tag}_auc"] = aux_auc
            if base_auc is not None and aux_auc is not None:
                row[f"{tag}_delta"] = aux_auc - base_auc
        rows.append(row)
    overall = pd.DataFrame(rows)
    overall.to_csv(args.out / "overall_compare.csv", index=False)
    logger.info("overall_compare.csv 保存:\n%s", overall.to_string(index=False))

    # ── 2. pair-level paired bootstrap CI ──
    boot_rows = []
    for h in HORIZONS:
        base_df = read_pair_predictions(args.base, h)
        if base_df is None or "irl_dir_prob_calibrated" not in base_df.columns:
            continue
        y_true = base_df["label"].astype(int).to_numpy()
        p_base = base_df["irl_dir_prob_calibrated"].to_numpy()
        for aux_root in args.aux_roots:
            tag = aux_root.name
            aux_df = read_pair_predictions(aux_root, h)
            if aux_df is None or "irl_dir_prob_calibrated" not in aux_df.columns:
                continue
            # align by (developer, directory)
            merged = base_df[["developer", "directory", "label", "irl_dir_prob_calibrated"]].merge(
                aux_df[["developer", "directory", "irl_dir_prob_calibrated"]],
                on=["developer", "directory"],
                suffixes=("_base", "_aux"),
            )
            if len(merged) < 30:
                continue
            yt = merged["label"].astype(int).to_numpy()
            pa = merged["irl_dir_prob_calibrated_base"].to_numpy()
            pb = merged["irl_dir_prob_calibrated_aux"].to_numpy()
            mean_d, lo, hi = bootstrap_auc_diff(yt, pa, pb, n_boot=args.n_boot)
            boot_rows.append({
                "horizon": h,
                "aux_variant": tag,
                "n_pairs": len(merged),
                "delta_auc_mean": mean_d,
                "ci_lo": lo,
                "ci_hi": hi,
                "significant": (lo > 0 or hi < 0),
            })
    if boot_rows:
        boot = pd.DataFrame(boot_rows)
        boot.to_csv(args.out / "paired_bootstrap_compare.csv", index=False)
        logger.info("paired_bootstrap_compare.csv 保存:\n%s", boot.to_string(index=False))

    # ── 3. archetype 別 AUC ──
    if args.archetype_csv and args.archetype_csv.exists():
        arche = pd.read_csv(args.archetype_csv, index_col=0)
        if "dominant" not in arche.columns:
            logger.warning("archetype CSV に 'dominant' 列がない、スキップ")
        else:
            arche["arche_int"] = label_to_int(arche["dominant"])
            arche_map = arche["dominant"].to_dict()

            per_arche_rows = []
            for h in HORIZONS:
                base_df = read_pair_predictions(args.base, h)
                if base_df is None:
                    continue
                base_df = base_df.copy()
                base_df["archetype"] = base_df["developer"].map(arche_map)
                for tag, root in [("baseline", args.base)] + [(r.name, r) for r in args.aux_roots]:
                    df = base_df if tag == "baseline" else read_pair_predictions(root, h)
                    if df is None:
                        continue
                    df = df.copy()
                    if "archetype" not in df.columns:
                        df["archetype"] = df["developer"].map(arche_map)
                    for arche_name in ["A1", "A2", "A3", "A4"]:
                        sub = df[df["archetype"] == arche_name]
                        if len(sub) < 10:
                            continue
                        y = sub["label"].astype(int).to_numpy()
                        if "irl_dir_prob_calibrated" not in sub.columns:
                            continue
                        p = sub["irl_dir_prob_calibrated"].to_numpy()
                        if len(np.unique(y)) < 2:
                            continue
                        per_arche_rows.append({
                            "horizon": h,
                            "variant": tag,
                            "archetype": arche_name,
                            "n": int(len(sub)),
                            "auc": float(roc_auc_score(y, p)),
                        })
            if per_arche_rows:
                per_arche = pd.DataFrame(per_arche_rows)
                per_arche.to_csv(args.out / "per_archetype_compare.csv", index=False)
                pivot = per_arche.pivot_table(
                    index=["archetype", "horizon"], columns="variant", values="auc"
                )
                pivot.to_csv(args.out / "per_archetype_pivot.csv")
                logger.info(
                    "per_archetype_compare.csv 保存 (%d 行), pivot 形式も保存",
                    len(per_arche),
                )

    logger.info("完了: %s", args.out)


if __name__ == "__main__":
    main()
