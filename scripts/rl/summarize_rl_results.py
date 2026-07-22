#!/usr/bin/env python3
"""RL 実験結果の横断集約スクリプト

指定ディレクトリ以下の summary_metrics.json (eval_assignment.py の出力) を
再帰的に集め、run × 手法 × 評価軸の long 形式 CSV と、
主要軸 (step_label AUC-ROC) の run × 手法 markdown 表を出力する。

run のパラメータ (λ, γ, seed) は eval と同階層の train_config.json
(train_offline_rl.py が出力) から拾う。無ければパス名から推定する。

使い方:
    uv run python scripts/rl/summarize_rl_results.py outputs/rl/sweep \\
        --output-csv outputs/rl/sweep/summary.csv \\
        --output-md  outputs/rl/sweep/summary.md
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# summary_metrics.json のキー形式: RL_<method>_<axis>
AXES = ("step_label", "step_label_dev", "monthly")
METHODS = (
    "random", "r_theta_baseline", "irl_pi", "rf_supervised", "logreg_supervised",
    "bc_pi", "iql_pi", "iql_q_advantage", "iql_v", "ensemble_irl_v",
)


def _load_train_config(eval_dir: Path) -> Dict[str, Any]:
    """eval ディレクトリの親 (run ディレクトリ) から train_config.json を探す。"""
    for parent in (eval_dir.parent, eval_dir.parent.parent):
        cfg = parent / "train_config.json"
        if cfg.exists():
            try:
                with open(cfg) as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def _params_from_path(path: Path) -> Dict[str, Any]:
    """sweep ディレクトリ名 (lam<λ>_gam<γ>) からパラメータを推定する。"""
    m = re.search(r"lam([0-9.]+)_gam([0-9.]+)", str(path))
    if m:
        return {"reward_shaping": float(m.group(1)), "gamma": float(m.group(2))}
    return {}


def collect(root: Path) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for summary_path in sorted(root.rglob("summary_metrics.json")):
        try:
            with open(summary_path) as f:
                summary = json.load(f)
        except Exception as e:
            logger.warning("読み込み失敗 %s: %s", summary_path, e)
            continue

        eval_dir = summary_path.parent
        run_dir = eval_dir.parent
        meta = summary.get("_meta", {})
        cfg = _load_train_config(eval_dir)
        params = {**_params_from_path(run_dir), **{
            k: cfg[k] for k in ("reward_shaping", "gamma", "tau", "beta", "seed")
            if k in cfg and cfg[k] is not None
        }}

        base = {
            "run": str(run_dir.relative_to(root)) if run_dir.is_relative_to(root) else str(run_dir),
            "eval_kind": eval_dir.name,  # eval / eval_devholdout
            "is_held_out": meta.get("is_held_out"),
            "train_n": meta.get("train_n"),
            "test_n": meta.get("test_n"),
            "ensemble_best_beta": meta.get("ensemble_best_beta"),
            **params,
        }

        for axis in AXES:
            for method in METHODS:
                entry = summary.get(f"RL_{method}_{axis}")
                if not entry:
                    continue
                records.append({
                    **base,
                    "axis": axis,
                    "method": method,
                    "auc_roc": entry.get("clf_auc_roc"),
                    "auc_pr": entry.get("clf_auc_pr"),
                    "f1": entry.get("clf_f1"),
                    "n_pairs": entry.get("clf_n_pairs"),
                })
        lt = summary.get("RL_iql_v_long_term")
        if lt:
            records.append({
                **base,
                "axis": "v_long_term",
                "method": "iql_v",
                "spearman": lt.get("spearman_v_vs_remaining_accept"),
                "n_pairs": lt.get("clf_n_pairs"),
            })
    return pd.DataFrame(records)


def to_markdown(df: pd.DataFrame) -> str:
    """主要軸の run × 手法 AUC-ROC ピボット表を markdown で返す。"""
    lines: List[str] = ["# RL sweep 結果サマリ", ""]
    for axis in AXES:
        sub = df[(df["axis"] == axis) & df["auc_roc"].notna()]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index=["run", "eval_kind"], columns="method", values="auc_roc",
        ).round(4)
        cols = [m for m in METHODS if m in pivot.columns]
        lines += [f"## {axis} (AUC-ROC)", "", pivot[cols].to_markdown(), ""]
    lt = df[df["axis"] == "v_long_term"]
    if not lt.empty and "spearman" in lt.columns:
        lines += [
            "## V(s) 長期価値 (Spearman vs remaining_accept)", "",
            lt.set_index(["run", "eval_kind"])["spearman"].round(4).to_markdown(), "",
        ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="RL 実験結果の横断集約")
    p.add_argument("root", nargs="?", default="outputs/rl", help="走査ルート")
    p.add_argument("--output-csv", default=None, help="long 形式 CSV の出力先")
    p.add_argument("--output-md", default=None, help="markdown 表の出力先")
    args = p.parse_args()

    root = Path(args.root)
    df = collect(root)
    if df.empty:
        logger.warning("summary_metrics.json が見つかりません: %s", root)
        return
    logger.info("%d run × 軸 × 手法 のレコードを集約", len(df))

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("CSV: %s", out)

    md = to_markdown(df)
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        logger.info("Markdown: %s", out)
    print(md)


if __name__ == "__main__":
    main()
