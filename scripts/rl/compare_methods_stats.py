#!/usr/bin/env python3
"""手法間 AUC 差の統計的検定 (paired bootstrap)

pair_predictions.csv (eval_assignment.py の出力) から、各手法の AUC と
ベースラインとの差 ΔAUC を paired bootstrap で推定し、95% CI と
bootstrap p 値を出す。IRL 論文 (表8) の bootstrap 95%CI と同じ作法で、
「IQL > IRL」等の主張に有意性を付けるために使う。

使い方:
    uv run python scripts/rl/compare_methods_stats.py \\
        outputs/rl/sweep/<...>/lam0.3_gam0.99/eval/pair_predictions.csv \\
        --target step_label \\
        --baseline score_irl_pi \\
        --output-dir outputs/rl/analysis/<run名>
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def bootstrap_auc_table(
    y: np.ndarray,
    scores: pd.DataFrame,
    baseline: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """paired bootstrap で各手法の AUC / ΔAUC(vs baseline) の分布を推定する。

    同じ resample インデックスを全手法に使う (paired) ことで、
    サンプル揺らぎを共通化して差の分散を正しく縮める。
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    methods = list(scores.columns)
    boot = {m: np.empty(n_boot) for m in methods}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        for m in methods:
            boot[m][b] = _auc(yb, scores[m].to_numpy()[idx])

    records = []
    base_boot = boot.get(baseline)
    for m in methods:
        point = _auc(y, scores[m].to_numpy())
        bm = boot[m]
        rec = {
            "method": m,
            "auc": point,
            "auc_ci_lo": float(np.nanpercentile(bm, 2.5)),
            "auc_ci_hi": float(np.nanpercentile(bm, 97.5)),
        }
        if base_boot is not None and m != baseline:
            delta = bm - base_boot
            delta = delta[np.isfinite(delta)]
            if len(delta) > 0:
                # 両側 bootstrap p 値: Δ の符号が反転する割合の2倍
                p_two = 2 * min(
                    float(np.mean(delta <= 0)), float(np.mean(delta >= 0))
                )
                rec.update({
                    "delta_vs_baseline": point - _auc(y, scores[baseline].to_numpy()),
                    "delta_ci_lo": float(np.nanpercentile(delta, 2.5)),
                    "delta_ci_hi": float(np.nanpercentile(delta, 97.5)),
                    "p_boot_two_sided": min(p_two, 1.0),
                })
        records.append(rec)
    return pd.DataFrame(records).sort_values("auc", ascending=False)


def main() -> None:
    p = argparse.ArgumentParser(description="手法間 AUC 差の paired bootstrap 検定")
    p.add_argument("pair_predictions", help="pair_predictions.csv のパス")
    p.add_argument(
        "--target", default="step_label",
        help="正解列 (step_label / step_label_dev / monthly_accept)",
    )
    p.add_argument("--baseline", default="score_irl_pi", help="ΔAUC の基準列")
    p.add_argument("--methods", default=None,
                   help="対象スコア列 (comma区切り)。未指定なら score_* 全部")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None, help="CSV/MD の出力先 (省略時は標準出力のみ)")
    args = p.parse_args()

    df = pd.read_csv(args.pair_predictions)
    if args.target not in df.columns:
        raise SystemExit(f"正解列がありません: {args.target}")
    y_raw = pd.to_numeric(df[args.target], errors="coerce")
    valid = y_raw.notna()

    if args.methods:
        method_cols: List[str] = [m.strip() for m in args.methods.split(",")]
    else:
        method_cols = [c for c in df.columns if c.startswith("score_")]
    missing = [m for m in method_cols if m not in df.columns]
    if missing:
        raise SystemExit(f"スコア列がありません: {missing}")
    if args.baseline not in method_cols:
        logger.warning("baseline %s がスコア列に無いため ΔAUC は出ません", args.baseline)

    y = y_raw[valid].astype(int).to_numpy()
    scores = df.loc[valid, method_cols]
    logger.info(
        "target=%s N=%d (有効 %d/%d), methods=%d, baseline=%s",
        args.target, len(y), int(valid.sum()), len(df), len(method_cols), args.baseline,
    )

    table = bootstrap_auc_table(y, scores, args.baseline, args.n_boot, args.seed)
    md = "\n".join([
        f"# 手法間比較: target={args.target}, baseline={args.baseline}, "
        f"N={len(y)}, bootstrap={args.n_boot}",
        "",
        table.round(4).to_markdown(index=False),
        "",
        "読み方: delta_ci が 0 を跨がず p_boot < 0.05 なら「baseline との差は",
        "bootstrap 95%CI で有意」と主張できる (IRL 論文 表8 と同じ流儀)。",
    ])
    print(md)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = f"methods_stats_{args.target}"
        table.to_csv(out / f"{stem}.csv", index=False)
        (out / f"{stem}.md").write_text(md, encoding="utf-8")
        logger.info("出力: %s", out / f"{stem}.csv")


if __name__ == "__main__":
    main()
