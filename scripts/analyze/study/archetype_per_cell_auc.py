"""(train, eval) セル単位で archetype 別 AUC を計算する。

archetype_ensemble.py が 4 eval 窓を pool した AUC を出すのに対し、
こちらは各セルを独立に評価し「同期間 (train_W / eval_W) で何が見えるか」を出す。

使い方:
    uv run python scripts/analyze/study/archetype_per_cell_auc.py \
        --alpha-csv outputs/analysis/archetype_irl_rf_volat_coldonly/k4_dev_alpha_with_metrics.csv \
        --pred-base outputs/main32_mce_volat \
        --output outputs/analysis/archetype_irl_rf_volat_coldonly/per_cell_archetype_auc.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def auc_pair(y, s):
    if pd.Series(y).nunique() < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, s), average_precision_score(y, s)


def evaluate_cell(pred_csv: Path, alpha_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(pred_csv)
    df = df.dropna(subset=["label", "irl_dir_prob_calibrated", "rf_dir_prob"]).copy()
    df = df.merge(alpha_df[["developer", "hard_archetype"]], on="developer", how="inner")

    rows = []
    for arch, sub in df.groupby("hard_archetype"):
        y = sub["label"].astype(int).to_numpy()
        irl = sub["irl_dir_prob_calibrated"].to_numpy()
        rf = sub["rf_dir_prob"].to_numpy()
        avg = (irl + rf) / 2

        n = len(sub)
        n_pos = int(y.sum())
        a_irl, p_irl = auc_pair(y, irl)
        a_rf, p_rf = auc_pair(y, rf)
        a_avg, p_avg = auc_pair(y, avg)

        rows.append({
            "archetype": arch, "n_pairs": n, "n_pos": n_pos,
            "auc_irl": a_irl, "auc_rf": a_rf, "auc_avg": a_avg,
            "pr_irl": p_irl, "pr_rf": p_rf, "pr_avg": p_avg,
        })

    # ALL archetypes pooled in this cell (sanity)
    y = df["label"].astype(int).to_numpy()
    irl = df["irl_dir_prob_calibrated"].to_numpy()
    rf = df["rf_dir_prob"].to_numpy()
    avg = (irl + rf) / 2
    a_irl, p_irl = auc_pair(y, irl)
    a_rf, p_rf = auc_pair(y, rf)
    a_avg, p_avg = auc_pair(y, avg)
    rows.append({
        "archetype": "ALL", "n_pairs": len(df), "n_pos": int(y.sum()),
        "auc_irl": a_irl, "auc_rf": a_rf, "auc_avg": a_avg,
        "pr_irl": p_irl, "pr_rf": p_rf, "pr_avg": p_avg,
    })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha-csv", type=Path, required=True)
    p.add_argument("--pred-base", type=Path, required=True,
                   help="例: outputs/main32_mce_volat")
    p.add_argument("--variant", default="lstm_baseline")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    alpha = pd.read_csv(args.alpha_csv, index_col=0)
    alpha.index.name = "developer"
    alpha = alpha.reset_index()
    if "dominant" in alpha.columns:
        alpha["hard_archetype"] = alpha["dominant"]
    elif "hard_archetype" not in alpha.columns:
        # fallback: A1..A4 列から argmax
        arch_cols = [c for c in alpha.columns if c in ("A1", "A2", "A3", "A4", "A5", "A6")]
        if arch_cols:
            alpha["hard_archetype"] = alpha[arch_cols].idxmax(axis=1)
        else:
            raise SystemExit(f"hard_archetype/dominant 列が無い: {args.alpha_csv}")

    print(f"[alpha] devs={len(alpha)}, archetypes={sorted(alpha['hard_archetype'].unique())}")

    base = args.pred_base / args.variant
    cells = sorted(base.glob("train_*/eval_*/pair_predictions.csv"))
    print(f"[cells] {len(cells)}")

    all_rows = []
    for csv in cells:
        parts = csv.parts
        train = next(x for x in parts if x.startswith("train_"))
        ev = next(x for x in parts if x.startswith("eval_"))
        cell_df = evaluate_cell(csv, alpha)
        cell_df.insert(0, "train", train)
        cell_df.insert(1, "eval", ev)
        all_rows.append(cell_df)

    full = pd.concat(all_rows, ignore_index=True)

    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("\n=== per-cell archetype AUC ===")
    print(full.to_string(index=False))

    # 注目: diagonal (train_W = eval_W) だけ抜粋
    diag = full[full["train"].str.replace("train_", "") == full["eval"].str.replace("eval_", "")]
    print("\n=== diagonal (train_W = eval_W) ===")
    print(diag.to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        full.to_csv(args.output, index=False)
        print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
