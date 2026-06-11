"""IRL_Dir × RF_Dir のスコアアンサンブルを試す。

各 eval セル (train_X/eval_Y/pair_predictions.csv) を読み、
- irl_dir_prob_calibrated 単体
- rf_dir_prob 単体
- 単純平均
- ランク平均
- 重み付き線形 (w を 0.0〜1.0 で sweep、最良 w を採用)
の AUC-ROC / AUC-PR を比較する。

最良 w は **同一セル内の最適**であり、本来は train で選んで eval で評価すべきだが、
まずは「アンサンブルが理論上どこまで効くか」を見る上限値推定として実行する。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def rank_normalize(s: pd.Series) -> np.ndarray:
    """値を [0, 1] のランク (percentile) に変換。同値は平均ランク。"""
    return s.rank(pct=True).to_numpy()


def auc(y, s) -> tuple[float, float]:
    if y.nunique() < 2:
        return float("nan"), float("nan")
    return roc_auc_score(y, s), average_precision_score(y, s)


def evaluate_cell(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["label", "irl_dir_prob_calibrated", "rf_dir_prob"]).copy()
    if df.empty or df["label"].nunique() < 2:
        return {"n": 0}

    y = df["label"].astype(int)
    irl = df["irl_dir_prob_calibrated"].to_numpy()
    rf = df["rf_dir_prob"].to_numpy()
    irl_r = rank_normalize(df["irl_dir_prob_calibrated"])
    rf_r = rank_normalize(df["rf_dir_prob"])

    out: dict[str, float] = {
        "n": len(df),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
    }

    out["auc_irl"], out["pr_irl"] = auc(y, irl)
    out["auc_rf"], out["pr_rf"] = auc(y, rf)
    out["auc_avg"], out["pr_avg"] = auc(y, (irl + rf) / 2)
    out["auc_rank_avg"], out["pr_rank_avg"] = auc(y, (irl_r + rf_r) / 2)

    # weighted sweep on calibrated probs
    best_w, best_auc = 0.5, -1.0
    for w in np.linspace(0, 1, 21):
        s = w * irl + (1 - w) * rf
        a, _ = auc(y, s)
        if a > best_auc:
            best_auc = a
            best_w = w
    out["auc_wopt"] = best_auc
    out["w_opt"] = best_w
    _, out["pr_wopt"] = auc(y, best_w * irl + (1 - best_w) * rf)

    # weighted sweep on rank-normalized
    best_w_r, best_auc_r = 0.5, -1.0
    for w in np.linspace(0, 1, 21):
        s = w * irl_r + (1 - w) * rf_r
        a, _ = auc(y, s)
        if a > best_auc_r:
            best_auc_r = a
            best_w_r = w
    out["auc_rank_wopt"] = best_auc_r
    out["w_opt_rank"] = best_w_r
    _, out["pr_rank_wopt"] = auc(
        y, best_w_r * irl_r + (1 - best_w_r) * rf_r
    )

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    csvs = sorted(args.base_dir.glob("*/train_*/eval_*/pair_predictions.csv"))
    if not csvs:
        csvs = sorted(args.base_dir.glob("train_*/eval_*/pair_predictions.csv"))
    if not csvs:
        raise SystemExit(f"pair_predictions.csv が見つからない: {args.base_dir}")

    rows = []
    for csv in csvs:
        parts = csv.parts
        train = next(p for p in parts if p.startswith("train_"))
        ev = next(p for p in parts if p.startswith("eval_"))
        variant = parts[parts.index(train) - 1] if parts.index(train) > 0 else ""
        r = evaluate_cell(csv)
        r["variant"] = variant
        r["train"] = train
        r["eval"] = ev
        rows.append(r)

    df = pd.DataFrame(rows)
    cols = [
        "variant", "train", "eval", "n", "n_pos",
        "auc_irl", "auc_rf", "auc_avg", "auc_rank_avg",
        "auc_wopt", "w_opt", "auc_rank_wopt", "w_opt_rank",
        "pr_irl", "pr_rf", "pr_avg", "pr_rank_avg",
        "pr_wopt", "pr_rank_wopt",
    ]
    df = df[[c for c in cols if c in df.columns]]

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))

    print("\n=== mean (over cells) ===")
    means = df[[c for c in df.columns if c.startswith("auc_") or c.startswith("pr_") or c.startswith("w_")]].mean(numeric_only=True)
    print(means.to_string())

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        summary = {
            "per_cell": json.loads(df.to_json(orient="records")),
            "means": json.loads(means.to_json()),
        }
        (args.output.with_suffix(".json")).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        print(f"\nwrote: {args.output}")


if __name__ == "__main__":
    main()
