# -*- coding: utf-8 -*-
"""Activity pattern classification on the warm-start MCE-IRL (4 future windows).

The bachelor-thesis analysis (scripts/analyze/plot/plot_activity_patterns.py)
classified reviewers present in **all 4 diagonal cells** as
  - 一貫承諾型 (consistent acceptors): true_label=1 in every window
  - 一貫非承諾型 (consistent rejectors): true_label=0 in every window
  - 変動型 (variable): true_label changes across windows
and compared IRL / RF accuracy per group.

The KD warm-start run (outputs/main32_mce_kd_warmstart) only produced eval
files for the train_0-3m model so far. We therefore evaluate the *same*
warm-start checkpoint (trained on 0-3m future labels) on the four future
windows 0-3m / 3-6m / 6-9m / 9-12m and compute per-pattern accuracy for
three models that all share the same (developer, directory) eval rows:

  warm-start  (irl_dir_prob_calibrated  in outputs/main32_mce_kd_warmstart/...)
  cold IRL    (irl_dir_prob_calibrated  in outputs/main32_mce/...)
  RF_Dir      (rf_dir_prob              in outputs/main32_mce_kd_warmstart/...)

Both pair-level and dev-level (max-prob aggregation) classifications are
emitted. Thresholds are taken from each cell's summary_metrics.json
(F1-optimal threshold the eval pipeline selected per metric block).

Usage:
    uv run python scripts/analyze/study/activity_patterns_warmstart.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Dict, List

import japanize_matplotlib  # noqa: F401  日本語フォント登録（副作用 import）
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]

PATTERN_CONSISTENT_ACCEPT = "一貫承諾型"
PATTERN_CONSISTENT_REJECT = "一貫非承諾型"
PATTERN_VARIABLE = "変動型"
PATTERN_ORDER = [PATTERN_CONSISTENT_ACCEPT, PATTERN_CONSISTENT_REJECT, PATTERN_VARIABLE]

# モデル定義: (列名, 凡例ラベル, summary_metrics.json のメトリクスキー)
MODELS = [
    ("warmstart", "Warm-start IRL", "irl_dir_prob_calibrated", "IRL_Dir_calibrated"),
    ("cold_irl", "Cold MCE-IRL",    "irl_dir_prob_calibrated", "IRL_Dir_calibrated"),
    ("rf",       "RF_Dir",          "rf_dir_prob",             "RF_Dir"),
]


def load_window(base_dir: pathlib.Path, train_win: str, eval_win: str) -> pd.DataFrame:
    csv = base_dir / f"lstm_baseline/train_{train_win}/eval_{eval_win}/pair_predictions.csv"
    if not csv.exists():
        raise FileNotFoundError(f"missing pair_predictions.csv: {csv}")
    return pd.read_csv(csv)


def load_thresholds(base_dir: pathlib.Path, train_win: str, eval_win: str) -> Dict[str, float]:
    js = base_dir / f"lstm_baseline/train_{train_win}/eval_{eval_win}/summary_metrics.json"
    with open(js) as f:
        d = json.load(f)
    return {k: float(d[k]["clf_threshold"]) for k in
            ("IRL_Dir_calibrated", "IRL_Dir", "RF_Dir") if k in d}


def classify_pairs(labels_by_window: Dict[str, np.ndarray]) -> np.ndarray:
    """Classify each row by its 4 labels into 3 patterns."""
    stacked = np.stack([labels_by_window[w] for w in PERIODS], axis=1)
    n_pos = stacked.sum(axis=1)
    n_win = stacked.shape[1]
    out = np.empty(stacked.shape[0], dtype=object)
    out[n_pos == n_win] = PATTERN_CONSISTENT_ACCEPT
    out[n_pos == 0] = PATTERN_CONSISTENT_REJECT
    out[(n_pos > 0) & (n_pos < n_win)] = PATTERN_VARIABLE
    return out


def per_pattern_accuracy(
    pattern: np.ndarray,
    correct_by_window: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Average per-row correct rate over the 4 windows, then group by pattern."""
    stacked = np.stack([correct_by_window[w].astype(float) for w in PERIODS], axis=1)
    row_acc = stacked.mean(axis=1)
    df = pd.DataFrame({"pattern": pattern, "accuracy": row_acc})
    summary = (df.groupby("pattern")["accuracy"]
                 .agg(["mean", "count"])
                 .reindex(PATTERN_ORDER))
    return summary


def plot_grouped_bar(table: pd.DataFrame, counts: pd.Series, out_path: pathlib.Path,
                     title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(PATTERN_ORDER))
    width = 0.27
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, (col, color) in enumerate(zip(table.columns, colors)):
        bars = ax.bar(x + (i - 1) * width, table[col].values * 100, width,
                       label=col, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)
    labels = [f"{p}\n(n={int(counts[p])})" for p in PATTERN_ORDER]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("正解率 (%)")
    ax.set_ylim(0, 110)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def aggregate_dev(df_pair: pd.DataFrame, prob_col: str, label_col: str = "label") -> pd.DataFrame:
    """Dev-level: max prob across directories, any label=1 → dev label=1."""
    agg = df_pair.groupby("developer").agg(
        prob=(prob_col, "max"),
        label=(label_col, "max"),
    )
    return agg


def run(warm_base: pathlib.Path, cold_base: pathlib.Path, out_dir: pathlib.Path,
        train_win: str, diagonal: bool = False) -> None:
    if diagonal:
        print(f"[load] diagonal train_W/eval_W for W in {PERIODS}")
        warm = {w: load_window(warm_base, w, w) for w in PERIODS}
        cold = {w: load_window(cold_base, w, w) for w in PERIODS}
        warm_thr = {w: load_thresholds(warm_base, w, w) for w in PERIODS}
        cold_thr = {w: load_thresholds(cold_base, w, w) for w in PERIODS}
    else:
        print(f"[load] train_{train_win} × {PERIODS}")
        warm = {w: load_window(warm_base, train_win, w) for w in PERIODS}
        cold = {w: load_window(cold_base, train_win, w) for w in PERIODS}
        warm_thr = {w: load_thresholds(warm_base, train_win, w) for w in PERIODS}
        cold_thr = {w: load_thresholds(cold_base, train_win, w) for w in PERIODS}

    # 4 窓で同じ (dev, dir) を持つことを確認
    key = warm[PERIODS[0]][["developer", "directory"]].copy()
    key["row_id"] = np.arange(len(key))
    for w in PERIODS:
        for src in (warm, cold):
            df = src[w]
            assert len(df) == len(key)
            df.set_index(["developer", "directory"], inplace=True)
            df = df.loc[list(zip(key["developer"], key["directory"]))].reset_index()
            src[w] = df

    # ---------------- pair-level ----------------
    labels = {w: warm[w]["label"].to_numpy().astype(int) for w in PERIODS}
    pattern_pair = classify_pairs(labels)
    counts_pair = pd.Series(pattern_pair).value_counts().reindex(PATTERN_ORDER).fillna(0).astype(int)

    pair_acc_table = {}
    per_window_acc = {}  # for diagnostic CSV
    for key_name, label, prob_col_warm, metric_key in MODELS:
        correct = {}
        win_acc_rows = []
        for w in PERIODS:
            if key_name == "warmstart":
                df = warm[w]
                prob = df[prob_col_warm].to_numpy()
                thr = warm_thr[w][metric_key]
            elif key_name == "cold_irl":
                df = cold[w]
                prob = df["irl_dir_prob_calibrated"].to_numpy()
                thr = cold_thr[w]["IRL_Dir_calibrated"]
            elif key_name == "rf":
                # 2026-05-18 訂正: 以前は warm dir の RF (warm pipeline 独立学習) を使っていたが、
                # cold IRL との apples-to-apples 比較のため cold pipeline の RF に統一。
                df = cold[w]
                prob = df["rf_dir_prob"].to_numpy()
                thr = cold_thr[w]["RF_Dir"]
            pred = (prob >= thr).astype(int)
            ok = (pred == labels[w]).astype(int)
            correct[w] = ok
            win_acc_rows.append({"window": w, "model": label, "threshold": thr,
                                  "acc": float(ok.mean()), "n": len(ok)})
        per_window_acc[label] = win_acc_rows
        summary = per_pattern_accuracy(pattern_pair, correct)
        pair_acc_table[label] = summary["mean"]

    table_pair = pd.DataFrame(pair_acc_table)
    table_pair.index.name = "pattern"
    print("\n[pair-level] accuracy by activity pattern (avg over 4 windows)")
    print((table_pair * 100).round(2).to_string())
    print(f"  counts: {counts_pair.to_dict()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    table_pair.assign(count=counts_pair).to_csv(out_dir / "pair_per_pattern_accuracy.csv")
    title_suffix = "diagonal train_W/eval_W" if diagonal else f"train_{train_win}"
    plot_grouped_bar(table_pair, counts_pair,
                      out_dir / "fig_activity_pattern_pair.pdf",
                      title=f"活動パターン別 正解率 (pair-level, {title_suffix})")

    pd.DataFrame([r for rows in per_window_acc.values() for r in rows]
                 ).to_csv(out_dir / "per_window_accuracy.csv", index=False)

    # ---------------- dev-level ----------------
    dev_labels = {}
    dev_probs = {m[0]: {} for m in MODELS}
    for w in PERIODS:
        # warm
        agg = aggregate_dev(warm[w], "irl_dir_prob_calibrated", "label")
        dev_probs["warmstart"][w] = agg["prob"]
        dev_labels[w] = agg["label"]
        # 2026-05-18 訂正: rf も cold dir から取る (cold IRL との apples-to-apples 比較のため)
        agg_rf = aggregate_dev(cold[w], "rf_dir_prob", "label")
        dev_probs["rf"][w] = agg_rf["prob"]
        # cold
        agg_c = aggregate_dev(cold[w], "irl_dir_prob_calibrated", "label")
        dev_probs["cold_irl"][w] = agg_c["prob"]

    common_devs = set(dev_labels[PERIODS[0]].index)
    for w in PERIODS:
        common_devs &= set(dev_labels[w].index)
    common_devs = sorted(common_devs)

    dev_label_mat = np.stack(
        [dev_labels[w].loc[common_devs].to_numpy().astype(int) for w in PERIODS], axis=1)
    n_pos = dev_label_mat.sum(axis=1)
    n_win = dev_label_mat.shape[1]
    dev_pattern = np.empty(len(common_devs), dtype=object)
    dev_pattern[n_pos == n_win] = PATTERN_CONSISTENT_ACCEPT
    dev_pattern[n_pos == 0] = PATTERN_CONSISTENT_REJECT
    dev_pattern[(n_pos > 0) & (n_pos < n_win)] = PATTERN_VARIABLE
    counts_dev = pd.Series(dev_pattern).value_counts().reindex(PATTERN_ORDER).fillna(0).astype(int)

    # threshold: use the per-window pair-level threshold (same prob scale).
    # For dev-level we re-binarize the max prob with the same threshold.
    dev_acc_table = {}
    for key_name, label, _, metric_key in MODELS:
        correct = {}
        for w in PERIODS:
            if key_name == "warmstart":
                thr = warm_thr[w]["IRL_Dir_calibrated"]
            elif key_name == "cold_irl":
                thr = cold_thr[w]["IRL_Dir_calibrated"]
            elif key_name == "rf":
                # 2026-05-18 訂正: cold dir の RF threshold に統一
                thr = cold_thr[w]["RF_Dir"]
            prob = dev_probs[key_name][w].loc[common_devs].to_numpy()
            pred = (prob >= thr).astype(int)
            ok = (pred == dev_label_mat[:, PERIODS.index(w)]).astype(int)
            correct[w] = ok
        summary = per_pattern_accuracy(dev_pattern, correct)
        dev_acc_table[label] = summary["mean"]

    table_dev = pd.DataFrame(dev_acc_table)
    table_dev.index.name = "pattern"
    print("\n[dev-level] accuracy by activity pattern (avg over 4 windows, max-prob agg)")
    print((table_dev * 100).round(2).to_string())
    print(f"  counts: {counts_dev.to_dict()}")
    table_dev.assign(count=counts_dev).to_csv(out_dir / "dev_per_pattern_accuracy.csv")
    plot_grouped_bar(table_dev, counts_dev,
                      out_dir / "fig_activity_pattern_dev.pdf",
                      title=f"活動パターン別 正解率 (dev-level, {title_suffix})")

    # ---------------- raw pattern dump ----------------
    raw = pd.DataFrame({
        "developer": [d for d, _ in zip(key["developer"], key["directory"])],
        "directory": [r for _, r in zip(key["developer"], key["directory"])],
        "pattern": pattern_pair,
        **{f"label_{w}": labels[w] for w in PERIODS},
    })
    raw.to_csv(out_dir / "pair_classified.csv", index=False)

    dev_raw = pd.DataFrame({
        "developer": common_devs,
        "pattern": dev_pattern,
        **{f"label_{w}": dev_label_mat[:, i] for i, w in enumerate(PERIODS)},
    })
    dev_raw.to_csv(out_dir / "dev_classified.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warm-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_kd_warmstart"))
    p.add_argument("--cold-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce"))
    p.add_argument("--train-win", default="0-3m",
                   help="Training window to evaluate over the 4 future windows (default 0-3m)")
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/activity_patterns_warmstart"))
    p.add_argument("--diagonal", action="store_true",
                   help="For each window W, load train_W/eval_W instead of fixed --train-win")
    args = p.parse_args()
    run(args.warm_dir, args.cold_dir, args.output_dir, args.train_win, args.diagonal)


if __name__ == "__main__":
    main()
