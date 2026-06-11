"""A2 (Strict Reviewer-Only) の訓練期と評価期での行動シフトを検証する。

仮説 (新規): archetype 割当は訓練期 (2019-2022) の累積特徴から計算されているが、
2022 年中の評価期では A2 dev が「ほぼ拒否」を維持しない可能性がある。
もしそうなら:
  - RF は訓練累積統計 (低 base rate) に依存 → 評価期の振る舞いを捕捉できない
  - IRL は直近の時系列文脈を持つ → シフトを追従できる
これが A2 で IRL が RF に勝つ本質的機序かもしれない。

検証指標:
  1. dev 単位: 訓練期と評価期の受諾率の変化
  2. archetype 単位: 訓練期 cumulative accept rate vs 評価期 accept rate
  3. A2 dev で「シフト量」と「IRL 予測精度」の相関 (シフト大ほど IRL 有利か)

入力:
  - data/combined_raw_main32.csv
  - outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv
  - outputs/main32_mce_volat_ep150/lstm_baseline/train_*m/eval_*m/pair_predictions.csv

出力:
  - outputs/analysis/a2_train_eval_shift/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

TRAIN_END = pd.Timestamp("2022-01-01")
EVAL_END = pd.Timestamp("2023-01-01")
WINDOWS = ["0-3m", "3-6m", "6-9m", "9-12m"]


def load(reviews_csv, arche_csv):
    rev = pd.read_csv(reviews_csv, parse_dates=["timestamp"])
    rev["label"] = rev["label"].astype(int)
    arche = pd.read_csv(arche_csv)
    dev_arch = (arche.groupby("developer")["dominant"]
                .agg(lambda s: s.value_counts().idxmax())
                .reset_index().rename(columns={"developer": "email", "dominant": "archetype"}))
    return rev, dev_arch


def per_dev_train_eval(rev, dev_arch):
    """各 dev の訓練期 vs 評価期の受諾率を計算。"""
    train = rev[rev["timestamp"] < TRAIN_END]
    eval_ = rev[(rev["timestamp"] >= TRAIN_END) & (rev["timestamp"] < EVAL_END)]

    tg = train.groupby("email")["label"].agg(["mean", "size"]).reset_index()
    tg.columns = ["email", "train_rate", "train_n"]

    eg = eval_.groupby("email")["label"].agg(["mean", "size"]).reset_index()
    eg.columns = ["email", "eval_rate", "eval_n"]

    m = tg.merge(eg, on="email", how="inner").merge(dev_arch, on="email", how="inner")
    m["shift"] = m["eval_rate"] - m["train_rate"]
    return m[m["train_n"] >= 5]  # min 5 train events to compute meaningful rate


def archetype_shift_summary(merged):
    rows = []
    for arch in sorted(merged["archetype"].unique()):
        sub = merged[merged["archetype"] == arch]
        rows.append({
            "archetype": arch, "n_dev": len(sub),
            "train_rate_mean": sub["train_rate"].mean(),
            "train_rate_median": sub["train_rate"].median(),
            "eval_rate_mean": sub["eval_rate"].mean(),
            "eval_rate_median": sub["eval_rate"].median(),
            "shift_mean": sub["shift"].mean(),
            "shift_median": sub["shift"].median(),
            "shift_abs_mean": sub["shift"].abs().mean(),
            "n_dev_with_shift_gt_0.2": int((sub["shift"].abs() > 0.2).sum()),
        })
    return pd.DataFrame(rows)


def load_predictions(base):
    """A2 dev で IRL 予測精度と shift の関係を見るための準備。"""
    base_dir = Path(base) / "lstm_baseline"
    frames = []
    for h in WINDOWS:
        p = base_dir / f"train_{h}" / f"eval_{h}" / "pair_predictions.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p).dropna(
            subset=["irl_dir_prob_calibrated", "rf_dir_prob", "label"]).copy()
        d["window"] = h
        d["ensemble"] = (d["irl_dir_prob_calibrated"] + d["rf_dir_prob"]) / 2
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def shift_vs_irl_advantage(merged, predictions, dev_arch):
    """A2 dev について、(eval_rate - train_rate) のシフト量と
       IRL - RF の per-dev AUC 差の相関を見る。"""
    # 各 A2 dev の per-dev AUC (IRL, RF) を計算
    a2_devs = set(merged[merged["archetype"] == "A2"]["email"])
    pred = predictions[predictions["developer"].isin(a2_devs)].copy()

    dev_rows = []
    for dev, g in pred.groupby("developer"):
        if g["label"].nunique() < 2:
            continue
        y = g["label"].astype(int).to_numpy()
        try:
            ai = roc_auc_score(y, g["irl_dir_prob_calibrated"])
            ar = roc_auc_score(y, g["rf_dir_prob"])
        except ValueError:
            continue
        dev_rows.append({
            "developer": dev, "n_pairs": len(g),
            "auc_irl": ai, "auc_rf": ar, "irl_minus_rf": ai - ar,
        })
    dev_auc = pd.DataFrame(dev_rows)
    m = dev_auc.merge(
        merged[["email", "shift", "train_rate", "eval_rate"]].rename(columns={"email": "developer"}),
        on="developer", how="inner",
    )
    return m


def shift_and_window_auc(merged, predictions):
    """各予測窓で、shift の大小と IRL の AUC を見る。
       シフト大の A2 dev を含む窓で IRL が強いかを確認。"""
    a2_devs = set(merged[merged["archetype"] == "A2"]["email"])
    rows = []
    for h in WINDOWS:
        pred = predictions[(predictions["window"] == h) &
                           (predictions["developer"].isin(a2_devs))].copy()
        if len(pred) == 0:
            continue
        # dev 別 shift を pred に結合
        pred = pred.merge(
            merged[["email", "shift"]].rename(columns={"email": "developer"}),
            on="developer", how="left",
        )
        # shift quartile 別 IRL vs RF AUC
        pred["shift_q"] = pd.qcut(pred["shift"].abs(), 3,
                                   labels=["low", "mid", "high"], duplicates="drop")
        for q in ["low", "mid", "high"]:
            sub = pred[pred["shift_q"] == q]
            if len(sub) < 30 or sub["label"].nunique() < 2:
                continue
            y = sub["label"].astype(int).to_numpy()
            ai = roc_auc_score(y, sub["irl_dir_prob_calibrated"])
            ar = roc_auc_score(y, sub["rf_dir_prob"])
            rows.append({
                "window": h, "shift_bucket": q, "n": len(sub),
                "auc_irl": ai, "auc_rf": ar, "irl_minus_rf": ai - ar,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", type=Path, default=Path("data/combined_raw_main32.csv"))
    ap.add_argument("--archetype-csv", type=Path,
                    default=Path("outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv"))
    ap.add_argument("--predictions-base", type=Path,
                    default=Path("outputs/main32_mce_volat_ep150"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/analysis/a2_train_eval_shift"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[load]")
    rev, dev_arch = load(args.reviews, args.archetype_csv)
    merged = per_dev_train_eval(rev, dev_arch)
    print(f"  archetype-assigned devs with train data = {len(merged):,}")
    merged.to_csv(args.output_dir / "per_dev_train_eval.csv", index=False)

    # ─── 1. archetype 別 train/eval シフト ──────────────
    summary = archetype_shift_summary(merged)
    summary.to_csv(args.output_dir / "archetype_shift_summary.csv", index=False)
    print("\n=== 1. Archetype 別: 訓練期 vs 評価期の受諾率 ===")
    print(summary.to_string(index=False))

    # ─── 2. A2 dev のシフト分布 ─────────────────────────
    a2 = merged[merged["archetype"] == "A2"].copy()
    print(f"\n=== 2. A2 dev (n={len(a2)}) シフト分布 ===")
    print(f"  train_rate: mean={a2['train_rate'].mean():.3f}  median={a2['train_rate'].median():.3f}")
    print(f"  eval_rate : mean={a2['eval_rate'].mean():.3f}  median={a2['eval_rate'].median():.3f}")
    print(f"  shift     : mean={a2['shift'].mean():+.3f}  median={a2['shift'].median():+.3f}")
    print(f"  |shift| > 0.2 の dev = {(a2['shift'].abs() > 0.2).sum()} / {len(a2)}")
    print(f"  |shift| > 0.4 の dev = {(a2['shift'].abs() > 0.4).sum()} / {len(a2)}")

    # ─── 3. シフトと IRL 優位の相関 (per-dev) ───────────
    print("\n[load predictions]")
    pred = load_predictions(args.predictions_base)
    if pred is None:
        print("  predictions not found, skip 3 & 4")
        return

    dev_corr = shift_vs_irl_advantage(merged, pred, dev_arch)
    dev_corr.to_csv(args.output_dir / "a2_shift_vs_irl_advantage.csv", index=False)
    if len(dev_corr) > 5:
        r, p = spearmanr(dev_corr["shift"].abs(), dev_corr["irl_minus_rf"])
        print(f"\n=== 3. A2 dev (n={len(dev_corr)}): |shift| と IRL_minus_RF per-dev AUC 差 ===")
        print(f"  Spearman 相関 = {r:+.3f}  (p={p:.4f})")
        if r > 0.2 and p < 0.05:
            print("  → シフト大の dev ほど IRL が RF より勝つ → 仮説支持")
        elif abs(r) < 0.1:
            print("  → 相関なし → シフトと IRL 優位は無関係")
        else:
            print("  → 弱い相関")

    # ─── 4. 各窓 x シフト bucket で IRL-RF gap ──────────
    bucket = shift_and_window_auc(merged, pred)
    bucket.to_csv(args.output_dir / "window_shift_bucket_auc.csv", index=False)
    print("\n=== 4. 予測窓 x A2 dev のシフト 3 分位 別 AUC ===")
    print(bucket.to_string(index=False))

    print(f"\n[done] outputs at {args.output_dir}")


if __name__ == "__main__":
    main()
