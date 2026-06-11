"""Archetype 別に「切り替えパターン」の有無を実データで検証する。

論文 §5.1 の仮説:
  「A2（厳格な査読専従者）では受諾／拒否の切替パターンが明確に存在し，
   それを LSTM hidden state がよく符号化するため IRL が優位となる。」

これを検証するため、開発者ごとに月次の受諾／拒否系列を構築し、
以下の指標を archetype 別に比較する:

  1. 月次受諾率の CV (Coefficient of Variation): 受諾率の揺らぎの大きさ
  2. モード遷移回数: 受諾優位月 ⇄ 拒否優位月 の切り替え回数
  3. ラン長 (run length) 平均: 同モードが連続する月数の平均

入力:
  - data/combined_raw_main32.csv          (email, timestamp, label, ...)
  - outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/
        dev_window_pooled_with_alpha.csv  (developer x window -> dominant archetype)

出力:
  - outputs/analysis/archetype_switching_patterns/
        per_dev.csv      開発者ごとの 3 指標
        per_archetype.csv archetype 別の中央値・平均
        summary.txt      可読サマリ
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_END = "2022-01-01"  # 訓練期間カットオフ
MIN_MONTHS = 3            # 切替指標を計算するための最低月数


def load_reviews(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    df = df[df["timestamp"] < TRAIN_END].copy()
    df["ym"] = df["timestamp"].dt.to_period("M")
    df["label"] = df["label"].astype(int)
    return df[["email", "ym", "label"]]


def monthly_series(reviews: pd.DataFrame) -> pd.DataFrame:
    """(email, ym) -> n_accept, n_reject, accept_rate, dominant_mode."""
    g = reviews.groupby(["email", "ym"]).agg(
        n_accept=("label", "sum"),
        n_total=("label", "size"),
    ).reset_index()
    g["n_reject"] = g["n_total"] - g["n_accept"]
    g["accept_rate"] = g["n_accept"] / g["n_total"]
    # dominant mode: accept-month (acc>=rej) vs reject-month (acc<rej)
    g["mode"] = np.where(g["n_accept"] >= g["n_reject"], "A", "R")
    return g


def compute_dev_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    """開発者ごとに切替指標 3 種を計算。"""
    rows = []
    for dev, g in monthly.groupby("email"):
        g = g.sort_values("ym")
        n_months = len(g)
        if n_months < MIN_MONTHS:
            continue
        rates = g["accept_rate"].to_numpy()
        modes = g["mode"].tolist()
        # 1. 受諾率 CV
        mean_rate = rates.mean()
        cv = rates.std() / mean_rate if mean_rate > 0 else 0.0
        # 2. モード遷移回数
        n_trans = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])
        # 3. ラン長平均
        runs, cur = [], 1
        for i in range(1, len(modes)):
            if modes[i] == modes[i-1]:
                cur += 1
            else:
                runs.append(cur); cur = 1
        runs.append(cur)
        mean_run = float(np.mean(runs))
        rows.append({
            "developer": dev, "n_months": n_months,
            "mean_accept_rate": float(mean_rate),
            "accept_rate_cv": float(cv),
            "n_transitions": int(n_trans),
            "transitions_per_month": float(n_trans / (n_months - 1)) if n_months > 1 else 0.0,
            "mean_run_length": mean_run,
            "n_accept_months": sum(1 for m in modes if m == "A"),
            "n_reject_months": sum(1 for m in modes if m == "R"),
        })
    return pd.DataFrame(rows)


def join_archetype(dev_metrics: pd.DataFrame, arche_csv: Path) -> pd.DataFrame:
    arche = pd.read_csv(arche_csv)
    # dev -> dominant archetype (window 横断で最頻のものを採用)
    dev_arch = (
        arche.groupby("developer")["dominant"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"dominant": "archetype"})
    )
    return dev_metrics.merge(dev_arch, on="developer", how="inner")


def summarize_by_archetype(merged: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "n_months", "mean_accept_rate", "accept_rate_cv",
        "transitions_per_month", "mean_run_length",
        "n_accept_months", "n_reject_months",
    ]
    rows = []
    for arch in sorted(merged["archetype"].unique()):
        sub = merged[merged["archetype"] == arch]
        d = {"archetype": arch, "n_dev": len(sub)}
        for c in cols:
            d[f"{c}_median"] = float(sub[c].median())
            d[f"{c}_mean"] = float(sub[c].mean())
        rows.append(d)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviews", type=Path,
                    default=Path("data/combined_raw_main32.csv"))
    ap.add_argument("--archetype-csv", type=Path,
                    default=Path("outputs/analysis/archetype_irl_rf_volat_ep150_coldonly/dev_window_pooled_with_alpha.csv"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/analysis/archetype_switching_patterns"))
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] load reviews: {args.reviews}")
    reviews = load_reviews(args.reviews)
    print(f"  rows={len(reviews):,}  devs={reviews['email'].nunique():,}")

    print("[2/4] build monthly series")
    monthly = monthly_series(reviews)
    print(f"  (dev, month) rows = {len(monthly):,}")

    print("[3/4] compute dev metrics")
    dev_metrics = compute_dev_metrics(monthly)
    print(f"  developers with >= {MIN_MONTHS} months = {len(dev_metrics):,}")

    print("[4/4] join archetype + summarize")
    merged = join_archetype(dev_metrics, args.archetype_csv)
    print(f"  archetype-assigned developers = {len(merged):,}")
    merged.to_csv(args.output_dir / "per_dev.csv", index=False)

    summary = summarize_by_archetype(merged)
    summary.to_csv(args.output_dir / "per_archetype.csv", index=False)

    print("\n=== Archetype 別の中央値（解釈用、外れ値の影響を抑える） ===")
    cols_show = ["archetype", "n_dev",
                 "n_months_median", "mean_accept_rate_median",
                 "accept_rate_cv_median", "transitions_per_month_median",
                 "mean_run_length_median"]
    print(summary[cols_show].to_string(index=False))

    print("\n=== Archetype 別の平均 ===")
    cols_show_mean = ["archetype", "n_dev",
                      "accept_rate_cv_mean", "transitions_per_month_mean",
                      "mean_run_length_mean"]
    print(summary[cols_show_mean].to_string(index=False))

    # 仮説検証: A2 の切替指標が他 archetype より大きいか?
    print("\n=== 仮説検証: A2 の切替指標は他より顕著か？ ===")
    for metric in ["accept_rate_cv", "transitions_per_month", "mean_run_length"]:
        med = merged.groupby("archetype")[metric].median().to_dict()
        order = sorted(med.items(), key=lambda kv: kv[1], reverse=(metric != "mean_run_length"))
        rank = [k for k,_ in order]
        marker = " ← A2 最大" if (metric != "mean_run_length" and rank[0]=="A2") else (
                 " ← A2 最小 (= 切替多い)" if (metric == "mean_run_length" and rank[-1]=="A2") else "")
        vals = "  ".join(f"{k}={med[k]:.3f}" for k in ["A1","A2","A3","A4"] if k in med)
        print(f"  {metric:<26s}: {vals}{marker}")

    with open(args.output_dir / "summary.txt", "w") as f:
        f.write("Archetype switching patterns (training period <= 2022-01-01)\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary.to_string(index=False) + "\n")

    print(f"\n[done] outputs at {args.output_dir}")


if __name__ == "__main__":
    main()
