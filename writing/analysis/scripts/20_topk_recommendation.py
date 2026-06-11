# -*- coding: utf-8 -*-
"""B-3: Top-K 推薦評価 (HitRate@K, Precision@K, Recall@K, MRR, NDCG@K).

AUC 以外のレンズで補完性とペア予測の実用価値を測る。各 horizon ごとに以下 2 シナリオ
で評価する:

  (1) Global ranking: 全 (dev, dir) ペアを prob で sort → top-K に正例が何件含まれるか
      → タスク割当の総合的な精度
  (2) Per-dir ranking: 各 dir 内で dev を prob で sort → 上位 K に正例 dev が何人いるか
      → 「次にこの dir をレビューしそうな人」の特定精度 (枯渇予測の前段)

モデル (RULES.md 準拠): cold IRL (irl_<h>), RF (rf_<h>), アンサンブル ((irl+rf)/2)

入力:
  data/pair_trajectories.csv
  outputs/analysis/bot_detection/bot_candidates_main32.csv

出力:
  results/topk_recommendation_metrics.csv
  results/topk_per_dir_metrics.csv
  results/fig_topk_recommendation.pdf
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
REPO_ROOT = ROOT.parents[1]
BOT_CSV = REPO_ROOT / "outputs/analysis/bot_detection/bot_candidates_main32.csv"

HORIZONS = ["0-3m", "3-6m", "6-9m", "9-12m"]
MODELS = {"irl": "cold IRL", "rf": "RF", "ens": "アンサンブル"}
K_VALUES = [10, 25, 50, 100, 200, 500]


def add_ens(pair: pd.DataFrame) -> pd.DataFrame:
    pair = pair.copy()
    for h in HORIZONS:
        pair[f"ens_{h}"] = (pair[f"irl_{h}"] + pair[f"rf_{h}"]) / 2
    return pair


def metrics_global(y: np.ndarray, s: np.ndarray, ks: list[int]) -> dict:
    """Global ranking metrics."""
    order = np.argsort(-s)
    y_sorted = y[order]
    n_pos_total = int(y.sum())
    out = {"n": len(y), "n_pos": n_pos_total}
    # MRR (over positives)
    pos_ranks = np.where(y_sorted == 1)[0] + 1
    out["mrr"] = float((1.0 / pos_ranks).mean()) if len(pos_ranks) > 0 else np.nan
    # nDCG@max(K)
    K_max = max(ks)
    relevance = y_sorted[:K_max].astype(float)
    discounts = 1.0 / np.log2(np.arange(2, K_max + 2))
    dcg = (relevance * discounts).sum()
    ideal = np.zeros(K_max); ideal[:min(K_max, n_pos_total)] = 1
    idcg = (ideal * discounts).sum()
    out["ndcg_max_k"] = float(dcg / idcg) if idcg > 0 else np.nan
    out["ndcg_at_k_value"] = K_max
    for k in ks:
        hits = int(y_sorted[:k].sum())
        out[f"hits_{k}"] = hits
        out[f"hit_rate_{k}"] = float(hits / k)  # = Precision@K
        out[f"recall_{k}"] = float(hits / n_pos_total) if n_pos_total > 0 else np.nan
        # nDCG@K
        relevance_k = y_sorted[:k].astype(float)
        discounts_k = 1.0 / np.log2(np.arange(2, k + 2))
        dcg_k = (relevance_k * discounts_k).sum()
        ideal_k = np.zeros(k); ideal_k[:min(k, n_pos_total)] = 1
        idcg_k = (ideal_k * discounts_k).sum()
        out[f"ndcg_{k}"] = float(dcg_k / idcg_k) if idcg_k > 0 else np.nan
    return out


def per_dir_topk(pair: pd.DataFrame, h: str, model: str,
                 ks: list[int]) -> pd.DataFrame:
    """各 dir 内で dev を prob で sort し、上位 K に正例 dev が何人いるか."""
    rows = []
    for d, g in pair.groupby("directory"):
        y = g[f"label_{h}"].values
        s = g[f"{model}_{h}"].values
        n_pos = int(y.sum())
        if n_pos == 0 or len(y) < 2:
            continue
        order = np.argsort(-s); y_sorted = y[order]
        row = {"directory": d, "n_dev": len(y), "n_pos": n_pos,
               "base_rate": float(y.mean())}
        for k in ks:
            kk = min(k, len(y))
            row[f"hits_{k}"] = int(y_sorted[:kk].sum())
            row[f"recall_{k}"] = float(y_sorted[:kk].sum() / n_pos)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_all(pair: pd.DataFrame, label: str) -> tuple[pd.DataFrame,
                                                          pd.DataFrame]:
    rows = []
    per_dir_rows = []
    print(f"\n========== {label} (n_pair={len(pair):,}) ==========")
    for h in HORIZONS:
        y = pair[f"label_{h}"].values.astype(int)
        base = float(y.mean())
        print(f"\n[{h}] base_rate={base:.3f}")
        for m, m_label in MODELS.items():
            s = pair[f"{m}_{h}"].values
            metrics = metrics_global(y, s, K_VALUES)
            metrics.update({
                "bot_filter": label, "horizon": h,
                "model": m, "model_label": m_label,
                "base_rate": base,
            })
            rows.append(metrics)
            # per-dir
            pd_topk = per_dir_topk(pair, h, m, K_VALUES)
            agg = {f"recall_{k}": float(pd_topk[f"recall_{k}"].mean())
                   for k in K_VALUES}
            agg.update({
                "bot_filter": label, "horizon": h, "model": m,
                "model_label": m_label, "n_dir": len(pd_topk),
                "mean_n_pos": float(pd_topk["n_pos"].mean()),
            })
            per_dir_rows.append(agg)
            # 表示
            print(f"  {m_label}: HR@10={metrics['hit_rate_10']:.3f}, "
                  f"HR@50={metrics['hit_rate_50']:.3f}, "
                  f"HR@100={metrics['hit_rate_100']:.3f}, "
                  f"MRR={metrics['mrr']:.3f}, "
                  f"nDCG@100={metrics['ndcg_100']:.3f}, "
                  f"per-dir recall@5={agg['recall_10']:.3f}/{agg['recall_25']:.3f}")
    return pd.DataFrame(rows), pd.DataFrame(per_dir_rows)


def plot_figure(g_w: pd.DataFrame, g_wo: pd.DataFrame,
                d_w: pd.DataFrame, out_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))

    # (a-d) Global HitRate@K curves per horizon (bot 込み)
    for j, h in enumerate(HORIZONS):
        ax = axes[0, j]
        for m, m_label in MODELS.items():
            sub = g_w[(g_w["horizon"] == h) & (g_w["model"] == m)].iloc[0]
            hrs = [sub[f"hit_rate_{k}"] for k in K_VALUES]
            ax.plot(K_VALUES, hrs, marker="o", lw=1.8, label=m_label)
        ax.axhline(sub["base_rate"], color="gray", linestyle=":",
                   label=f"base={sub['base_rate']:.2f}")
        ax.set_xscale("log")
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels(K_VALUES, fontsize=8)
        ax.set_xlabel("K")
        ax.set_ylabel("HitRate@K (= Precision@K)")
        ax.set_title(f"(a-{j+1}) Global HitRate@K, horizon={h}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    # (e-h) Per-dir mean recall@K
    for j, h in enumerate(HORIZONS):
        ax = axes[1, j]
        for m, m_label in MODELS.items():
            sub = d_w[(d_w["horizon"] == h) & (d_w["model"] == m)].iloc[0]
            rs = [sub[f"recall_{k}"] for k in K_VALUES]
            ax.plot(K_VALUES, rs, marker="s", lw=1.8, label=m_label)
        ax.set_xscale("log")
        ax.set_xticks(K_VALUES)
        ax.set_xticklabels(K_VALUES, fontsize=8)
        ax.set_xlabel("K (per dir 内の上位)")
        ax.set_ylabel("per-dir mean Recall@K")
        ax.set_title(f"(b-{j+1}) Per-dir Recall@K, horizon={h}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)

    fig.suptitle("Top-K 推薦評価: Global (上段) と Per-dir (下段), bot 込み",
                 fontsize=12.5, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def main() -> None:
    print("loading data...")
    pair = pd.read_csv(DATA / "pair_trajectories.csv")
    pair = add_ens(pair)
    bot_df = pd.read_csv(BOT_CSV)
    bots = set(bot_df.loc[bot_df["is_bot"] == True, "email"].tolist())  # noqa: E712

    g_w, d_w = compute_all(pair, "with_bots")
    pair_nb = pair[~pair["developer"].isin(bots)].copy()
    g_wo, d_wo = compute_all(pair_nb, "without_bots")

    OUT.mkdir(exist_ok=True)
    pd.concat([g_w, g_wo], ignore_index=True).to_csv(
        OUT / "topk_recommendation_metrics.csv", index=False)
    pd.concat([d_w, d_wo], ignore_index=True).to_csv(
        OUT / "topk_per_dir_metrics.csv", index=False)
    print(f"\nsaved CSVs to {OUT}")

    # bot 影響
    print("\n========== bot 除外影響 (|delta hit_rate_50|) ==========")
    m = g_w.merge(g_wo, on=["horizon", "model"], suffixes=("_w", "_wo"))
    m["d"] = (m["hit_rate_50_w"] - m["hit_rate_50_wo"]).abs()
    print(f"|delta hit_rate_50| max={m['d'].max():.4f}, "
          f"mean={m['d'].mean():.4f}")

    plot_figure(g_w, g_wo, d_w, OUT / "fig_topk_recommendation.pdf")


if __name__ == "__main__":
    main()
