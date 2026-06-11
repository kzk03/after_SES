# -*- coding: utf-8 -*-
"""ディレクトリ × 予測区間遷移の「危険セル」マップ — 期間別予測のメリット定量化.

§12 では「IRL が当てやすい開発者タイプ」を 16 アーキタイプセルで示したが、
「**4 区間別に予測することのメリット**」は示せていなかった。
本節は (directory × 予測区間遷移) を「危険セル」と捉え、

  - その期間にレビュアー脱落 (label=1→0) が集中する dir × 遷移 を同定
  - その危険セルで IRL/RF/Ens の予測が「脱落者を事前に低スコアにできるか」を測る
  - 期間別予測でなければ言えない「○月から○月の間にこの dir が危ない」が当たることを示す

定義:
  - 遷移: (0-3m → 3-6m), (3-6m → 6-9m), (6-9m → 9-12m) の 3 つ
  - attrition rate = P(label_curr=0 | label_prev=1) = 過去期間で承諾していた人が次期間で承諾しなくなった割合
  - n_active_prev = label_prev=1 のペア数 (この部分集合で予測を評価)
  - per-cell AUC = roc_auc(label_curr, prob_curr) — active_prev 内で
  - recall@K の attrited = prob_curr 昇順で K 件取った時の脱落者捕捉率
                        (K = n_attrited で precision = recall)

ディレクトリは全 dir 対象 (MIN_N フィルタなし、ノイズは表示時に切る)。

入力:
  data/pair_trajectories.csv

出力:
  results/horizon_dir_attrition.csv               全 (dir × 遷移) の attrition + AUC + recall@K
  results/horizon_dir_attrition_top_cases.csv     危険セル上位の dev 例
  results/fig_horizon_dir_attrition_heatmap.pdf   メイン: 上位 dir の attrition と Ens AUC
  results/fig_horizon_dir_attrition_top_cases.pdf 上位 5 危険セルの recall@K 曲線
  results/fig_horizon_dir_attrition_scatter.pdf   attrition rate vs Ens AUC 散布図
"""
from __future__ import annotations

import pathlib

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"

TRANSITIONS = [("0-3m", "3-6m"), ("3-6m", "6-9m"), ("6-9m", "9-12m")]
TRANSITION_LABEL_JP = {
    ("0-3m", "3-6m"): "0-3m→3-6m",
    ("3-6m", "6-9m"): "3-6m→6-9m",
    ("6-9m", "9-12m"): "6-9m→9-12m",
}
MODELS = ["irl", "rf", "ens"]
DISPLAY_MIN_N_PREV = 10  # 表示時の安定性線 (集計には影響しない)


def safe_auc(y, s):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def per_cell_stats(df: pd.DataFrame) -> pd.DataFrame:
    """全 directory × 全 遷移 で集計 (フィルタなし)."""
    df = df.copy()
    if "avg_0-3m" in df.columns:
        # 既存の avg_{h} 列が ensemble (= (irl+rf)/2)
        pass
    rows = []
    for (h_prev, h_curr) in TRANSITIONS:
        sub_all = df.dropna(subset=[f"label_{h_prev}", f"label_{h_curr}"])
        for d, gr in sub_all.groupby("directory"):
            active = gr[gr[f"label_{h_prev}"] == 1]
            n_prev = len(active)
            if n_prev == 0:
                continue
            attrited_mask = active[f"label_{h_curr}"] == 0
            n_attrited = int(attrited_mask.sum())
            attrition_rate = n_attrited / n_prev
            row = {
                "directory": d, "h_prev": h_prev, "h_curr": h_curr,
                "n_active_prev": n_prev, "n_attrited": n_attrited,
                "attrition_rate": attrition_rate,
            }
            # AUC for predicting label_curr in active_prev 部分集合
            y = active[f"label_{h_curr}"].astype(int).values
            for m in MODELS:
                col = f"avg_{h_curr}" if m == "ens" else f"{m}_{h_curr}"
                s = active[col].values
                row[f"{m}_auc"] = safe_auc(y, s)
            # recall@K of attrited at K = n_attrited
            if n_attrited > 0 and n_attrited < n_prev:
                K = n_attrited
                for m in MODELS:
                    col = f"avg_{h_curr}" if m == "ens" else f"{m}_{h_curr}"
                    order = np.argsort(active[col].values)  # 昇順 (低 prob から)
                    top_K_idx = order[:K]
                    n_hit = int(attrited_mask.values[top_K_idx].sum())
                    row[f"{m}_recall_at_attrited"] = n_hit / K
                    # baseline = K / n_prev (ランダムに K 件取った時の attrited 比率)
                row["random_baseline_recall"] = K / n_prev * (n_attrited / n_prev) * n_prev / K
                # 整理: random_baseline_recall = n_attrited / n_prev (= attrition_rate)
                # K 件取って attrited を含む期待値 = K * attrition_rate, recall = K*attrition_rate / K = attrition_rate
                row["random_baseline_recall"] = attrition_rate
            else:
                for m in MODELS:
                    row[f"{m}_recall_at_attrited"] = np.nan
                row["random_baseline_recall"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap_top_dirs(stats: pd.DataFrame, out_path: pathlib.Path,
                          top_k: int = 25) -> None:
    """attrition rate 上位 dir (n_active_prev>=DISPLAY_MIN_N_PREV) で 2 枚並べる."""
    sub = stats[stats["n_active_prev"] >= DISPLAY_MIN_N_PREV].copy()
    # dir ごとの最大 attrition rate で並べ替え
    dir_max = sub.groupby("directory")["attrition_rate"].max().sort_values(ascending=False)
    top_dirs = dir_max.head(top_k).index.tolist()

    attr_mat = np.full((len(top_dirs), 3), np.nan)
    auc_mat = np.full((len(top_dirs), 3), np.nan)
    n_mat = np.full((len(top_dirs), 3), np.nan)
    for i, d in enumerate(top_dirs):
        for j, (h_prev, h_curr) in enumerate(TRANSITIONS):
            row = sub[(sub["directory"] == d) & (sub["h_prev"] == h_prev)
                      & (sub["h_curr"] == h_curr)]
            if not row.empty:
                attr_mat[i, j] = row["attrition_rate"].values[0]
                auc_mat[i, j] = row["ens_auc"].values[0]
                n_mat[i, j] = row["n_active_prev"].values[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, 0.32 * len(top_dirs))))

    for ax, mat, cmap, vmin, vmax, title, fmt in [
        (axes[0], attr_mat, "Reds", 0, 1, "attrition rate\n(承諾→拒否 切替割合)", "{:.0%}"),
        (axes[1], auc_mat, "RdYlGn", 0.3, 1.0,
         "Ens AUC\n(active_prev 内で label_curr を予測)", "{:.2f}"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(3))
        ax.set_xticklabels([TRANSITION_LABEL_JP[t] for t in TRANSITIONS], fontsize=10)
        ax.set_yticks(range(len(top_dirs)))
        ax.set_yticklabels(top_dirs, fontsize=8)
        ax.set_title(title, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=7, color="gray")
                else:
                    n = int(n_mat[i, j]) if not np.isnan(n_mat[i, j]) else 0
                    txt = f"{fmt.format(mat[i, j])}\nn={n}"
                    color = "white" if (mat[i, j] > 0.7 if cmap == "Reds"
                                        else mat[i, j] < 0.5) else "black"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7, color=color)
    fig.suptitle(f"ディレクトリ × 予測区間遷移の「危険セル」マップ "
                 f"(上位 {top_k} dir、n_active_prev≥{DISPLAY_MIN_N_PREV} で表示絞り)",
                 fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"saved: {out_path}")


def plot_scatter_attrition_vs_auc(stats: pd.DataFrame, out_path: pathlib.Path) -> None:
    """attrition rate vs Ens AUC (危険セルが予測できているか)."""
    sub = stats.dropna(subset=["ens_auc", "attrition_rate"]).copy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (h_prev, h_curr) in zip(axes, TRANSITIONS):
        s = sub[(sub["h_prev"] == h_prev) & (sub["h_curr"] == h_curr)]
        sz = np.clip(s["n_active_prev"].values * 3, 10, 300)
        # n_active_prev 別に色分け
        is_reliable = s["n_active_prev"] >= DISPLAY_MIN_N_PREV
        ax.scatter(s.loc[~is_reliable, "attrition_rate"], s.loc[~is_reliable, "ens_auc"],
                   s=sz[~is_reliable.values], alpha=0.3, color="lightgray",
                   label=f"n<{DISPLAY_MIN_N_PREV} (参考)")
        ax.scatter(s.loc[is_reliable, "attrition_rate"], s.loc[is_reliable, "ens_auc"],
                   s=sz[is_reliable.values], alpha=0.6, color="steelblue",
                   label=f"n≥{DISPLAY_MIN_N_PREV}")
        # 危険セル: attrition>=0.5 かつ n>=DISPLAY_MIN_N_PREV
        danger = s[(s["attrition_rate"] >= 0.5) & is_reliable]
        ax.scatter(danger["attrition_rate"], danger["ens_auc"],
                   s=np.clip(danger["n_active_prev"].values * 3, 10, 300),
                   facecolors="none", edgecolors="red", linewidth=1.5,
                   label="危険セル (attrition≥0.5)")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlabel("attrition rate", fontsize=10)
        ax.set_ylabel("Ens AUC (active_prev 内)", fontsize=10)
        ax.set_title(TRANSITION_LABEL_JP[(h_prev, h_curr)], fontsize=11)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("予測区間遷移ごとの attrition rate vs Ens AUC "
                 "(右上 = 危険セルを予測できている = 期間別予測のメリット)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"saved: {out_path}")


def extract_top_cases(df_pair: pd.DataFrame, stats: pd.DataFrame,
                       top_k: int = 5, min_n_prev: int = 10) -> pd.DataFrame:
    """attrition 上位の (dir × 遷移) で attrited 開発者を例示."""
    sub = stats[stats["n_active_prev"] >= min_n_prev].copy()
    sub = sub.sort_values("attrition_rate", ascending=False).head(top_k)
    rows = []
    for _, cell in sub.iterrows():
        d = cell["directory"]
        h_prev, h_curr = cell["h_prev"], cell["h_curr"]
        active = df_pair[
            (df_pair["directory"] == d) & (df_pair[f"label_{h_prev}"] == 1)
        ].copy()
        # prob_curr 昇順 (IRL, RF, Ens) で順位を付ける
        for m, col in [("irl", f"irl_{h_curr}"), ("rf", f"rf_{h_curr}"),
                       ("ens", f"avg_{h_curr}")]:
            active[f"{m}_rank"] = active[col].rank(method="min", ascending=True)
        active["attrited"] = (active[f"label_{h_curr}"] == 0).astype(int)
        K = int(cell["n_attrited"])
        for _, dev in active.iterrows():
            rows.append({
                "directory": d, "h_prev": h_prev, "h_curr": h_curr,
                "attrition_rate": cell["attrition_rate"],
                "n_active_prev": cell["n_active_prev"],
                "n_attrited": K,
                "developer": dev["developer"],
                "label_0-3m": int(dev["label_0-3m"]) if pd.notna(dev["label_0-3m"]) else None,
                "label_3-6m": int(dev["label_3-6m"]) if pd.notna(dev["label_3-6m"]) else None,
                "label_6-9m": int(dev["label_6-9m"]) if pd.notna(dev["label_6-9m"]) else None,
                "label_9-12m": int(dev["label_9-12m"]) if pd.notna(dev["label_9-12m"]) else None,
                "attrited": int(dev["attrited"]),
                "irl_prob_curr": float(dev[f"irl_{h_curr}"]),
                "rf_prob_curr": float(dev[f"rf_{h_curr}"]),
                "ens_prob_curr": float(dev[f"avg_{h_curr}"]),
                "irl_rank": int(dev["irl_rank"]),
                "rf_rank": int(dev["rf_rank"]),
                "ens_rank": int(dev["ens_rank"]),
                "K_budget": K,
                "ens_in_topK": int(dev["ens_rank"] <= K),
                "irl_in_topK": int(dev["irl_rank"] <= K),
                "rf_in_topK": int(dev["rf_rank"] <= K),
            })
    return pd.DataFrame(rows)


def plot_recall_curves(df_pair: pd.DataFrame, stats: pd.DataFrame,
                       out_path: pathlib.Path, top_k: int = 5,
                       min_n_prev: int = 10) -> None:
    """上位 5 危険セルで recall@K 曲線を IRL/RF/Ens で重ねる."""
    sub = stats[stats["n_active_prev"] >= min_n_prev].copy()
    sub = sub.sort_values("attrition_rate", ascending=False).head(top_k)
    fig, axes = plt.subplots(1, len(sub), figsize=(4.2 * len(sub), 4.2))
    if len(sub) == 1:
        axes = [axes]
    for ax, (_, cell) in zip(axes, sub.iterrows()):
        d = cell["directory"]
        h_prev, h_curr = cell["h_prev"], cell["h_curr"]
        active = df_pair[
            (df_pair["directory"] == d) & (df_pair[f"label_{h_prev}"] == 1)
        ].copy()
        attrited_mask = (active[f"label_{h_curr}"] == 0).values.astype(int)
        n_prev = len(active)
        n_attr = int(cell["n_attrited"])
        K_range = np.arange(1, n_prev + 1)
        for m, col, color in [("IRL", f"irl_{h_curr}", "tab:blue"),
                              ("RF", f"rf_{h_curr}", "tab:red"),
                              ("Ens", f"avg_{h_curr}", "tab:green")]:
            order = np.argsort(active[col].values)  # 昇順
            sorted_attr = attrited_mask[order]
            cum_hits = np.cumsum(sorted_attr)
            recall = cum_hits / n_attr if n_attr > 0 else np.zeros_like(cum_hits)
            ax.plot(K_range / n_prev, recall, label=m, color=color, linewidth=1.5)
        # ランダム基準
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1,
                label="ランダム")
        # K = n_attrited の縦線
        ax.axvline(n_attr / n_prev, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(f"{d}\n{TRANSITION_LABEL_JP[(h_prev, h_curr)]}\n"
                     f"attr={cell['attrition_rate']:.0%}, "
                     f"n={n_prev} ({n_attr} attrited)",
                     fontsize=9)
        ax.set_xlabel("K / n_active_prev", fontsize=9)
        ax.set_ylabel("recall of attrited", fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"上位 {top_k} 危険セルでの脱落者捕捉率 (低 prob 側から取る)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"saved: {out_path}")


def main() -> None:
    OUT.mkdir(exist_ok=True, parents=True)
    df = pd.read_csv(DATA / "pair_trajectories.csv")
    print(f"pairs: {len(df)}, dirs: {df['directory'].nunique()}")

    stats = per_cell_stats(df)
    stats = stats.sort_values(["h_prev", "attrition_rate"], ascending=[True, False])
    stats.to_csv(OUT / "horizon_dir_attrition.csv", index=False)
    print(f"saved: {OUT/'horizon_dir_attrition.csv'} ({len(stats)} rows)")

    # 集計サマリ
    print("\n=== 集計 (全 dir, 遷移別) ===")
    for (h_prev, h_curr) in TRANSITIONS:
        sub = stats[(stats["h_prev"] == h_prev) & (stats["h_curr"] == h_curr)]
        # 重み付き median (n_active_prev で)
        print(f"  {h_prev}→{h_curr}: dirs={len(sub)}, "
              f"median attrition_rate (全 dir)={sub['attrition_rate'].median():.2%}, "
              f"重み付き (n_active_prev) attrition (cell の sum)="
              f"{sub['n_attrited'].sum() / sub['n_active_prev'].sum():.2%}")

    # n_active_prev>=10 の安定 dir に絞った場合の attrition の中央値
    print("\n=== n_active_prev>=10 で絞った場合 ===")
    rel = stats[stats["n_active_prev"] >= DISPLAY_MIN_N_PREV]
    for (h_prev, h_curr) in TRANSITIONS:
        sub = rel[(rel["h_prev"] == h_prev) & (rel["h_curr"] == h_curr)]
        print(f"  {h_prev}→{h_curr}: dirs={len(sub)}, "
              f"median attrition_rate={sub['attrition_rate'].median():.2%}, "
              f"median Ens AUC={sub['ens_auc'].median():.3f}, "
              f"median IRL AUC={sub['irl_auc'].median():.3f}, "
              f"median RF AUC={sub['rf_auc'].median():.3f}")

    # 危険セル top 10 (n>=10)
    print("\n=== 危険セル top 10 (n_active_prev>=10) ===")
    rel = stats[stats["n_active_prev"] >= DISPLAY_MIN_N_PREV].copy()
    top10 = rel.sort_values("attrition_rate", ascending=False).head(10)
    print(top10[["directory", "h_prev", "h_curr", "n_active_prev", "n_attrited",
                 "attrition_rate", "irl_auc", "rf_auc", "ens_auc",
                 "irl_recall_at_attrited", "rf_recall_at_attrited",
                 "ens_recall_at_attrited", "random_baseline_recall"]].to_string(index=False))

    # 小規模 dir の危険セル top 10 (n<10)
    print("\n=== 小規模 dir (n_active_prev<10) の危険セル top 10 ===")
    small = stats[stats["n_active_prev"] < DISPLAY_MIN_N_PREV].copy()
    small_top10 = small.sort_values("attrition_rate", ascending=False).head(10)
    print(small_top10[["directory", "h_prev", "h_curr", "n_active_prev",
                       "n_attrited", "attrition_rate"]].to_string(index=False))

    # 図
    plot_heatmap_top_dirs(stats, OUT / "fig_horizon_dir_attrition_heatmap.pdf",
                          top_k=25)
    plot_scatter_attrition_vs_auc(stats,
                                  OUT / "fig_horizon_dir_attrition_scatter.pdf")

    # 危険セル top 5 の dev 例
    cases = extract_top_cases(df, stats, top_k=5, min_n_prev=10)
    cases.to_csv(OUT / "horizon_dir_attrition_top_cases.csv", index=False)
    print(f"\nsaved: {OUT/'horizon_dir_attrition_top_cases.csv'}")

    plot_recall_curves(df, stats,
                       OUT / "fig_horizon_dir_attrition_top_cases.pdf",
                       top_k=5, min_n_prev=10)


if __name__ == "__main__":
    main()
