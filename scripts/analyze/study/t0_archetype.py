# -*- coding: utf-8 -*-
"""t0 掃引用 cold-only アーキタイプ分析 (dev-level + pair-level).

論文のアーキタイプ分析を t0 掃引向けに作り直したもの:
  - dev-level  : archetype_irl_rf.py の cold-only / B定義ラベル版
  - pair-level : archetype_pair_level.py の cold-only / B定義ラベル版

掃引特有の差分 (run_t0_sweep が産む出力構造に整合):
  - IRL は cold のみ (warm-start は掃引で再学習しない) → 2 モデル比較 cold_irl vs rf_dir (+ens)
  - IRL 予測: <irl_base>/train_{h}m/eval_{h}m/pair_predictions.csv (対角, headline と同じ)
  - RF  予測: <rf_base>/fs_{h}_to_m/pair_predictions.csv (別ディレクトリ)
  - (developer, directory) で inner join し B定義(承諾)ラベルを raw から再計算
    ([[blabel-recompute]] / verify_blabel_headline と同一手順) → t0_compare の AUC と直接比較可
  - prob は irl_dir_prob / rf_dir_prob の生値 (headline と同じ。calibrated や閾値は使わない)
  - archetype 入力は feature_end = train_end (= t0 − 12mo) 時点の 1 snapshot

基準 t0=2023-01 は IRL/RF が揃っているので、このスクリプトでそのまま流せる
(スモークテスト兼ねる。全 t0 を同一方法に揃える意味でも基準も本スクリプトで再計算する)。

acc は出さない: cold-only/B定義では焼き込み閾値 (アサイン定義 label で較正) を
B定義ラベルへ適用するのが整合しないため、AUC ベース (per-dev/pair AUC + pooled
bootstrap AUC + Wilcoxon) に統一する。これは論文の IRL-vs-RF 地形比較の主指標と一致。

使い方:
  uv run python scripts/analyze/study/t0_archetype.py \
      --irl-base outputs/dynstartB_ep150/lstm_baseline \
      --rf-base  outputs/dynstart_rf_1snap_s777 \
      --prediction-time 2023-01-01 --train-end 2022-01-01 \
      --output-dir outputs/t0_sweep/20230101/archetype \
      --level both --k-grid 2 3 4 5 6
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from glob import glob
from typing import Dict, List, Tuple

import japanize_matplotlib  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))

from archetype_irl_rf import (  # noqa: E402
    archetypal_analysis,
    build_dev_feature_matrix,
)
from archetype_pair_level import (  # noqa: E402
    build_pair_feature_matrix,
    pooled_auc_with_ci,
)
from verify_blabel_headline import (  # noqa: E402
    FS,
    HZ,
    load_df_with_dirs,
    recompute_blabel,
)
from review_predictor.IRL.features.common_features import STATE_FEATURES  # noqa: E402
from review_predictor.IRL.features.path_features import (  # noqa: E402
    PATH_FEATURE_NAMES,
    PathFeatureExtractor,
)

MODELS = ["cold_irl", "rf_dir", "ens"]
IRL_COL = {"cold_irl": "irl_dir_prob", "rf_dir": "rf_dir_prob"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 予測の読み込み + B定義ラベル付け (headline と同じ inner-join 整合)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_cold_long(
    df_dirs: pd.DataFrame, irl_base: str, rf_base: str, t0, delta: int,
    responded_in_window: bool = False,
) -> pd.DataFrame:
    """全 (developer, directory, horizon, model) の (label[B], prob) を long 形式で。

    IRL(対角) と RF(fs_) を (developer, directory) で inner join し、
    compute_headline と同じ pair 集合・同じ生 prob・同じ B定義ラベルにそろえる。
    responded_in_window=True: label==1(14日以内)の代わりに反応あり(first_response_time notna)で判定。
    """
    t0 = pd.Timestamp(t0)
    rows = []
    for h in HZ:
        irl = pd.read_csv(f"{irl_base}/train_{h}m/eval_{h}m/pair_predictions.csv")
        rf = pd.read_csv(f"{rf_base}/fs_{h.replace('-', '_to_')}m/pair_predictions.csv")
        m = irl[["developer", "directory", "irl_dir_prob"]].merge(
            rf[["developer", "directory", "rf_dir_prob"]],
            on=["developer", "directory"], how="inner",
        )
        dir_devs = recompute_blabel(df_dirs, t0, FS[h], delta, assign=False,
                                    responded_in_window=responded_in_window)
        m["label"] = [
            1 if dev in dir_devs.get(d, set()) else 0
            for dev, d in zip(m["developer"], m["directory"])
        ]
        m["ens"] = (m["irl_dir_prob"] + m["rf_dir_prob"]) / 2.0
        for model in MODELS:
            col = "ens" if model == "ens" else IRL_COL[model]
            d = m[["developer", "directory", "label"]].copy()
            d["prob"] = m[col].to_numpy()
            d["horizon"] = h
            d["model"] = model
            rows.append(d)
    long = pd.concat(rows, ignore_index=True)
    return long


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. per-unit AUC 集約 (dev / pair 共通ロジック)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def per_unit_auc(long: pd.DataFrame, unit_cols: List[str]) -> pd.DataFrame:
    """unit (= dev or (dev,dir)) × model: horizon 横断の per-unit AUC。

    dev-level は (dev, horizon) で max-prob 集約してから 4 点で AUC を取る
    (archetype_irl_rf と同じ)。pair-level は (dev, dir, horizon) が既に 1 点ずつ
    なので horizon 4 点で AUC を取る (archetype_pair_level と同じ)。
    label が全 horizon 同一の unit は AUC=NaN (定義不能)。
    """
    if unit_cols == ["developer"]:
        agg = long.groupby(["developer", "horizon", "model"]).agg(
            prob=("prob", "max"), label=("label", "max"),
        ).reset_index()
    else:
        agg = long.rename(columns={}).copy()
    out = []
    for key, g in agg.groupby(unit_cols + ["model"]):
        labels = g["label"].to_numpy()
        probs = g["prob"].to_numpy()
        if len(np.unique(labels)) >= 2:
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = np.nan
        else:
            auc = np.nan
        rec = dict(zip(unit_cols + ["model"], key if isinstance(key, tuple) else (key,)))
        rec["auc"] = auc
        out.append(rec)
    return pd.DataFrame(out), agg


def auc_pivot(per_unit: pd.DataFrame, unit_cols: List[str]) -> pd.DataFrame:
    piv = per_unit.pivot_table(index=unit_cols, columns="model", values="auc")
    piv.columns = [f"auc_{c}" for c in piv.columns]
    return piv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. archetype 別の集計 (hard mean AUC / pooled bootstrap AUC / Wilcoxon)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hard_archetype_auc(
    alpha_df: pd.DataFrame, dom: np.ndarray, k: int, auc_cols: List[str], n_label: str
) -> pd.DataFrame:
    rows = []
    for j in range(k):
        mask = dom == j
        row = {"archetype": f"A{j+1}", n_label: int(mask.sum())}
        for col in auc_cols:
            vals = alpha_df.loc[mask, col].dropna()
            row[col] = float(vals.mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pooled_bootstrap_auc(
    agg: pd.DataFrame, unit_to_arch: Dict, unit_cols: List[str], k: int, rng
) -> pd.DataFrame:
    """archetype × model の pooled AUC + bootstrap 95%CI (全サンプルをプール)。"""
    a = agg.copy()
    a["archetype"] = a.apply(
        lambda r: unit_to_arch.get(tuple(r[c] for c in unit_cols)), axis=1
    )
    a = a.dropna(subset=["archetype"])
    a["archetype"] = a["archetype"].astype(int)
    rows = []
    for j in range(k):
        for model in MODELS:
            sub = a[(a["archetype"] == j) & (a["model"] == model)][["label", "prob"]].to_numpy()
            if len(sub) < 10 or len(np.unique(sub[:, 0])) < 2:
                auc, lo, hi = np.nan, np.nan, np.nan
            else:
                auc, lo, hi = pooled_auc_with_ci(
                    sub[:, 0].astype(int), sub[:, 1].astype(float), rng, n_boot=1000
                )
            rows.append({
                "archetype": f"A{j+1}", "model": model, "n_samples": int(len(sub)),
                "auc": auc, "auc_ci_lo": lo, "auc_ci_hi": hi,
            })
    return pd.DataFrame(rows)


def wilcoxon_irl_vs_rf(
    alpha_df: pd.DataFrame, dom: np.ndarray, k: int, n_label: str
) -> pd.DataFrame:
    rows = []
    for j in range(k):
        mask = dom == j
        sub = alpha_df.loc[mask, ["auc_cold_irl", "auc_rf_dir"]].dropna()
        n = len(sub)
        if n < 6 or sub["auc_cold_irl"].std() < 1e-9 or sub["auc_rf_dir"].std() < 1e-9:
            p_val, stat, med = np.nan, np.nan, np.nan
        else:
            try:
                res = wilcoxon(sub["auc_cold_irl"], sub["auc_rf_dir"], zero_method="zsplit")
                p_val, stat = float(res.pvalue), float(res.statistic)
                med = float((sub["auc_cold_irl"] - sub["auc_rf_dir"]).median())
            except ValueError:
                p_val, stat, med = np.nan, np.nan, np.nan
        rows.append({
            "archetype": f"A{j+1}", n_label: int(n), "comparison": "cold_irl_vs_rf",
            "median_auc_diff": med, "wilcoxon_stat": stat, "p_value": p_val,
        })
    return pd.DataFrame(rows)


def save_elbow(elbow_rows: List[dict], out_dir: pathlib.Path, unit: str) -> None:
    df = pd.DataFrame(elbow_rows)
    df.to_csv(out_dir / "rss_elbow.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["k"], df["rss_per_unit"], "o-", color="#1f77b4")
    ax.set_xlabel("number of archetypes k")
    ax.set_ylabel(f"RSS / N_{unit}")
    ax.set_title(f"{unit}-level Archetypal Analysis: elbow")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_elbow.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. dev-level
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_dev(
    long: pd.DataFrame, df_dirs: pd.DataFrame, feat_start, feat_end,
    out_dir: pathlib.Path, k_grid: List[int],
) -> Dict[int, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dev] feature window: {feat_start.date()} 〜 {feat_end.date()}  state_dim={len(STATE_FEATURES)}")

    per_dev, agg = per_unit_auc(long, ["developer"])
    piv = auc_pivot(per_dev, ["developer"])
    auc_cols = [c for c in ["auc_cold_irl", "auc_rf_dir", "auc_ens"] if c in piv.columns]

    common_devs = sorted(set(long["developer"]))
    X, kept = build_dev_feature_matrix(df_dirs, common_devs, feat_end, feat_start)
    print(f"[dev] X shape: {X.shape} (devs kept after non-zero filter)")
    piv = piv.reindex(kept)

    elbow_rows, by_k = [], {}
    for k in k_grid:
        Z, alpha, beta, rss = archetypal_analysis(X, k=k, max_iter=120, seed=777)
        elbow_rows.append({"k": k, "rss": rss, "rss_per_unit": rss / len(X)})
        by_k[k] = {"Z": Z, "alpha": alpha}
        print(f"  [dev] k={k}  RSS={rss:.3f}  RSS/dev={rss/len(X):.4f}")
    save_elbow(elbow_rows, out_dir, "dev")

    dom_by_k = {}
    rng = np.random.default_rng(777)
    for k in k_grid:
        Z, alpha = by_k[k]["Z"], by_k[k]["alpha"]
        dom = alpha.argmax(axis=1)
        dom_by_k[k] = dom

        pd.DataFrame(Z, columns=STATE_FEATURES,
                     index=[f"A{j+1}" for j in range(k)]).to_csv(
            out_dir / f"k{k}_archetype_profile.csv")

        dev_alpha = pd.DataFrame(alpha, index=kept,
                                 columns=[f"A{j+1}" for j in range(k)])
        dev_alpha["dominant"] = [f"A{j+1}" for j in dom]
        dev_alpha = dev_alpha.join(piv, how="left")
        dev_alpha.to_csv(out_dir / f"k{k}_dev_alpha_with_metrics.csv")

        hard_archetype_auc(dev_alpha, dom, k, auc_cols, "n_dev").to_csv(
            out_dir / f"k{k}_hard_archetype_metrics.csv", index=False)

        unit_to_arch = {(d,): int(dom[i]) for i, d in enumerate(kept)}
        pooled_bootstrap_auc(agg, unit_to_arch, ["developer"], k, rng).to_csv(
            out_dir / f"k{k}_pooled_auc_bootstrap.csv", index=False)

        wilcoxon_irl_vs_rf(dev_alpha, dom, k, "n_dev_with_auc").to_csv(
            out_dir / f"k{k}_wilcoxon.csv", index=False)

        # 代表 dev (α 上位 5)
        ex = []
        for j in range(k):
            col = f"A{j+1}"
            top5 = dev_alpha.sort_values(col, ascending=False).head(5)
            for rank, (email, r) in enumerate(top5.iterrows(), 1):
                rec = {"archetype": col, "rank": rank, "developer": email,
                       "alpha": float(r[col])}
                for c in auc_cols:
                    rec[c] = float(r[c]) if pd.notna(r[c]) else np.nan
                ex.append(rec)
        pd.DataFrame(ex).to_csv(out_dir / f"k{k}_archetype_top5_devs.csv", index=False)

        if k == 3:
            _plot_simplex_k3(alpha, dev_alpha, out_dir / "fig_k3_simplex.pdf")
        print(f"  [dev] k={k}: 出力済み 分布={[(f'A{j+1}', int((dom==j).sum())) for j in range(k)]}")
    return dom_by_k


def _plot_simplex_k3(alpha: np.ndarray, dev_alpha: pd.DataFrame, out_path: pathlib.Path) -> None:
    a1, a2, a3 = alpha[:, 0], alpha[:, 1], alpha[:, 2]
    s = a1 + a2 + a3
    x = 0.5 * (2 * a2 + a3) / s
    y = (np.sqrt(3) / 2) * a3 / s
    diff = (dev_alpha.get("auc_cold_irl", pd.Series(np.nan, index=dev_alpha.index)).to_numpy()
            - dev_alpha.get("auc_rf_dir", pd.Series(np.nan, index=dev_alpha.index)).to_numpy())
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=diff, cmap="RdBu_r", vmin=-0.3, vmax=0.3, s=22,
                    edgecolor="k", linewidth=0.3)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], "k-", linewidth=1)
    ax.text(-0.03, -0.04, "A1", ha="right")
    ax.text(1.03, -0.04, "A2", ha="left")
    ax.text(0.5, np.sqrt(3)/2 + 0.03, "A3", ha="center")
    ax.set_aspect("equal"); ax.axis("off")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("AUC(cold IRL) − AUC(RF_Dir)\n  + → IRL strong   − → RF strong")
    ax.set_title("Archetype simplex (k=3): IRL vs RF 勝敗分布")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. pair-level
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_pair(
    long: pd.DataFrame, df_dirs: pd.DataFrame, path_extractor: PathFeatureExtractor,
    feat_start, feat_end, out_dir: pathlib.Path, k_grid: List[int], state_only: bool,
) -> Dict[int, np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = list(STATE_FEATURES) if state_only else list(STATE_FEATURES) + list(PATH_FEATURE_NAMES)
    dim_desc = "state-only 26" if state_only else f"{len(feat_cols)} (state 26 + path {len(PATH_FEATURE_NAMES)})"
    print(f"[pair] feature dim: {dim_desc}")

    per_pair, agg = per_unit_auc(long, ["developer", "directory"])
    piv = auc_pivot(per_pair, ["developer", "directory"])
    auc_cols = [c for c in ["auc_cold_irl", "auc_rf_dir", "auc_ens"] if c in piv.columns]

    pairs = sorted(set(zip(per_pair["developer"], per_pair["directory"])))
    print(f"[pair] unique (dev, dir): {len(pairs)}")
    X, kept = build_pair_feature_matrix(
        pairs, df_dirs, path_extractor, feat_start, feat_end, state_only=state_only)
    print(f"[pair] X shape: {X.shape}")
    piv = piv.reindex(kept)

    elbow_rows, by_k = [], {}
    for k in k_grid:
        Z, alpha, beta, rss = archetypal_analysis(X, k=k, max_iter=80, seed=777)
        elbow_rows.append({"k": k, "rss": rss, "rss_per_unit": rss / len(X)})
        by_k[k] = {"Z": Z, "alpha": alpha}
        print(f"  [pair] k={k}  RSS={rss:.2f}  RSS/N={rss/len(X):.4f}")
    save_elbow(elbow_rows, out_dir, "pair")

    dom_by_k = {}
    rng = np.random.default_rng(777)
    for k in k_grid:
        Z, alpha = by_k[k]["Z"], by_k[k]["alpha"]
        dom = alpha.argmax(axis=1)
        dom_by_k[k] = dom

        pd.DataFrame(Z, columns=feat_cols,
                     index=[f"A{j+1}" for j in range(k)]).to_csv(
            out_dir / f"k{k}_archetype_profile.csv")

        pa = pd.DataFrame(alpha, columns=[f"A{j+1}" for j in range(k)])
        pa["developer"] = [d for d, _ in kept]
        pa["directory"] = [p for _, p in kept]
        pa["dominant"] = [f"A{j+1}" for j in dom]
        pa = pa.merge(piv.reset_index(), on=["developer", "directory"], how="left")
        pa.to_csv(out_dir / f"k{k}_pair_alpha_with_metrics.csv", index=False)

        hard_archetype_auc(pa, dom, k, auc_cols, "n_pair").to_csv(
            out_dir / f"k{k}_hard_metrics.csv", index=False)

        unit_to_arch = {(d, p): int(dom[i]) for i, (d, p) in enumerate(kept)}
        pooled_bootstrap_auc(agg, unit_to_arch, ["developer", "directory"], k, rng).to_csv(
            out_dir / f"k{k}_pooled_auc.csv", index=False)

        wilcoxon_irl_vs_rf(pa, dom, k, "n_pair_with_auc").to_csv(
            out_dir / f"k{k}_wilcoxon.csv", index=False)
        print(f"  [pair] k={k}: 出力済み 分布={[(f'A{j+1}', int((dom==j).sum())) for j in range(k)]}")
    return dom_by_k


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--irl-base", required=True,
                   help="IRL 出力ベース (例 outputs/t0_sweep/20221001/irl/lstm_baseline)")
    p.add_argument("--rf-base", required=True,
                   help="RF 出力ベース (例 outputs/t0_sweep/20221001/rf)")
    p.add_argument("--prediction-time", required=True, help="t0 (YYYY-MM-DD)")
    p.add_argument("--train-end", required=True,
                   help="archetype 特徴量の feature_end (= t0 − 12mo)")
    p.add_argument("--output-dir", type=pathlib.Path, required=True)
    p.add_argument("--level", choices=["both", "dev", "pair"], default="both")
    p.add_argument("--k-grid", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--delta-months", type=int, default=3)
    p.add_argument("--feature-window-days", type=int, default=180)
    p.add_argument("--state-only", action="store_true",
                   help="pair-level で path_features を抜き state 26 のみ (ablation)")
    p.add_argument("--responded-in-window", action="store_true",
                   help="B定義ラベルを label==1(14日以内) の代わりに first_response_time notna で計算")
    p.add_argument("--data", default="data/combined_raw_main32.csv")
    p.add_argument("--raw-json-list", default="data/combined_raw_main32.raw_json_list.txt")
    p.add_argument("--raw-json-glob", default="data/raw_json/openstack__*.json")
    p.add_argument("--hub-scores",
                   default="experiments/dependency_analysis/results/hub_scores_main32.csv")
    p.add_argument("--cochange-neighbors",
                   default="experiments/dependency_analysis/results/cochange_neighbors_main32.csv")
    args = p.parse_args()

    feat_end = pd.Timestamp(args.train_end)
    feat_start = feat_end - pd.Timedelta(days=args.feature_window_days)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] raw + dirs (t0={args.prediction_time}) ...")
    df_dirs = load_df_with_dirs(args.data, args.raw_json_list)
    # build_dev/pair_feature_matrix は timestamp が datetime であることを要求
    df_dirs["timestamp"] = pd.to_datetime(df_dirs["timestamp"])

    print("[load] cold predictions + B定義ラベル付け ...")
    long = build_cold_long(df_dirs, args.irl_base, args.rf_base,
                           args.prediction_time, args.delta_months,
                           responded_in_window=args.responded_in_window)
    print(f"[load] long rows: {len(long)}  (models={MODELS})")

    if args.level in ("both", "dev"):
        run_dev(long, df_dirs, feat_start, feat_end,
                args.output_dir / "dev", args.k_grid)

    if args.level in ("both", "pair"):
        hub = pathlib.Path(args.hub_scores)
        coch = pathlib.Path(args.cochange_neighbors)
        path_extractor = PathFeatureExtractor(
            df_dirs, window_days=args.feature_window_days,
            hub_scores_path=str(hub) if hub.exists() else None,
            cochange_neighbors_path=str(coch) if coch.exists() else None,
        )
        run_pair(long, df_dirs, path_extractor, feat_start, feat_end,
                 args.output_dir / "pair", args.k_grid, args.state_only)

    print(f"\n[done] outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
