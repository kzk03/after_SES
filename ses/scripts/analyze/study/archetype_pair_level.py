# -*- coding: utf-8 -*-
"""Pair-level (dev × dir) Archetypal Analysis — B-28/29 の dev-level 版を pair 粒度に拡張。

dev-level (B-28) との違い:
  - 1 dev = 1 archetype 帰属 → 1 (dev, dir) ペア = 1 archetype 帰属
  - 入力次元: state 26 + path 6 = 32
  - サンプル数: 4534 pair (dev-level 289 の 16 倍弱)
  - AUC 推定が安定 (per-pair 4 sample → bootstrap CI が狭くなる)

期待される追加知見:
  1. 同じ dev でも dir によって IRL/RF 優位が変わるか (dir 特性が分離されるか)
  2. dev-level archetype A2/A4 の per-dev AUC 圧勝が pair-level でも頑健か
  3. path_features (hub_score, owner_overlap, neighbor_coverage) が archetype 分離に寄与するか

入力:
  - outputs/main32_mce_v2_phase2/lstm_baseline/train_0-3m/eval_*/pair_predictions.csv (cold + RF)
  - outputs/main32_mce_kd_warmstart_v2/lstm_baseline/train_0-3m/eval_*/pair_predictions.csv (warm)
  - data/combined_raw_main32.csv
  - data/raw_json/openstack__*.json (change_dir_map)
  - experiments/dependency_analysis/results/hub_scores_main32.csv
  - experiments/dependency_analysis/results/cochange_neighbors_main32.csv

出力:
  outputs/analysis/archetype_pair_level_v21/

使い方:
  uv run python scripts/analyze/study/archetype_pair_level.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from glob import glob
from typing import Dict, List, Tuple

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "src"))
from review_predictor.IRL.features.common_features import (  # noqa: E402
    STATE_FEATURES,
    extract_common_features,
    normalize_features,
)
from review_predictor.IRL.features.path_features import (  # noqa: E402
    PATH_FEATURE_NAMES,
    PathFeatureExtractor,
    attach_dirs_to_df,
    load_change_dir_map_multi,
)

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]
RANGE_NEG1_POS1 = {"activity_trend", "response_time_trend", "accept_rate_drift_30d"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Archetypal ALS (B-28 と同じ自前実装)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _solve_convex_lsq(A, b):
    lam = 1e3
    A_aug = np.vstack([A, lam * np.ones((1, A.shape[1]))])
    b_aug = np.concatenate([b, [lam]])
    x, _ = nnls(A_aug, b_aug)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)


def archetypal_analysis(X, k, max_iter=80, tol=1e-5, seed=777):
    n, m = X.shape
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=k, replace=False)
    beta = np.zeros((k, n))
    for j, i in enumerate(idx):
        beta[j, i] = 1.0
    Z = beta @ X
    prev_rss = np.inf
    for it in range(max_iter):
        alpha = np.zeros((n, k))
        for i in range(n):
            alpha[i] = _solve_convex_lsq(Z.T, X[i])
        try:
            Z_target = np.linalg.solve(alpha.T @ alpha + 1e-8 * np.eye(k), alpha.T @ X)
        except np.linalg.LinAlgError:
            Z_target = np.linalg.lstsq(alpha, X, rcond=None)[0]
        beta = np.zeros((k, n))
        for j in range(k):
            beta[j] = _solve_convex_lsq(X.T, Z_target[j])
        Z = beta @ X
        rss = float(np.sum((X - alpha @ Z) ** 2))
        if abs(prev_rss - rss) / max(prev_rss, 1e-10) < tol:
            break
        prev_rss = rss
    return Z, alpha, beta, rss


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# データロード & feature
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_pair_long(cold_dir, warm_dir):
    """全 (dev, dir, window, model) の (label, prob) を long 形式で。"""
    rows = []
    for w in PERIODS:
        cold = pd.read_csv(cold_dir / f"lstm_baseline/train_0-3m/eval_{w}/pair_predictions.csv")
        warm = pd.read_csv(warm_dir / f"lstm_baseline/train_0-3m/eval_{w}/pair_predictions.csv")
        for df, model, prob_col in [
            (cold, "cold_irl", "irl_dir_prob_calibrated"),
            (warm, "warm_irl", "irl_dir_prob_calibrated"),
            (warm, "rf_dir", "rf_dir_prob"),
        ]:
            d = df[["developer", "directory", "label", prob_col]].copy()
            d.rename(columns={prob_col: "prob"}, inplace=True)
            d["window"] = w
            d["model"] = model
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def build_pair_feature_matrix(pairs, df_raw, path_extractor, feat_start, feat_end, state_only=False):
    """pair (dev, dir) × 32 dim normalized 特徴量。

    dev の state 26 + path 6 を結合。同 dev の複数 pair は state 共通、path 異なる。
    """
    # dev ごとの state を 1 回だけ計算してキャッシュ
    dev_set = sorted(set(p[0] for p in pairs))
    state_cache = {}
    n_dev_active = 0
    for email in dev_set:
        # 履歴フィルタ (active dev only)
        rev_mask = (
            (df_raw["email"] == email)
            & (df_raw["timestamp"] >= feat_start)
            & (df_raw["timestamp"] < feat_end)
        )
        if not bool(rev_mask.any()):
            continue
        feats = extract_common_features(
            df_raw, email, feat_start.to_pydatetime(), feat_end.to_pydatetime(),
            normalize=True,
        )
        feats = normalize_features(feats)
        vec = []
        for k in STATE_FEATURES:
            v = float(feats.get(k, 0.0))
            if k in RANGE_NEG1_POS1:
                v = (v + 1.0) / 2.0
            vec.append(v)
        state_cache[email] = np.asarray(vec, dtype=float)
        n_dev_active += 1
    print(f"  active devs (with history in window): {n_dev_active}/{len(dev_set)}")

    # pair × 32 を構築
    rows, kept_pairs = [], []
    feat_end_dt = feat_end.to_pydatetime()
    n_path_feats = len(PATH_FEATURE_NAMES)
    for dev, dir_path in pairs:
        if dev not in state_cache:
            continue
        state_vec = state_cache[dev]
        if state_only:
            full = state_vec
        else:
            # path features
            try:
                path_vec = path_extractor.compute(
                    dev, frozenset({dir_path}), feat_end_dt
                )
                path_vec = np.asarray(path_vec, dtype=float)
            except Exception:
                path_vec = np.zeros(n_path_feats, dtype=float)
            # 既に各 path 特徴量は extractor 内で 0-1 化されている前提
            full = np.concatenate([state_vec, path_vec])
        rows.append(full)
        kept_pairs.append((dev, dir_path))
    X = np.asarray(rows, dtype=float)
    return X, kept_pairs


def aggregate_per_pair(long):
    """per-pair × model: 4 窓平均 accuracy + (label 変動なら) AUC。"""
    out = []
    for (dev, dir_p, model), grp in long.groupby(["developer", "directory", "model"]):
        acc = grp["correct"].mean() if "correct" in grp.columns else np.nan
        labels = grp["label"].to_numpy()
        probs = grp["prob"].to_numpy()
        if len(np.unique(labels)) >= 2:
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = np.nan
        else:
            auc = np.nan
        out.append({"developer": dev, "directory": dir_p, "model": model,
                     "acc": acc, "auc": auc})
    return pd.DataFrame(out)


def pooled_auc_with_ci(labels, probs, rng, n_boot=500):
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return np.nan, np.nan, np.nan
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        return np.nan, np.nan, np.nan
    boots = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bl = labels[idx]
        if len(np.unique(bl)) < 2:
            continue
        try:
            boots.append(roc_auc_score(bl, probs[idx]))
        except ValueError:
            continue
    lo = float(np.percentile(boots, 2.5)) if len(boots) > 50 else np.nan
    hi = float(np.percentile(boots, 97.5)) if len(boots) > 50 else np.nan
    return auc, lo, hi


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cold-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_v2_phase2"))
    p.add_argument("--warm-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_kd_warmstart_v2"))
    p.add_argument("--raw-csv", type=pathlib.Path,
                   default=pathlib.Path("data/combined_raw_main32.csv"))
    p.add_argument("--raw-json-glob",
                   default="data/raw_json/openstack__*.json")
    p.add_argument("--hub-scores", type=pathlib.Path,
                   default=pathlib.Path("experiments/dependency_analysis/results/hub_scores_main32.csv"))
    p.add_argument("--cochange-neighbors", type=pathlib.Path,
                   default=pathlib.Path("experiments/dependency_analysis/results/cochange_neighbors_main32.csv"))
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_pair_level_v21"))
    p.add_argument("--train-end", default="2022-01-01")
    p.add_argument("--feature-window-days", type=int, default=180)
    p.add_argument("--k-grid", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--state-only", action="store_true",
                   help="path_features を抜いて state 26 のみで archetype を計算 (ablation)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feat_end = pd.Timestamp(args.train_end)
    feat_start = feat_end - pd.Timedelta(days=args.feature_window_days)
    print(f"[setup] feature window: {feat_start.date()} 〜 {feat_end.date()}")

    # ── 1. pair-level predictions ──
    print("[load] pair_predictions × 4 windows × 3 models ...")
    long = load_pair_long(args.cold_dir, args.warm_dir)
    # 閾値で 2 値化 + correct 計算 (window ごとの threshold を取得)
    thr_cache = {}
    for w in PERIODS:
        for base, model_thr_key in [
            (args.cold_dir, ("cold_irl", "IRL_Dir_calibrated")),
            (args.warm_dir, ("warm_irl", "IRL_Dir_calibrated")),
            (args.warm_dir, ("rf_dir", "RF_Dir")),
        ]:
            f = base / f"lstm_baseline/train_0-3m/eval_{w}/summary_metrics.json"
            if f.exists():
                d = json.load(f.open())
                key = model_thr_key[1]
                if key in d:
                    thr_cache[(model_thr_key[0], w)] = float(d[key]["clf_threshold"])
    long["thr"] = long.apply(lambda r: thr_cache.get((r["model"], r["window"]), 0.5), axis=1)
    long["pred"] = (long["prob"] >= long["thr"]).astype(int)
    long["correct"] = (long["pred"] == long["label"]).astype(int)

    per_pair = aggregate_per_pair(long)
    pairs = sorted(set(zip(per_pair["developer"], per_pair["directory"])))
    print(f"[pairs] unique (dev, dir): {len(pairs)}")

    # ── 2. raw CSV + change_dir_map + path_extractor ──
    print(f"[features] loading raw csv: {args.raw_csv}")
    df_raw = pd.read_csv(args.raw_csv, parse_dates=["timestamp"])
    if "first_response_time" in df_raw.columns:
        df_raw["first_response_time"] = pd.to_datetime(df_raw["first_response_time"], errors="coerce")

    raw_json_files = sorted(glob(args.raw_json_glob))
    print(f"[features] loading {len(raw_json_files)} raw_json files for change_dir_map ...")
    cdm = load_change_dir_map_multi(raw_json_files, depth=2)
    df_raw = attach_dirs_to_df(df_raw, cdm)

    print(f"[features] PathFeatureExtractor (hub + neighbors) ...")
    path_extractor = PathFeatureExtractor(
        df_raw, window_days=args.feature_window_days,
        hub_scores_path=str(args.hub_scores) if args.hub_scores.exists() else None,
        cochange_neighbors_path=str(args.cochange_neighbors) if args.cochange_neighbors.exists() else None,
    )

    dim_desc = "state-only 26" if args.state_only else "32 (state 26 + path 6)"
    print(f"[features] building pair feature matrix ({len(pairs)} pairs × {dim_desc}) ...")
    X, kept_pairs = build_pair_feature_matrix(
        pairs, df_raw, path_extractor, feat_start, feat_end,
        state_only=args.state_only,
    )
    print(f"  X shape: {X.shape}")

    # ── 3. archetype ──
    elbow_rows, archetypes_by_k = [], {}
    for k in args.k_grid:
        print(f"  k={k} ...", end=" ", flush=True)
        Z, alpha, beta, rss = archetypal_analysis(X, k=k, max_iter=60, seed=777)
        elbow_rows.append({"k": k, "rss": rss, "rss_per_pair": rss / len(X)})
        archetypes_by_k[k] = {"Z": Z, "alpha": alpha, "rss": rss}
        print(f"RSS={rss:.2f}, RSS/N={rss/len(X):.4f}")
    elbow_df = pd.DataFrame(elbow_rows)
    elbow_df.to_csv(args.output_dir / "rss_elbow.csv", index=False)

    # elbow plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(elbow_df["k"], elbow_df["rss_per_pair"], "o-", color="#1f77b4")
    ax.set_xlabel("number of archetypes k")
    ax.set_ylabel("RSS / N_pairs")
    ax.set_title("Pair-level Archetypal Analysis: elbow")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig_elbow.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # ── 4. per-k 評価 ──
    FEATURE_COLS = list(STATE_FEATURES) if args.state_only else list(STATE_FEATURES) + list(PATH_FEATURE_NAMES)
    pair_keys = [f"{d}|{p}" for d, p in kept_pairs]

    per_pair_pivot = per_pair.pivot_table(
        index=["developer", "directory"], columns="model", values=["acc", "auc"]
    )
    per_pair_pivot.columns = [f"{a}_{b}" for a, b in per_pair_pivot.columns]
    # pair index に合わせる
    per_pair_pivot = per_pair_pivot.reindex(kept_pairs)

    rng = np.random.default_rng(777)

    for k in args.k_grid:
        info = archetypes_by_k[k]
        alpha = info["alpha"]
        Z = info["Z"]
        dom = alpha.argmax(axis=1)

        # profile
        z_df = pd.DataFrame(Z, columns=FEATURE_COLS,
                             index=[f"A{j+1}" for j in range(k)])
        z_df.to_csv(args.output_dir / f"k{k}_archetype_profile.csv")

        # per-pair α + metrics
        pair_alpha = pd.DataFrame(alpha, columns=[f"A{j+1}" for j in range(k)])
        pair_alpha["developer"] = [d for d, _ in kept_pairs]
        pair_alpha["directory"] = [dp for _, dp in kept_pairs]
        pair_alpha["dominant"] = [f"A{j+1}" for j in dom]
        pair_alpha = pair_alpha.merge(
            per_pair_pivot.reset_index(),
            on=["developer", "directory"], how="left",
        )
        pair_alpha.to_csv(args.output_dir / f"k{k}_pair_alpha_with_metrics.csv", index=False)

        # hard aggregation
        hard_rows = []
        for j in range(k):
            mask = (dom == j)
            n = int(mask.sum())
            row = {"archetype": f"A{j+1}", "n_pair": n}
            sub = pair_alpha[mask]
            for col in per_pair_pivot.columns:
                vals = sub[col].dropna()
                row[col] = float(vals.mean()) if len(vals) else np.nan
            hard_rows.append(row)
        pd.DataFrame(hard_rows).to_csv(args.output_dir / f"k{k}_hard_metrics.csv", index=False)

        # pooled AUC + bootstrap CI
        pair_to_arch = {(d, dp): dom[i] for i, (d, dp) in enumerate(kept_pairs)}
        long_with_arch = long.assign(
            archetype=long.apply(lambda r: pair_to_arch.get((r["developer"], r["directory"])), axis=1)
        ).dropna(subset=["archetype"])
        long_with_arch["archetype"] = long_with_arch["archetype"].astype(int)

        boot_rows = []
        for j in range(k):
            for model in ["cold_irl", "warm_irl", "rf_dir"]:
                sub = long_with_arch[
                    (long_with_arch["archetype"] == j)
                    & (long_with_arch["model"] == model)
                ][["label", "prob"]].to_numpy()
                if len(sub) < 10 or len(np.unique(sub[:, 0])) < 2:
                    auc, lo, hi = np.nan, np.nan, np.nan
                else:
                    labels = sub[:, 0].astype(int)
                    probs = sub[:, 1].astype(float)
                    auc, lo, hi = pooled_auc_with_ci(labels, probs, rng)
                boot_rows.append({"archetype": f"A{j+1}", "model": model,
                                   "n_samples": int(len(sub)), "auc": auc,
                                   "auc_ci_lo": lo, "auc_ci_hi": hi})
        pd.DataFrame(boot_rows).to_csv(args.output_dir / f"k{k}_pooled_auc.csv", index=False)

        # Wilcoxon per-pair AUC (cold vs RF + warm vs RF)
        wilc_rows = []
        for j in range(k):
            mask = (dom == j)
            sub = pair_alpha[mask][["auc_cold_irl", "auc_warm_irl", "auc_rf_dir"]].dropna()
            n = len(sub)
            for irl_col, tag in [("auc_cold_irl", "cold_vs_rf"), ("auc_warm_irl", "warm_vs_rf")]:
                if n < 6 or sub[irl_col].std() < 1e-9 or sub["auc_rf_dir"].std() < 1e-9:
                    p_val, stat, med = np.nan, np.nan, np.nan
                else:
                    try:
                        res = wilcoxon(sub[irl_col], sub["auc_rf_dir"], zero_method="zsplit")
                        p_val, stat = float(res.pvalue), float(res.statistic)
                        med = float((sub[irl_col] - sub["auc_rf_dir"]).median())
                    except ValueError:
                        p_val, stat, med = np.nan, np.nan, np.nan
                wilc_rows.append({"archetype": f"A{j+1}", "n_pair_with_auc": int(n),
                                   "comparison": tag, "median_diff": med,
                                   "wilcoxon_stat": stat, "p_value": p_val})
        pd.DataFrame(wilc_rows).to_csv(args.output_dir / f"k{k}_wilcoxon.csv", index=False)

        print(f"  k={k}: hard / pooled / wilcoxon 出力, archetype 分布: "
              f"{[(f'A{j+1}', int((dom==j).sum())) for j in range(k)]}")

    print(f"\n[done] outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
