# -*- coding: utf-8 -*-
"""Archetypal Analysis (Cutler & Breiman, 1994) for Cold IRL vs RF_Dir comparison.

主目的:
    既存の「一貫承諾/一貫非承諾/変動型」3 分類は label 系列のみで定義された hard
    分類だが、これを 状態特徴量 (21 or 26 dim) 全体を使った data-driven な soft
    分類 (α 係数 = k-simplex 上の確率) に拡張する。各 dev に archetype 所属率を
    付与し、archetype 別に Cold IRL / Warm IRL / RF_Dir の AUC・accuracy を比較
    → どの archetype で IRL が勝つ/RF が勝つかを地形図化。

設計 (2026-05-18 合意):
    - 入力:  dev × 21 dim normalized state features
              (volat 学習後の 26 dim 切替は --feature-version v26 で対応)
    - 時間:  train_0-3m 時点 (feature_end = 2022-01-01) で 1 snapshot
    - k:     2..6 で全部走らせ RSS-vs-k elbow plot を出力
    - 評価:  Cold IRL + Warm IRL + RF_Dir の per-dev 4 窓平均 AUC / accuracy
    - 集約:  hard (argmax-α) メイン + soft (α-weighted) も併記
    - 検定:  Wilcoxon signed-rank で IRL vs RF の per-dev AUC 差の有意性

実装メモ:
    py_pcha は NumPy 2.x 非互換 (np.mat 削除) のため自前で ALS 実装。
    Cutler & Breiman 1994 の交互最小二乗 + 凸制約 (sum=1, nonneg) を scipy.nnls
    で素朴に解く。N=数百, k=2..6 で十分速い。

使い方:
    uv run python scripts/analyze/study/archetype_irl_rf.py \
        --cold-dir outputs/main32_mce_v2_phase2 \
        --warm-dir outputs/main32_mce_kd_warmstart_v2 \
        --output-dir outputs/analysis/archetype_irl_rf_v21
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
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

PERIODS = ["0-3m", "3-6m", "6-9m", "9-12m"]
# -1〜+1 値域の特徴量 → archetype 入力前に [0,1] へ linear shift
RANGE_NEG1_POS1 = {"activity_trend", "response_time_trend", "accept_rate_drift_30d"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Archetypal Analysis (Cutler & Breiman 1994 — ALS 自前実装)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _solve_convex_lsq(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """min ||A x - b||^2 s.t. x >= 0, sum(x) = 1 (simplex constraint via penalty + nnls)."""
    # sum(x)=1 を soft constraint で nnls に乗せる
    lam = 1e3
    A_aug = np.vstack([A, lam * np.ones((1, A.shape[1]))])
    b_aug = np.concatenate([b, [lam]])
    x, _ = nnls(A_aug, b_aug)
    s = x.sum()
    return x / s if s > 0 else np.ones_like(x) / len(x)


def archetypal_analysis(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-5,
    seed: int = 777,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Cutler & Breiman 1994 の交互最小二乗。

    X: (n, m) データ行列
    Returns:
        Z: (k, m) アーケタイプ行列 (Z = β^T X)
        alpha: (n, k) 各観測の archetype 帰属確率 (rows sum to 1)
        beta: (k, n) 各 archetype の観測点係数 (rows sum to 1)
        rss: 最終 RSS
    """
    n, m = X.shape
    rng = np.random.default_rng(seed)

    # β 初期化: k 個の観測点に rng で割当
    idx_init = rng.choice(n, size=k, replace=False)
    beta = np.zeros((k, n))
    for j, i in enumerate(idx_init):
        beta[j, i] = 1.0
    Z = beta @ X  # (k, m)

    prev_rss = np.inf
    for it in range(max_iter):
        # Step A: α 更新 (各 i について min ||X_i - α_i Z||^2, α_i >= 0, sum=1)
        alpha = np.zeros((n, k))
        for i in range(n):
            alpha[i] = _solve_convex_lsq(Z.T, X[i])

        # Step B: β 更新 (各 j について min ||Z_j - β_j X||^2, β_j >= 0, sum=1)
        # Z = α^T X が真の archetype の場合に近づくよう Z_target = (α^T α)^-1 α^T X を計算してから
        # β で Z_target を観測点凸結合で表現
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
# 2. データ準備
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_pair_predictions(base: pathlib.Path, train_win: str, eval_win: str) -> pd.DataFrame:
    f = base / f"lstm_baseline/train_{train_win}/eval_{eval_win}/pair_predictions.csv"
    if not f.exists():
        raise FileNotFoundError(f"missing: {f}")
    return pd.read_csv(f)


def load_thresholds(base: pathlib.Path, train_win: str, eval_win: str) -> Dict[str, float]:
    f = base / f"lstm_baseline/train_{train_win}/eval_{eval_win}/summary_metrics.json"
    with open(f) as fh:
        d = json.load(fh)
    return {k: float(d[k]["clf_threshold"]) for k in d if "clf_threshold" in d.get(k, {})}


def build_dev_feature_matrix(
    df_raw: pd.DataFrame,
    devs: List[str],
    feature_end: pd.Timestamp,
    feature_start: pd.Timestamp,
) -> Tuple[np.ndarray, List[str]]:
    """dev × len(STATE_FEATURES) の normalized 特徴量行列。

    レビュー履歴ゼロ dev (extract_common_features が _get_default_features を返す
    ケース) は archetype を歪めるため除外。raw データで feature_start..end に
    1 件以上のレビュー or PR がある dev のみを保持。
    """
    rows, kept = [], []
    feat_start_ts = pd.Timestamp(feature_start)
    feat_end_ts = pd.Timestamp(feature_end)
    has_owner = "owner_email" in df_raw.columns

    for email in devs:
        # 履歴フィルタ: reviewer/owner として 1 件以上あるか
        rev_mask = (
            (df_raw["email"] == email)
            & (df_raw["timestamp"] >= feat_start_ts)
            & (df_raw["timestamp"] < feat_end_ts)
        )
        has_rev = bool(rev_mask.any())
        has_own = False
        if has_owner:
            own_mask = (
                (df_raw["owner_email"] == email)
                & (df_raw["timestamp"] >= feat_start_ts)
                & (df_raw["timestamp"] < feat_end_ts)
            )
            has_own = bool(own_mask.any())
        if not (has_rev or has_own):
            continue  # 履歴ゼロ → default features は raw 値で archetype を歪める

        feats = extract_common_features(
            df_raw, email, feat_start_ts.to_pydatetime(), feat_end_ts.to_pydatetime(),
            normalize=True,
        )
        # 念のため再正規化 (_get_default_features 経路の漏れ対策)
        feats = normalize_features(feats)

        vec = []
        for k in STATE_FEATURES:
            v = float(feats.get(k, 0.0))
            if k in RANGE_NEG1_POS1:
                v = (v + 1.0) / 2.0  # [-1,+1] → [0,1]
            vec.append(v)
        rows.append(vec)
        kept.append(email)
    X = np.asarray(rows, dtype=float)
    return X, kept


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. per-dev model metrics (4 窓平均)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def compute_per_dev_metrics(
    cold_base: pathlib.Path,
    warm_base: pathlib.Path,
    diagonal: bool,
) -> pd.DataFrame:
    """per-dev × 各モデル の 4 窓平均 accuracy + AUC (max-prob aggregation)。"""
    train_eval_pairs = (
        [(w, w) for w in PERIODS] if diagonal
        else [("0-3m", w) for w in PERIODS]
    )

    rows = []  # list of {developer, model, window, acc, auc, n_pos, n_neg}
    for train_w, eval_w in train_eval_pairs:
        cold_df = load_pair_predictions(cold_base, train_w, eval_w)
        warm_df = load_pair_predictions(warm_base, train_w, eval_w)
        cold_thr = load_thresholds(cold_base, train_w, eval_w)
        warm_thr = load_thresholds(warm_base, train_w, eval_w)

        for model_name, df, thr_dict, prob_col, thr_key in [
            ("cold_irl", cold_df, cold_thr, "irl_dir_prob_calibrated", "IRL_Dir_calibrated"),
            ("warm_irl", warm_df, warm_thr, "irl_dir_prob_calibrated", "IRL_Dir_calibrated"),
            ("rf_dir",   warm_df, warm_thr, "rf_dir_prob",             "RF_Dir"),
        ]:
            thr = thr_dict.get(thr_key, 0.5)
            agg = df.groupby("developer").agg(
                prob=(prob_col, "max"),
                label=("label", "max"),
                n=("label", "count"),
            ).reset_index()
            agg["pred"] = (agg["prob"] >= thr).astype(int)
            agg["correct"] = (agg["pred"] == agg["label"]).astype(int)
            for _, r in agg.iterrows():
                rows.append({
                    "developer": r["developer"], "model": model_name, "window": eval_w,
                    "label": int(r["label"]), "prob": float(r["prob"]),
                    "correct": int(r["correct"]),
                })
    long = pd.DataFrame(rows)
    return long


def aggregate_per_dev(long: pd.DataFrame) -> pd.DataFrame:
    """per-dev × model: 4 窓平均 accuracy + (label の variability があれば) per-dev AUC。
    label が 4 窓全部同じ dev は AUC=NaN (定義不能)。"""
    out = []
    for (dev, model), grp in long.groupby(["developer", "model"]):
        acc = grp["correct"].mean()
        labels = grp["label"].to_numpy()
        probs = grp["prob"].to_numpy()
        if len(np.unique(labels)) >= 2:
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = np.nan
        else:
            auc = np.nan
        out.append({"developer": dev, "model": model, "acc": acc, "auc": auc})
    return pd.DataFrame(out)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. メイン処理
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run(
    cold_dir: pathlib.Path,
    warm_dir: pathlib.Path,
    raw_csv: pathlib.Path,
    out_dir: pathlib.Path,
    train_end: str,
    feature_window_days: int,
    k_grid: List[int],
    diagonal: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_end = pd.Timestamp(train_end)
    feat_start = feat_end - pd.Timedelta(days=feature_window_days)
    print(f"[setup] feature window: {feat_start.date()} 〜 {feat_end.date()}")
    print(f"[setup] state_dim = {len(STATE_FEATURES)}")

    # --- 1. per-dev model metrics ---
    print("[load] pair_predictions.csv × 4 windows × 2 models ...")
    long = compute_per_dev_metrics(cold_dir, warm_dir, diagonal)
    per_dev = aggregate_per_dev(long)
    common_devs = sorted(set(per_dev["developer"]))
    print(f"[devs] common evaluated: {len(common_devs)}")

    # --- 2. dev 特徴量行列 ---
    print(f"[features] loading raw csv: {raw_csv}")
    df_raw = pd.read_csv(raw_csv, parse_dates=["timestamp"])
    if "first_response_time" in df_raw.columns:
        df_raw["first_response_time"] = pd.to_datetime(df_raw["first_response_time"], errors="coerce")
    X, kept_devs = build_dev_feature_matrix(df_raw, common_devs, feat_end, feat_start)
    print(f"[features] X shape: {X.shape} (devs kept after non-zero filter)")

    # --- 3. archetype: k_grid 全部走らせて RSS-elbow ---
    print(f"[archetype] running k = {k_grid} ...")
    elbow_rows = []
    archetypes_by_k: Dict[int, Dict] = {}
    for k in k_grid:
        Z, alpha, beta, rss = archetypal_analysis(X, k=k, max_iter=120, seed=777)
        elbow_rows.append({"k": k, "rss": rss, "rss_per_dev": rss / len(X)})
        archetypes_by_k[k] = {"Z": Z, "alpha": alpha, "beta": beta, "rss": rss}
        print(f"  k={k}  RSS={rss:.3f}  RSS/dev={rss/len(X):.4f}")

    elbow_df = pd.DataFrame(elbow_rows)
    elbow_df.to_csv(out_dir / "rss_elbow.csv", index=False)

    # elbow plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(elbow_df["k"], elbow_df["rss_per_dev"], "o-", color="#1f77b4")
    ax.set_xlabel("number of archetypes k")
    ax.set_ylabel("RSS / N (lower = better fit)")
    ax.set_title("Archetypal Analysis: elbow plot")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_elbow.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir/'fig_elbow.pdf'}")

    # --- 4. 各 k で per-archetype 評価表を作る ---
    per_dev_pivot = per_dev.pivot_table(
        index="developer", columns="model", values=["acc", "auc"]
    )
    per_dev_pivot.columns = [f"{a}_{b}" for a, b in per_dev_pivot.columns]
    per_dev_pivot = per_dev_pivot.loc[kept_devs]

    for k in k_grid:
        info = archetypes_by_k[k]
        alpha = info["alpha"]
        Z = info["Z"]
        dev_alpha = pd.DataFrame(alpha, index=kept_devs,
                                  columns=[f"A{j+1}" for j in range(k)])
        dom = alpha.argmax(axis=1)
        dev_alpha["dominant"] = [f"A{j+1}" for j in dom]
        dev_alpha = dev_alpha.join(per_dev_pivot, how="left")
        dev_alpha.to_csv(out_dir / f"k{k}_dev_alpha_with_metrics.csv")

        # archetype プロファイル (k × len(STATE_FEATURES))
        z_df = pd.DataFrame(Z, columns=STATE_FEATURES,
                             index=[f"A{j+1}" for j in range(k)])
        z_df.to_csv(out_dir / f"k{k}_archetype_profile.csv")

        # hard aggregation: 各 archetype 内で各モデルの mean acc/auc
        hard_rows = []
        for j in range(k):
            mask = (dom == j)
            n = int(mask.sum())
            row = {"archetype": f"A{j+1}", "n_dev": n}
            for col in per_dev_pivot.columns:
                vals = dev_alpha.loc[mask, col].dropna()
                row[col] = float(vals.mean()) if len(vals) else np.nan
            hard_rows.append(row)
        hard_df = pd.DataFrame(hard_rows)
        hard_df.to_csv(out_dir / f"k{k}_hard_archetype_metrics.csv", index=False)

        # soft aggregation: α 加重平均
        soft_rows = []
        for j in range(k):
            w = alpha[:, j]
            wsum = w.sum()
            row = {"archetype": f"A{j+1}", "effective_n": float(wsum)}
            for col in per_dev_pivot.columns:
                vals = per_dev_pivot[col].to_numpy()
                mask = ~np.isnan(vals)
                if mask.sum() > 0 and w[mask].sum() > 0:
                    row[col] = float(np.sum(w[mask] * vals[mask]) / w[mask].sum())
                else:
                    row[col] = np.nan
            soft_rows.append(row)
        soft_df = pd.DataFrame(soft_rows)
        soft_df.to_csv(out_dir / f"k{k}_soft_archetype_metrics.csv", index=False)

        # Wilcoxon signed-rank: per-archetype IRL vs RF (accuracy + AUC 両方)
        # accuracy は binary 化で差が潰れる傾向 → AUC でも検定
        wilc_rows = []
        for j in range(k):
            mask = (dom == j)
            for metric, irl_cold, irl_warm, rf_col in [
                ("acc", "acc_cold_irl", "acc_warm_irl", "acc_rf_dir"),
                ("auc", "auc_cold_irl", "auc_warm_irl", "auc_rf_dir"),
            ]:
                sub = dev_alpha.loc[mask, [irl_cold, irl_warm, rf_col]].dropna()
                n = len(sub)
                for irl_col, tag in [(irl_cold, f"{metric}_cold_vs_rf"),
                                       (irl_warm, f"{metric}_warm_vs_rf")]:
                    if n < 6 or sub[irl_col].std() < 1e-9 or sub[rf_col].std() < 1e-9:
                        p, stat, med = np.nan, np.nan, np.nan
                    else:
                        try:
                            res = wilcoxon(sub[irl_col], sub[rf_col], zero_method="zsplit")
                            p, stat = float(res.pvalue), float(res.statistic)
                            med = float((sub[irl_col] - sub[rf_col]).median())
                        except ValueError:
                            p, stat, med = np.nan, np.nan, np.nan
                    wilc_rows.append({"archetype": f"A{j+1}", "n_dev_with_metric": int(n),
                                       "metric": metric, "comparison": tag,
                                       "median_diff": med,
                                       "wilcoxon_stat": stat, "p_value": p})
        pd.DataFrame(wilc_rows).to_csv(out_dir / f"k{k}_wilcoxon.csv", index=False)

        # Pooled archetype AUC + bootstrap CI (全 dev×window pair をプールして archetype 単位 AUC)
        # per-dev AUC は 4 sample しかなくノイジー、pooled は archetype 全体の真の AUC を推定
        dev_to_arch = dict(zip(kept_devs, dom))
        long_with_arch = long.assign(
            archetype=long["developer"].map(lambda d: dev_to_arch.get(d))
        ).dropna(subset=["archetype"])
        long_with_arch["archetype"] = long_with_arch["archetype"].astype(int)

        boot_rows = []
        rng = np.random.default_rng(777)
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
                    try:
                        auc = float(roc_auc_score(labels, probs))
                    except ValueError:
                        auc = np.nan
                    # bootstrap 1000 回
                    boot_aucs = []
                    n_samp = len(sub)
                    for _ in range(1000):
                        idx = rng.integers(0, n_samp, n_samp)
                        bl, bp = labels[idx], probs[idx]
                        if len(np.unique(bl)) < 2:
                            continue
                        try:
                            boot_aucs.append(roc_auc_score(bl, bp))
                        except ValueError:
                            continue
                    if len(boot_aucs) > 50:
                        lo = float(np.percentile(boot_aucs, 2.5))
                        hi = float(np.percentile(boot_aucs, 97.5))
                    else:
                        lo, hi = np.nan, np.nan
                boot_rows.append({
                    "archetype": f"A{j+1}", "model": model,
                    "n_samples": int(len(sub)), "auc": auc,
                    "auc_ci_lo": lo, "auc_ci_hi": hi,
                })
        pd.DataFrame(boot_rows).to_csv(out_dir / f"k{k}_pooled_auc_bootstrap.csv", index=False)

        # 各 archetype の代表 dev (α 最大上位 5 名) を抽出
        examples_rows = []
        for j in range(k):
            arch_col = f"A{j+1}"
            top5 = dev_alpha.sort_values(arch_col, ascending=False).head(5)
            for rank, (email, row) in enumerate(top5.iterrows(), 1):
                examples_rows.append({
                    "archetype": arch_col, "rank": rank, "developer": email,
                    "alpha": float(row[arch_col]),
                    **{c: float(row[c]) if pd.notna(row[c]) else np.nan
                        for c in ["acc_cold_irl", "acc_warm_irl", "acc_rf_dir",
                                   "auc_cold_irl", "auc_warm_irl", "auc_rf_dir"]},
                })
        pd.DataFrame(examples_rows).to_csv(out_dir / f"k{k}_archetype_top5_devs.csv", index=False)

        # k=3 のときは三角プロット (simplex)
        if k == 3:
            try:
                from matplotlib.tri import Triangulation  # noqa: F401
                _plot_simplex_k3(alpha, dev_alpha, out_dir / "fig_k3_simplex.pdf")
            except Exception as e:
                print(f"  simplex plot skipped: {e}")

        print(f"  k={k}: hard/soft/wilcoxon table 出力済み (n_dev/archetype: "
              f"{[(hard_rows[i]['archetype'], hard_rows[i]['n_dev']) for i in range(k)]})")

    print(f"[done] outputs: {out_dir}")


def _plot_simplex_k3(alpha: np.ndarray, dev_alpha: pd.DataFrame, out_path: pathlib.Path) -> None:
    # 三角座標へ射影
    a1, a2, a3 = alpha[:, 0], alpha[:, 1], alpha[:, 2]
    x = 0.5 * (2 * a2 + a3) / (a1 + a2 + a3)
    y = (np.sqrt(3) / 2) * a3 / (a1 + a2 + a3)
    diff = dev_alpha.get("acc_cold_irl", pd.Series(np.nan, index=dev_alpha.index)).to_numpy() \
           - dev_alpha.get("acc_rf_dir", pd.Series(np.nan, index=dev_alpha.index)).to_numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(x, y, c=diff, cmap="RdBu_r", vmin=-0.3, vmax=0.3, s=22, edgecolor="k", linewidth=0.3)
    # 三角形枠
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], "k-", linewidth=1)
    ax.text(-0.03, -0.04, "A1", ha="right")
    ax.text(1.03, -0.04, "A2", ha="left")
    ax.text(0.5, np.sqrt(3)/2 + 0.03, "A3", ha="center")
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("acc(Cold IRL) − acc(RF_Dir)\n  + → IRL strong   − → RF strong")
    ax.set_title("Archetype simplex (k=3): IRL vs RF 勝敗分布")
    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cold-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_v2_phase2"))
    p.add_argument("--warm-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/main32_mce_kd_warmstart_v2"))
    p.add_argument("--raw-csv", type=pathlib.Path,
                   default=pathlib.Path("data/combined_raw_main32.csv"))
    p.add_argument("--output-dir", type=pathlib.Path,
                   default=pathlib.Path("outputs/analysis/archetype_irl_rf_v21"))
    p.add_argument("--train-end", default="2022-01-01",
                   help="特徴量計算の feature_end (= train_end)")
    p.add_argument("--feature-window-days", type=int, default=180)
    p.add_argument("--k-grid", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    p.add_argument("--diagonal", action="store_true",
                   help="train_W/eval_W 評価 (v2 は train_0-3m のみフル diagonal)")
    args = p.parse_args()
    run(args.cold_dir, args.warm_dir, args.raw_csv, args.output_dir,
        args.train_end, args.feature_window_days, args.k_grid, args.diagonal)


if __name__ == "__main__":
    main()
