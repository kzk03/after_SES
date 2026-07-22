#!/usr/bin/env python3
"""V(s) の駆動要因分析 — RQ2「どの状態遷移が離脱前兆か」の推奨考察手法

学習済み IQL の V(s)（長期エンゲージメント価値）を「何が動かしているか」を
3つの相補的な分析で定量化する:

  (1) 特徴グループ感度 (permutation sensitivity)
      意味クラスタ (負荷 / 承諾傾向 / 応答・遅延 / 活動量 / 協業 / 親和度) ごとに
      列をシャッフルし、V の平均変化量 |ΔV| を測る。
      → 「V はどの信号群を見ているか」(モデル依存・ラベル不要)

  (2) ΔV ↔ Δ特徴 の遷移相関
      同一軌跡内の連続ステップで ΔV = V(s_{t+1})−V(s_t) と各特徴の変化量
      Δx の Spearman 相関を取る。
      → 「拒否が増える / 応答が遅くなる / 重いタスクが続くと V が下がるか」
        (ユーザーの中心仮説を遷移レベルで直接検証する)

  (3) V 急落遷移のプロファイル
      ΔV 下位 q% の遷移について、特徴変化の標準化平均を全体平均と比較。
      → 「離脱前兆の遷移では実際に何が起きているか」(論文の考察・実例の種)

使い方 (test set を推奨 — 学習に使っていない遷移で分析する):
    uv run python scripts/rl/analyze_value_drivers.py \\
        --dataset  outputs/rl/sweep/<...>/dataset/event_transitions_test.parquet \\
        --iql-model outputs/rl/sweep/<...>/lam0.3_gam0.99/iql_agent.pt \\
        --obs-dim 32 \\
        --output-dir outputs/rl/analysis/<run名>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from review_predictor.IRL.features.common_features import (
    PATH_FEATURE_NAMES,
    STATE_FEATURES,
)
from review_predictor.RL.agent.iql import IQLAgent
from review_predictor.RL.data.loading import load_transitions_df, states_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# 観測ベクトルの特徴名 (build_obs の組み立て順と一致させる)
OBS_FEATURE_NAMES: List[str] = list(STATE_FEATURES) + list(PATH_FEATURE_NAMES)

# RQ2 の意味クラスタ。§1-5 (docs/rl_redesign_plan.md) の分類と対応
FEATURE_GROUPS: Dict[str, List[str]] = {
    "workload(負荷)": [
        "recent_load_ratio_30d_all", "recent_load_ratio_7d_30d",
        "concurrent_open_reviews_7d",
    ],
    "acceptance(承諾傾向)": [
        "overall_acceptance_rate", "recent_acceptance_rate",
        "acceptance_rate_last10", "accept_rate_drift_30d", "recent_switch_rate",
    ],
    "latency(応答・間隔)": [
        "response_time_trend", "days_since_last_activity",
        "time_since_last_accept", "time_since_last_reject", "avg_activity_gap",
    ],
    "activity(活動量・経験)": [
        "window_tenure_days", "total_changes", "total_reviews",
        "recent_activity_frequency", "activity_trend", "active_months_ratio",
        "n_projects",
    ],
    "social(協業・役割)": [
        "unique_collaborator_count", "reciprocity_score", "core_reviewer_ratio",
        "reviewer_owner_ratio", "cross_project_review_share", "same_domain_share",
    ],
    "path(ディレクトリ親和度)": list(PATH_FEATURE_NAMES),
}


def _feature_names(obs_dim: int) -> List[str]:
    names = OBS_FEATURE_NAMES[:obs_dim]
    names += [f"extra_{i}" for i in range(len(names), obs_dim)]
    return names


def _v_values(
    agent: IQLAgent, states: np.ndarray, device: str, batch_size: int = 8192,
) -> np.ndarray:
    """V(s) をバッチ分割で計算する (event 級の大規模データでの OOM 防止)。"""
    agent.v_net.eval()
    dev = torch.device(device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(states), batch_size):
            t = torch.tensor(states[i:i + batch_size], dtype=torch.float32, device=dev)
            outs.append(agent.v_net(t).cpu().numpy())
    return np.concatenate(outs).astype(np.float64)


def group_sensitivity(
    agent: IQLAgent,
    states: np.ndarray,
    v_base: np.ndarray,
    names: List[str],
    device: str,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    """(1) 特徴グループ / 単一特徴の permutation 感度。"""
    rng = np.random.default_rng(seed)
    idx_of = {n: i for i, n in enumerate(names)}
    records = []

    targets = [(g, [idx_of[n] for n in cols if n in idx_of])
               for g, cols in FEATURE_GROUPS.items()]
    targets += [(f"single::{n}", [idx_of[n]]) for n in names]

    for label, cols in targets:
        if not cols:
            continue
        deltas = []
        for _ in range(n_repeats):
            perm = rng.permutation(len(states))
            shuffled = states.copy()
            shuffled[:, cols] = states[perm][:, cols]
            v_perm = _v_values(agent, shuffled, device)
            deltas.append(float(np.mean(np.abs(v_perm - v_base))))
        records.append({
            "target": label,
            "kind": "group" if not label.startswith("single::") else "feature",
            "n_features": len(cols),
            "mean_abs_dV": float(np.mean(deltas)),
            "std_abs_dV": float(np.std(deltas)),
            "mean_abs_dV_per_feature": float(np.mean(deltas)) / len(cols),
        })
    df = pd.DataFrame(records).sort_values("mean_abs_dV", ascending=False)
    return df


def transition_deltas(
    df: pd.DataFrame, states: np.ndarray, v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(2)(3) 用: 同一軌跡内の連続ステップの (Δ特徴, ΔV) を集める。"""
    order_cols = [c for c in ("email", "directory") if c in df.columns]
    fe = pd.to_datetime(df.get("feature_end"), errors="coerce")
    work = pd.DataFrame({"fe": fe, "_row": np.arange(len(df))})
    for c in order_cols:
        work[c] = df[c].values
    work = work.sort_values(order_cols + ["fe"], kind="stable")

    dx_list, dv_list = [], []
    if order_cols:
        grouped = work.groupby(order_cols, sort=False)
    else:
        grouped = [(None, work)]
    for _, g in grouped:
        rows = g["_row"].to_numpy()
        if len(rows) < 2:
            continue
        dx_list.append(states[rows[1:]] - states[rows[:-1]])
        dv_list.append(v[rows[1:]] - v[rows[:-1]])
    if not dx_list:
        return np.zeros((0, states.shape[1])), np.zeros(0)
    return np.concatenate(dx_list), np.concatenate(dv_list)


def dv_dx_correlation(dx: np.ndarray, dv: np.ndarray, names: List[str]) -> pd.DataFrame:
    from scipy.stats import spearmanr
    records = []
    for i, name in enumerate(names):
        col = dx[:, i]
        if np.std(col) < 1e-12 or np.std(dv) < 1e-12:
            rho, pval = float("nan"), float("nan")
        else:
            rho, pval = spearmanr(col, dv)
        records.append({
            "feature": name,
            "spearman_dV": float(rho) if rho == rho else float("nan"),
            "p_value": float(pval) if pval == pval else float("nan"),
            "mean_abs_dx": float(np.mean(np.abs(col))),
        })
    return (
        pd.DataFrame(records)
        .assign(abs_rho=lambda d: d["spearman_dV"].abs())
        .sort_values("abs_rho", ascending=False)
        .drop(columns=["abs_rho"])
    )


def v_drop_profile(
    dx: np.ndarray, dv: np.ndarray, names: List[str], quantile: float,
) -> pd.DataFrame:
    """(3) ΔV 下位 quantile の遷移における特徴変化の標準化プロファイル。"""
    if len(dv) == 0:
        return pd.DataFrame()
    thr = np.quantile(dv, quantile)
    drop_mask = dv <= thr
    std = dx.std(axis=0)
    std[std < 1e-12] = np.nan
    z_drop = (dx[drop_mask].mean(axis=0) - dx.mean(axis=0)) / std
    return (
        pd.DataFrame({
            "feature": names,
            "z_mean_dx_in_drops": z_drop,
            "mean_dx_in_drops": dx[drop_mask].mean(axis=0),
            "mean_dx_overall": dx.mean(axis=0),
        })
        .assign(abs_z=lambda d: d["z_mean_dx_in_drops"].abs())
        .sort_values("abs_z", ascending=False)
        .drop(columns=["abs_z"])
    )


def _save_bar_figure(df: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        logger.warning("matplotlib 不在のため図をスキップ")
        return
    g = df[df["kind"] == "group"].sort_values("mean_abs_dV")
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(g) + 1.5))
    ax.barh(g["target"], g["mean_abs_dV"], xerr=g["std_abs_dV"])
    ax.set_xlabel("mean |ΔV| under group permutation")
    ax.set_title("V(s) feature-group sensitivity")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    logger.info("図: %s", out)


def main() -> None:
    p = argparse.ArgumentParser(description="V(s) 駆動要因の考察分析 (RQ2)")
    p.add_argument("--dataset", required=True, help="transitions parquet (test 推奨)")
    p.add_argument("--iql-model", required=True, help="IQL チェックポイント")
    p.add_argument("--obs-dim", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-repeats", type=int, default=5, help="permutation 反復回数")
    p.add_argument("--drop-quantile", type=float, default=0.1, help="V急落の下位分位")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_transitions_df(Path(args.dataset))
    states = states_matrix(df["state"], args.obs_dim)
    names = _feature_names(args.obs_dim)
    agent = IQLAgent.load(args.iql_model, device=args.device)
    v = _v_values(agent, states, args.device)
    logger.info("N=%d, V: mean=%.4f std=%.4f", len(v), v.mean(), v.std())

    # (1) グループ/特徴 permutation 感度
    sens = group_sensitivity(
        agent, states, v, names, args.device, args.n_repeats, args.seed,
    )
    sens.to_csv(out_dir / "value_sensitivity.csv", index=False)

    # (2)(3) 遷移ベース分析
    dx, dv = transition_deltas(df, states, v)
    logger.info("軌跡内の連続遷移ペア: %d", len(dv))
    corr = dv_dx_correlation(dx, dv, names)
    corr.to_csv(out_dir / "dv_dx_spearman.csv", index=False)
    drops = v_drop_profile(dx, dv, names, args.drop_quantile)
    drops.to_csv(out_dir / "v_drop_profile.csv", index=False)

    _save_bar_figure(sens, out_dir / "fig_value_group_sensitivity.pdf")

    # summary.md (考察の下書きに貼れる形)
    lines = [
        "# V(s) 駆動要因分析", "",
        f"- dataset: `{args.dataset}` (N={len(v)}, 遷移ペア={len(dv)})",
        f"- model: `{args.iql_model}`", "",
        "## (1) 特徴グループ感度 (mean |ΔV|)", "",
        sens[sens['kind'] == 'group'][
            ["target", "mean_abs_dV", "std_abs_dV", "mean_abs_dV_per_feature"]
        ].round(4).to_markdown(index=False), "",
        "## (2) ΔV ↔ Δ特徴 Spearman (上位15)", "",
        corr.head(15).round(4).to_markdown(index=False), "",
        f"## (3) V 急落遷移 (下位 {args.drop_quantile:.0%}) の特徴変化プロファイル (上位15)", "",
        drops.head(15).round(4).to_markdown(index=False), "",
        "解釈メモ: (2)(3) は相関であり介入効果ではない。論文では「〜と共起する」",
        "「〜が前兆となる」という表現に留める (docs/rl_experiment_pipeline.md §考察)。",
    ]
    (out_dir / "value_drivers_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("出力: %s", out_dir)
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
