"""S3軸(個人役割移動型)の深掘り: 元のSES研究の reviewer_owner_ratio 特徴量をそのまま使い、
「自分が担当するモジュールのAI関与度が上がると、個人はレビュー(評価)側にシフトするか」を検証する。

元のSES研究(レビュー継続予測)は reviewer_owner_ratio を「開発者が生成側か評価側かの個人特性」
として使っていた。Loop6のS3(reviewer_ratio_delta = 最初と最後の単純差分)をさらに深掘りし、
個人×月パネルでモジュールのAI関与度(A_prime)との関係を固定効果回帰で見る。

モデル:
  reviewer_ratio(actor, t) ~ A_prime_weighted(actorの活動モジュール, t) + actor_FE + time_FE

A_prime_weightedは、そのactorがその月に活動した各モジュールのA_primeを、
活動量(G+Eイベント数)で加重平均したもの。

実行: uv run python scripts/pilot/s3_deepdive.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_DIR = REPO_ROOT / "pilot" / "interim"
PROCESSED_DIR = REPO_ROOT / "pilot" / "processed"
CONFIG_DIR = REPO_ROOT / "pilot" / "config"

FIRST_BATCH_REPOS = [
    "home-assistant/core",
    "grafana/grafana",
    "juspay/hyperswitch",
    "ray-project/ray",
    "python/cpython",
    "ghostty-org/ghostty",
    "apache/airflow",
    "godotengine/godot",
]


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def load_repos() -> list[str]:
    repos = list(FIRST_BATCH_REPOS)
    batch2_csv = CONFIG_DIR / "loop7_selected_repos_batch2_final.csv"
    if batch2_csv.exists():
        with batch2_csv.open() as f:
            repos += [row["full_name"] for row in csv.DictReader(f)]
    return repos


def compute_for_repo(repo: str) -> pd.DataFrame | None:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    processed_d = PROCESSED_DIR / repo_dir_name(repo)
    events_path = interim_d / "events.parquet"
    panel_path = processed_d / "panel.parquet"
    if not events_path.exists() or not panel_path.exists():
        return None

    events = pd.read_parquet(events_path)
    ge = events[events["stage"].isin(["G", "E"]) & (events["actor_category"] == "human")]
    if ge.empty:
        return None

    panel = pd.read_parquet(panel_path)[["module_id", "month", "A_prime"]]
    module_a = dict(zip(zip(panel["module_id"], panel["month"]), panel["A_prime"]))

    # actor x module x month の活動量(G+E件数)
    activity = ge.groupby(["actor_login", "module_id", "month", "stage"]).size().unstack(fill_value=0)
    for col in ["G", "E"]:
        if col not in activity.columns:
            activity[col] = 0
    activity = activity.reset_index()
    activity["n_total"] = activity["G"] + activity["E"]
    activity["a_prime"] = activity.apply(lambda r: module_a.get((r["module_id"], r["month"])), axis=1)
    activity = activity.dropna(subset=["a_prime"])
    if activity.empty:
        return None

    # actor x month に集約: reviewer_ratio(全モジュール合算) と A_primeの活動量加重平均
    rows = []
    for (actor, month), grp in activity.groupby(["actor_login", "month"]):
        n_g = grp["G"].sum()
        n_e = grp["E"].sum()
        n_total = n_g + n_e
        if n_total == 0:
            continue
        reviewer_ratio = n_e / n_total
        weights = grp["n_total"]
        a_weighted = float((grp["a_prime"] * weights).sum() / weights.sum()) if weights.sum() > 0 else np.nan
        rows.append(
            {
                "actor_login": actor,
                "month": month,
                "reviewer_ratio": reviewer_ratio,
                "A_prime_weighted": a_weighted,
                "n_total": n_total,
            }
        )
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["repo"] = repo
    return df


def main() -> None:
    repos = load_repos()
    frames = []
    for repo in repos:
        df = compute_for_repo(repo)
        if df is not None:
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "s3_deepdive.parquet"
    all_df.to_parquet(out_path, index=False)

    model_df = all_df.dropna(subset=["reviewer_ratio", "A_prime_weighted"]).copy()
    model_df["actor_id_global"] = model_df["repo"] + "::" + model_df["actor_login"]
    model_df["month_period"] = pd.PeriodIndex(model_df["month"], freq="M").to_timestamp()
    model_df["log_n_total"] = np.log1p(model_df["n_total"])

    # 2回以上活動しているactorのみ(固定効果推定に必要)
    counts = model_df.groupby("actor_id_global").size()
    keep = counts[counts >= 2].index
    model_df = model_df[model_df["actor_id_global"].isin(keep)]

    idx_df = model_df.set_index(["actor_id_global", "month_period"])
    exog = idx_df[["A_prime_weighted", "log_n_total"]]
    endog = idx_df["reviewer_ratio"]

    mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    n_repo_clusters = idx_df.reset_index()["actor_id_global"].str.split("::").str[0].nunique()
    model_summary = {
        "n_obs": int(res.nobs),
        "n_actors(entity)": int(idx_df.index.get_level_values(0).nunique()),
        "n_repo_clusters": int(n_repo_clusters),
        "params": res.params.to_dict(),
        "pvalues(cluster=actor)": res.pvalues.to_dict(),
        "rsquared_within": res.rsquared_within,
    }
    if n_repo_clusters >= 5:
        repo_series = idx_df.reset_index()["actor_id_global"].str.split("::").str[0]
        repo_series.index = idx_df.index
        mod2 = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
        res2 = mod2.fit(cov_type="clustered", clusters=repo_series)
        model_summary["pvalues(cluster=repo)"] = res2.pvalues.to_dict()
        model_summary["std_errors(cluster=repo)"] = res2.std_errors.to_dict()

    summary = {
        "n_repos_used": len(frames),
        "n_obs_raw": len(all_df),
        "reviewer_ratio_mean": float(all_df["reviewer_ratio"].mean()),
        "model(reviewer_ratio ~ A_prime_weighted + log_n_total + actor_FE + time_FE)": model_summary,
    }
    (PROCESSED_DIR / "s3_deepdive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
