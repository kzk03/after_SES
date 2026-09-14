"""Loop 8 — 主解析モデルの試運転(パイロット規模)。

研究設計書§5.1: H_E(m,t) ~ A(m,t) + controls + module_FE + time_FE
  controls: モジュール規模(ファイル数), 活動量(G+Eイベント数), 担い手数(G+Eユニークactor数)
  標準誤差: リポジトリ単位でクラスタリング(設計書の指定)

2工程(生成・評価)決定(gates.md Loop3)に伴い、events集計はG/Eのみを対象とする。
agentはtrailer込み判定(gates.md Loop1)、つまり A = A_prime を主指標として使う。

★モジュール規模(n_files)は月ごとに変化しない(Loop2の最新月スナップショットのみで算出しているため)。
module固定効果モデルでは時間不変な変数は原理的に識別不能なため、メインモデルのexogからは外し、
between推定(モジュール単位の平均をpooled OLSで見る)で別途その効果を確認する。

実行: uv run python scripts/pilot/loop8_main_model.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
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


def load_repos() -> list[str]:
    repos = list(FIRST_BATCH_REPOS)
    batch2_csv = CONFIG_DIR / "loop7_selected_repos_batch2_final.csv"
    if batch2_csv.exists():
        with batch2_csv.open() as f:
            reader = csv.DictReader(f)
            repos += [row["full_name"] for row in reader]
    return repos


REPOS = load_repos()


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def build_controls(repo: str) -> pd.DataFrame:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    modules = pd.read_parquet(interim_d / "modules.parquet")
    n_files = modules.groupby("module_id").size().rename("n_files")

    events = pd.read_parquet(interim_d / "events.parquet")
    ge = events[events["stage"].isin(["G", "E"])]  # 2工程決定(gates.md Loop3)
    grp = ge.groupby(["module_id", "month"]).agg(
        n_events=("actor_login", "size"), n_unique_actors=("actor_login", "nunique")
    )
    grp = grp.join(n_files, on="module_id")
    return grp.reset_index()


def main() -> None:
    frames = []
    for repo in REPOS:
        panel = pd.read_parquet(PROCESSED_DIR / repo_dir_name(repo) / "panel.parquet")
        controls = build_controls(repo)
        merged = panel.merge(controls, on=["module_id", "month"], how="inner")
        merged["repo"] = repo
        merged["module_id_global"] = repo + "::" + merged["module_id"]
        frames.append(merged)

    df = pd.concat(frames, ignore_index=True)

    # A = A_prime(trailer込み, gates.md Loop1で確定した主指標)を使う
    df["A_main"] = df["A_prime"]
    df["log_n_files"] = np.log1p(df["n_files"])
    df["log_n_events"] = np.log1p(df["n_events"])
    df["log_n_unique_actors"] = np.log1p(df["n_unique_actors"])

    model_df = df.dropna(subset=["H_E", "A_main", "log_n_files", "log_n_events", "log_n_unique_actors"]).copy()
    model_df["month_period"] = pd.PeriodIndex(model_df["month"], freq="M").to_timestamp()
    model_df = model_df.set_index(["module_id_global", "month_period"])

    # log_n_files はモジュール固定効果に完全吸収される(時間不変のため)ので、メインモデルには入れない
    exog_cols = ["A_main", "log_n_events", "log_n_unique_actors"]
    exog = model_df[exog_cols]
    endog = model_df["H_E"]

    mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
    res_cluster_entity = mod.fit(cov_type="clustered", cluster_entity=True)

    n_repo_clusters = model_df.reset_index()["module_id_global"].str.split("::").str[0].nunique()

    summary = {
        "n_obs": int(res_cluster_entity.nobs),
        "n_modules(global)": int(model_df.index.get_level_values(0).nunique()),
        "n_months": int(model_df.index.get_level_values(1).nunique()),
        "n_repo_clusters": n_repo_clusters,
        "params": res_cluster_entity.params.to_dict(),
        "std_errors(cluster=module)": res_cluster_entity.std_errors.to_dict(),
        "pvalues(cluster=module)": res_cluster_entity.pvalues.to_dict(),
        "rsquared_within": res_cluster_entity.rsquared_within,
    }

    # リポジトリ単位クラスタ標準誤差(研究設計書§5.1が指定する本来の方法)。n_repo_clusters>=8になったので試す
    if n_repo_clusters >= 5:
        repo_series = model_df.reset_index()["module_id_global"].str.split("::").str[0]
        repo_series.index = model_df.index
        mod2 = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
        res_cluster_repo = mod2.fit(cov_type="clustered", clusters=repo_series)
        summary["std_errors(cluster=repo, 設計書指定の本来の方法)"] = res_cluster_repo.std_errors.to_dict()
        summary["pvalues(cluster=repo)"] = res_cluster_repo.pvalues.to_dict()
        print(res_cluster_repo)
        print()

    # 補助分析: モジュール規模(n_files)の効果は between推定(モジュール単位平均, repo固定効果込みpooled OLS)で見る
    mod_avg = (
        model_df.reset_index()
        .groupby("module_id_global")
        .agg(
            H_E=("H_E", "mean"),
            A_main=("A_main", "mean"),
            log_n_files=("log_n_files", "mean"),
            log_n_events=("log_n_events", "mean"),
            log_n_unique_actors=("log_n_unique_actors", "mean"),
        )
    )
    mod_avg["repo"] = mod_avg.index.str.split("::").str[0]
    repo_dummies = pd.get_dummies(mod_avg["repo"], drop_first=True, dtype=float)
    X_between = sm.add_constant(pd.concat([mod_avg[["A_main", "log_n_files", "log_n_events", "log_n_unique_actors"]], repo_dummies], axis=1))
    between_res = sm.OLS(mod_avg["H_E"], X_between).fit(cov_type="HC1")
    summary["between_estimation(モジュール規模の効果, repo固定効果込みpooled OLS)"] = {
        "params": between_res.params.to_dict(),
        "pvalues": between_res.pvalues.to_dict(),
        "n_obs": int(between_res.nobs),
    }

    out_path = PROCESSED_DIR / "loop8_main_model_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    print(res_cluster_entity)
    print()
    print("=== Between推定(モジュール規模の効果) ===")
    print(between_res.summary())
    print()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
