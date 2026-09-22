"""参入障壁低下 vs ホットスポット効果の切り分け(追加分析)。

「AI関与度が高いモジュールで新規参入が多い」(H2bの逆方向)の解釈を検証する:
  参入障壁低下説: AIが土台を作り、新規参入者が最初に触るタスクが小さく・簡単になっている
  ホットスポット効果説: 単に活発なモジュールに人が集まっているだけ(活動量を統制しても消えない別の交絡)

各(module, actor)で「そのモジュールへの最初のPR(生成)」を特定し、その規模(変更行数・ファイル数)を
S1_EG_prime(m, 参入月)に回帰する。参入障壁低下説が正しければ、AI関与度が高い(S1が低い)モジュールほど
初回PRの規模が小さい=係数が正(S1が上がるほど初回PR規模も上がる)になるはず。

実行: uv run python scripts/pilot/rq2_entry_task_size.py
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
RAW_DIR = REPO_ROOT / "pilot" / "raw"
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


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def compute_entry_sizes_for_repo(repo: str) -> pd.DataFrame | None:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    raw_d = RAW_DIR / repo_dir_name(repo)
    events_path = interim_d / "events.parquet"
    prs_path = raw_d / "prs_full.jsonl"
    if not events_path.exists() or not prs_path.exists():
        return None

    events = pd.read_parquet(events_path)
    g_human = events[(events["stage"] == "G") & (events["actor_category"] == "human")]
    if g_human.empty:
        return None

    g_sorted = g_human.sort_values(["month", "pr_number"])
    first_entries = g_sorted.groupby(["actor_login", "module_id"], as_index=False).first()

    pr_size: dict[int, dict[str, int]] = {}
    for pr in read_jsonl(prs_path):
        files = (pr.get("files") or {}).get("nodes", [])
        total_lines = sum(f.get("additions", 0) + f.get("deletions", 0) for f in files)
        pr_size[pr["number"]] = {"total_lines": total_lines, "n_files": len(files)}

    rows = []
    for _, row in first_entries.iterrows():
        size = pr_size.get(row["pr_number"])
        if size is None:
            continue
        rows.append(
            {
                "module_id": row["module_id"],
                "month": row["month"],
                "actor_login": row["actor_login"],
                "entry_pr_lines": size["total_lines"],
                "entry_pr_files": size["n_files"],
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
        df = compute_entry_sizes_for_repo(repo)
        if df is not None:
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    agg = (
        all_df.groupby(["repo", "module_id", "month"])
        .agg(
            avg_entry_lines=("entry_pr_lines", "mean"),
            avg_entry_files=("entry_pr_files", "mean"),
            n_new_entrants_g=("actor_login", "count"),
        )
        .reset_index()
    )
    out_path = PROCESSED_DIR / "rq2_entry_task_size.parquet"
    agg.to_parquet(out_path, index=False)

    # separation_axes + panel(活動量control)とマージ
    sep_frames, ctrl_frames = [], []
    for repo in repos:
        sep_path = PROCESSED_DIR / repo_dir_name(repo) / "separation_axes.parquet"
        panel_path = PROCESSED_DIR / repo_dir_name(repo) / "panel.parquet"
        if sep_path.exists():
            s = pd.read_parquet(sep_path)[["module_id", "month", "S1_EG_prime"]]
            s["repo"] = repo
            sep_frames.append(s)
        if panel_path.exists():
            p = pd.read_parquet(panel_path)[["module_id", "month", "H_G_n", "H_E_n"]]
            p["repo"] = repo
            ctrl_frames.append(p)
    sep_all = pd.concat(sep_frames, ignore_index=True)
    ctrl_all = pd.concat(ctrl_frames, ignore_index=True)
    ctrl_all["n_activity"] = ctrl_all["H_G_n"].fillna(0) + ctrl_all["H_E_n"].fillna(0)

    model_df = agg.merge(sep_all, on=["module_id", "month", "repo"], how="left")
    model_df = model_df.merge(ctrl_all[["module_id", "month", "repo", "n_activity"]], on=["module_id", "month", "repo"], how="left")
    model_df = model_df.dropna(subset=["avg_entry_lines", "avg_entry_files", "S1_EG_prime", "n_activity"])
    model_df = model_df[model_df["avg_entry_lines"] >= 0]

    model_df["log_entry_lines"] = np.log1p(model_df["avg_entry_lines"])
    model_df["log_entry_files"] = np.log1p(model_df["avg_entry_files"])
    model_df["log_activity"] = np.log1p(model_df["n_activity"])
    model_df["module_id_global"] = model_df["repo"] + "::" + model_df["module_id"]
    model_df["month_period"] = pd.PeriodIndex(model_df["month"], freq="M").to_timestamp()
    idx_df = model_df.set_index(["module_id_global", "month_period"])

    def run_within(dep_col: str) -> dict:
        exog = idx_df[["S1_EG_prime", "log_activity"]]
        endog = idx_df[dep_col]
        mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        n_repo_clusters = idx_df.reset_index()["repo"].nunique()
        out = {
            "n_obs": int(res.nobs),
            "n_repo_clusters": int(n_repo_clusters),
            "params": res.params.to_dict(),
            "pvalues(cluster=module)": res.pvalues.to_dict(),
            "rsquared_within": res.rsquared_within,
        }
        if n_repo_clusters >= 5:
            repo_series = idx_df.reset_index()["repo"]
            repo_series.index = idx_df.index
            mod2 = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
            res2 = mod2.fit(cov_type="clustered", clusters=repo_series)
            out["pvalues(cluster=repo)"] = res2.pvalues.to_dict()
            out["std_errors(cluster=repo)"] = res2.std_errors.to_dict()
        return out

    within_lines = run_within("log_entry_lines")
    within_files = run_within("log_entry_files")

    # between推定(モジュール単位平均、repo固定効果込みpooled OLS)。Loop8主解析と同じ手法で頑健性を追加確認
    mod_avg = (
        model_df.groupby("module_id_global")
        .agg(
            log_entry_lines=("log_entry_lines", "mean"),
            log_entry_files=("log_entry_files", "mean"),
            S1_EG_prime=("S1_EG_prime", "mean"),
            log_activity=("log_activity", "mean"),
            repo=("repo", "first"),
        )
    )
    repo_dummies = pd.get_dummies(mod_avg["repo"], drop_first=True, dtype=float)

    def run_between(dep_col: str) -> dict:
        X = sm.add_constant(pd.concat([mod_avg[["S1_EG_prime", "log_activity"]], repo_dummies], axis=1))
        res = sm.OLS(mod_avg[dep_col], X).fit(cov_type="HC1")
        return {
            "n_obs": int(res.nobs),
            "S1_EG_prime_coef": float(res.params["S1_EG_prime"]),
            "S1_EG_prime_pvalue": float(res.pvalues["S1_EG_prime"]),
            "log_activity_coef": float(res.params["log_activity"]),
            "log_activity_pvalue": float(res.pvalues["log_activity"]),
        }

    between_lines = run_between("log_entry_lines")
    between_files = run_between("log_entry_files")

    summary = {
        "n_repos_used": len(frames),
        "n_obs_raw": len(agg),
        "avg_entry_lines_overall_mean": float(agg["avg_entry_lines"].mean()),
        "avg_entry_files_overall_mean": float(agg["avg_entry_files"].mean()),
        "within(log_entry_lines ~ S1_EG_prime + log_activity + module_FE + time_FE)": within_lines,
        "within(log_entry_files ~ S1_EG_prime + log_activity + module_FE + time_FE)": within_files,
        "between(log_entry_lines ~ S1_EG_prime + log_activity + repo_FE, module平均)": between_lines,
        "between(log_entry_files ~ S1_EG_prime + log_activity + repo_FE, module平均)": between_files,
    }
    (PROCESSED_DIR / "rq2_entry_task_size_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
