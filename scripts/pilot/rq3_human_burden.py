"""追加分析: 「コメントはAI、最終承認は人間」構造での人間の負担を測る。

ユーザー仮説: AIがレビューコメントの多くを担うようになった結果、
人間の最終承認者はより多くのAIコメントを読んでから判断する必要があり、
承認までの時間も延びているのではないか。

指標:
  n_ai_reviews_before_approval: そのPRで、人間の最初のAPPROVEDより前に付いていたagentレビュー数
  human_approval_lag_days: PR作成から人間の最初のAPPROVEDまでの日数

これらをモジュールのAI関与度(A_prime)や分離度(S1_EG_prime)と関係づける。

実行: uv run python scripts/pilot/rq3_human_burden.py
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
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


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def compute_for_repo(repo: str, actor_category: dict[str, str]) -> pd.DataFrame | None:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    raw_d = RAW_DIR / repo_dir_name(repo)
    events_path = interim_d / "events.parquet"
    prs_path = raw_d / "prs_full.jsonl"
    if not events_path.exists() or not prs_path.exists():
        return None

    events = pd.read_parquet(events_path)
    g_events = events[events["stage"] == "G"]
    pr_to_module = dict(zip(g_events["pr_number"], g_events["module_id"]))

    rows = []
    for pr in read_jsonl(prs_path):
        number = pr["number"]
        module_id = pr_to_module.get(number)
        if module_id is None:
            continue
        reviews = (pr.get("reviews") or {}).get("nodes", [])
        if not reviews:
            continue
        # submittedAtでソート
        parsed = []
        for rv in reviews:
            if not rv.get("submittedAt") or not rv.get("author"):
                continue
            login = rv["author"]["login"]
            cat = actor_category.get(login, "unknown")
            parsed.append((parse_dt(rv["submittedAt"]), cat, rv.get("state")))
        parsed.sort(key=lambda x: x[0])

        # 最初の human APPROVED を探す
        human_approval_idx = None
        for i, (dt, cat, state) in enumerate(parsed):
            if cat == "human" and state == "APPROVED":
                human_approval_idx = i
                break
        if human_approval_idx is None:
            continue

        n_ai_before = sum(1 for dt, cat, state in parsed[:human_approval_idx] if cat == "agent")
        n_human_before = sum(1 for dt, cat, state in parsed[:human_approval_idx] if cat == "human")

        created_at = pr.get("createdAt")
        if not created_at:
            continue
        approval_dt = parsed[human_approval_idx][0]
        lag_days = (approval_dt - parse_dt(created_at)).total_seconds() / 86400
        if lag_days < 0:
            continue

        # PR規模(複雑度の統制用): 変更行数・変更ファイル数
        files = (pr.get("files") or {}).get("nodes", [])
        pr_lines = sum(f.get("additions", 0) + f.get("deletions", 0) for f in files)
        pr_files = len(files)

        rows.append(
            {
                "module_id": module_id,
                "month": created_at[:7],
                "pr_lines": pr_lines,
                "pr_files": pr_files,
                "pr_number": number,
                "n_ai_reviews_before_approval": n_ai_before,
                "n_human_reviews_before_approval": n_human_before,
                "human_approval_lag_days": lag_days,
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
        actors_path = INTERIM_DIR / repo_dir_name(repo) / "actors.parquet"
        if not actors_path.exists():
            continue
        actors = pd.read_parquet(actors_path)
        actor_category = dict(zip(actors["actor_login"], actors["category"]))
        df = compute_for_repo(repo, actor_category)
        if df is None:
            continue
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "rq3_human_burden.parquet"
    all_df.to_parquet(out_path, index=False)

    summary_basic = {
        "n_repos_used": len(frames),
        "n_prs_with_human_approval": len(all_df),
        "n_ai_reviews_before_approval_mean": float(all_df["n_ai_reviews_before_approval"].mean()),
        "pct_prs_with_any_ai_review_before_approval": float((all_df["n_ai_reviews_before_approval"] > 0).mean()),
        "human_approval_lag_days_mean": float(all_df["human_approval_lag_days"].mean()),
        "human_approval_lag_days_median": float(all_df["human_approval_lag_days"].median()),
    }

    # モジュール×月に集計し、A_prime(生成側AI関与度)との関係を見る
    agg = (
        all_df.groupby(["repo", "module_id", "month"])
        .agg(
            avg_ai_reviews_before_approval=("n_ai_reviews_before_approval", "mean"),
            avg_approval_lag_days=("human_approval_lag_days", "mean"),
            avg_pr_lines=("pr_lines", "mean"),
            avg_pr_files=("pr_files", "mean"),
            n_prs=("pr_number", "count"),
        )
        .reset_index()
    )

    panel_frames = []
    for repo in repos:
        panel_path = PROCESSED_DIR / repo_dir_name(repo) / "panel.parquet"
        if panel_path.exists():
            p = pd.read_parquet(panel_path)[["module_id", "month", "A_prime"]]
            p["repo"] = repo
            panel_frames.append(p)
    panel_all = pd.concat(panel_frames, ignore_index=True)

    model_df = agg.merge(panel_all, on=["module_id", "month", "repo"], how="left")
    model_df = model_df.dropna(subset=["avg_ai_reviews_before_approval", "A_prime", "avg_pr_lines"])
    model_df["log_ai_reviews"] = np.log1p(model_df["avg_ai_reviews_before_approval"])
    model_df["log_n_prs"] = np.log1p(model_df["n_prs"])
    model_df["log_pr_lines"] = np.log1p(model_df["avg_pr_lines"])  # PR複雑度の統制(変更行数)
    model_df["log_pr_files"] = np.log1p(model_df["avg_pr_files"])  # PR複雑度の統制(変更ファイル数)
    model_df["module_id_global"] = model_df["repo"] + "::" + model_df["module_id"]
    model_df["month_period"] = pd.PeriodIndex(model_df["month"], freq="M").to_timestamp()
    idx_df = model_df.set_index(["module_id_global", "month_period"])

    def run_within(dep_col: str, with_pr_size_control: bool) -> dict:
        cols = ["A_prime", "log_n_prs"]
        if with_pr_size_control:
            cols += ["log_pr_lines", "log_pr_files"]
        exog = idx_df[cols]
        endog = idx_df[dep_col]
        mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
        res = mod.fit(cov_type="clustered", cluster_entity=True)
        n_repo_clusters = idx_df.reset_index()["repo"].nunique()
        out = {
            "n_obs": int(res.nobs),
            "n_repo_clusters": int(n_repo_clusters),
            "params": res.params.to_dict(),
            "pvalues(cluster=module)": res.pvalues.to_dict(),
        }
        if n_repo_clusters >= 5:
            repo_series = idx_df.reset_index()["repo"]
            repo_series.index = idx_df.index
            mod2 = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
            res2 = mod2.fit(cov_type="clustered", clusters=repo_series)
            out["pvalues(cluster=repo)"] = res2.pvalues.to_dict()
        return out

    model_ai_reviews = run_within("log_ai_reviews", with_pr_size_control=False)
    model_lag = run_within("avg_approval_lag_days", with_pr_size_control=False)
    model_ai_reviews_ctrl = run_within("log_ai_reviews", with_pr_size_control=True)
    model_lag_ctrl = run_within("avg_approval_lag_days", with_pr_size_control=True)

    summary = {
        **summary_basic,
        "model(log(1+AIレビュー数_承認前) ~ A_prime + log_n_prs + FE)": model_ai_reviews,
        "model(承認までの日数 ~ A_prime + log_n_prs + FE)": model_lag,
        "model_PR規模統制済み(log(1+AIレビュー数_承認前) ~ A_prime + log_n_prs + log_pr_lines + log_pr_files + FE)": model_ai_reviews_ctrl,
        "model_PR規模統制済み(承認までの日数 ~ A_prime + log_n_prs + log_pr_lines + log_pr_files + FE)": model_lag_ctrl,
    }
    (PROCESSED_DIR / "rq3_human_burden_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
