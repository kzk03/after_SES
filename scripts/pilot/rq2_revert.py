"""RQ2 — 差し戻し率(補助)の実測。

研究設計書§5.3: 差し戻し率(m,t+k) ~ 分離度(m,t) + controls + FE
  差し戻し率の定義: revertコミット数 / 全コミット数(m,tで)

revert検出: commit messageに "revert" を含む(大文字小文字無視)。
Git標準の "Revert \"...\"" / "This reverts commit <sha>" パターンを広く拾う簡易ルール。

実行: uv run python scripts/pilot/rq2_revert.py
"""
from __future__ import annotations

import csv
import json
import re
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

REVERT_PATTERN = re.compile(r"^revert\b|this reverts commit", re.IGNORECASE)


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


def compute_revert_for_repo(repo: str) -> pd.DataFrame | None:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    raw_d = RAW_DIR / repo_dir_name(repo)
    events_path = interim_d / "events.parquet"
    prs_path = raw_d / "prs_full.jsonl"
    if not events_path.exists() or not prs_path.exists():
        return None

    events = pd.read_parquet(events_path)
    g_events = events[events["stage"] == "G"]
    pr_to_module = dict(zip(g_events["pr_number"], g_events["module_id"]))

    counts: dict[tuple[str, str], dict[str, int]] = {}
    for pr in read_jsonl(prs_path):
        number = pr["number"]
        module_id = pr_to_module.get(number)
        if module_id is None:
            continue
        for c in (pr.get("commits") or {}).get("nodes", []):
            commit = c["commit"]
            msg = commit.get("message", "")
            date = commit.get("committedDate")
            if not date:
                continue
            month = date[:7]
            key = (module_id, month)
            d = counts.setdefault(key, {"n_commits": 0, "n_revert": 0})
            d["n_commits"] += 1
            if REVERT_PATTERN.search(msg):
                d["n_revert"] += 1

    if not counts:
        return None

    rows = [{"module_id": k[0], "month": k[1], **v} for k, v in counts.items()]
    df = pd.DataFrame(rows)
    df["revert_rate"] = df["n_revert"] / df["n_commits"]
    df["repo"] = repo
    return df


def main() -> None:
    repos = load_repos()
    frames = []
    for repo in repos:
        df = compute_revert_for_repo(repo)
        if df is not None:
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    out_path = PROCESSED_DIR / "rq2_revert.parquet"
    all_df.to_parquet(out_path, index=False)

    summary = {
        "n_repos_used": len(frames),
        "n_obs": len(all_df),
        "revert_rate_mean": float(all_df["revert_rate"].mean()),
        "n_commits_total": int(all_df["n_commits"].sum()),
        "n_revert_total": int(all_df["n_revert"].sum()),
    }

    # separation_axesとマージして固定効果回帰
    sep_frames = []
    for repo in repos:
        sep_path = PROCESSED_DIR / repo_dir_name(repo) / "separation_axes.parquet"
        if sep_path.exists():
            s = pd.read_parquet(sep_path)[["module_id", "month", "S1_EG_prime"]]
            s["repo"] = repo
            sep_frames.append(s)
    sep_all = pd.concat(sep_frames, ignore_index=True)

    model_df = all_df.merge(sep_all, on=["module_id", "month", "repo"], how="left")
    model_df = model_df.dropna(subset=["revert_rate", "S1_EG_prime"])
    model_df = model_df[model_df["n_commits"] >= 3]  # 分母が小さすぎるセルは除外

    model_df["log_n_commits"] = np.log1p(model_df["n_commits"])
    model_df["module_id_global"] = model_df["repo"] + "::" + model_df["module_id"]
    model_df["month_period"] = pd.PeriodIndex(model_df["month"], freq="M").to_timestamp()
    idx_df = model_df.set_index(["module_id_global", "month_period"])

    exog = idx_df[["S1_EG_prime", "log_n_commits"]]
    endog = idx_df["revert_rate"]
    mod = PanelOLS(endog, exog, entity_effects=True, time_effects=True, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)

    n_repo_clusters = idx_df.reset_index()["repo"].nunique()
    model_summary = {
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
        model_summary["pvalues(cluster=repo)"] = res2.pvalues.to_dict()
        model_summary["std_errors(cluster=repo)"] = res2.std_errors.to_dict()

    summary["model"] = model_summary
    (PROCESSED_DIR / "rq2_revert_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
