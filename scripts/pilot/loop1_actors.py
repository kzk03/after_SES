"""Loop 1 — 担い手同定。

pilot/raw/<repo>/{issues_full,prs_full}.jsonl から全actorを抽出し、
pilot/config/agent_identifiers.yaml に基づいて human / agent / legacy_bot に分類する。

実行: uv run python scripts/pilot/loop1_actors.py <owner>/<repo>
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "pilot" / "raw"
INTERIM_DIR = REPO_ROOT / "pilot" / "interim"
REPORTS_DIR = REPO_ROOT / "pilot" / "reports"
CONFIG_PATH = REPO_ROOT / "pilot" / "config" / "agent_identifiers.yaml"

random.seed(42)


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def normalize_login(login: str) -> str:
    return re.sub(r"\[bot\]$", "", login).strip()


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def classify(login: str, agent_accounts: set[str], legacy_accounts: set[str], legacy_patterns: list[re.Pattern]) -> str:
    norm = normalize_login(login).lower()
    if norm in agent_accounts:
        return "agent"
    if norm in legacy_accounts:
        return "legacy_bot"
    for pat in legacy_patterns:
        if pat.match(login):
            return "legacy_bot"
    return "human"


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
                continue  # 収集中(バックグラウンド書き込み)の不完全な最終行をスキップ


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: loop1_actors.py <owner>/<repo>")
        sys.exit(1)
    repo = sys.argv[1]
    d = RAW_DIR / repo_dir_name(repo)

    cfg = load_config()
    agent_accounts = {a.lower() for a in cfg["agent"]["accounts"]}
    legacy_accounts = {a.lower() for a in cfg["legacy_bot"]["accounts"]}
    legacy_patterns = [re.compile(p) for p in cfg["legacy_bot"].get("patterns", [])]
    trailer_patterns = [re.compile(p, re.IGNORECASE) for p in cfg["agent"].get("trailer_patterns", [])]

    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    trailer_hits: dict[str, int] = defaultdict(int)

    for issue in read_jsonl(d / "issues_full.jsonl"):
        author = issue.get("author")
        if author:
            stats[author["login"]]["n_issue_authored"] += 1

    n_prs = 0
    for pr in read_jsonl(d / "prs_full.jsonl"):
        n_prs += 1
        author = pr.get("author")
        if author:
            stats[author["login"]]["n_pr_authored"] += 1
        for rv in (pr.get("reviews") or {}).get("nodes", []):
            rv_author = rv.get("author")
            if rv_author:
                stats[rv_author["login"]]["n_reviews_submitted"] += 1
        for c in (pr.get("commits") or {}).get("nodes", []):
            commit = c["commit"]
            user = (commit.get("author") or {}).get("user")
            login = user["login"] if user else None
            if login:
                stats[login]["n_commits"] += 1
            msg = commit.get("message", "")
            for pat in trailer_patterns:
                m = pat.search(msg)
                if m:
                    trailer_hits[pat.pattern] += 1

    rows = []
    for login, s in stats.items():
        category = classify(login, agent_accounts, legacy_accounts, legacy_patterns)
        rows.append(
            {
                "actor_login": login,
                "category": category,
                "n_issue_authored": s.get("n_issue_authored", 0),
                "n_pr_authored": s.get("n_pr_authored", 0),
                "n_reviews_submitted": s.get("n_reviews_submitted", 0),
                "n_commits": s.get("n_commits", 0),
            }
        )
    df = pd.DataFrame(rows).sort_values("n_pr_authored", ascending=False).reset_index(drop=True)

    interim_d = INTERIM_DIR / repo_dir_name(repo)
    interim_d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(interim_d / "actors.parquet", index=False)

    # 目視検証用サンプル: 各カテゴリから最大50件を無作為抽出
    sample_rows = []
    for cat in ["human", "agent", "legacy_bot"]:
        sub = df[df["category"] == cat]
        n = min(50, len(sub))
        sample_rows.append(sub.sample(n=n, random_state=42) if n > 0 else sub)
    sample_df = pd.concat(sample_rows) if sample_rows else df.iloc[0:0]

    summary = {
        "repo": repo,
        "n_prs_scanned": n_prs,
        "n_unique_actors": len(df),
        "category_counts": df["category"].value_counts().to_dict(),
        "trailer_pattern_hits": dict(trailer_hits),
    }
    (interim_d / "loop1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_df.to_csv(interim_d / "loop1_visual_check_sample.csv", index=False)

    print(f"=== {repo} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"actors.parquet -> {interim_d / 'actors.parquet'}")
    print(f"目視検証サンプル -> {interim_d / 'loop1_visual_check_sample.csv'}")


if __name__ == "__main__":
    main()
