"""D5: PR 単位のアウトカムを作る(人間の負担・品質・選択バイアス対策)。

既存の rq3_human_burden.parquet は「人間の APPROVED があった PR」しか含まないため、
「AI 導入後に承認されない PR が増えた」ような選択が起きていると見落とす。
ここでは全 PR を対象に、承認の有無そのものもアウトカムとして持たせる。

出力: pilot/phase2/outputs/d5_pr_level.parquet
  module_id_global, repo, month, pr_number
  has_human_approval, approval_lag_days, merged, time_to_merge_days, closed_unmerged
  n_ai_reviews_before_approval, n_human_reviews_before_approval
  n_ai_reviews_total, n_human_reviews_total, n_review_rounds
  pr_lines, pr_files

実行: uv run python pilot/phase2/scripts/d5_pr_outcomes.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from common import INTERIM_DIR, OUT_DIR, RAW_DIR, REPO_ROOT, load_repos, repo_dir_name

# RAW_DIR は common 側で環境変数 PILOT_RAW により切り替わる


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


def compute_for_repo(repo: str) -> pd.DataFrame | None:
    interim_d = INTERIM_DIR / repo_dir_name(repo)
    prs_path = RAW_DIR / repo_dir_name(repo) / "prs_full.jsonl"
    actors_path = interim_d / "actors.parquet"
    events_path = interim_d / "events.parquet"
    if not (prs_path.exists() and actors_path.exists() and events_path.exists()):
        return None

    actors = pd.read_parquet(actors_path)
    cat_of = dict(zip(actors["actor_login"], actors["category"]))
    events = pd.read_parquet(events_path)
    g = events[events["stage"] == "G"]
    pr_to_module = dict(zip(g["pr_number"], g["module_id"]))

    rows = []
    for pr in read_jsonl(prs_path):
        number = pr["number"]
        module_id = pr_to_module.get(number)
        if module_id is None:
            continue
        created_at = pr.get("createdAt")
        if not created_at:
            continue
        created = parse_dt(created_at)

        parsed = []
        for rv in (pr.get("reviews") or {}).get("nodes", []):
            if not rv.get("submittedAt") or not rv.get("author"):
                continue
            parsed.append(
                (parse_dt(rv["submittedAt"]), cat_of.get(rv["author"]["login"], "unknown"), rv.get("state"))
            )
        parsed.sort(key=lambda x: x[0])

        approval_idx = None
        for i, (_, cat, state) in enumerate(parsed):
            if cat == "human" and state == "APPROVED":
                approval_idx = i
                break

        if approval_idx is None:
            lag = None
            n_ai_before = None
            n_human_before = None
        else:
            lag = (parsed[approval_idx][0] - created).total_seconds() / 86400
            if lag < 0:
                lag = None
            n_ai_before = sum(1 for _, c, _ in parsed[:approval_idx] if c == "agent")
            n_human_before = sum(1 for _, c, _ in parsed[:approval_idx] if c == "human")

        merged_at = pr.get("mergedAt")
        ttm = (parse_dt(merged_at) - created).total_seconds() / 86400 if merged_at else None
        if ttm is not None and ttm < 0:
            ttm = None

        # レビューの種別(state)を担い手カテゴリ別に数える。
        # 「AI が先に指摘するから人間は指摘しなくてよくなる」なら
        # 人間の CHANGES_REQUESTED が減るはずで、メカニズムの手がかりになる
        def count_state(cat: str, state: str) -> int:
            return sum(1 for _, c, s in parsed if c == cat and s == state)

        files = (pr.get("files") or {}).get("nodes", [])
        rows.append(
            {
                "repo": repo,
                "module_id": module_id,
                "month": created_at[:7],
                "pr_number": number,
                "has_human_approval": approval_idx is not None,
                "approval_lag_days": lag,
                "n_ai_reviews_before_approval": n_ai_before,
                "n_human_reviews_before_approval": n_human_before,
                "n_ai_reviews_total": sum(1 for _, c, _ in parsed if c == "agent"),
                "n_human_reviews_total": sum(1 for _, c, _ in parsed if c == "human"),
                "n_review_rounds": len(parsed),
                "n_human_approved": count_state("human", "APPROVED"),
                "n_human_changes_requested": count_state("human", "CHANGES_REQUESTED"),
                "n_human_commented": count_state("human", "COMMENTED"),
                "n_ai_approved": count_state("agent", "APPROVED"),
                "n_ai_changes_requested": count_state("agent", "CHANGES_REQUESTED"),
                "n_ai_commented": count_state("agent", "COMMENTED"),
                "merged": merged_at is not None,
                "time_to_merge_days": ttm,
                "closed_unmerged": (pr.get("state") == "CLOSED") and merged_at is None,
                "pr_lines": sum(f.get("additions", 0) + f.get("deletions", 0) for f in files),
                "pr_files": len(files),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def main() -> None:
    frames = []
    for repo in load_repos():
        df = compute_for_repo(repo)
        if df is not None:
            frames.append(df)
            print(f"  {repo}: {len(df)} PRs")
    all_df = pd.concat(frames, ignore_index=True)
    all_df["module_id_global"] = all_df["repo"] + "::" + all_df["module_id"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "d5_pr_level.parquet"
    all_df.to_parquet(out_path, index=False)

    print(f"\n総 PR 数: {len(all_df)}  リポジトリ: {all_df['repo'].nunique()}")
    print(f"人間 APPROVED あり: {all_df['has_human_approval'].mean():.1%}")
    print(f"マージ済み: {all_df['merged'].mean():.1%}")
    print(f"クローズ(未マージ): {all_df['closed_unmerged'].mean():.1%}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
