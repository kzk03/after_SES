"""Loop 3 — 工程の帰属。

PR(生成G・評価E)とissue(意図I)をモジュールに帰属させ、pilot/interim/<repo>/events.parquet を作る。

方針(研究設計書 §3.3 / パイロット手順.md Loop3 の「まず」の決め方に従う):
  - 跨るPR: 変更行数(additions+deletions)最大のモジュールに寄せる(主モジュール方式)
  - リンクなしissue: 除外
  - モジュール定義は Loop2 で作った「最新月」のfile->module割当を全期間に対して適用する近似
    (CODEOWNERSの経年変化を厳密に反映していない。感度分析の対象としてLoop8で扱う)

実行: uv run python scripts/pilot/loop3_stage.py <owner>/<repo>
"""
from __future__ import annotations

import json
import re
import sys
import os
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "pilot" / os.environ.get("PILOT_RAW", "raw")
INTERIM_DIR = REPO_ROOT / "pilot" / os.environ.get("PILOT_INTERIM", "interim")
CONFIG_PATH = REPO_ROOT / "pilot" / "config" / "agent_identifiers.yaml"


def repo_dir_name(repo: str) -> str:
    return repo.replace("/", "__")


def load_trailer_patterns() -> list[re.Pattern]:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    return [re.compile(p, re.IGNORECASE) for p in cfg["agent"].get("trailer_patterns", [])]


def pr_has_agent_trailer(pr: dict, patterns: list[re.Pattern]) -> bool:
    for c in (pr.get("commits") or {}).get("nodes", []):
        msg = c["commit"].get("message", "")
        if any(p.search(msg) for p in patterns):
            return True
    return False


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


def month_of(iso_ts: str) -> str:
    return iso_ts[:7]  # "YYYY-MM"


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: loop3_stage.py <owner>/<repo>")
        sys.exit(1)
    repo = sys.argv[1]
    raw_d = RAW_DIR / repo_dir_name(repo)
    interim_d = INTERIM_DIR / repo_dir_name(repo)

    modules_df = pd.read_parquet(interim_d / "modules.parquet")
    file_to_module = dict(zip(modules_df["file"], modules_df["module_id"]))

    actors_df = pd.read_parquet(interim_d / "actors.parquet")
    actor_category = dict(zip(actors_df["actor_login"], actors_df["category"]))
    trailer_patterns = load_trailer_patterns()

    events = []
    pr_module: dict[int, str] = {}  # PR番号 -> 主モジュール
    n_pr_total = 0
    n_pr_matched = 0
    n_pr_multi_module = 0  # 2つ以上の異なるモジュールにファイルが跨っているPR数(参考値)

    for pr in read_jsonl(raw_d / "prs_full.jsonl"):
        n_pr_total += 1
        number = pr["number"]
        created_at = pr["createdAt"]
        mon = month_of(created_at)

        # 主モジュール方式: モジュールごとの変更行数(additions+deletions)を集計し最大を採用
        module_lines: dict[str, int] = {}
        # 変更ファイル数が非常に多いPRではGraphQLがfilesにnullを返すことがある(GitHub側の既知の制限)
        for f in (pr.get("files") or {}).get("nodes", []):
            mod = file_to_module.get(f["path"])
            if mod is None:
                continue
            module_lines[mod] = module_lines.get(mod, 0) + f.get("additions", 0) + f.get("deletions", 0)
        if not module_lines:
            continue  # このPRはどのモジュールにも帰属できない
        if len(module_lines) >= 2:
            n_pr_multi_module += 1
        main_module = max(module_lines, key=module_lines.get)
        pr_module[number] = main_module
        n_pr_matched += 1

        has_trailer = pr_has_agent_trailer(pr, trailer_patterns)
        author = pr.get("author")
        if author:
            login = author["login"]
            events.append(
                {
                    "stage": "G",
                    "actor_login": login,
                    "actor_category": actor_category.get(login, "unknown"),
                    "module_id": main_module,
                    "month": mon,
                    "pr_number": number,
                    "trailer_agent_detected": has_trailer,
                }
            )
        for rv in (pr.get("reviews") or {}).get("nodes", []):
            rv_author = rv.get("author")
            if not rv_author:
                continue
            login = rv_author["login"]
            rv_mon = month_of(rv["submittedAt"]) if rv.get("submittedAt") else mon
            events.append(
                {
                    "stage": "E",
                    "actor_login": login,
                    "actor_category": actor_category.get(login, "unknown"),
                    "module_id": main_module,
                    "month": rv_mon,
                    "pr_number": number,
                }
            )

    n_issue_total = 0
    n_issue_linked = 0
    n_issue_matched = 0
    for issue in read_jsonl(raw_d / "issues_full.jsonl"):
        n_issue_total += 1
        linked = issue.get("closedByPullRequestsReferences", {}).get("nodes", [])
        if not linked:
            continue  # リンクなしissueは除外(未決事項。まず除外の方針)
        n_issue_linked += 1
        # 最初にリンクされたPRのモジュールを採用
        linked_pr_number = linked[0]["number"]
        mod = pr_module.get(linked_pr_number)
        if mod is None:
            continue
        n_issue_matched += 1
        author = issue.get("author")
        if not author:
            continue
        login = author["login"]
        events.append(
            {
                "stage": "I",
                "actor_login": login,
                "actor_category": actor_category.get(login, "unknown"),
                "module_id": mod,
                "month": month_of(issue["createdAt"]),
                "pr_number": linked_pr_number,
            }
        )

    events_df = pd.DataFrame(events)
    events_df.to_parquet(interim_d / "events.parquet", index=False)

    summary = {
        "repo": repo,
        "n_pr_total": n_pr_total,
        "n_pr_matched_to_module": n_pr_matched,
        "pr_match_rate": n_pr_matched / n_pr_total if n_pr_total else None,
        "n_pr_multi_module": n_pr_multi_module,
        "pr_multi_module_rate": n_pr_multi_module / n_pr_matched if n_pr_matched else None,
        "n_issue_total": n_issue_total,
        "n_issue_linked_to_pr": n_issue_linked,
        "issue_link_rate": n_issue_linked / n_issue_total if n_issue_total else None,
        "n_issue_matched_to_module": n_issue_matched,
        "issue_module_attribution_rate": n_issue_matched / n_issue_total if n_issue_total else None,
        "n_events": len(events_df),
        "events_by_stage": events_df["stage"].value_counts().to_dict() if len(events_df) else {},
    }
    (interim_d / "loop3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"=== {repo} ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"events.parquet -> {interim_d / 'events.parquet'}")


if __name__ == "__main__":
    main()
