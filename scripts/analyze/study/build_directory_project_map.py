"""raw_json から directory → 主要 project マッピングを再構築する。

pair_predictions.csv の `directory` 列はファイルパスの先頭 2 セグメント
(例: `doc/source`, `nova/compute`) で project 情報が欠落している。
本スクリプトでは 32 main repos の raw_json から各 change のファイルパスを抽出し、
directory ごとにどの project (= リポジトリ) に最も多く現れるかを集計する。

出力 (outputs/analysis/per_repo_basic_stats/):
  - directory_project_map.csv  : directory ごとの project 出現数 + 主要 project + 純度
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def extract_dir(file_path: str) -> str:
    """ファイルパスから top-2 segments を抽出。1 segment しかなければそのまま返す。"""
    parts = file_path.split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def process_repo_json(json_path: Path) -> Counter:
    """1 つの raw_json から (project, directory) ごとの change 出現数を集計。"""
    counter: Counter = Counter()
    if not json_path.exists():
        return counter
    try:
        data = json.loads(json_path.read_text())
    except Exception as e:
        print(f"  failed to parse {json_path.name}: {e}")
        return counter
    if not isinstance(data, list):
        return counter
    for change in data:
        project = change.get("project", "")
        if not project:
            continue
        repo = project.split("/")[-1]  # openstack/nova -> nova
        revs = change.get("revisions", {})
        if not isinstance(revs, dict):
            continue
        # 最新 revision の files を使う (created 最大)
        latest = None
        latest_num = -1
        for rev in revs.values():
            n = rev.get("_number", 0)
            if isinstance(n, int) and n > latest_num:
                latest_num = n
                latest = rev
        if latest is None:
            continue
        files = latest.get("files", {})
        if not isinstance(files, dict):
            continue
        # ファイル単位でなく change 単位で count (重複カウント避ける)
        dirs_in_change = set()
        for fp in files.keys():
            if fp == "/COMMIT_MSG":
                continue
            dirs_in_change.add(extract_dir(fp))
        for d in dirs_in_change:
            counter[(d, repo)] += 1
    return counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-json-dir", default="data/raw_json")
    ap.add_argument("--main-repos-csv", default="data/service_teams_main_repos.csv")
    ap.add_argument("--output", default="outputs/analysis/per_repo_basic_stats/directory_project_map.csv")
    args = ap.parse_args()

    mr_df = pd.read_csv(args.main_repos_csv).dropna(subset=["main_repo"])
    main_repos = sorted(set(mr_df["main_repo"].astype(str).str.split("/").str[-1]) - {"nan"})
    print(f"target main repos: {len(main_repos)}")

    counter_total: Counter = Counter()
    for repo in main_repos:
        json_path = Path(args.raw_json_dir) / f"openstack__{repo}.json"
        cnt = process_repo_json(json_path)
        print(f"  {repo}: {sum(cnt.values()):>6} change-dir entries from {json_path.name}")
        counter_total.update(cnt)

    # DataFrame 化: (directory, project, n_changes)
    rows = [
        {"directory": d, "project": p, "n_changes": n}
        for (d, p), n in counter_total.items()
    ]
    df = pd.DataFrame(rows)
    print(f"\ntotal (directory, project) pairs: {len(df)}")

    # directory ごとに集計
    summary_rows = []
    for d, g in df.groupby("directory"):
        g_sorted = g.sort_values("n_changes", ascending=False)
        total = int(g_sorted["n_changes"].sum())
        top1 = g_sorted.iloc[0]
        top1_project = str(top1["project"])
        top1_n = int(top1["n_changes"])
        top1_pct = top1_n / total * 100 if total else 0
        top3 = g_sorted.head(3)
        top3_str = ", ".join(
            f"{r['project']}({int(r['n_changes'])})" for _, r in top3.iterrows()
        )
        n_projects = int(g_sorted["project"].nunique())
        summary_rows.append({
            "directory": d,
            "total_changes": total,
            "n_projects": n_projects,
            "top1_project": top1_project,
            "top1_n": top1_n,
            "top1_pct": top1_pct,
            "top3": top3_str,
            "category": "single" if top1_pct >= 80 else ("dominant" if top1_pct >= 50 else "shared"),
        })
    summary = pd.DataFrame(summary_rows).sort_values("total_changes", ascending=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")

    # 主要 directory 例
    print("\n=== サンプル: top 20 by total_changes ===")
    print(summary.head(20)[
        ["directory", "total_changes", "n_projects", "top1_project", "top1_pct", "category"]
    ].to_string(index=False))
    print("\n=== shared (top1 < 50%) の例 ===")
    shared = summary[summary["category"] == "shared"].head(15)
    print(shared[
        ["directory", "total_changes", "n_projects", "top3"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
