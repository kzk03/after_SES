"""Loop 7 — 各候補リポジトリのAを軽量に推定し、3層(高/中/低)に層化する。

全期間平均のAを厳密に計算するには全PR収集が要り非常に重いため、
直近100件のPRサンプルで「agent(login or trailer)によるPRの割合」を代理指標として使う。

実行: uv run python scripts/pilot/loop7_estimate_A.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphql_client import GraphQLClient

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "pilot" / "config"

QUERY = """
query($owner:String!,$repo:String!){
  repository(owner:$owner,name:$repo){
    pullRequests(first:100, orderBy:{field:CREATED_AT, direction:DESC}){
      nodes{
        author{login}
        commits(first:10){nodes{commit{message}}}
      }
    }
  }
  rateLimit{cost remaining}
}
"""


def load_agent_matchers() -> tuple[set[str], list[re.Pattern]]:
    cfg = yaml.safe_load((CONFIG_DIR / "agent_identifiers.yaml").read_text())
    accounts = {a.lower() for a in cfg["agent"]["accounts"]}
    patterns = [re.compile(p, re.IGNORECASE) for p in cfg["agent"]["trailer_patterns"]]
    return accounts, patterns


def is_agent_pr(pr: dict, accounts: set[str], patterns: list[re.Pattern]) -> bool:
    author = pr.get("author")
    if author and author["login"].lower() in accounts:
        return True
    for c in pr.get("commits", {}).get("nodes", []):
        msg = c["commit"].get("message", "")
        if any(p.search(msg) for p in patterns):
            return True
    return False


def main() -> None:
    accounts, patterns = load_agent_matchers()
    client = GraphQLClient()

    in_path = CONFIG_DIR / "loop7_candidates_filtered.csv"
    with in_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{len(rows)}件のリポジトリでAを推定中...")
    results = []
    for i, row in enumerate(rows):
        owner, name = row["full_name"].split("/")
        try:
            data = client.query(QUERY, {"owner": owner, "repo": name})
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] {row['full_name']}: {e}")
            continue
        prs = data["repository"]["pullRequests"]["nodes"]
        if not prs:
            continue
        n_agent = sum(1 for pr in prs if is_agent_pr(pr, accounts, patterns))
        a_estimate = n_agent / len(prs)
        results.append({**row, "n_pr_sampled": len(prs), "A_estimate": round(a_estimate, 4)})
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(rows)}件")

    results.sort(key=lambda r: -r["A_estimate"])
    out_path = CONFIG_DIR / "loop7_candidates_with_A.csv"
    fieldnames = list(results[0].keys()) if results else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"-> {out_path} ({len(results)}件)")

    # 3層に層化(A_estimateの分布を3分位で区切る)
    a_values = sorted(r["A_estimate"] for r in results)
    n = len(a_values)
    if n >= 3:
        low_cut = a_values[n // 3]
        high_cut = a_values[2 * n // 3]
        print(f"層化の目安: low<{low_cut:.3f}, mid, high>={high_cut:.3f}")


if __name__ == "__main__":
    main()
