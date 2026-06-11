"""dev-centric raw_json から、各 dev × feature_end ごとの全期間プロフィール特徴量を計算する。

入力: data/raw_json_by_dev/<safe_email>.json (build_dataset_by_dev.py 出力)
出力: data/dev_history_features.parquet (long table: dev, feature_end, features)

特徴量:
  - historical_tenure_days       : feature_end - earliest_activity_in_window
  - historical_review_count_all  : 全 Gerrit 累積レビュー数 (feature_end まで)
  - historical_n_unique_repos    : 触った unique repo 数
  - out_of_main32_share          : 32 main 外での活動比率
  - domain_diversity             : repo の domain 別 entropy

domain 分類は `data/service_teams_repos.csv` から tier or team を使う。
"""

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_main32_repos(path: Path) -> set[str]:
    """32 main repos の project 名セット (例: 'openstack/nova')。"""
    df = pd.read_csv(path)
    for c in ("repo", "main_repo"):
        if c in df.columns:
            col = c
            break
    else:
        col = df.columns[-1]
    return set(df[col].dropna().str.lower())


def load_repo_to_team(path: Path) -> dict[str, str]:
    """244 repo → team 名のマッピング (domain として team を使う)。"""
    df = pd.read_csv(path)
    repo_col = "repo" if "repo" in df.columns else df.columns[-1]
    team_col = "team" if "team" in df.columns else df.columns[0]
    return dict(zip(df[repo_col].str.lower(), df[team_col].str.lower()))


def safe_email(email: str) -> str:
    import re
    s = email.lower().strip()
    s = re.sub(r"[^a-z0-9._-]", "_", s)
    return s[:120]


def load_dev_changes(json_path: Path) -> list[dict]:
    """build_dataset_by_dev.py の出力を読み、change list を返す。"""
    with open(json_path) as f:
        data = json.load(f)
    return data.get("changes", [])


def get_dev_role(change: dict, email: str) -> dict:
    """change と email から dev のロール / 関連メタを取り出す。

    Returns:
      {
        'role': 'owner' | 'reviewer' | 'both',
        'is_label': True/False (このペアが review 依頼レコードとして label を持つか),
        'timestamp': datetime,
        'project': str,
      }
    """
    email = email.lower()
    role = None
    owner_email = (change.get("owner", {}) or {}).get("email", "").lower()
    if owner_email == email:
        role = "owner"

    # reviewer (DETAILED_ACCOUNTS で来る)
    reviewers = change.get("reviewers", {}).get("REVIEWER", []) or []
    reviewer_emails = [r.get("email", "").lower() for r in reviewers]
    if email in reviewer_emails:
        role = "both" if role == "owner" else "reviewer"

    return {
        "role": role,
        "project": change.get("project", "").lower(),
        "created": change.get("created"),
        "updated": change.get("updated"),
    }


def compute_features_at(
    dev_changes: list[dict],
    email: str,
    feature_end: datetime,
    main32_repos: set[str],
    repo_to_team: dict[str, str],
) -> dict:
    """指定 feature_end における dev の全期間プロフィール特徴量を計算する。

    feature_end より過去 (created < feature_end) のレコードのみ使用。
    """
    feat_end_ts = pd.Timestamp(feature_end)
    earliest = None
    review_count = 0
    repos_set = set()
    out_of_main32_count = 0
    domain_counter = {}

    for c in dev_changes:
        info = get_dev_role(c, email)
        if not info["role"]:
            continue
        # created を datetime に
        try:
            created_str = info["created"]
            if "." in created_str:
                created_str = created_str.split(".")[0]
            created = pd.Timestamp(created_str)
        except Exception:
            continue
        if created >= feat_end_ts:
            continue

        if earliest is None or created < earliest:
            earliest = created

        # role が reviewer or both ならレビュー扱い (owner-only は PR 投稿、別途扱い)
        if info["role"] in ("reviewer", "both"):
            review_count += 1

        repos_set.add(info["project"])
        if info["project"] not in main32_repos:
            out_of_main32_count += 1
        team = repo_to_team.get(info["project"], "unknown")
        domain_counter[team] = domain_counter.get(team, 0) + 1

    tenure_days = (feat_end_ts - earliest).days if earliest is not None else 0
    total = sum(domain_counter.values())
    if total > 0:
        entropy = -sum((c / total) * math.log(c / total + 1e-12) for c in domain_counter.values())
    else:
        entropy = 0.0

    return {
        "developer": email,
        "feature_end": feature_end.strftime("%Y-%m-%d"),
        "historical_tenure_days": tenure_days,
        "historical_review_count_all": review_count,
        "historical_n_unique_repos": len(repos_set),
        "out_of_main32_share": out_of_main32_count / total if total > 0 else 0.0,
        "domain_diversity": entropy,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("data/raw_json_by_dev"))
    p.add_argument("--main32-csv", type=Path, default=Path("data/service_teams_main_repos.csv"))
    p.add_argument("--all-repos-csv", type=Path, default=Path("data/service_teams_repos.csv"))
    p.add_argument("--feature-ends", nargs="+", default=["2022-01-01", "2022-04-01", "2022-07-01", "2022-10-01"],
                   help="計算する feature_end の日付 (multiple)")
    p.add_argument("--output", type=Path, default=Path("data/dev_history_features.parquet"))
    args = p.parse_args()

    logger.info(f"main32_repos: {args.main32_csv}")
    main32 = load_main32_repos(args.main32_csv)
    logger.info(f"  {len(main32)} repos")

    logger.info(f"all_repos team mapping: {args.all_repos_csv}")
    repo_to_team = load_repo_to_team(args.all_repos_csv)
    logger.info(f"  {len(repo_to_team)} repo → team mappings")

    json_files = sorted(args.input_dir.glob("*.json"))
    json_files = [f for f in json_files if not f.name.startswith("_")]
    logger.info(f"dev json files: {len(json_files)}")

    feature_ends = [datetime.strptime(d, "%Y-%m-%d") for d in args.feature_ends]
    logger.info(f"feature_ends: {args.feature_ends}")

    rows = []
    for jf in tqdm(json_files):
        with open(jf) as f:
            data = json.load(f)
        email = data["email"]
        changes = data.get("changes", [])
        if not changes:
            continue
        for fe in feature_ends:
            row = compute_features_at(changes, email, fe, main32, repo_to_team)
            rows.append(row)

    out_df = pd.DataFrame(rows)
    logger.info(f"output rows: {len(out_df)}")
    logger.info(f"unique devs: {out_df['developer'].nunique()}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix == ".parquet":
        out_df.to_parquet(args.output, index=False)
    else:
        out_df.to_csv(args.output, index=False)
    logger.info(f"wrote: {args.output}")


if __name__ == "__main__":
    main()
