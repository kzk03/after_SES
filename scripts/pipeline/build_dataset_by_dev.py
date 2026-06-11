"""dev-centric な Gerrit データ収集。

repo-centric の build_dataset.py を踏襲し、Gerrit クエリを
`(owner:<email> OR reviewer:<email>) after:... before:...` に変更して
1 dev の全 change を Gerrit 全体から取得する。

特徴:
  - 既存 build_dataset.py の GerritDataFetcher を再利用 (クエリのみ変更)
  - dev リスト (1 列 csv) を受け取り、各 dev について sleep を入れつつ収集
  - 既取得 (output file 存在) はスキップ → 再開可能
  - ネットワークエラー時はリトライ + バックオフ

使い方:
  uv run python scripts/pipeline/build_dataset_by_dev.py \\
      --devs data/active_devs.csv \\
      --start-date 2018-01-01 --end-date 2023-01-01 \\
      --output-dir data/raw_json_by_dev/ \\
      --sleep 5

引数:
  --devs       : 1 列 csv (email)
  --start-date : 取得期間開始 (YYYY-MM-DD)
  --end-date   : 取得期間終了 (YYYY-MM-DD)
  --output-dir : 各 dev の json 出力先 (default: data/raw_json_by_dev/)
  --sleep      : dev 間の sleep (秒、default: 5)
  --gerrit-url : default https://review.opendev.org
  --skip-existing: 既取得 dev をスキップ (default: True)
  --limit-n    : テスト用、N dev だけ処理 (default: 全部)
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# 既存 build_dataset.py から GerritDataFetcher を再利用
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_dataset import GerritDataFetcher  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def sanitize_email(email: str) -> str:
    """email を file name 用に sanitize する。"""
    s = email.lower().strip()
    s = re.sub(r"[^a-z0-9._-]", "_", s)  # @ . - _ 以外を _ に
    return s[:120]  # 長すぎる名前を防ぐ


def fetch_for_dev(
    fetcher: GerritDataFetcher,
    email: str,
    start_date: datetime,
    end_date: datetime,
    limit: int = 500,
) -> list[dict]:
    """1 dev の全 change を Gerrit 全体から取得する。

    クエリ: (owner:<email> OR reviewer:<email>) after:... before:...
    """
    all_changes = []
    start = 0
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    # email 中の特殊文字は escape (Gerrit クエリでは @ は通常そのまま OK だが念のため)
    query = f"(owner:{email} OR reviewer:{email}) after:{start_str} before:{end_str}"

    while True:
        params = {
            "q": query,
            # dev-centric は profile 計算用なので最小限の payload に絞る:
            #   - DETAILED_ACCOUNTS: reviewer/owner email を入れる
            #   - DETAILED_LABELS: 承諾/拒否の vote 取得
            #   - REVIEWER_UPDATES: reviewer 追加/削除履歴
            # 除外 (大きすぎる):
            #   ALL_REVISIONS / ALL_FILES / CURRENT_FILES (file paths)
            #   MESSAGES (commit messages, huge)
            #   TRACKING_IDS / SUBMITTABLE / SUBMIT_REQUIREMENTS
            "o": [
                "DETAILED_ACCOUNTS", "DETAILED_LABELS", "REVIEWER_UPDATES",
            ],
            "n": limit,
            "start": start,
        }
        try:
            changes = fetcher._make_request("changes/", params)
            if not changes:
                break
            all_changes.extend(changes)
            if not changes[-1].get("_more_changes", False):
                break
            start += limit
        except Exception as e:
            logger.warning(f"  fetch error for {email} at start={start}: {e}")
            return None  # 上位でリトライ判断
    return all_changes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devs", type=Path, required=True,
                   help="1 列 csv (email)")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/raw_json_by_dev/"))
    p.add_argument("--sleep", type=float, default=5.0,
                   help="dev 間の sleep (秒)")
    p.add_argument("--retry-sleep", type=float, default=30.0,
                   help="エラー時の追加 sleep (秒)")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--gerrit-url", default="https://review.opendev.org")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="既取得ファイルがあっても再取得 (default: スキップ)")
    p.add_argument("--limit-n", type=int, default=0,
                   help="テスト用、N dev だけ処理 (0 = 全部)")
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    devs_df = pd.read_csv(args.devs)
    col = "email" if "email" in devs_df.columns else devs_df.columns[0]
    devs = devs_df[col].dropna().unique().tolist()
    if args.limit_n > 0:
        devs = devs[: args.limit_n]
    logger.info(f"対象 dev 数: {len(devs)}")
    logger.info(f"期間: {args.start_date} ~ {args.end_date}")
    logger.info(f"出力先: {args.output_dir}")
    logger.info(f"sleep: {args.sleep}s / dev")

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    fetcher = GerritDataFetcher(args.gerrit_url, timeout=args.timeout)

    skipped = 0
    fetched = 0
    failed = []

    for i, email in enumerate(devs):
        safe = sanitize_email(email)
        outfile = args.output_dir / f"{safe}.json"
        if outfile.exists() and not args.no_skip_existing:
            skipped += 1
            continue

        logger.info(f"[{i+1}/{len(devs)}] {email} → {outfile.name}")
        retries = 0
        changes = None
        while retries < args.max_retries:
            changes = fetch_for_dev(fetcher, email, start_date, end_date)
            if changes is not None:
                break
            retries += 1
            logger.warning(f"  retry {retries}/{args.max_retries} after {args.retry_sleep}s")
            time.sleep(args.retry_sleep)

        if changes is None:
            logger.error(f"  failed after {args.max_retries} retries: {email}")
            failed.append(email)
            continue

        logger.info(f"  {len(changes)} changes fetched")
        # 出力 (raw json をそのまま保存)
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump({
                "email": email,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "n_changes": len(changes),
                "changes": changes,
            }, f, ensure_ascii=False)
        fetched += 1

        # sleep
        if i < len(devs) - 1:
            time.sleep(args.sleep)

    logger.info("")
    logger.info(f"=== summary ===")
    logger.info(f"fetched: {fetched} devs")
    logger.info(f"skipped (existing): {skipped} devs")
    logger.info(f"failed: {len(failed)} devs")
    if failed:
        failed_path = args.output_dir / "_failed.txt"
        with open(failed_path, "w") as f:
            f.write("\n".join(failed))
        logger.info(f"failed dev list saved: {failed_path}")


if __name__ == "__main__":
    main()
