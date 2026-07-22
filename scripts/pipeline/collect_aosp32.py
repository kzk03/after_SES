"""AOSP main32 本収集ドライバ

data/aosp_candidate_repos.csv の selected=1 の32repoについて、
build_dataset.py を repo ごとに実行する(2020-01-01〜2025-01-01, response_window=14)。

- 出力: data/aosp/csv/<repo>.csv, data/aosp/raw_json/aosp__<repo>.json
- 再開可能: raw_json が既に存在する repo はスキップ
- 完了後の結合は merge 側(combine手順)で行う

実行:
    uv run python scripts/pipeline/collect_aosp32.py
"""
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GERRIT_URL = "https://android-review.googlesource.com"
START, END = "2020-01-01", "2025-01-01"
CSV_DIR = ROOT / "data/aosp/csv"
RAW_DIR = ROOT / "data/aosp/raw_json"


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(ROOT / "data/aosp_candidate_repos.csv") as f:
        repos = [r["repo"] for r in csv.DictReader(f) if r.get("selected") == "1"]
    print(f"対象: {len(repos)} repos", flush=True)

    failed = []
    for i, repo in enumerate(repos, 1):
        safe = repo.replace("/", "__")
        raw_out = RAW_DIR / f"aosp__{safe}.json"
        csv_out = CSV_DIR / f"{safe}.csv"
        if raw_out.exists():
            print(f"[{i}/{len(repos)}] skip (exists): {repo}", flush=True)
            continue
        print(f"[{i}/{len(repos)}] collecting: {repo}", flush=True)
        t0 = time.time()
        cmd = [sys.executable, str(ROOT / "scripts/pipeline/build_dataset.py"),
               "--gerrit-url", GERRIT_URL, "--project", repo,
               "--start-date", START, "--end-date", END,
               "--response-window", "14",
               "--output", str(csv_out), "--raw-output", str(raw_out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if res.returncode != 0:
            failed.append(repo)
            # 失敗時は中途半端な出力を残さない(再開時にスキップされないように)
            raw_out.unlink(missing_ok=True)
            csv_out.unlink(missing_ok=True)
            print(f"  FAILED ({dt:.0f}s): {repo}\n{res.stderr[-2000:]}", flush=True)
        else:
            tail = [l for l in res.stderr.splitlines() if "Total changes" in l or "Extracted" in l]
            print(f"  done ({dt:.0f}s): " + " / ".join(tail), flush=True)
        time.sleep(5)

    print(f"完了: 成功 {len(repos) - len(failed)} / 失敗 {len(failed)} {failed}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
