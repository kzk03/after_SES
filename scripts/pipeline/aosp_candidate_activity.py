# -*- coding: utf-8 -*-
"""AOSP 候補リポジトリの活動量を集計し直す（論文誌 査読 2-条4 対応）.

2026-07-13 の選定時の集計（/tmp/gerrit_activity*.py、消失）と同じ方法で、
各年 6月第1週（6/1〜6/7）にマージされた change を Gerrit API で取得し、
作成者（owner）が bot でないものを「人間の活動量」として数える。

  GET /changes/?q=status:merged+project:<repo>+after:YYYY-06-01+before:YYYY-06-08&n=500&o=DETAILED_ACCOUNTS

bot 判定は選定時と同じ正規表現（Obsidian 調査方針_OpenStack類似プロジェクト調査.md §4）を
owner の email・name・username に適用する。

出力: data/aosp_candidate_activity_recomputed.csv
  repo, h2020..h2026（各年のサンプル週の人間 change 数）, human_7wk（7週合計）,
  pre_t0（2020〜2022 の3週合計）, selected（元CSVの採用フラグ）, flags
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://android-review.googlesource.com"
YEARS = list(range(2020, 2027))
BOT_RE = re.compile(r"robot|worker|presubmit|gserviceaccount|roller|copybara|automerg|autosubmit|buildbot|\bbot\b", re.I)


def fetch_week(repo: str, year: int, sleep: float) -> list[dict]:
    q = f"status:merged project:{repo} after:{year}-06-01 before:{year}-06-08"
    out, start = [], 0
    while True:
        for attempt in range(5):
            r = requests.get(f"{URL}/changes/", params={"q": q, "n": 500, "S": start, "o": "DETAILED_ACCOUNTS"}, timeout=60)
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        data = json.loads(r.text.split("\n", 1)[1])
        out.extend(data)
        time.sleep(sleep)
        if data and data[-1].get("_more_changes"):
            start += len(data)
        else:
            return out


def is_bot(owner: dict) -> bool:
    text = " ".join(str(owner.get(k, "")) for k in ("email", "name", "username"))
    return (not owner.get("email")) or bool(BOT_RE.search(text))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="data/aosp_candidate_repos.csv")
    p.add_argument("--out", default="data/aosp_candidate_activity_recomputed.csv")
    p.add_argument("--sleep", type=float, default=1.0)
    a = p.parse_args()

    cand = pd.read_csv(a.candidates)
    rows = []
    for i, r in cand.iterrows():
        row = {"repo": r["repo"]}
        for y in YEARS:
            ch = fetch_week(r["repo"], y, a.sleep)
            row[f"h{y}"] = sum(1 for c in ch if not is_bot(c.get("owner", {})))
            row[f"all{y}"] = len(ch)
        row["human_7wk"] = sum(row[f"h{y}"] for y in YEARS)
        row["pre_t0"] = sum(row[f"h{y}"] for y in (2020, 2021, 2022))
        row["human_7wk_orig"] = r["human_7wk"]
        row["selected"] = int(r["selected"] == 1)
        row["flags"] = r["flags"] if isinstance(r["flags"], str) else ""
        rows.append(row)
        print(f"[{i+1}/{len(cand)}] {r['repo']}: human_7wk={row['human_7wk']} (orig {r['human_7wk']}) pre_t0={row['pre_t0']}", flush=True)
    out = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
