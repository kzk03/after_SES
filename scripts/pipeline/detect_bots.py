#!/usr/bin/env python3
"""combined_raw_*.csv の email を集計し、ぼっとを判定して除外リストを生成する。

判定ロジック:
    1. パターンマッチ:
       既存 build_dataset.py の bot_patterns を継承しつつ、main32 で取りこぼしている
       3rd-party CI (aristanetworks 等) を網羅する追加パターンを足す。
    2. 挙動ヒューリスティック:
       「多くの異なる owner に短時間で機械的に承諾」している email を bot とみなす。
       人間 reviewer はチーム内偏りがあるためここに乗りにくい。

       閾値 (全て同時に満たすと bot 判定):
         - n_requests >= 100
         - owner_diversity (= n_unique_owners / n_requests) >= 0.8
         - median_latency <= 0.5 (日)
         - accept_rate >= 0.9

出力:
    data/bot_emails_<tag>.txt:
        除外対象 email の一覧 (1 行 1 email)。filter_bots.py が読む。
    outputs/analysis/bot_detection/email_stats_<tag>.csv:
        全 email の集計と判定根拠 (matched_pattern / behavior_flag)。目視確認用。
    outputs/analysis/bot_detection/bot_candidates_<tag>.csv:
        bot 判定された email を行数降順で出力。

使い方:
    uv run python scripts/pipeline/detect_bots.py --input data/combined_raw_main32.csv --tag main32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PATTERNS = [
    'zuul', 'jenkins', 'noreply', 'openstack-infra',
    'gserviceaccount.com', 'appspot.gserviceaccount.com',
    'luci-project-accounts', 'luci-bisection',
    'autoroll', 'auto-roller', 'automerger', 'autosubmit', 'autorerun',
    'system.gserviceaccount',
    'rubber-stamper', 'findit-for-me', 'tricium', 'chromeperf',
    'sheriffs-robots',
    'android-build-', 'android-test-infra-', 'boq-android-',
    'culprit-assistant', 'stale-change-watcher', 'presubmit-', 'workplan-finisher',
    'ci@', 'bot@', 'gerrit@', 'review@',
    '-ci@', '-ci-', '-ci.',
    '-bot@', '-bot-', '-bot.',
    '-robot', '-robots',
    '-test@', '-tests@', 'testbot', 'test-bot', 'testing-bot',
    'infra-', 'test-infra-', '-infra@',
    'automation@', '-automation@', 'continuous-integration',
    'aristanetworks', 'arista-openstack',
    '@review.opendev.org', '@review.openstack.org',
]


def matches_pattern(email):
    if not isinstance(email, str):
        return None
    el = email.lower()
    for p in PATTERNS:
        if p in el:
            return p
    return None


def behavior_flag(row, *, min_requests=100, min_diversity=0.8,
                  max_latency=0.5, min_accept=0.9):
    if (row["n_requests"] >= min_requests
            and row["owner_diversity"] >= min_diversity
            and pd.notna(row["median_latency"])
            and row["median_latency"] <= max_latency
            and row["accept_rate"] >= min_accept):
        return "auto_accept_high_diversity"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="combined_raw CSV")
    ap.add_argument("--tag", required=True, help="出力ファイルの suffix (例: main32)")
    ap.add_argument("--outdir", default="outputs/analysis/bot_detection")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    df = pd.read_csv(args.input, low_memory=False)
    print(f"[detect_bots] 入力: {args.input}  行数={len(df):,}")
    print(f"[detect_bots]   unique email = {df['email'].nunique():,}")

    agg = df.groupby("email", dropna=False).agg(
        n_requests=("change_id", "count"),
        n_accepted=("label", "sum"),
        n_unique_owners=("owner_email", "nunique"),
        n_unique_projects=("project", "nunique"),
        median_latency=("response_latency_days", "median"),
    ).reset_index()
    agg["accept_rate"] = agg["n_accepted"] / agg["n_requests"]
    agg["owner_diversity"] = agg["n_unique_owners"] / agg["n_requests"]

    agg["matched_pattern"] = agg["email"].map(matches_pattern)
    agg["behavior_flag"] = agg.apply(behavior_flag, axis=1)
    agg["is_bot"] = agg["matched_pattern"].notna() | agg["behavior_flag"].notna()

    stats_path = outdir / f"email_stats_{args.tag}.csv"
    agg.sort_values("n_requests", ascending=False).to_csv(stats_path, index=False)

    cand = agg[agg["is_bot"]].sort_values("n_requests", ascending=False)
    cand_path = outdir / f"bot_candidates_{args.tag}.csv"
    cand.to_csv(cand_path, index=False)

    bots_path = data_dir / f"bot_emails_{args.tag}.txt"
    cand["email"].dropna().to_csv(bots_path, index=False, header=False)

    bot_set = set(cand["email"].dropna().tolist())
    bot_rows = df["email"].isin(bot_set).sum()

    print()
    print(f"[detect_bots] bot 判定 email 数  : {len(bot_set):,} / {len(agg):,}")
    print(f"[detect_bots] bot 由来の行数      : {bot_rows:,} / {len(df):,}  ({100*bot_rows/len(df):.2f}%)")
    print()
    print("[detect_bots] 判定内訳 (matched_pattern, 上位 15):")
    print(agg.loc[agg["is_bot"], "matched_pattern"].value_counts(dropna=False).head(15).to_string())
    print()
    print("[detect_bots] 判定内訳 (behavior_flag のみ):")
    only_behavior = agg[(agg["behavior_flag"].notna()) & (agg["matched_pattern"].isna())]
    print(f"  パターン外で挙動だけで判定された email = {len(only_behavior):,}")
    if len(only_behavior) > 0:
        cols = ["email", "n_requests", "n_unique_owners", "owner_diversity",
                "median_latency", "accept_rate"]
        print(only_behavior.sort_values("n_requests", ascending=False)
              [cols].head(20).to_string(index=False))
    print()
    print(f"[detect_bots] 出力:")
    print(f"  {stats_path}")
    print(f"  {cand_path}")
    print(f"  {bots_path}")


if __name__ == "__main__":
    main()
