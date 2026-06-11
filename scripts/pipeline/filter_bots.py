#!/usr/bin/env python3
"""combined_raw_*.csv から bot_emails_<tag>.txt の email を除外して新 CSV を出力する。

raw_json は触らない (path_features の change_id → directory マッピングしか使わず、
bot reviewer の email を読まないため、CSV 側の filter だけで MCE-IRL の学習軌跡から
bot は完全に排除される)。

使い方:
    uv run python scripts/pipeline/filter_bots.py \\
        --input data/combined_raw_main32.csv \\
        --bots  data/bot_emails_main32.txt \\
        --output data/combined_raw_main32_nobot.csv

raw_json_list ファイルは入力と同じものを使う想定 (リスト内容は CSV 側に依存しない)。
このスクリプトはコピーまで面倒は見ないので、必要なら呼び出し側で:
    cp data/combined_raw_main32.raw_json_list.txt data/combined_raw_main32_nobot.raw_json_list.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--bots", required=True, help="1 行 1 email の除外リスト")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    bots_path = Path(args.bots)
    out_path = Path(args.output)

    bot_set = {line.strip() for line in bots_path.read_text().splitlines() if line.strip()}
    print(f"[filter_bots] 除外対象 email = {len(bot_set):,}")

    df = pd.read_csv(in_path, low_memory=False)
    n_before = len(df)
    mask = df["email"].isin(bot_set)
    n_drop = int(mask.sum())
    df_kept = df.loc[~mask].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_kept.to_csv(out_path, index=False)

    n_after = len(df_kept)
    print(f"[filter_bots] 入力 : {in_path}  ({n_before:,} 行, {df['email'].nunique():,} email)")
    print(f"[filter_bots] 除外 : {n_drop:,} 行  ({100*n_drop/n_before:.2f}%)")
    print(f"[filter_bots] 出力 : {out_path}  ({n_after:,} 行, {df_kept['email'].nunique():,} email)")


if __name__ == "__main__":
    main()
