"""AOSP32のrepo別CSVを結合して combined_raw_aosp32.csv を作る(main32のcombined_raw形式)

- 出力: data/combined_raw_aosp32.csv
- 併せて raw_json のパス一覧を data/combined_raw_aosp32.raw_json_list.txt に書く
  (main32 の combined_raw_main32.raw_json_list.txt と同じ運用)

実行: uv run python scripts/pipeline/combine_aosp32.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "data/aosp/csv"
RAW_DIR = ROOT / "data/aosp/raw_json"
OUT_CSV = ROOT / "data/combined_raw_aosp32.csv"
OUT_LIST = ROOT / "data/combined_raw_aosp32.raw_json_list.txt"

csvs = sorted(CSV_DIR.glob("*.csv"))
assert len(csvs) == 32, f"CSVが32個ない: {len(csvs)}"

dfs = []
cols = None
for p in csvs:
    df = pd.read_csv(p)
    if cols is None:
        cols = list(df.columns)
    assert list(df.columns) == cols, f"カラム不一致: {p.name}"
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined = combined.sort_values("timestamp").reset_index(drop=True)
combined.to_csv(OUT_CSV, index=False)
print(f"combined: {len(combined):,} rows, {combined['project'].nunique()} projects -> {OUT_CSV}")
print(f"label分布: {combined['label'].mean():.1%} positive")

raws = sorted(RAW_DIR.glob("aosp__*.json"))
assert len(raws) == 32, f"raw_jsonが32個ない: {len(raws)}"
# main32と同じ「スペース区切り1行」形式(run_mce_irl_variant_single.sh が read -r -a で読む)
OUT_LIST.write_text(" ".join(str(p.relative_to(ROOT)) for p in raws) + "\n")
print(f"raw_json list ({len(raws)} files) -> {OUT_LIST}")
