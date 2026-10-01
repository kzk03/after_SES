#!/usr/bin/env bash
# 全マージ済み PR の mergedBy を収集する。
# 実測で 100 PR あたり cost=1・1.3秒。約70万件で 7,000ポイント、2〜3時間。
# レート制限(5,000/時)が律速なので並列は3本までにする。
#
# 実行: bash pilot/phase2/scripts/run_collect_mergedby.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
LOG_DIR="pilot/phase2/logs"
mkdir -p "$LOG_DIR"

# 対象が少ないリポジトリから処理してカバー率を優先する
REPOS=$(uv run python -c "
import pandas as pd
pr = pd.read_parquet('pilot/phase2/outputs_full/d5_pr_level.parquet')
c = pr[pr['merged']].groupby('repo').size().sort_values(ascending=True)
print('\n'.join(c.index))
")

PARALLEL=3
for repo in $REPOS; do
  safe=${repo//\//__}
  ckpt="pilot/raw_full/${safe}/.checkpoint_mergedby.json"
  if [ -f "$ckpt" ] && grep -q '"complete": true' "$ckpt" 2>/dev/null; then
    echo "skip (done): $repo"
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  echo "start: $repo"
  uv run python pilot/phase2/scripts/d29_collect_merged_by.py "$repo" 100 \
    > "$LOG_DIR/mergedby_${safe}.log" 2>&1 &
  sleep 4
done
wait
echo "=== mergedBy の収集が終了しました ==="
tot=0
for d in pilot/raw_full/*/; do
  n=$(wc -l < "$d/merged_by.jsonl" 2>/dev/null || echo 0)
  tot=$((tot + n))
done
echo "取得件数: $tot"
