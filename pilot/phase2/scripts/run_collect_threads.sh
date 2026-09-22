#!/usr/bin/env bash
# レビュースレッドの解決状態(isResolved / isOutdated)を収集する。
# 対象は AI レビューが付いた 41,969 PR。cost≈12/20PR なので約5時間かかる。
# レート制限に当たらないよう並列は2本まで。
#
# 実行: bash pilot/phase2/scripts/run_collect_threads.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
LOG_DIR="pilot/phase2/logs"
mkdir -p "$LOG_DIR"

REPOS=$(uv run python -c "
import pandas as pd
pr = pd.read_parquet('pilot/phase2/outputs_full/d5_pr_level.parquet')
c = pr[pr['n_ai_reviews_total'] > 0].groupby('repo').size().sort_values(ascending=True)  # 小さい順: リポジトリ数のカバーを優先
print('\n'.join(c.index))
")

PARALLEL=2
for repo in $REPOS; do
  safe=${repo//\//__}
  ckpt="pilot/raw_full/${safe}/.checkpoint_threads.json"
  if [ -f "$ckpt" ] && grep -q '"complete": true' "$ckpt" 2>/dev/null; then
    echo "skip (done): $repo"
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  echo "start: $repo"
  uv run python pilot/phase2/scripts/d22_collect_thread_status.py "$repo" 20 \
    > "$LOG_DIR/threads_${safe}.log" 2>&1 &
  sleep 6
done
wait
echo "=== レビュースレッド状態の収集が終了しました ==="
tot=0
for d in pilot/raw_full/*/; do
  n=$(wc -l < "$d/thread_status.jsonl" 2>/dev/null || echo 0)
  tot=$((tot + n))
done
echo "取得した PR 数: $tot"
