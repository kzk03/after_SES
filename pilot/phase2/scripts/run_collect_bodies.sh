#!/usr/bin/env bash
# AI レビューが付いた 41,970 PR について、レビュー本文と指摘位置を収集する。
# 大きいリポジトリから先に投入し、4並列のジョブプールで回す。
# 各リポジトリはチェックポイントを持つので、中断しても再実行すれば続きから進む。
#
# 実行: bash pilot/phase2/scripts/run_collect_bodies.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
LOG_DIR="pilot/phase2/logs"
mkdir -p "$LOG_DIR"

# 対象 PR が多い順に並べる(ロングテールを避ける)
REPOS=$(uv run python -c "
import pandas as pd
pr = pd.read_parquet('pilot/phase2/outputs_full/d5_pr_level.parquet')
c = pr[pr['n_ai_reviews_total'] > 0].groupby('repo').size().sort_values(ascending=False)
print('\n'.join(c.index))
")

PARALLEL=2  # レート制限に当たったので絞る(1クエリ cost≈4、20PR/クエリ)
for repo in $REPOS; do
  safe=${repo//\//__}
  ckpt="pilot/raw_full/${safe}/.checkpoint_bodies.json"
  if [ -f "$ckpt" ] && grep -q '"complete": true' "$ckpt" 2>/dev/null; then
    echo "skip (done): $repo"
    continue
  fi
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do sleep 5; done
  echo "start: $repo"
  uv run python pilot/phase2/scripts/d20_collect_review_bodies.py "$repo" 20 \
    > "$LOG_DIR/bodies_${safe}.log" 2>&1 &
  sleep 6
done
wait
echo "=== レビュー本文の収集が終了しました ==="
tot=0
for d in pilot/raw_full/*/; do
  n=$(wc -l < "$d/review_bodies.jsonl" 2>/dev/null || echo 0)
  tot=$((tot + n))
done
echo "取得した PR 数: $tot"
