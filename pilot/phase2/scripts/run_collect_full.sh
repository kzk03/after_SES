#!/usr/bin/env bash
# 39リポジトリの 2023-01〜 全 PR を収集する。
#
# 実測: 現行クエリは 75PR あたり cost=2。GraphQL のレート制限は 5000ポイント/時なので
# 理論上は毎時 18万件まで取れる。実際の律速はリクエストのレイテンシなので、並列数を上げる価値がある。
# phase1 で6並列にしたときセカンダリレート制限(403)に抵触したため、5並列までに留める。
#
# 1本終わったら次を投入するジョブプール方式(3本ずつ wait する方式だと、
# 大きいリポジトリ1本に他が引きずられて待たされるため)。
# 大きいリポジトリから先に投入してロングテールを避ける。
# 各リポジトリはチェックポイントを持つので、中断しても再実行すれば続きから進む。
#
# 実行: bash pilot/phase2/scripts/run_collect_full.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
LOG_DIR="pilot/phase2/logs"
mkdir -p "$LOG_DIR"

# 既存(直近13ヶ月)の件数が多い順に並べる。全期間でも大きいはずなので先に始める
REPOS=$(uv run python -c "
import sys; sys.path.insert(0, 'pilot/phase2/scripts')
from pathlib import Path
from common import load_repos, repo_dir_name, REPO_ROOT
def size(r):
    p = REPO_ROOT / 'pilot' / 'raw' / repo_dir_name(r) / 'prs_full.jsonl'
    return p.stat().st_size if p.exists() else 0
for r in sorted(load_repos(), key=size, reverse=True):
    print(r)
")

PARALLEL=5
for repo in $REPOS; do
  safe=${repo//\//__}
  ckpt="pilot/raw_full/${safe}/.checkpoint_prs.json"
  if [ -f "$ckpt" ] && grep -q '"done": true' "$ckpt" 2>/dev/null; then
    echo "skip (done): $repo"
    continue
  fi
  # 空きが出るまで待つ
  while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do
    sleep 5
  done
  echo "start: $repo"
  uv run python pilot/phase2/scripts/collect_full_history.py "$repo" \
    > "$LOG_DIR/collect_${safe}.log" 2>&1 &
  sleep 8  # 起動間隔をずらしてセカンダリレート制限を避ける
done
wait
echo "=== 全リポジトリの収集が終了しました ==="
for f in "$LOG_DIR"/collect_*.log; do
  tail -1 "$f"
done
