#!/usr/bin/env bash
# 収集に失敗したリポジトリを、page_size を下げながら再試行する。
#
# GitHub の GraphQL は、特定の PR を含むページで INTERNAL エラーを繰り返すことがある
# (phase1 では node.js がこれで最後まで取れず対象から除外した)。
# 1ページに詰め込むノード数を減らすと通ることが多いので、75 → 30 → 15 → 8 と下げて再挑戦する。
#
# run_collect_full.sh が全リポジトリを投入し終わるまで待ってから動き出す
# (未着手のリポジトリを二重起動しないため)。
#
# 実行: bash pilot/phase2/scripts/retry_failed_collect.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
LOG_DIR="pilot/phase2/logs"
mkdir -p "$LOG_DIR"
RETRY_LOG="$LOG_DIR/retry.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RETRY_LOG"; }

log "run_collect_full.sh の完了を待っています..."
while pgrep -f "run_collect_full.sh" > /dev/null 2>&1; do
  sleep 120
done
log "投入が完了しました。失敗したリポジトリの再試行に入ります"

REPOS=$(uv run python -c "
import sys; sys.path.insert(0, 'pilot/phase2/scripts')
from common import load_repos
print('\n'.join(load_repos()))
")

for PS in 30 15 8; do
  # 走っている収集が終わるのを待つ
  while pgrep -f "collect_full_history.py" > /dev/null 2>&1; do sleep 60; done

  FAILED=""
  for repo in $REPOS; do
    safe=${repo//\//__}
    ckpt="pilot/raw_full/${safe}/.checkpoint_prs.json"
    if [ ! -f "$ckpt" ] || ! grep -q '"done": true' "$ckpt" 2>/dev/null; then
      FAILED="$FAILED $repo"
    fi
  done

  if [ -z "$FAILED" ]; then
    log "未完了のリポジトリはありません。再試行を終了します"
    break
  fi
  log "page_size=$PS で再試行:$FAILED"

  for repo in $FAILED; do
    safe=${repo//\//__}
    while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 5; done
    uv run python pilot/phase2/scripts/collect_full_history.py "$repo" "$PS" \
      >> "$LOG_DIR/collect_${safe}.log" 2>&1 &
    sleep 8
  done
  wait
  log "page_size=$PS のラウンドが終了しました"
done

REMAIN=""
for repo in $REPOS; do
  safe=${repo//\//__}
  ckpt="pilot/raw_full/${safe}/.checkpoint_prs.json"
  if [ ! -f "$ckpt" ] || ! grep -q '"done": true' "$ckpt" 2>/dev/null; then
    n=$(wc -l < "pilot/raw_full/${safe}/prs_full.jsonl" 2>/dev/null || echo 0)
    REMAIN="$REMAIN\n  $repo (取得済み ${n} 件)"
  fi
done
if [ -n "$REMAIN" ]; then
  log "最後まで取り切れなかったリポジトリ:$(echo -e "$REMAIN")"
  log "→ 取得済みの範囲だけで分析に進みます(期間が短いので該当リポジトリは結果の解釈に注意)"
else
  log "全リポジトリの収集が完了しました"
fi
