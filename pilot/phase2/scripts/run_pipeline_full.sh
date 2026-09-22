#!/usr/bin/env bash
# 2023-01〜の全期間データ(pilot/raw_full)で、loop1〜4 と phase2 の全分析を回す。
#
# 収集(run_collect_full.sh)の完了を待ってから始める。
# 出力は phase1/phase2 の既存成果物と衝突しないよう、すべて別ディレクトリに出す。
#   pilot/interim_full, pilot/processed_full, pilot/phase2/outputs_full
#
# 実行: bash pilot/phase2/scripts/run_pipeline_full.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
LOG_DIR="pilot/phase2/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/pipeline_full.log"

export PILOT_RAW=raw_full
export PILOT_INTERIM=interim_full
export PILOT_PROCESSED=processed_full
export PHASE2_OUT=outputs_full
export PHASE2_FIG=figures_full

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

# ---------------------------------------------------------------- 0. 収集完了待ち
log "収集の完了を待っています..."
WAITED=0
MAX_WAIT=108000  # 30時間で打ち切る(llvm-project のような巨大リポジトリは遡りに時間がかかる)
while pgrep -f "collect_full_history.py|run_collect_full.sh|retry_failed_collect.sh" > /dev/null 2>&1; do
  sleep 300
  WAITED=$((WAITED + 300))
  n=$(cat pilot/raw_full/*/prs_full.jsonl 2>/dev/null | wc -l | tr -d ' ')
  d=$(grep -l '"done": true' pilot/raw_full/*/.checkpoint_prs.json 2>/dev/null | wc -l | tr -d ' ')
  log "  待機中 ${WAITED}秒 / 完了 ${d} リポジトリ / 総PR ${n} 件"
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    log "  最大待機時間に達したため、取得済みの分で先に進みます"
    break
  fi
done
log "収集が終了しました。取得済み: $(grep -l '\"done\": true' pilot/raw_full/*/.checkpoint_prs.json 2>/dev/null | wc -l | tr -d ' ') リポジトリ"

REPOS=$(uv run python -c "
import sys; sys.path.insert(0, 'pilot/phase2/scripts')
from common import load_repos
print('\n'.join(load_repos()))
")

# -------------------------------------------- 1. CODEOWNERS と file_tree を持ち込む
# raw_full には PR しか入れていない。モジュール定義(CODEOWNERS)は期間に依存しないので
# phase1 で収集済みのものをそのまま使う。
log "CODEOWNERS / file_tree を raw から raw_full へ複製します"
for repo in $REPOS; do
  safe=${repo//\//__}
  src="pilot/raw/$safe"
  dst="pilot/raw_full/$safe"
  [ -d "$dst" ] || continue
  [ -f "$src/file_tree.json" ] && [ ! -f "$dst/file_tree.json" ] && cp "$src/file_tree.json" "$dst/"
  [ -d "$src/codeowners" ] && [ ! -d "$dst/codeowners" ] && cp -R "$src/codeowners" "$dst/"
done

# ------------------------------------------------------------------ 2. loop1〜4
# loop3 は出力先を自分で mkdir しないので先に作っておく
for repo in $REPOS; do
  safe=${repo//\//__}
  mkdir -p "pilot/interim_full/$safe" "pilot/processed_full/$safe"
done

run_parallel() {  # $1=スクリプト名, $2=並列数
  local script=$1 par=$2 i=0
  log "--- $script を実行します(並列 $par) ---"
  for repo in $REPOS; do
    local safe=${repo//\//__}
    while [ "$(jobs -rp | wc -l)" -ge "$par" ]; do sleep 2; done
    uv run python "scripts/pilot/$script" "$repo" > "$LOG_DIR/${script%.py}_${safe}.log" 2>&1 &
    i=$((i + 1))
  done
  wait
  log "--- $script 完了($i リポジトリ) ---"
}

run_parallel loop1_actors.py 4
run_parallel loop2_modules.py 4
run_parallel loop3_stage.py 4
run_parallel loop4_metrics.py 4

ok=$(ls pilot/processed_full/*/panel.parquet 2>/dev/null | wc -l | tr -d ' ')
log "panel.parquet が生成されたリポジトリ: $ok"
if [ "$ok" -lt 5 ]; then
  log "パネルがほとんど作れていません。ログを確認してください。ここで中断します。"
  exit 1
fi

# --------------------------------------------------------------- 3. phase2 の分析
mkdir -p "pilot/phase2/outputs_full"
run_phase2() {
  local s=$1
  log "  phase2: $s"
  uv run python "pilot/phase2/scripts/$s" > "$LOG_DIR/full_${s%.py}.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && log "    !! $s が失敗しました (exit $rc)"
  return 0
}

log "--- phase2 の分析を全期間データで再実行します ---"
run_phase2 d5_pr_outcomes.py      # 他が依存するので最初
run_phase2 d1_decomposition.py
run_phase2 d2_adoption_timing.py
run_phase2 d4_repo_event_study.py
run_phase2 d6_robustness.py
run_phase2 d7_burden_event_study.py
run_phase2 d8_talent_flow.py
run_phase2 d10_review_kind.py
run_phase2 d11_agent_profile.py
run_phase2 d13_review_response.py
run_phase2 d14_dose_response.py
run_phase2 d15_burden_distribution.py
run_phase2 d16_generation_side.py
run_phase2 d17_asymmetry.py
run_phase2 d18_redundancy.py
run_phase2 d9_equivalence.py      # D4/D7/D8 の出力を読むので後
run_phase2 d12_figures.py

log "=== 全期間パイプラインが完了しました ==="
log "結果: pilot/phase2/outputs_full/"
