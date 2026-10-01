#!/usr/bin/env bash
# run_pipeline_full.sh が回さない D19・D21〜D42 を全期間データで回す。
#
# run_pipeline_full.sh は D1〜D18 までしか実行しない。D19 以降は後から足した分析なので、
# actors.parquet や d5/d24 が作り直されたら、こちらも回さないと古い結果が残る。
#
# 収集スクリプト(D20/D22/D29)は含めない。生データが変わらない限り再実行不要。
#
# 実行: bash pilot/phase2/scripts/run_post_pipeline.sh
set -u

cd "$(dirname "$0")/../../.." || exit 1
LOG_DIR="pilot/phase2/logs"
mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/post_pipeline.log"

export PILOT_RAW=raw_full
export PILOT_INTERIM=interim_full
export PILOT_PROCESSED=processed_full
export PHASE2_OUT=outputs_full
export PHASE2_FIG=figures_full

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MAIN_LOG"; }

run() {
  local s=$1
  log "  $s"
  uv run python "pilot/phase2/scripts/$s" > "$LOG_DIR/post_${s%.py}.log" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && log "    !! $s が失敗しました (exit $rc) -> $LOG_DIR/post_${s%.py}.log"
  return 0
}

log "=== D19・D21〜D31 を回します ==="
run d24_approval_substitution.py   # D30/D31 が読むので先
run d19_prediction_decay_full.py
run d21_comment_overlap.py
run d23_thread_resolution.py
run d25_habituation.py
run d26_triage_model.py
run d27_behavior_shift.py
run d28_ai_role_shift.py
run d30_merge_authority.py
run d31_classification_audit.py
run d33_targeted_delegation.py
run d34_attention_coverage.py
run d35_revert_outcome.py
run d36_who_said_it.py          # 素朴な推定値の再現用(結論は d38)
run d38_resolver_analysis.py
run d41_delegation_policy.py
run d42_fix_inducing.py
run d43_causal_robustness.py
run d44_author_excluded.py
run d45_verdict_boundary.py
run d12_figures.py              # fig7 は d44 に依存するので最後に描き直す
log "=== 完了 ==="
log "結果: pilot/phase2/outputs_full/"
