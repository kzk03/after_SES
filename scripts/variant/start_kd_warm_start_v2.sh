#!/bin/bash
# KD warm-start v2: 1から再実行 (別コンテナ想定)。
#
# v1 (start_kd_warm_start.sh) との違い:
#   - EPOCHS=300 (v1 は 150) — v1 は全 4 窓とも上限で打ち切られ未収束だった
#   - PATIENCE=20 (v1 は 15) — cosine warm restarts を 2-3 サイクル許容
#   - OUTBASE/LOG/PID をすべて _v2 系に分離 (v1 の出力を上書きしない)
#   - trajectory cache (CACHE_TAG=main32) は v1 と共有 (state_dim=27 で互換)
#
# 使い方 (別コンテナ内で):
#   bash scripts/variant/start_kd_warm_start_v2.sh
#
# 確認:  tail -f logs/kd_warm_start_v2.log
# 中断:  kill $(cat logs/kd_warm_start_v2.pid)

cd "$(dirname "$0")/../.."
mkdir -p logs

# ── 二重起動防止 ──
PID_FILE="logs/kd_warm_start_v2.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && [ -d "/proc/$OLD_PID" ]; then
        echo "ERROR: 既に PID=$OLD_PID で実行中です。"
        echo "       終了するなら: kill $OLD_PID"
        echo "       強制再起動するなら: rm $PID_FILE して再実行"
        exit 1
    fi
fi

# ── 設定 ──
VARIANT="lstm_baseline"
OUTBASE="outputs/main32_mce_kd_warmstart_v2"
GPU_ID=0

LOG_FILE="logs/kd_warm_start_v2.log"
{
    echo ""
    echo "============================================================"
    echo "  start_kd_warm_start_v2.sh: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  variant=$VARIANT  outbase=$OUTBASE  gpu=$GPU_ID"
    echo "  EPOCHS=300, PATIENCE=20 (収束確保)"
    echo "============================================================"
} >> "$LOG_FILE"

EPOCHS=300 \
PATIENCE=20 \
HUB_SCORES_CSV=experiments/dependency_analysis/results/hub_scores_main32.csv \
COCHANGE_NEIGHBORS_CSV=experiments/dependency_analysis/results/cochange_neighbors_main32.csv \
CACHE_TAG=main32 \
nohup bash scripts/variant/run_mce_irl_kd_warm_start.sh \
    "$VARIANT" "$OUTBASE" "$GPU_ID" \
    >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "started: PID=$PID"
echo "log:     $LOG_FILE"
echo ""
echo "確認:    tail -f $LOG_FILE"
echo "中断:    kill $PID  (子プロセスごと: pkill -TERM -P $PID; kill $PID)"
