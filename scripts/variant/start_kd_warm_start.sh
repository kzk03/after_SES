#!/bin/bash
# KD warm-start (RF_Dir → KD pretrain → MCE-IRL) を nohup でバックグラウンド起動。
#
# 使い方 (サーバの Docker コンテナ内で):
#   bash scripts/variant/start_kd_warm_start.sh
#
# 確認:  tail -f logs/kd_warm_start.log
# 中断:  kill $(cat logs/kd_warm_start.pid)

cd "$(dirname "$0")/../.."
mkdir -p logs

# ── 二重起動防止 ──
PID_FILE="logs/kd_warm_start.pid"
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
OUTBASE="outputs/main32_mce_kd_warmstart"
GPU_ID=0

LOG_FILE="logs/kd_warm_start.log"
{
    echo ""
    echo "============================================================"
    echo "  start_kd_warm_start.sh: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  variant=$VARIANT  outbase=$OUTBASE  gpu=$GPU_ID"
    echo "============================================================"
} >> "$LOG_FILE"

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
