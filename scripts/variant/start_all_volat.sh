#!/bin/bash
# [サーバ用] 変動型対応特徴量 (2026-05-18 追加 5 特徴量) の全パイプラインを連鎖実行。
#
# 実行内容 (sequential, 単一 GPU):
#   1. Cold MCE-IRL  : 4 train window × 4 eval window (RF_Dir も同 pipeline 内で同時学習)
#       → outputs/main32_mce_volat/
#   2. KD Warm-start : 4 train window × 4 eval window (RF_Dir も同時学習)
#       → outputs/main32_mce_kd_warmstart_volat/
#   3. Diagonal eval : 活動パターン別 accuracy (cold IRL / warm IRL / RF_Dir の 3 モデル比較)
#       → outputs/analysis/activity_patterns_warmstart_volat/
#
# 順次実行する理由:
#   - 軌跡キャッシュ (CACHE_TAG=main32_volat) を cold 側で初回生成 → warm-start で再利用
#   - 単一 GPU でも resource contention 回避
#
# 使い方 (サーバの Docker コンテナ内で):
#   bash scripts/variant/start_all_volat.sh           # GPU 0 で実行
#   GPU_ID=1 bash scripts/variant/start_all_volat.sh  # GPU 1 で実行
#
# 確認: tail -f logs/all_volat.log
# 中断: kill $(cat logs/all_volat.pid)
#       (子プロセスごと: pkill -TERM -P $(cat logs/all_volat.pid); kill $(cat logs/all_volat.pid))

set -e
cd "$(dirname "$0")/../.."
mkdir -p logs

# ── 設定 ──
VARIANT="${VARIANT:-lstm_baseline}"
GPU_ID="${GPU_ID:-0}"
CACHE_TAG="${CACHE_TAG:-main32_volat}"

# MCE-IRL 学習設定 (先に定義: OUTBASE/log のサフィックスに使う)
# v1 設定 (EPOCHS=150/PATIENCE=15) がベスト、v2 の EPOCHS=300 は AUC を 2.5-5pt 悪化させたため不採用。
# v2 vs v1 検証 (2026-05-18): train_0-3m × 4 eval で IRL AUC が全窓 -2.5〜-5.0pt → 過学習。
# run_mce_irl_variant_single.sh の default (EPOCHS=150/PATIENCE=15) と一致。
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-15}"

# エポック数を含む OUTBASE / log サフィックス（上書き可）
# 例: EPOCHS=200 で走らせれば *_ep200 系に出力 → 設定違いで上書きが起きない
SUFFIX="ep${EPOCHS}"

COLD_OUTBASE="${COLD_OUTBASE:-outputs/main32_mce_volat_${SUFFIX}}"
WARM_OUTBASE="${WARM_OUTBASE:-outputs/main32_mce_kd_warmstart_volat_${SUFFIX}}"
EVAL_OUTDIR="${EVAL_OUTDIR:-outputs/analysis/activity_patterns_warmstart_volat_${SUFFIX}}"

HUB_SCORES_CSV="${HUB_SCORES_CSV:-experiments/dependency_analysis/results/hub_scores_main32.csv}"
COCHANGE_NEIGHBORS_CSV="${COCHANGE_NEIGHBORS_CSV:-experiments/dependency_analysis/results/cochange_neighbors_main32.csv}"

# ── 二重起動防止 (PID file もサフィックス付きで設定違いを並列に走らせ可) ──
PID_FILE="logs/all_volat_${SUFFIX}.pid"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$OLD_PID" ] && [ -d "/proc/$OLD_PID" ]; then
        echo "ERROR: 既に PID=$OLD_PID で実行中です ($SUFFIX)。"
        echo "       終了するなら: kill $OLD_PID"
        echo "       強制再起動するなら: rm $PID_FILE して再実行"
        exit 1
    fi
fi

LOG_FILE="logs/all_volat_${SUFFIX}.log"
COLD_LOG="logs/all_volat_${SUFFIX}_step1_cold.log"
WARM_LOG="logs/all_volat_${SUFFIX}_step2_warm.log"
EVAL_LOG="logs/all_volat_${SUFFIX}_step3_eval.log"

{
    echo ""
    echo "============================================================"
    echo "  start_all_volat.sh: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  variant=$VARIANT  gpu=$GPU_ID  cache_tag=$CACHE_TAG"
    echo "  cold_outbase=$COLD_OUTBASE"
    echo "  warm_outbase=$WARM_OUTBASE"
    echo "  eval_outdir=$EVAL_OUTDIR"
    echo "  EPOCHS=$EPOCHS  PATIENCE=$PATIENCE"
    echo "============================================================"
} >> "$LOG_FILE"

# ── 連鎖実行 (cold → warm → eval) ──
# 失敗時は次に進まないよう && で連結。各 step は専用 log にも出す。
nohup bash -c "
    set -e
    echo \"[\$(date '+%F %T')] === STEP 1/3: Cold MCE-IRL ===\" >> '$LOG_FILE'
    CACHE_TAG='$CACHE_TAG' \\
    HUB_SCORES_CSV='$HUB_SCORES_CSV' \\
    COCHANGE_NEIGHBORS_CSV='$COCHANGE_NEIGHBORS_CSV' \\
    bash scripts/variant/run_mce_irl_variant_single.sh \\
        0 '$VARIANT' '$COLD_OUTBASE' '$GPU_ID' \\
        2>&1 | tee '$COLD_LOG' | tail -50 >> '$LOG_FILE'

    echo \"[\$(date '+%F %T')] === STEP 2/3: KD Warm-start ===\" >> '$LOG_FILE'
    EPOCHS='$EPOCHS' \\
    PATIENCE='$PATIENCE' \\
    CACHE_TAG='$CACHE_TAG' \\
    HUB_SCORES_CSV='$HUB_SCORES_CSV' \\
    COCHANGE_NEIGHBORS_CSV='$COCHANGE_NEIGHBORS_CSV' \\
    bash scripts/variant/run_mce_irl_kd_warm_start.sh \\
        '$VARIANT' '$WARM_OUTBASE' '$GPU_ID' \\
        2>&1 | tee '$WARM_LOG' | tail -50 >> '$LOG_FILE'

    echo \"[\$(date '+%F %T')] === STEP 3/3: Diagonal Eval ===\" >> '$LOG_FILE'
    uv run python scripts/analyze/study/activity_patterns_warmstart.py \\
        --diagonal \\
        --warm-dir '$WARM_OUTBASE' \\
        --cold-dir '$COLD_OUTBASE' \\
        --output-dir '$EVAL_OUTDIR' \\
        2>&1 | tee '$EVAL_LOG' >> '$LOG_FILE'

    echo \"[\$(date '+%F %T')] === ALL DONE ===\" >> '$LOG_FILE'
" >> "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$PID_FILE"

echo "started: PID=$PID  (EPOCHS=$EPOCHS / PATIENCE=$PATIENCE → suffix='$SUFFIX')"
echo ""
echo "main log:  $LOG_FILE"
echo "step logs: $COLD_LOG / $WARM_LOG / $EVAL_LOG"
echo ""
echo "確認:   tail -f $LOG_FILE"
echo "中断:   kill $PID  (子プロセスごと: pkill -TERM -P $PID; kill $PID)"
echo ""
echo "完了後の結果:"
echo "  - Cold IRL + RF: $COLD_OUTBASE/lstm_baseline/train_*/eval_*/pair_predictions.csv"
echo "  - Warm IRL + RF: $WARM_OUTBASE/lstm_baseline/train_*/eval_*/pair_predictions.csv"
echo "  - 活動パターン別 acc: $EVAL_OUTDIR/{dev,pair}_per_pattern_accuracy.csv"
