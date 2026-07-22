#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# SES査読 A-2/B-3 対応: 退化版(最終有効ステップ教師のみ)系列分類器の逐次実行キュー
#
# 1つのGPUで以下を順に実行する（各ステップは学習済みならスキップ＝再実行安全）:
#   1. variant 0: lstm_final_only        (A-2/B-3: LSTM退化版)
#   2. variant 8: gru_final_only         (B-3: RNN=GRU)
#   3. variant 2: transformer_final_only (B-3: Transformer)
#
# 1つ失敗しても残りを続行し、最後に成否一覧を表示する。
#
# 使い方（サーバ、リポジトリルートで）:
#   nohup bash scripts/variant/run_ses_b3_ablation_queue.sh 0 > logs/ses_b3_queue.log 2>&1 &
#
# env（いずれも上書き可。既定は論文ヘッドライン dynstartB_ep150 と同条件）:
#   OUTBASE          : 出力ベース (default outputs/ses_review/b3_ablation)
#   DYNAMIC_START    : 学習窓等化方式 (default 1 = Phase B, dynstartB と同じ)
#                      注: 論文本文の「訓練期間3年」と異なり、3-6m以降の horizon は
#                      train_start を FS ヶ月前倒しして学習窓 33 ステップに等化する。
#                      比較相手のヘッドライン IRL (dynstartB) がこの条件なので既定は 1。
#                      本文どおり厳密 3 年にするなら 0（ただしステップ数が 33/30/27/24 とズレる）。
#                      リーク防止バッファ 1 年 (train_end=t0-12M, ラベルは train_end で
#                      後方クリップ) はどちらでも保たれる。
#   EPOCHS           : (default 150)
#   FINAL_STEP_ONLY  : (default 1 = 退化版。0 にすると IRL 定式化のままバックボーン比較)
#   CACHE_TAG        : 軌跡キャッシュのタグ。既存の通常IRL実行と同じ値にすると
#                      キャッシュを再利用できる (default "default")
#   HUB_SCORES_CSV / COCHANGE_NEIGHBORS_CSV : path特徴量CSV (default main32用)
# ──────────────────────────────────────────────────────────────────────────
set -uo pipefail

GPU_ID="${1:-0}"
OUTBASE="${OUTBASE:-outputs/ses_review/b3_ablation}"

export DYNAMIC_START="${DYNAMIC_START:-1}"
export EPOCHS="${EPOCHS:-150}"
export FINAL_STEP_ONLY="${FINAL_STEP_ONLY:-1}"
export HUB_SCORES_CSV="${HUB_SCORES_CSV:-experiments/dependency_analysis/results/hub_scores_main32.csv}"
export COCHANGE_NEIGHBORS_CSV="${COCHANGE_NEIGHBORS_CSV:-experiments/dependency_analysis/results/cochange_neighbors_main32.csv}"

# "variant_id:variant_name" の実行キュー
QUEUE=(
    "0:lstm_final_only"
    "8:gru_final_only"
    "2:transformer_final_only"
)

echo "============================================================"
echo "  SES B-3 ablation queue (GPU=$GPU_ID)"
echo "  OUTBASE=$OUTBASE DYNAMIC_START=$DYNAMIC_START EPOCHS=$EPOCHS FINAL_STEP_ONLY=$FINAL_STEP_ONLY"
echo "  queue: ${QUEUE[*]}"
echo "============================================================"

declare -a RESULTS=()
for entry in "${QUEUE[@]}"; do
    vtype="${entry%%:*}"
    vname="${entry#*:}"
    echo ""
    echo "──── [queue] variant $vtype: $vname 開始 ($(date '+%F %T')) ────"
    if bash scripts/variant/run_mce_irl_variant_single.sh "$vtype" "$vname" "$OUTBASE" "$GPU_ID"; then
        RESULTS+=("OK   $vname")
    else
        RESULTS+=("FAIL $vname (exit=$?)")
        echo "[queue] $vname が失敗。次のバリアントへ進む。"
    fi
done

echo ""
echo "============================================================"
echo "  キュー完了 ($(date '+%F %T')) 成否一覧:"
for r in "${RESULTS[@]}"; do
    echo "    $r"
done
echo "  結果: $OUTBASE/<variant_name>/train_*m/eval_*m/"
echo "============================================================"
