#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# RF_Dir のみローカル検証 (IRL なし)
# ──────────────────────────────────────────────────────────────────────────
# 目的:
#   IRL モデルを学習せず、RF_Dir の予測性能だけ評価する。
#   eval_mce_irl_path_prediction.py は --irl-dir-model 未指定なら IRL_Dir を
#   スキップし、--irl-model が存在しなければグローバル IRL もスキップする。
#   ⇒ RF_Dir + Naive + Linear + (RF global) のみが走る。
#
# 期間/ラベル sensitivity 用に以下を env で上書き可:
#   PREDICTION_TIME  : 推論時点               (default 2023-01-01)
#   RF_TRAIN_END     : RF 学習スナップショット (default 2022-01-01)
#   WINDOW_DAYS      : RF 特徴量窓 (日)        (default 180)
#   DELTA_MONTHS     : ラベル窓長 (月)         (default 3)
#   HORIZONS         : "0 3 6 9" 形式の future_start_months 列
#                                              (default "0 3 6 9")
#   OUTBASE          : 出力ベース              (default outputs/local_rf_only)
#   DEVICE           : (RF は CPU、互換のため受ける)  (default cpu)
#   N_JOBS           : 並列数                  (default 4)
#
# 例:
#   bash scripts/local/run_rf_only_local.sh
#   WINDOW_DAYS=365 OUTBASE=outputs/local_rf_w365 bash scripts/local/run_rf_only_local.sh
#   RF_TRAIN_END=2021-07-01 PREDICTION_TIME=2022-07-01 bash scripts/local/run_rf_only_local.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 設定 ──
REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"

PREDICTION_TIME="${PREDICTION_TIME:-2023-01-01}"
RF_TRAIN_END="${RF_TRAIN_END:-2022-01-01}"
WINDOW_DAYS="${WINDOW_DAYS:-180}"
DELTA_MONTHS="${DELTA_MONTHS:-3}"
HORIZONS="${HORIZONS:-0 3 6 9}"
OUTBASE="${OUTBASE:-outputs/local_rf_only}"
DEVICE="${DEVICE:-cpu}"
N_JOBS="${N_JOBS:-4}"

# ── 入力存在チェック ──
if [ ! -f "$REVIEWS" ]; then
    echo "ERROR: REVIEWS が見つからない: $REVIEWS" >&2
    exit 1
fi
if [ ! -f "$RAW_JSON_LIST_FILE" ]; then
    echo "ERROR: RAW_JSON_LIST_FILE が見つからない: $RAW_JSON_LIST_FILE" >&2
    exit 1
fi
read -r -a RAW_JSON <<< "$(cat "$RAW_JSON_LIST_FILE")"

# IRL モデルを指さない (絶対に存在しないパスにする)
DUMMY_IRL_MODEL="/dev/null/no_irl_model.pt"

echo "============================================================"
echo "  [Local] RF_Dir 単独検証"
echo "  PREDICTION_TIME = $PREDICTION_TIME"
echo "  RF_TRAIN_END    = $RF_TRAIN_END"
echo "  WINDOW_DAYS     = $WINDOW_DAYS"
echo "  DELTA_MONTHS    = $DELTA_MONTHS"
echo "  HORIZONS (fs)   = $HORIZONS"
echo "  OUTBASE         = $OUTBASE"
echo "============================================================"

for fs in $HORIZONS; do
    fs_end=$((fs + DELTA_MONTHS))
    eval_dir="$OUTBASE/fs_${fs}_to_${fs_end}m"
    mkdir -p "$eval_dir"

    if [ -f "$eval_dir/summary_metrics.json" ]; then
        echo "[fs=${fs}-${fs_end}m] スキップ (既存)"
        continue
    fi

    echo ""
    echo "[fs=${fs}-${fs_end}m] RF_Dir 評価開始..."
    uv run python scripts/analyze/eval/eval_mce_irl_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --irl-model "$DUMMY_IRL_MODEL" \
        --prediction-time "$PREDICTION_TIME" \
        --delta-months "$DELTA_MONTHS" \
        --future-start-months "$fs" \
        --rf-future-start-months "$fs" \
        --rf-train-end "$RF_TRAIN_END" \
        --window-days "$WINDOW_DAYS" \
        --device "$DEVICE" \
        --n-jobs "$N_JOBS" \
        --output-dir "$eval_dir"
    echo "[fs=${fs}-${fs_end}m] 完了 → $eval_dir"
done

echo ""
echo "============================================================"
echo "  全 horizon 完了。結果一覧:"
for d in "$OUTBASE"/fs_*; do
    [ -d "$d" ] || continue
    if [ -f "$d/summary_metrics.json" ]; then
        echo "  $d/summary_metrics.json"
    fi
done
echo "============================================================"
