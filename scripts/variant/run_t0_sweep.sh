#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# [SES 追加分析] t0（スナップショット時刻）掃引ドライバ
# ──────────────────────────────────────────────────────────────────────────
# 論文§7の妥当性の脅威「t0を2023-01-01単一に固定」に対する感度分析。
# [訓練3年][buffer1年][評価1年] の塊を t0 ごと丸ごと平行移動して再学習・再評価する。
#
# t0 から各日付を算出（論文と同じ相対関係）:
#   EVAL_CUTOFF / PREDICTION_TIME = t0
#   TRAIN_END   = t0 − 12ヶ月
#   TRAIN_START = t0 − 48ヶ月  (=訓練窓3年, FS=0 base。DYNAMIC_START=1 で horizon 別に後方延長)
#
# 使い方:
#   bash scripts/variant/run_t0_sweep.sh <t0> <mode> [gpu_id]
#     mode: irl   … IRL を再学習＋評価（GPU推奨, DEVICE 既定 cuda）
#           rf    … RF を再学習＋評価（CPU, ローカル可）
#           verify… B定義ラベルで headline 表を集計（irl/rf 出力が揃っている前提）
#           all   … irl → rf → verify を順に
# 例:
#   # GPUサーバで IRL のみ
#   bash scripts/variant/run_t0_sweep.sh 2022-10-01 irl 0
#   # ローカルで RF と集計
#   bash scripts/variant/run_t0_sweep.sh 2022-10-01 rf
#   bash scripts/variant/run_t0_sweep.sh 2022-10-01 verify
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

T0="${1:?Usage: $0 <t0 YYYY-MM-DD> <irl|rf|verify|all> [gpu_id]}"
MODE="${2:?Usage: $0 <t0> <irl|rf|verify|all> [gpu_id]}"
GPU_ID="${3:-0}"

# [align系列] 「buffer消去＋label tail 12統一」設計（ALIGN=1）は選択肢から外した（2026-10-06）。
#   論文系列（Phase B: [訓練3年][buffer1年][評価1年]、outputs/t0_sweep/）のみで実行する。
#   戻す場合は以下のコメントアウトを解除する。
# ALIGN="${ALIGN:-0}"
# FEATURE_STEPS="${FEATURE_STEPS:-24}"    # align時の feature 月次ステップ数 (24既定/36拡張可)
ALIGN=0

# [対象プロジェクト] PROJECT=main32(既定, OpenStack) / aosp32(AOSP)。
#   aosp32 では AOSP 本実験(2026-07, outputs/aosp32/dynstartB_ep150)と同じ設定に切り替える:
#   データ・raw_json・共変更グラフ(results_aosp)、DIR_PREFIX_PROJECT=1、COCHANGE_TOPK=100、COCHANGE_MIN_WEIGHT=2。
#   出力は outputs/aosp32/t0_sweep/、軌跡キャッシュは aosp32_t0_<日付> に分離（OpenStack と混ざらない）。
#   注意: AOSP のデータは 2020-01-01〜2025-01-01 なので t0 は 2024-01-01 まで。
PROJECT="${PROJECT:-main32}"
if [ "$PROJECT" = "aosp32" ]; then
    export REVIEWS="data/combined_raw_aosp32.csv"
    export RAW_JSON_LIST_FILE="data/combined_raw_aosp32.raw_json_list.txt"
    export DIR_PREFIX_PROJECT=1 COCHANGE_TOPK=100 COCHANGE_MIN_WEIGHT=2
    HUB_SCORES_CSV="${HUB_SCORES_CSV:-experiments/dependency_analysis/results_aosp/hub_scores_main32.csv}"
    COCHANGE_NEIGHBORS_CSV="${COCHANGE_NEIGHBORS_CSV:-experiments/dependency_analysis/results_aosp/cochange_neighbors_main32.csv}"
    DEFAULT_SWEEP_ROOT="outputs/aosp32/t0_sweep"
    CACHE_PREFIX="aosp32_"
elif [ "$PROJECT" = "main32" ]; then
    export REVIEWS="data/combined_raw_main32.csv"
    export RAW_JSON_LIST_FILE="data/combined_raw_main32.raw_json_list.txt"
    DEFAULT_SWEEP_ROOT="outputs/t0_sweep"
    CACHE_PREFIX=""
else
    echo "ERROR: PROJECT は main32|aosp32 (指定: $PROJECT)" >&2; exit 1
fi

VNAME="lstm_baseline"
# if [ "$ALIGN" = "1" ]; then
#     # 出力は step 数を明示した _fs<steps> に分離（fs24=旧 outputs/t0_sweep_align を
#     # 2026-07-02 にリネーム。系列間の上書き事故防止）。
#     SWEEP_ROOT="${SWEEP_ROOT:-outputs/t0_sweep_align_fs${FEATURE_STEPS}}"
# else
SWEEP_ROOT="${SWEEP_ROOT:-$DEFAULT_SWEEP_ROOT}"
# fi
T0TAG="${T0//-/}"                       # 2022-10-01 → 20221001
IRL_OUTBASE="$SWEEP_ROOT/$T0TAG/irl"
RF_OUTBASE="$SWEEP_ROOT/$T0TAG/rf"

HUB_SCORES_CSV="${HUB_SCORES_CSV:-experiments/dependency_analysis/results/hub_scores_main32.csv}"
COCHANGE_NEIGHBORS_CSV="${COCHANGE_NEIGHBORS_CSV:-experiments/dependency_analysis/results/cochange_neighbors_main32.csv}"

# t0 → 日付（dateutil）
# if [ "$ALIGN" = "1" ]; then
#     # align: train_end=t0, train_start=(t0-12)-FEATURE_STEPS（label tail 12 を別途確保）
#     read -r TRAIN_START TRAIN_END < <(uv run python -c "
# from datetime import date
# from dateutil.relativedelta import relativedelta
# t0 = date.fromisoformat('$T0')
# print((t0 - relativedelta(months=12 + $FEATURE_STEPS)).isoformat(), t0.isoformat())
# ")
#     IRL_DYNAMIC_START=0
#     IRL_LABEL_TAIL=12
#     IRL_CACHE_TAG="t0align_${T0TAG}_s${FEATURE_STEPS}"
# else
# 従来 Phase B: [訓練3年][buffer1年][評価1年]
read -r TRAIN_START TRAIN_END < <(uv run python -c "
from datetime import date
from dateutil.relativedelta import relativedelta
t0 = date.fromisoformat('$T0')
print((t0 - relativedelta(months=48)).isoformat(), (t0 - relativedelta(months=12)).isoformat())
")
IRL_DYNAMIC_START=1
IRL_LABEL_TAIL=""
IRL_CACHE_TAG="${CACHE_PREFIX}t0_${T0TAG}"
# fi
EVAL_CUTOFF="$T0"

echo "============================================================"
echo "  [t0掃引] project=$PROJECT  t0=$T0  mode=$MODE"
echo "  TRAIN_START=$TRAIN_START  TRAIN_END=$TRAIN_END  EVAL_CUTOFF=$EVAL_CUTOFF"
echo "  IRL_OUTBASE=$IRL_OUTBASE"
echo "  RF_OUTBASE =$RF_OUTBASE"
echo "============================================================"

run_irl() {
    echo ">>> IRL 再学習＋評価 (ALIGN=$ALIGN, DYNAMIC_START=$IRL_DYNAMIC_START, LABEL_TAIL=${IRL_LABEL_TAIL:-none}, EPOCHS=150, DEVICE=${DEVICE:-cuda})"
    PYTHONHASHSEED=0 DYNAMIC_START="$IRL_DYNAMIC_START" EPOCHS="${EPOCHS:-150}" \
        TRAIN_START="$TRAIN_START" TRAIN_END="$TRAIN_END" EVAL_CUTOFF="$EVAL_CUTOFF" \
        LABEL_TAIL_MONTHS="$IRL_LABEL_TAIL" \
        DEVICE="${DEVICE:-cuda}" \
        CACHE_TAG="$IRL_CACHE_TAG" \
        HUB_SCORES_CSV="$HUB_SCORES_CSV" \
        COCHANGE_NEIGHBORS_CSV="$COCHANGE_NEIGHBORS_CSV" \
        bash scripts/variant/run_mce_irl_variant_single.sh 0 "$VNAME" "$IRL_OUTBASE" "$GPU_ID" 0
}

run_rf() {
    # if [ "$ALIGN" = "1" ]; then
    #     echo ">>> RF 再学習＋評価 (align: train_end=t0-12, train_start=t0-$((12 + FEATURE_STEPS)), tail12, seed777)"
    # else
    echo ">>> RF 再学習＋評価 (1snap, 学習窓33ヶ月で IRL と等化, seed777)"
    # fi
    PREDICTION_TIME="$EVAL_CUTOFF" \
        BASE_TRAIN_START="$TRAIN_START" BASE_TRAIN_END="$TRAIN_END" \
        LABEL_TAIL_MONTHS="$IRL_LABEL_TAIL" \
        POOL_WINDOW_DAYS=180 DELTA_MONTHS=3 HORIZONS="0 3 6 9" \
        OUTBASE="$RF_OUTBASE" DEVICE=cpu \
        HUB_SCORES_CSV="$HUB_SCORES_CSV" \
        COCHANGE_NEIGHBORS_CSV="$COCHANGE_NEIGHBORS_CSV" \
        bash scripts/local/run_rf_dynstart_1snap_local.sh
}

run_verify() {
    echo ">>> B定義ラベルで headline 表を集計"
    mkdir -p "$SWEEP_ROOT/$T0TAG"
    uv run python scripts/analyze/study/verify_blabel_headline.py \
        --data "$REVIEWS" --raw-json-list "$RAW_JSON_LIST_FILE" \
        --prediction-time "$EVAL_CUTOFF" \
        --irl-base "$IRL_OUTBASE/$VNAME" \
        --rf-base "$RF_OUTBASE" \
        --out-csv "$SWEEP_ROOT/$T0TAG/headline_blabel.csv"
}

case "$MODE" in
    irl)    run_irl ;;
    rf)     run_rf ;;
    verify) run_verify ;;
    all)    run_irl; run_rf; run_verify ;;
    *) echo "ERROR: mode は irl|rf|verify|all のいずれか (指定: $MODE)" >&2; exit 1 ;;
esac

echo "[t0掃引 t0=$T0 mode=$MODE] 完了"
