#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# RF_Dir 1snap Phase C 整合 (短 horizon の rf_train_start を後ろにずらし
#   全 horizon を STEP_MONTHS=24 ヶ月学習窓に揃える、9-12m モデル基準)
# ──────────────────────────────────────────────────────────────────────────
# 目的:
#   サーバの IRL Phase C (DYNAMIC_START=2, outputs/dynstartC_ep150/) と整合させる。
#   rf_train_end(h) = BASE_TRAIN_END − (FS+DELTA) ヶ月  (Phase B と同じ)
#   rf_train_start(h) = rf_train_end(h) − STEP_MONTHS ヶ月  (Phase B から変更)
#     FS=0: rf_train_end=2021-10, rf_train_start=2019-10  (24 月)
#     FS=3: rf_train_end=2021-07, rf_train_start=2019-07  (24 月)
#     FS=6: rf_train_end=2021-04, rf_train_start=2019-04  (24 月)
#     FS=9: rf_train_end=2021-01, rf_train_start=2019-01  (24 月, Phase B と一致)
#   全 horizon で 学習窓 = 24 月 = IRL Phase C と等化。
#   候補ペアプール (推論対象 dev-dir) は POOL_WINDOW_DAYS=180 で固定。
#
# 出力:
#   $OUTBASE/fs_<fs>_to_<fs+delta>m/pair_predictions.csv (rf_dir_prob 列)
#
# env:
#   PREDICTION_TIME      : 推論時点                          (default 2023-01-01)
#   BASE_TRAIN_END       : FS=0 時の rf_train_end             (default 2022-01-01)
#   STEP_MONTHS          : 等化目標の学習窓長 (月)           (default 24)
#   POOL_WINDOW_DAYS     : 候補プール窓 (日)                  (default 180)
#   DELTA_MONTHS         : ラベル窓長 (月)                    (default 3)
#   HORIZONS             : "0 3 6 9" 形式の FS 列             (default "0 3 6 9")
#   OUTBASE              : 出力ベース                         (default outputs/dynstart_rf_1snap_trim)
#   DEVICE               : (RF は CPU)                        (default cpu)
#   N_JOBS               : 並列数                             (default 4)
#   HUB_SCORES_CSV       : path 特徴量 hub_scores
#   COCHANGE_NEIGHBORS_CSV: path 特徴量 cochange
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"

PREDICTION_TIME="${PREDICTION_TIME:-2023-01-01}"
BASE_TRAIN_END="${BASE_TRAIN_END:-${RF_TRAIN_END:-2022-01-01}}"
STEP_MONTHS="${STEP_MONTHS:-24}"
POOL_WINDOW_DAYS="${POOL_WINDOW_DAYS:-180}"
DELTA_MONTHS="${DELTA_MONTHS:-3}"
HORIZONS="${HORIZONS:-0 3 6 9}"
OUTBASE="${OUTBASE:-outputs/dynstart_rf_1snap_trim}"
DEVICE="${DEVICE:-cpu}"
N_JOBS="${N_JOBS:-4}"

HUB_SCORES_CSV="${HUB_SCORES_CSV:-experiments/dependency_analysis/results/hub_scores_main32.csv}"
COCHANGE_NEIGHBORS_CSV="${COCHANGE_NEIGHBORS_CSV:-experiments/dependency_analysis/results/cochange_neighbors_main32.csv}"

if [ ! -f "$REVIEWS" ]; then
    echo "ERROR: REVIEWS が見つからない: $REVIEWS" >&2
    exit 1
fi
if [ ! -f "$RAW_JSON_LIST_FILE" ]; then
    echo "ERROR: RAW_JSON_LIST_FILE が見つからない: $RAW_JSON_LIST_FILE" >&2
    exit 1
fi
if [ ! -f "$HUB_SCORES_CSV" ]; then
    echo "ERROR: HUB_SCORES_CSV が見つからない: $HUB_SCORES_CSV" >&2
    exit 1
fi
if [ ! -f "$COCHANGE_NEIGHBORS_CSV" ]; then
    echo "ERROR: COCHANGE_NEIGHBORS_CSV が見つからない: $COCHANGE_NEIGHBORS_CSV" >&2
    exit 1
fi
read -r -a RAW_JSON <<< "$(cat "$RAW_JSON_LIST_FILE")"

DUMMY_IRL_MODEL="/dev/null/no_irl_model.pt"

# rf_train_end(h) = BASE_TRAIN_END − (FS+DELTA) ヶ月
compute_train_end() {
    local fs=$1
    local delta=$2
    uv run python -c "
from datetime import date
from dateutil.relativedelta import relativedelta
base = date.fromisoformat('$BASE_TRAIN_END')
print((base - relativedelta(months=$fs + $delta)).isoformat())
"
}

# rf_train_start(h) = rf_train_end(h) − STEP_MONTHS ヶ月
compute_train_start_from_end() {
    local rf_train_end=$1
    uv run python -c "
from datetime import date
from dateutil.relativedelta import relativedelta
end = date.fromisoformat('$rf_train_end')
print((end - relativedelta(months=$STEP_MONTHS)).isoformat())
"
}

echo "============================================================"
echo "  [Local] RF_Dir 1snap Phase C 整合 (短 horizon 前方削り, 24 月等化)"
echo "  PREDICTION_TIME    = $PREDICTION_TIME"
echo "  BASE_TRAIN_END     = $BASE_TRAIN_END (FS=0)"
echo "  STEP_MONTHS        = $STEP_MONTHS (学習窓等化目標)"
echo "  POOL_WINDOW_DAYS   = $POOL_WINDOW_DAYS (候補プール固定)"
echo "  DELTA_MONTHS       = $DELTA_MONTHS"
echo "  HORIZONS (fs)      = $HORIZONS"
echo "  OUTBASE            = $OUTBASE"
echo "============================================================"

for fs in $HORIZONS; do
    fs_end=$((fs + DELTA_MONTHS))
    eval_dir="$OUTBASE/fs_${fs}_to_${fs_end}m"
    mkdir -p "$eval_dir"

    if [ -f "$eval_dir/summary_metrics.json" ]; then
        echo "[fs=${fs}-${fs_end}m] スキップ (既存)"
        continue
    fi

    rf_train_end=$(compute_train_end "$fs" "$DELTA_MONTHS")
    rf_train_start=$(compute_train_start_from_end "$rf_train_end")

    echo ""
    echo "[fs=${fs}-${fs_end}m] RF_Dir Phase C 評価開始 (rf_train_start=$rf_train_start, rf_train_end=$rf_train_end, window=${STEP_MONTHS}m)..."
    uv run python scripts/analyze/eval/eval_mce_irl_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --irl-model "$DUMMY_IRL_MODEL" \
        --prediction-time "$PREDICTION_TIME" \
        --delta-months "$DELTA_MONTHS" \
        --future-start-months "$fs" \
        --rf-future-start-months "$fs" \
        --rf-train-end "$rf_train_end" \
        --rf-train-start "$rf_train_start" \
        --pool-window-days "$POOL_WINDOW_DAYS" \
        --hub-scores "$HUB_SCORES_CSV" \
        --cochange-neighbors "$COCHANGE_NEIGHBORS_CSV" \
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
