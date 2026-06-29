#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# RF_Dir 1snap dynstart 検証 (両端 shift, 学習窓 33 月で IRL と完全等化)
# ──────────────────────────────────────────────────────────────────────────
# 目的:
#   サーバの IRL dynstartB 実験 (outputs/dynstartB_ep150/) の月次ステップ等化
#   (= 全 horizon で 33 ステップ) と整合させるため、RF_Dir 1snap の学習窓
#   両端を horizon (FS) 別にシフト:
#     rf_train_start(h) = BASE_TRAIN_START − FS(h)        月
#     rf_train_end(h)   = BASE_TRAIN_END   − (FS+delta)(h) 月
#                       = IRL dynstartB の最終 m_end と同じ
#   → 全 horizon で 学習窓 = 33 月に等化される。
#   候補ペアプール (推論対象 dev-dir) は POOL_WINDOW_DAYS=180 で固定し、
#   main の 4,534 ペアと整合させる。
#
# 出力:
#   $OUTBASE/fs_<fs>_to_<fs+delta>m/pair_predictions.csv (rf_dir_prob 列)
#
# env:
#   PREDICTION_TIME      : 推論時点                          (default 2023-01-01)
#   BASE_TRAIN_END       : FS=0 時の rf_train_end             (default 2022-01-01)
#   BASE_TRAIN_START     : FS=0 時の rf_train_start           (default 2019-01-01)
#   POOL_WINDOW_DAYS     : 候補プール窓 (日)                  (default 180)
#   DELTA_MONTHS         : ラベル窓長 (月)                    (default 3)
#   HORIZONS             : "0 3 6 9" 形式の FS 列             (default "0 3 6 9")
#   OUTBASE              : 出力ベース                         (default outputs/dynstart_rf_1snap)
#   DEVICE               : (RF は CPU)                        (default cpu)
#   N_JOBS               : 並列数                             (default 4)
#   HUB_SCORES_CSV       : path 特徴量 hub_scores             (default experiments/.../hub_scores_main32.csv)
#   COCHANGE_NEIGHBORS_CSV: path 特徴量 cochange              (default experiments/.../cochange_neighbors_main32.csv)
#
# 例:
#   bash scripts/local/run_rf_dynstart_1snap_local.sh
#   HORIZONS="0" OUTBASE=outputs/dynstart_rf_1snap_test bash scripts/local/run_rf_dynstart_1snap_local.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"

PREDICTION_TIME="${PREDICTION_TIME:-2023-01-01}"
BASE_TRAIN_END="${BASE_TRAIN_END:-${RF_TRAIN_END:-2022-01-01}}"
BASE_TRAIN_START="${BASE_TRAIN_START:-2019-01-01}"
POOL_WINDOW_DAYS="${POOL_WINDOW_DAYS:-180}"
DELTA_MONTHS="${DELTA_MONTHS:-3}"
HORIZONS="${HORIZONS:-0 3 6 9}"
# [align系列] 指定時、学習窓を全 horizon 共通に等化する:
#   rf_train_end(h)   = BASE_TRAIN_END  - LABEL_TAIL_MONTHS   （全 horizon 同一）
#   rf_train_start(h) = BASE_TRAIN_START                       （FS シフトなし）
# → IRL align の feature 月次集合 [BASE_TRAIN_START, BASE_TRAIN_END-12] と一致。
LABEL_TAIL_MONTHS="${LABEL_TAIL_MONTHS:-}"
OUTBASE="${OUTBASE:-outputs/dynstart_rf_1snap}"
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

# IRL モデルを指さない (絶対に存在しないパスにする)
DUMMY_IRL_MODEL="/dev/null/no_irl_model.pt"

# rf_train_start(h) = BASE_TRAIN_START - FS(h) months
# align (LABEL_TAIL_MONTHS 指定時): 全 horizon 同一 BASE_TRAIN_START（FS シフトなし）
compute_train_start() {
    local fs=$1
    local shift=$fs
    [ -n "$LABEL_TAIL_MONTHS" ] && shift=0
    uv run python -c "
from datetime import date
from dateutil.relativedelta import relativedelta
base = date.fromisoformat('$BASE_TRAIN_START')
print((base - relativedelta(months=$shift)).isoformat())
"
}

# rf_train_end(h) = BASE_TRAIN_END - (FS + delta)(h) months
# = IRL dynstartB の最終 m_end と同じ。これで学習窓 33 月で完全等化。
# align (LABEL_TAIL_MONTHS 指定時): 全 horizon 共通 BASE_TRAIN_END - LABEL_TAIL_MONTHS
compute_train_end() {
    local fs=$1
    local delta=$2
    local back=$(( fs + delta ))
    [ -n "$LABEL_TAIL_MONTHS" ] && back="$LABEL_TAIL_MONTHS"
    uv run python -c "
from datetime import date
from dateutil.relativedelta import relativedelta
base = date.fromisoformat('$BASE_TRAIN_END')
print((base - relativedelta(months=$back)).isoformat())
"
}

echo "============================================================"
echo "  [Local] RF_Dir 1snap dynstart 検証 (Option B: 両端 shift, 学習窓 33月)"
echo "  PREDICTION_TIME    = $PREDICTION_TIME"
echo "  BASE_TRAIN_END     = $BASE_TRAIN_END (FS=0)"
echo "  BASE_TRAIN_START   = $BASE_TRAIN_START (FS=0)"
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

    rf_train_start=$(compute_train_start "$fs")
    rf_train_end=$(compute_train_end "$fs" "$DELTA_MONTHS")

    echo ""
    echo "[fs=${fs}-${fs_end}m] RF_Dir dynstart 評価開始 (rf_train_start=$rf_train_start, rf_train_end=$rf_train_end)..."
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
