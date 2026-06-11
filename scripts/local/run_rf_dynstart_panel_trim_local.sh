#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# 月次パネル RF_Dir Phase C 整合 (短 horizon の train_start を後ろにずらし
#   全 horizon を STEP_MONTHS=24 ステップに揃える、9-12m モデル基準)
# ──────────────────────────────────────────────────────────────────────────
# 目的:
#   サーバの IRL Phase C (DYNAMIC_START=2, outputs/dynstartC_ep150/) と整合させる。
#   短 horizon ほど train_start を後ろにずらし、ラベルクリップで panel 月数を 24 に統一。
#     train_start(h) = TRAIN_END − (FS + DELTA + STEP_MONTHS) ヶ月
#       FS=0: 2022-01 − (0+3+24) = 2019-10
#       FS=3: 2022-01 − (3+3+24) = 2019-07
#       FS=6: 2022-01 − (6+3+24) = 2019-04
#       FS=9: 2022-01 − (9+3+24) = 2019-01  (= 旧 BASE_TRAIN_START)
#   Phase B (run_rf_dynstart_panel_local.sh, 後方延長) との違いは train_start
#   の計算方向のみ (前者は古いデータを足し、こちらは新しいデータだけ残す)。
#
# 出力:
#   $OUTBASE/fs_<fs>_to_<fs+delta>m/rf_panel_predictions.csv
#
# env:
#   TRAIN_END        : パネル終了月        (default 2022-01-01)
#   PREDICTION_TIME  : 推論時点            (default 2023-01-01)
#   DELTA_MONTHS     : ラベル窓長          (default 3)
#   STEP_MONTHS      : 等化目標の panel 月数 (default 24, 9-12m horizon 基準)
#   HORIZONS         : "0 3 6 9" FS 列     (default "0 3 6 9")
#   MONTH_STEP       : 何ヶ月おきか        (default 1)
#   N_JOBS           : 並列数              (default 4)
#   OUTBASE          : 出力先              (default outputs/dynstart_rf_panel_trim)
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"

TRAIN_END="${TRAIN_END:-2022-01-01}"
PREDICTION_TIME="${PREDICTION_TIME:-2023-01-01}"
DELTA_MONTHS="${DELTA_MONTHS:-3}"
STEP_MONTHS="${STEP_MONTHS:-24}"
HORIZONS="${HORIZONS:-0 3 6 9}"
MONTH_STEP="${MONTH_STEP:-1}"
N_JOBS="${N_JOBS:-4}"
OUTBASE="${OUTBASE:-outputs/dynstart_rf_panel_trim}"

if [ ! -f "$REVIEWS" ]; then
    echo "ERROR: REVIEWS が見つからない: $REVIEWS" >&2
    exit 1
fi
if [ ! -f "$RAW_JSON_LIST_FILE" ]; then
    echo "ERROR: RAW_JSON_LIST_FILE が見つからない: $RAW_JSON_LIST_FILE" >&2
    exit 1
fi
read -r -a RAW_JSON <<< "$(cat "$RAW_JSON_LIST_FILE")"

# train_start(h) = TRAIN_END − (FS + DELTA + STEP_MONTHS) ヶ月
compute_train_start() {
    local fs=$1
    uv run python -c "
from datetime import date
from dateutil.relativedelta import relativedelta
base = date.fromisoformat('$TRAIN_END')
print((base - relativedelta(months=$fs + $DELTA_MONTHS + $STEP_MONTHS)).isoformat())
"
}

echo "============================================================"
echo "  [Local] 月次パネル RF_Dir Phase C 整合 (短 horizon 前方削り)"
echo "  TRAIN_END        = $TRAIN_END"
echo "  PREDICTION_TIME  = $PREDICTION_TIME"
echo "  DELTA_MONTHS     = $DELTA_MONTHS"
echo "  STEP_MONTHS      = $STEP_MONTHS (panel 月数等化目標)"
echo "  MONTH_STEP       = $MONTH_STEP"
echo "  HORIZONS (fs)    = $HORIZONS"
echo "  N_JOBS           = $N_JOBS"
echo "  OUTBASE          = $OUTBASE"
echo "============================================================"

for fs in $HORIZONS; do
    fs_end=$((fs + DELTA_MONTHS))
    eval_dir="$OUTBASE/fs_${fs}_to_${fs_end}m"
    mkdir -p "$eval_dir"

    if [ -f "$eval_dir/summary_metrics.json" ]; then
        echo "[fs=${fs}-${fs_end}m] スキップ (既存)"
        continue
    fi

    horizon_train_start=$(compute_train_start "$fs")

    echo ""
    echo "[fs=${fs}-${fs_end}m] 月次パネル RF Phase C 評価開始 (train_start=$horizon_train_start, train_end=$TRAIN_END)..."
    uv run python scripts/analyze/eval/eval_rf_panel.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$horizon_train_start" \
        --train-end "$TRAIN_END" \
        --prediction-time "$PREDICTION_TIME" \
        --future-start-months "$fs" \
        --delta-months "$DELTA_MONTHS" \
        --month-step "$MONTH_STEP" \
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
        echo ""
        echo "  $d/summary_metrics.json:"
        uv run python -c "
import json
with open('$d/summary_metrics.json') as f: m = json.load(f)
print(f\"    pair_AUC_ROC={m['pair_auc_roc']:.4f}, AUC_PR={m['pair_auc_pr']:.4f}, \"
      f\"P={m.get('clf_precision', float('nan')):.4f}, R={m.get('clf_recall', float('nan')):.4f}, \"
      f\"F1={m.get('clf_f1', float('nan')):.4f}, thr={m.get('clf_threshold', float('nan')):.4f}, \"
      f\"base={m['base_rate']:.4f}, n={m['n_pairs']}, panel={m['train_panel_size']}\")
" 2>/dev/null || cat "$d/summary_metrics.json"
    fi
done
echo "============================================================"
