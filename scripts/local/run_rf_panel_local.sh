#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# 月次パネル RF_Dir 単独検証 (IRL 準拠の3年分累積データで RF を学習)
# ──────────────────────────────────────────────────────────────────────────
# 通常の RF_Dir (1スナップショット) ではなく、TRAIN_START〜TRAIN_END の
# 各月ごとに (dev, dir) ペアを再計算して学習データに積む。
# IRL と同じデータ露出量で RF を学習し、predictability を比較する。
#
# env:
#   TRAIN_START      : パネル開始月 (default 2019-01-01)
#   TRAIN_END        : パネル終了月 (default 2022-01-01)
#   PREDICTION_TIME  : 推論時点     (default 2023-01-01)
#   DELTA_MONTHS     : ラベル窓長   (default 3)
#   HORIZONS         : "0 3 6 9"    (future_start_months 列)
#   MONTH_STEP       : 何ヶ月おきか (default 1, データ削減なら 3 等)
#   N_JOBS           : 並列数       (default 4)
#   OUTBASE          : 出力先       (default outputs/local_rf_panel)
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"

TRAIN_START="${TRAIN_START:-2019-01-01}"
TRAIN_END="${TRAIN_END:-2022-01-01}"
PREDICTION_TIME="${PREDICTION_TIME:-2023-01-01}"
DELTA_MONTHS="${DELTA_MONTHS:-3}"
HORIZONS="${HORIZONS:-0 3 6 9}"
MONTH_STEP="${MONTH_STEP:-1}"
N_JOBS="${N_JOBS:-4}"
OUTBASE="${OUTBASE:-outputs/local_rf_panel}"

if [ ! -f "$REVIEWS" ]; then
    echo "ERROR: REVIEWS が見つからない: $REVIEWS" >&2
    exit 1
fi
if [ ! -f "$RAW_JSON_LIST_FILE" ]; then
    echo "ERROR: RAW_JSON_LIST_FILE が見つからない: $RAW_JSON_LIST_FILE" >&2
    exit 1
fi
read -r -a RAW_JSON <<< "$(cat "$RAW_JSON_LIST_FILE")"

echo "============================================================"
echo "  [Local] 月次パネル RF_Dir 単独検証"
echo "  TRAIN_START     = $TRAIN_START"
echo "  TRAIN_END       = $TRAIN_END"
echo "  PREDICTION_TIME = $PREDICTION_TIME"
echo "  DELTA_MONTHS    = $DELTA_MONTHS"
echo "  MONTH_STEP      = $MONTH_STEP"
echo "  HORIZONS (fs)   = $HORIZONS"
echo "  N_JOBS          = $N_JOBS"
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
    echo "[fs=${fs}-${fs_end}m] 月次パネル RF 評価開始..."
    uv run python scripts/analyze/eval/eval_rf_panel.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" \
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
print(f\"    pair_AUC_ROC={m['pair_auc_roc']:.4f}, AUC_PR={m['pair_auc_pr']:.4f}, base={m['base_rate']:.4f}, n={m['n_pairs']}, panel={m['train_panel_size']}\")
" 2>/dev/null || cat "$d/summary_metrics.json"
    fi
done
echo "============================================================"
