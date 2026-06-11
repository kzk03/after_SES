#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# 月次パネル RF_Dir dynstart 検証 (horizon ごとに TRAIN_START を前倒し)
# ──────────────────────────────────────────────────────────────────────────
# 目的:
#   サーバの IRL dynstartB (outputs/dynstartB_ep150/) と同じく、horizon (FS) 別に
#   TRAIN_START = BASE_TRAIN_START − FS(h) 月へ前倒して月次パネルを構築する。
#   結果として全 horizon で panel 月数が ~33 に揃う
#   (固定 train_start の従来版では 33/30/27/24 とズレていた)。
#
# 出力:
#   $OUTBASE/fs_<fs>_to_<fs+delta>m/rf_panel_predictions.csv
#
# env:
#   BASE_TRAIN_START : FS=0 時のパネル開始月  (default 2019-01-01)
#   TRAIN_END        : パネル終了月            (default 2022-01-01)
#   PREDICTION_TIME  : 推論時点                (default 2023-01-01)
#   DELTA_MONTHS     : ラベル窓長              (default 3)
#   HORIZONS         : "0 3 6 9" FS 列         (default "0 3 6 9")
#   MONTH_STEP       : 何ヶ月おきか            (default 1)
#   N_JOBS           : 並列数                  (default 4)
#   OUTBASE          : 出力先                  (default outputs/dynstart_rf_panel)
#
# 例:
#   bash scripts/local/run_rf_dynstart_panel_local.sh
#   HORIZONS="0" OUTBASE=outputs/dynstart_rf_panel_test bash scripts/local/run_rf_dynstart_panel_local.sh
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"

BASE_TRAIN_START="${BASE_TRAIN_START:-2019-01-01}"
TRAIN_END="${TRAIN_END:-2022-01-01}"
PREDICTION_TIME="${PREDICTION_TIME:-2023-01-01}"
DELTA_MONTHS="${DELTA_MONTHS:-3}"
HORIZONS="${HORIZONS:-0 3 6 9}"
MONTH_STEP="${MONTH_STEP:-1}"
N_JOBS="${N_JOBS:-4}"
OUTBASE="${OUTBASE:-outputs/dynstart_rf_panel}"

if [ ! -f "$REVIEWS" ]; then
    echo "ERROR: REVIEWS が見つからない: $REVIEWS" >&2
    exit 1
fi
if [ ! -f "$RAW_JSON_LIST_FILE" ]; then
    echo "ERROR: RAW_JSON_LIST_FILE が見つからない: $RAW_JSON_LIST_FILE" >&2
    exit 1
fi
read -r -a RAW_JSON <<< "$(cat "$RAW_JSON_LIST_FILE")"

compute_train_start() {
    local fs=$1
    uv run python -c "
from datetime import date
from dateutil.relativedelta import relativedelta
base = date.fromisoformat('$BASE_TRAIN_START')
print((base - relativedelta(months=$fs)).isoformat())
"
}

echo "============================================================"
echo "  [Local] 月次パネル RF_Dir dynstart 検証"
echo "  BASE_TRAIN_START = $BASE_TRAIN_START (FS=0)"
echo "  TRAIN_END        = $TRAIN_END"
echo "  PREDICTION_TIME  = $PREDICTION_TIME"
echo "  DELTA_MONTHS     = $DELTA_MONTHS"
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
    echo "[fs=${fs}-${fs_end}m] 月次パネル RF dynstart 評価開始 (train_start=$horizon_train_start)..."
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
print(f\"    pair_AUC_ROC={m['pair_auc_roc']:.4f}, AUC_PR={m['pair_auc_pr']:.4f}, base={m['base_rate']:.4f}, n={m['n_pairs']}, panel={m['train_panel_size']}\")
" 2>/dev/null || cat "$d/summary_metrics.json"
    fi
done
echo "============================================================"
