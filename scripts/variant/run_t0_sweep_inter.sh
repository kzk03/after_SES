#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# [SES 追加分析] t0掃引・中間設計（buffer消去・tail統一なし）専用ドライバ
# ──────────────────────────────────────────────────────────────────────────
# 論文(Phase B, buffer1年) から buffer だけ消した「中間」設計。
# 論文 → 中間 → align の3段で「buffer消去」と「tail統一(+2年学習)」の効果を段階分離する。
#   論文 : [訓練 t0-48〜t0-12][buffer 1年][評価]    (train_end=t0-12)
#   中間 : [訓練 t0-48〜t0(buffer区間もラベル観測へ)][評価]  ← これ
#   align: buffer消去 + ラベル観測 tail を全 horizon 12統一 + 2年学習
#
# 中間設計 (t0相対, 論文との差は TRAIN_END だけ):
#   EVAL_CUTOFF = t0
#   TRAIN_END   = t0          (buffer消去; 論文は t0-12)
#   TRAIN_START = t0 - 48     (3年相当, DYNAMIC_START=1 で horizon 別に後方延長)
#   LABEL_TAIL  = ""          (horizon依存 FE=3/6/9/12; tail統一しない=論文と同じ)
# 出力は別系列 outputs/t0_sweep_inter/ に分離（論文/align 系列を壊さない）。
#
# ★リーク注意: TRAIN_END=t0 のため IRL 実行の副産物 RF はリークする
#   (eval が --rf-train-end=TRAIN_END=t0 を受け、RF訓練ラベル窓 [t0+FS,t0+FS+3] が
#    評価窓と一致)。比較には使わず、ローカル専用 RF (run_rf; 両端shiftで
#    rf_train_end=t0-(FS+3)、leakなし) を使う。t0_compare.py は構造的に安全。
#   IRL 本体はラベルが train_end でクリップされ評価窓と被らない(leakなし)。
#
# 使い方 (既存 run_t0_sweep.sh と同じI/F、論文/align用スクリプトには非依存):
#   bash scripts/variant/run_t0_sweep_inter.sh <t0> <irl|rf|verify|all> [gpu_id]
# 例:
#   bash scripts/variant/run_t0_sweep_inter.sh 2022-10-01 irl 0   # GPUサーバ
#   bash scripts/variant/run_t0_sweep_inter.sh 2022-10-01 rf      # ローカル
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

T0="${1:?Usage: $0 <t0 YYYY-MM-DD> <irl|rf|verify|all> [gpu_id]}"
MODE="${2:?Usage: $0 <t0> <irl|rf|verify|all> [gpu_id]}"
GPU_ID="${3:-0}"

VNAME="lstm_baseline"
SWEEP_ROOT="${SWEEP_ROOT:-outputs/t0_sweep_inter}"
T0TAG="${T0//-/}"                       # 2022-10-01 → 20221001
IRL_OUTBASE="$SWEEP_ROOT/$T0TAG/irl"
RF_OUTBASE="$SWEEP_ROOT/$T0TAG/rf"

HUB_SCORES_CSV="${HUB_SCORES_CSV:-experiments/dependency_analysis/results/hub_scores_main32.csv}"
COCHANGE_NEIGHBORS_CSV="${COCHANGE_NEIGHBORS_CSV:-experiments/dependency_analysis/results/cochange_neighbors_main32.csv}"

# 中間設計: train_end=t0 (buffer消去), train_start=t0-48, tail未指定(horizon依存FE)
read -r TRAIN_START TRAIN_END < <(uv run python -c "
from datetime import date
from dateutil.relativedelta import relativedelta
t0 = date.fromisoformat('$T0')
print((t0 - relativedelta(months=48)).isoformat(), t0.isoformat())
")
IRL_DYNAMIC_START=1
IRL_LABEL_TAIL=""
IRL_CACHE_TAG="t0inter_${T0TAG}"
EVAL_CUTOFF="$T0"

echo "============================================================"
echo "  [t0掃引・中間] t0=$T0  mode=$MODE"
echo "  TRAIN_START=$TRAIN_START  TRAIN_END=$TRAIN_END (=t0, buffer消去)  EVAL_CUTOFF=$EVAL_CUTOFF"
echo "  DYNAMIC_START=$IRL_DYNAMIC_START  LABEL_TAIL=${IRL_LABEL_TAIL:-none}(horizon依存)"
echo "  IRL_OUTBASE=$IRL_OUTBASE"
echo "  RF_OUTBASE =$RF_OUTBASE"
echo "============================================================"

run_irl() {
    echo ">>> IRL 再学習＋評価 (中間: buffer消去, DYNAMIC_START=$IRL_DYNAMIC_START, LABEL_TAIL=none, EPOCHS=150, DEVICE=${DEVICE:-cuda})"
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
    echo ">>> RF 再学習＋評価 (中間 1snap: train_end=t0, train_start=t0-48, tail未指定, 両端shiftで leak回避, seed777)"
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
    echo ">>> B定義ラベルで headline 表を集計 (RF は rf_base=ローカル専用RF を使用; リークなし)"
    mkdir -p "$SWEEP_ROOT/$T0TAG"
    uv run python scripts/analyze/study/verify_blabel_headline.py \
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

echo "[t0掃引・中間 t0=$T0 mode=$MODE] 完了"
