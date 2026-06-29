#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# [SES 追加分析] t0掃引・中間設計 アーキタイプ分析ドライバ
# ──────────────────────────────────────────────────────────────────────────
# run_t0_sweep_inter.sh で再学習した中間設計(buffer消去)の各 t0 の IRL(cold)+RF
# 出力に対し、論文と同じアーキタイプ分析 (dev-level + pair-level, B定義) を回す。
# 出力は outputs/t0_sweep_inter/ 系列。論文/align 用スクリプトには非依存。
#
#   IRL_BASE  = outputs/t0_sweep_inter/<tag>/irl/lstm_baseline
#   RF_BASE   = outputs/t0_sweep_inter/<tag>/rf   (ローカル専用RF; リークなし)
#   TRAIN_END = t0 − 12ヶ月   (= archetype 特徴量の feature_end; 論文/align と統一して横断比較可)
# 中間系列では base 2023-01 も再学習物(SWEEP_ROOT 配下)を使う。
# IRL/RF 出力が未着の t0 は skip。
#
# 使い方:
#   bash scripts/variant/run_t0_archetype_inter.sh <t0 YYYY-MM-DD> [both|dev|pair] [k_grid...]
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

T0="${1:?Usage: $0 <t0 YYYY-MM-DD> [both|dev|pair] [k_grid...]}"
LEVEL="${2:-both}"
shift $(( $# >= 2 ? 2 : 1 ))
K_GRID=("$@")
[ ${#K_GRID[@]} -eq 0 ] && K_GRID=(2 3 4 5 6)

SWEEP_ROOT="${SWEEP_ROOT:-outputs/t0_sweep_inter}"
T0TAG="${T0//-/}"                       # 2022-10-01 → 20221001
OUT_DIR="$SWEEP_ROOT/$T0TAG/archetype"

# t0 → train_end (= feature_end)。論文/align と揃え t0-12（横断比較の一貫性のため）。
TRAIN_END="$(uv run python -c "
from datetime import date
from dateutil.relativedelta import relativedelta
print((date.fromisoformat('$T0') - relativedelta(months=12)).isoformat())
")"

# 中間系列では base 2023-01 も再学習物（SWEEP_ROOT 配下）を使う。
IRL_BASE="$SWEEP_ROOT/$T0TAG/irl/lstm_baseline"
RF_BASE="$SWEEP_ROOT/$T0TAG/rf"

echo "============================================================"
echo "  [t0アーキタイプ・中間] t0=$T0  level=$LEVEL  k=${K_GRID[*]}"
echo "  IRL_BASE  = $IRL_BASE"
echo "  RF_BASE   = $RF_BASE"
echo "  TRAIN_END = $TRAIN_END   OUT_DIR = $OUT_DIR"
echo "============================================================"

# IRL 出力が 4 horizon 揃っているか確認 (未着なら skip)
missing=0
for h in 0-3 3-6 6-9 9-12; do
    [ -f "$IRL_BASE/train_${h}m/eval_${h}m/pair_predictions.csv" ] || missing=1
done
if [ "$missing" -eq 1 ]; then
    echo "[skip] t0=$T0: IRL 出力が未着 ($IRL_BASE)。GPU 学習の完了後に再実行してください。"
    exit 0
fi
# RF も確認
for h in 0_to_3 3_to_6 6_to_9 9_to_12; do
    if [ ! -f "$RF_BASE/fs_${h}m/pair_predictions.csv" ]; then
        echo "[skip] t0=$T0: RF 出力が未着 ($RF_BASE/fs_${h}m)。" ; exit 0
    fi
done

uv run python scripts/analyze/study/t0_archetype.py \
    --irl-base "$IRL_BASE" \
    --rf-base "$RF_BASE" \
    --prediction-time "$T0" \
    --train-end "$TRAIN_END" \
    --output-dir "$OUT_DIR" \
    --level "$LEVEL" \
    --k-grid "${K_GRID[@]}"

echo "[t0アーキタイプ・中間 t0=$T0] 完了 → $OUT_DIR"
