#!/bin/bash
# Archetype 補助タスク (model_type=7) vs baseline (model_type=0) 比較実行ラッパー
#
# state_dim=32 (profile なし) で:
#   1. baseline (VTYPE=0) を 4 window × 4 eval 学習・評価
#   2. archetype_aux (VTYPE=7) を λ ∈ {0.1, 0.3, 1.0} で同様に実行
#   3. 比較レポート (compare_archetype_aux_vs_baseline.py) を実行
#
# キャッシュは archetype_aux_compare 専用 (CACHE_TAG=archetype_aux_compare) で
# 既存スコープと分離する。
#
# 使い方:
#   nohup bash scripts/variant/run_archetype_aux_compare.sh > logs/archetype_aux_compare.log 2>&1 &
#
# 環境変数で上書き可:
#   GPU_ID            : 0
#   OUTBASE           : outputs/archetype_aux_compare
#   AUX_ARCHETYPE_CSV : outputs/analysis/archetype_irl_rf_coldonly/k4_dev_alpha_with_metrics.csv
#   LAMBDAS           : "0.1 0.3 1.0" (スペース区切り)

set -e

GPU_ID="${GPU_ID:-0}"
OUTBASE="${OUTBASE:-outputs/archetype_aux_compare}"
AUX_ARCHETYPE_CSV="${AUX_ARCHETYPE_CSV:-outputs/analysis/archetype_irl_rf_coldonly/k4_dev_alpha_with_metrics.csv}"
LAMBDAS="${LAMBDAS:-0.1 0.3 1.0}"
CACHE_TAG="archetype_aux_compare"

mkdir -p "$OUTBASE" logs

if [ ! -f "$AUX_ARCHETYPE_CSV" ]; then
    echo "ERROR: archetype CSV が見つからない: $AUX_ARCHETYPE_CSV" >&2
    exit 1
fi

echo "============================================================"
echo "  Archetype Aux Comparison Pipeline"
echo "  OUTBASE          : $OUTBASE"
echo "  CACHE_TAG        : $CACHE_TAG"
echo "  GPU_ID           : $GPU_ID"
echo "  AUX_ARCHETYPE_CSV: $AUX_ARCHETYPE_CSV"
echo "  LAMBDAS          : $LAMBDAS"
echo "  開始時刻: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ── 1. baseline (VTYPE=0, profile なし) ──
echo ""
echo "[1/$((1 + $(echo $LAMBDAS | wc -w)))] baseline (VTYPE=0) 学習開始: $(date '+%H:%M:%S')"
CACHE_TAG="$CACHE_TAG" \
    bash scripts/variant/run_mce_irl_variant_single.sh \
        0 legacy_baseline "$OUTBASE" "$GPU_ID"
echo "[baseline] 完了: $(date '+%H:%M:%S')"

# ── 2. archetype_aux × 各 λ ──
i=1
total=$(echo $LAMBDAS | wc -w)
for L in $LAMBDAS; do
    i=$((i + 1))
    echo ""
    echo "[$((i))/$((1 + total))] archetype_aux λ=$L 学習開始: $(date '+%H:%M:%S')"
    CACHE_TAG="$CACHE_TAG" \
    AUX_ARCHETYPE_CSV="$AUX_ARCHETYPE_CSV" \
    AUX_LAMBDA="$L" \
        bash scripts/variant/run_mce_irl_variant_single.sh \
            7 "legacy_aux_l${L}" "$OUTBASE" "$GPU_ID"
    echo "[aux λ=$L] 完了: $(date '+%H:%M:%S')"
done

# ── 3. 比較レポート生成 ──
echo ""
echo "[$((i + 1))/$((1 + total + 1))] 比較レポート生成: $(date '+%H:%M:%S')"
AUX_ROOTS=()
for L in $LAMBDAS; do
    AUX_ROOTS+=("$OUTBASE/legacy_aux_l${L}")
done

REPORT_OUT="outputs/analysis/archetype_aux_compare"
mkdir -p "$REPORT_OUT"

uv run python scripts/analyze/study/compare_archetype_aux_vs_baseline.py \
    --base "$OUTBASE/legacy_baseline" \
    --aux-roots "${AUX_ROOTS[@]}" \
    --archetype-csv "$AUX_ARCHETYPE_CSV" \
    --out "$REPORT_OUT"

echo ""
echo "============================================================"
echo "  全パイプライン完了: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  モデル: $OUTBASE/legacy_baseline, $OUTBASE/legacy_aux_l*"
echo "  比較レポート: $REPORT_OUT"
echo "============================================================"
