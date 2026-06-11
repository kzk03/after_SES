#!/bin/bash
#SBATCH --partition=lab_gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#
# baseline + aux λ 全部終わってから比較レポートを生成する sbatch
# GPU 不要 (CPU のみで完結する)
#
# 使い方:
#   sbatch --job-name=compare --dependency=afterok:<JOBID1>:<JOBID2>:<JOBID3>:<JOBID4> \
#       scripts/variant/sbatch_compare_report.sh

set -e

OUTBASE="${OUTBASE:-outputs/archetype_aux_compare}"
AUX_ARCHETYPE_CSV="${AUX_ARCHETYPE_CSV:-outputs/analysis/archetype_irl_rf_coldonly/k4_dev_alpha_with_metrics.csv}"
LAMBDAS="${LAMBDAS:-0.1 0.3 1.0}"
REPORT_OUT="${REPORT_OUT:-outputs/analysis/archetype_aux_compare}"

echo "============================================================"
echo "  SLURM JOB: $SLURM_JOB_ID ($SLURM_JOB_NAME)"
echo "  COMPARE REPORT"
echo "  OUTBASE  : $OUTBASE"
echo "  REPORT   : $REPORT_OUT"
echo "  START    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

mkdir -p "$REPORT_OUT"

AUX_ROOTS=()
for L in $LAMBDAS; do
    AUX_ROOTS+=("$OUTBASE/legacy_aux_l${L}")
done

uv run python scripts/analyze/study/compare_archetype_aux_vs_baseline.py \
    --base "$OUTBASE/legacy_baseline" \
    --aux-roots "${AUX_ROOTS[@]}" \
    --archetype-csv "$AUX_ARCHETYPE_CSV" \
    --out "$REPORT_OUT"

echo ""
echo "  END      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  COMPARE DONE: $REPORT_OUT"
