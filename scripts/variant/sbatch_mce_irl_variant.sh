#!/bin/bash
#SBATCH --partition=lab_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=logs/slurm_%j_%x.out
#SBATCH --error=logs/slurm_%j_%x.err
#
# 1 つのバリアントだけ学習・評価する sbatch (lab_gpu MaxTime=24h 対応版)
#
# 使い方:
#   sbatch --job-name=base scripts/variant/sbatch_mce_irl_variant.sh 0 legacy_baseline
#   sbatch --job-name=aux_l0.3 --dependency=afterok:<JOBID> \
#       scripts/variant/sbatch_mce_irl_variant.sh 7 legacy_aux_l0.3 0.3
#
# 引数:
#   $1: VTYPE (0=baseline, 7=archetype aux)
#   $2: VNAME (出力ディレクトリ名, 例: legacy_baseline, legacy_aux_l0.3)
#   $3: AUX_LAMBDA (VTYPE=7 のときのみ使用, デフォルト 0.0)

set -e

VTYPE="${1:?Usage: sbatch ... script.sh <VTYPE> <VNAME> [AUX_LAMBDA]}"
VNAME="${2:?Usage: sbatch ... script.sh <VTYPE> <VNAME> [AUX_LAMBDA]}"
AUX_LAMBDA="${3:-0.0}"

OUTBASE="${OUTBASE:-outputs/archetype_aux_compare}"
AUX_ARCHETYPE_CSV="${AUX_ARCHETYPE_CSV:-outputs/analysis/archetype_irl_rf_coldonly/k4_dev_alpha_with_metrics.csv}"
CACHE_TAG="archetype_aux_compare"

# joblib loky / OpenMP / MKL の並列度を SLURM 割当 CPU 数に揃える
# (これがないと joblib が物理コア数まで広がって RAM OOM になる)
export LOKY_MAX_CPU_COUNT="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "============================================================"
echo "  SLURM JOB: $SLURM_JOB_ID ($SLURM_JOB_NAME)"
echo "  NODE     : $SLURM_NODELIST"
echo "  GPU      : $CUDA_VISIBLE_DEVICES"
echo "  CPUS     : $SLURM_CPUS_PER_TASK (LOKY_MAX_CPU_COUNT=$LOKY_MAX_CPU_COUNT)"
echo "  MEM      : ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "  WORKDIR  : $(pwd)"
echo "  VTYPE    : $VTYPE"
echo "  VNAME    : $VNAME"
echo "  AUX_LAMBDA: $AUX_LAMBDA"
echo "  CACHE_TAG: $CACHE_TAG"
echo "  START    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

mkdir -p logs

if [ "$VTYPE" = "7" ]; then
    CACHE_TAG="$CACHE_TAG" \
    AUX_ARCHETYPE_CSV="$AUX_ARCHETYPE_CSV" \
    AUX_LAMBDA="$AUX_LAMBDA" \
        bash scripts/variant/run_mce_irl_variant_single.sh \
            "$VTYPE" "$VNAME" "$OUTBASE" 0
else
    CACHE_TAG="$CACHE_TAG" \
        bash scripts/variant/run_mce_irl_variant_single.sh \
            "$VTYPE" "$VNAME" "$OUTBASE" 0
fi

echo ""
echo "  END      : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  JOB DONE: $VNAME"
