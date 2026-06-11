#!/bin/bash
# Archetype Aux 比較ジョブ群を dependency chain で投入するスクリプト
#
# lab_gpu の MaxTime=24h 制約に対応するため、4 バリアントを別ジョブにして
# 順次実行 (afterok dependency) する。最後に比較レポートジョブを追加。
#
# 投入後はキューに登録され、SSH 切断しても順次走り続ける。
#
# 使い方:
#   bash scripts/variant/submit_archetype_aux_chain.sh
#
# 進捗確認:
#   squeue -u $USER
#   tail -f logs/slurm_<JOBID>_<NAME>.out

set -e

mkdir -p logs

echo "============================================================"
echo "  Archetype Aux Compare Chain Submission"
echo "============================================================"

# ── 1. baseline (cache 抽出 + 学習 + 評価) ──
JOB_BASE=$(sbatch --parsable \
    --job-name=base \
    scripts/variant/sbatch_mce_irl_variant.sh 0 legacy_baseline)
echo "[1/5] baseline      submitted: JOB $JOB_BASE"

# ── 2-4. archetype_aux × 3 λ (baseline 後に逐次) ──
PREV="$JOB_BASE"
JOB_AUX_IDS=()
for L in 0.1 0.3 1.0; do
    JOB=$(sbatch --parsable \
        --job-name="aux_l${L}" \
        --dependency=afterok:${PREV} \
        scripts/variant/sbatch_mce_irl_variant.sh 7 "legacy_aux_l${L}" "$L")
    echo "[$((${#JOB_AUX_IDS[@]} + 2))/5] aux λ=$L      submitted: JOB $JOB (depends on $PREV)"
    JOB_AUX_IDS+=("$JOB")
    PREV="$JOB"
done

# ── 5. 比較レポート (全部終わってから, GPU 不要) ──
DEP_LIST="$JOB_BASE"
for j in "${JOB_AUX_IDS[@]}"; do
    DEP_LIST="${DEP_LIST}:${j}"
done
JOB_COMPARE=$(sbatch --parsable \
    --job-name=compare \
    --dependency=afterok:${DEP_LIST} \
    scripts/variant/sbatch_compare_report.sh)
echo "[5/5] compare       submitted: JOB $JOB_COMPARE (depends on $DEP_LIST)"

echo ""
echo "============================================================"
echo "  All jobs queued. SSH 切断しても順次走ります."
echo ""
echo "  確認:   squeue -u \$USER"
echo "  ログ:   tail -f logs/slurm_${JOB_BASE}_base.out"
echo "  停止:   scancel $JOB_BASE $JOB_COMPARE ${JOB_AUX_IDS[*]}"
echo "============================================================"
