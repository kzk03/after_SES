#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# [SES 追加分析] t0掃引「tail × 学習年数」グリッドの IRL 一括ランチャ (GPUサーバ用)
# ──────────────────────────────────────────────────────────────────────────
# 系列はカンマ区切りで複数指定可。指定順に「1系列5 t0 ずつ」直列で nohup 実行する
# （GPU 1枚運用を想定。並走させたい場合は本スクリプトを系列ごとに別 gpu_id で起動）。
# ログは系列ごとに logs/t0_grid_<series>_<起動日時>.log。
#
# 使い方:
#   bash scripts/variant/start_t0_grid.sh <series[,series...]> [gpu_id]
# 例 (GPU 1枚で3系列を順に):
#   bash scripts/variant/start_t0_grid.sh inter_fs36,align_fs36,inter_fs24 0
#
# 進捗確認: tail -f logs/t0_grid_<series>_*.log
# 完了後ローカルで:
#   rsync -avz --exclude='*.pkl' --progress \
#     socsel:/mnt/data1/kazuki-h/additional/outputs/ ~/projects/after_ses/outputs/
# ──────────────────────────────────────────────────────────────────────────
set -u

SERIES_LIST="${1:?Usage: $0 <inter_fs36|align_fs36|inter_fs24>[,series...] [gpu_id]}"
GPU_ID="${2:-0}"
T0S=(2022-01-01 2022-10-01 2023-01-01 2023-04-01 2024-01-01)
STAMP="$(date +%Y%m%d_%H%M%S)"

# 系列名 → (env, runner, log)。起動前に全系列を検証してから流す。
declare -a SERIES_ARR ENV_ARR RUNNER_ARR LOG_ARR
OLD_IFS="$IFS"
IFS=','
SERIES_ARR=($SERIES_LIST)
IFS="$OLD_IFS"
mkdir -p logs
for s in "${SERIES_ARR[@]}"; do
    case "$s" in
        inter_fs36) ENV_ARR+=("FEATURE_STEPS=36"); RUNNER_ARR+=("scripts/variant/run_t0_sweep_inter.sh") ;;
        inter_fs24) ENV_ARR+=("FEATURE_STEPS=24"); RUNNER_ARR+=("scripts/variant/run_t0_sweep_inter.sh") ;;
        align_fs36) ENV_ARR+=("ALIGN=1 FEATURE_STEPS=36"); RUNNER_ARR+=("scripts/variant/run_t0_sweep.sh") ;;
        *) echo "ERROR: series は inter_fs36|align_fs36|inter_fs24 (指定: $s)" >&2; exit 1 ;;
    esac
    LOG_ARR+=("logs/t0_grid_${s}_${STAMP}.log")
done

# 直列実行の本体スクリプトを一時生成して nohup で流す
CHAIN="logs/.t0_grid_chain_${STAMP}.sh"
{
    echo "#!/usr/bin/env bash"
    for i in "${!SERIES_ARR[@]}"; do
        s="${SERIES_ARR[$i]}"; e="${ENV_ARR[$i]}"; r="${RUNNER_ARR[$i]}"; log="${LOG_ARR[$i]}"
        cat <<EOS
{
  echo "============================================================"
  echo "  start_t0_grid.sh: \$(date '+%F %T')  series=$s  gpu=$GPU_ID"
  echo "  runner=$r  env=$e"
  echo "  t0: ${T0S[*]}"
  echo "============================================================"
} >> '$log'
for t0 in ${T0S[*]}; do
    echo "[\$(date '+%F %T')] === t0=\$t0 ($s) START ===" >> '$log'
    if env $e bash '$r' "\$t0" irl '$GPU_ID' >> '$log' 2>&1; then
        echo "[\$(date '+%F %T')] === t0=\$t0 DONE ===" >> '$log'
    else
        echo "[\$(date '+%F %T')] === t0=\$t0 FAILED (次の t0 へ続行) ===" >> '$log'
    fi
done
echo "[\$(date '+%F %T')] === SERIES DONE ($s) ===" >> '$log'
EOS
    done
    last_log="${LOG_ARR[${#LOG_ARR[@]}-1]}"
    echo "echo \"[\$(date '+%F %T')] === ALL SERIES DONE (${SERIES_LIST}) ===\" >> '$last_log'"
} > "$CHAIN"
chmod +x "$CHAIN"

nohup bash "$CHAIN" >/dev/null 2>&1 &
echo "launched: series=$SERIES_LIST (この順に直列) gpu=$GPU_ID pid=$!"
for log in "${LOG_ARR[@]}"; do echo "log: $log"; done
