#!/bin/bash
# RL フルパイプライン (軌跡抽出 → データセット構築 → λ×γ sweep → 評価 → 考察)
#
# キャッシュ作成から評価・考察まで1本で連続実行する。
# データセット構築 (重い特徴量抽出) は粒度ごとに 1 回だけ行い、
# λ (reward shaping) は train_offline_rl.py の --reward-shaping で
# 保存済み列 (step_label + r_theta(_next)) から再計算する。
# → λ を 5 点振ってもビルドは 1 回で済む。
#
# 使い方:
#   bash scripts/rl/run_rl_sweep.sh [t0_window] [variant] [gpu_id]
#
# 全体の流れ:
#   Step0 軌跡抽出   : mce_traj_<t0>.pkl が無ければ抽出 (train-end を延長し
#                       test split が空にならないようにする)
#   Step1 build      : 粒度 (event/monthly) ごとにデータセットを1回構築
#   Step2 train      : λ×γ の組ごとに BC→IQL を学習
#   Step3 eval       : train-fit → test 固定適用の評価 (+ dev holdout があれば追加評価)
#   Step4 summarize  : 全 run の summary_metrics.json を横断集約
#   Step5 analyze    : 代表点 (ANALYSIS_LAMBDA/GAMMA) について手法間 bootstrap 検定 +
#                       V(s) 駆動要因分析 (RQ2 考察の下書き) を自動生成
#
# 前提:
#   - 学習済み IRL モデル (OUTBASE/VARIANT/train_<t0>m/)
#   - 生データ (REVIEWS / RAW_JSON_LIST_FILE)。軌跡キャッシュが既にあれば不要
#
# 環境変数:
#   [Step0 軌跡抽出]
#   REVIEWS             : レビューCSV (default: data/combined_raw_main32.csv)
#   RAW_JSON_LIST_FILE  : raw JSON リストファイル (default: data/combined_raw_main32.raw_json_list.txt)
#   SKIP_TRAJ_EXTRACT   : 1 でキャッシュ必須化 (無ければエラー、抽出しない)
#   REBUILD_TRAJ        : 1 で既存キャッシュを破棄して再抽出
#   EXTRACT_TRAIN_START : 抽出の学習開始日 (default: 2019-01-01)
#   EXTRACT_TRAIN_END   : 抽出の学習終了日 (default: 2023-01-01)。
#                         IRL モデルの学習 train_end (通常 2022-01-01) より
#                         **後ろに延長**しておくこと。ここを IRL と同じ日付に
#                         すると train_model.py 側のリーク防止クリップにより
#                         test split (TEST_SPLIT_DATE 以降) が空になる。
#   [Step1-4 sweep]
#   GRANULARITIES       : "event monthly" (default) — 回す粒度
#   LAMBDAS             : λ のリスト (default: "0 0.1 0.3 0.5 1.0")
#   GAMMAS              : γ のリスト (default: "0.99"。感度を見るなら "0.9 0.95 0.99")
#   HOLDOUT_DEVS_RATIO  : dev holdout 割合 (default: 0.1)
#   TEST_SPLIT_DATE     : test 分割基準日 (default: IRL の train_end = 2022-01-01)
#   BC_EPOCHS / IQL_EPOCHS / SEED / OUTBASE / REBUILD_DATASET (=1 で再ビルド)
#   [Step5 考察]
#   ANALYSIS_LAMBDA     : 詳細分析の対象 λ (default: 0.3)
#   ANALYSIS_GAMMA      : 詳細分析の対象 γ (default: GAMMAS の先頭)
#   BOOT_N              : bootstrap 反復回数 (default: 1000)
#   VALUE_N_REPEATS     : permutation 感度の反復回数 (default: 5)
#   SKIP_ANALYSIS       : 1 で Step5 をスキップ
#
# 出力:
#   outputs/rl/sweep/<variant>_<t0>m_<gran>/dataset/          (共有データセット)
#   outputs/rl/sweep/<variant>_<t0>m_<gran>/lam<λ>_gam<γ>/    (モデル + eval)
#   outputs/rl/sweep/summary.csv / summary.md                 (横断集約)
#   outputs/rl/analysis/<variant>_<t0>m_<gran>/                (統計検定 + V(s)分析)

set -e
export PYTHONHASHSEED=0

T0_WINDOW="${1:-0-3}"
VARIANT="${2:-lstm_baseline}"
GPU_ARG="${3:-cpu}"

if [ "$GPU_ARG" = "cpu" ]; then
    DEVICE="cpu"; export CUDA_VISIBLE_DEVICES=""
else
    DEVICE="cuda"; export CUDA_VISIBLE_DEVICES="$GPU_ARG"
fi

OUTBASE="${OUTBASE:-outputs/main32_mce_volat_ep150}"
GRANULARITIES="${GRANULARITIES:-event monthly}"
LAMBDAS="${LAMBDAS:-0 0.1 0.3 0.5 1.0}"
GAMMAS="${GAMMAS:-0.99}"
HOLDOUT_DEVS_RATIO="${HOLDOUT_DEVS_RATIO:-0.1}"
TEST_SPLIT_DATE="${TEST_SPLIT_DATE:-}"
BC_EPOCHS="${BC_EPOCHS:-50}"
IQL_EPOCHS="${IQL_EPOCHS:-100}"
SEED="${SEED:-42}"
REBUILD_DATASET="${REBUILD_DATASET:-0}"

REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"
SKIP_TRAJ_EXTRACT="${SKIP_TRAJ_EXTRACT:-0}"
REBUILD_TRAJ="${REBUILD_TRAJ:-0}"
EXTRACT_TRAIN_START="${EXTRACT_TRAIN_START:-2019-01-01}"
EXTRACT_TRAIN_END="${EXTRACT_TRAIN_END:-2023-01-01}"

ANALYSIS_LAMBDA="${ANALYSIS_LAMBDA:-0.3}"
BOOT_N="${BOOT_N:-1000}"
VALUE_N_REPEATS="${VALUE_N_REPEATS:-5}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

case "$T0_WINDOW" in
    0-3)  TRAIN_END="2022-01-01"; FW_START=0; FW_END=3 ;;
    3-6)  TRAIN_END="2022-01-01"; FW_START=3; FW_END=6 ;;
    6-9)  TRAIN_END="2022-01-01"; FW_START=6; FW_END=9 ;;
    9-12) TRAIN_END="2022-01-01"; FW_START=9; FW_END=12 ;;
    *) echo "ERROR: t0_window は 0-3 / 3-6 / 6-9 / 9-12" >&2; exit 1 ;;
esac
if [ -z "$TEST_SPLIT_DATE" ]; then TEST_SPLIT_DATE="$TRAIN_END"; fi
# GAMMAS の先頭を ANALYSIS_GAMMA の既定値にする
ANALYSIS_GAMMA="${ANALYSIS_GAMMA:-$(echo "$GAMMAS" | awk '{print $1}')}"

IRL_MODEL_DIR="${OUTBASE}/${VARIANT}/train_${T0_WINDOW}m"
IRL_MODEL="${IRL_MODEL_DIR}/mce_irl_model.pt"
IRL_CONFIG="${IRL_MODEL_DIR}/model_metadata.json"
TRAJ_CACHE="outputs/mce_irl_trajectory_cache/mce_traj_${T0_WINDOW}.pkl"
SWEEP_ROOT="outputs/rl/sweep"
ANALYSIS_ROOT="outputs/rl/analysis"

if [ ! -f "$IRL_MODEL" ]; then
    echo "ERROR: IRL モデルが見つかりません: $IRL_MODEL" >&2; exit 1
fi

echo "========================================================"
echo " RL Full Pipeline: variant=${VARIANT} t0=${T0_WINDOW}m device=${DEVICE}"
echo "   granularities : ${GRANULARITIES}"
echo "   λ grid        : ${LAMBDAS}  (詳細考察は λ=${ANALYSIS_LAMBDA})"
echo "   γ grid        : ${GAMMAS}  (詳細考察は γ=${ANALYSIS_GAMMA})"
echo "   dev holdout   : ${HOLDOUT_DEVS_RATIO}"
echo "   test split    : ${TEST_SPLIT_DATE}"
echo "   extract range : ${EXTRACT_TRAIN_START} 〜 ${EXTRACT_TRAIN_END}"
echo "========================================================"

# ──────────────────────────────────────────────────────────────────────
# Step0: 軌跡抽出 (キャッシュが無ければ実行)
# ──────────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "$TRAJ_CACHE")"
if [ "$REBUILD_TRAJ" = "1" ]; then
    rm -f "$TRAJ_CACHE"
fi

if [ -f "$TRAJ_CACHE" ]; then
    echo "[Step0] 軌跡キャッシュを再利用: $TRAJ_CACHE"
elif [ "$SKIP_TRAJ_EXTRACT" = "1" ]; then
    echo "ERROR: SKIP_TRAJ_EXTRACT=1 だがキャッシュが見つかりません: $TRAJ_CACHE" >&2
    exit 1
else
    if [ ! -f "$RAW_JSON_LIST_FILE" ]; then
        echo "ERROR: RAW_JSON_LIST_FILE が見つかりません: $RAW_JSON_LIST_FILE" >&2; exit 1
    fi
    if [ ! -f "$REVIEWS" ]; then
        echo "ERROR: レビューデータが見つかりません: $REVIEWS" >&2; exit 1
    fi
    # 姉妹スクリプト (run_rl_pipeline.sh 等) と同じ配列パターン
    read -r -a RAW_JSON <<< "$(cat "$RAW_JSON_LIST_FILE")"

    echo "[Step0] 軌跡を抽出中 (train_end=${EXTRACT_TRAIN_END}, future_window=${FW_START}-${FW_END}m)..."
    uv run python scripts/train/extract_mce_trajectories.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$EXTRACT_TRAIN_START" \
        --train-end   "$EXTRACT_TRAIN_END" \
        --future-window-start "$FW_START" \
        --future-window-end   "$FW_END" \
        --n-jobs -1 \
        --output "$TRAJ_CACHE"
    echo "[Step0] 完了: $TRAJ_CACHE"
fi

STATE_DIM=$(python3 -c "import json; d=json.load(open('${IRL_CONFIG}')); print(d['state_dim'])" 2>/dev/null || echo "32")
ACTION_DIM=$(python3 -c "import json; d=json.load(open('${IRL_CONFIG}')); print(d['action_dim'])" 2>/dev/null || echo "5")
OBS_DIM=$STATE_DIM

# ──────────────────────────────────────────────────────────────────────
# Step1-3: 粒度ごとにデータセット構築 → λ×γ で train + eval
# ──────────────────────────────────────────────────────────────────────
for GRAN in $GRANULARITIES; do
    BASE_DIR="${SWEEP_ROOT}/${VARIANT}_${T0_WINDOW}m_${GRAN}"
    DATASET_DIR="${BASE_DIR}/dataset"
    mkdir -p "$DATASET_DIR"

    if [ "$GRAN" = "event" ]; then
        BUILD_SCRIPT="scripts/rl/build_rl_dataset_event.py"
        DATASET_PATH="${DATASET_DIR}/event_transitions.parquet"
        DATASET_TEST_PATH="${DATASET_DIR}/event_transitions_test.parquet"
        DATASET_DEVHOLD_PATH="${DATASET_DIR}/event_transitions_devholdout.parquet"
    else
        BUILD_SCRIPT="scripts/rl/build_rl_dataset.py"
        DATASET_PATH="${DATASET_DIR}/transitions.parquet"
        DATASET_TEST_PATH="${DATASET_DIR}/transitions_test.parquet"
        DATASET_DEVHOLD_PATH="${DATASET_DIR}/transitions_devholdout.parquet"
    fi

    # ── Step1: データセット構築 (粒度ごとに 1 回。λ は train 時に再計算するので任意値) ──
    if [ "$REBUILD_DATASET" != "1" ] && [ -f "$DATASET_PATH" ]; then
        echo "[Step1:${GRAN}] データセットを再利用: $DATASET_PATH"
    else
        echo "[Step1:${GRAN}] データセットを構築中 (1回のみ)..."
        uv run python "$BUILD_SCRIPT" \
            --trajectories       "$TRAJ_CACHE" \
            --irl-model          "$IRL_MODEL" \
            --irl-config         "$IRL_CONFIG" \
            --output             "$DATASET_PATH" \
            --state-dim          "$STATE_DIM" \
            --action-dim         "$ACTION_DIM" \
            --reward-shaping     0.3 \
            --test-split-date    "$TEST_SPLIT_DATE" \
            --dev-label-window   "${FW_START},${FW_END}" \
            --holdout-devs-ratio "$HOLDOUT_DEVS_RATIO" \
            --device             "$DEVICE"
    fi

    # ── Step2-3: λ × γ で train + eval ──
    for LAM in $LAMBDAS; do
        for GAM in $GAMMAS; do
            RUN_DIR="${BASE_DIR}/lam${LAM}_gam${GAM}"
            echo "[Step2-3:${GRAN}] λ=${LAM} γ=${GAM} → ${RUN_DIR}"
            mkdir -p "$RUN_DIR"

            uv run python scripts/rl/train_offline_rl.py \
                --dataset        "$DATASET_PATH" \
                --output-dir     "$RUN_DIR" \
                --obs-dim        "$OBS_DIM" \
                --n-actions      2 \
                --bc-epochs      "$BC_EPOCHS" \
                --iql-epochs     "$IQL_EPOCHS" \
                --gamma          "$GAM" \
                --reward-shaping "$LAM" \
                --seed           "$SEED" \
                --device         "$DEVICE"

            EVAL_ARGS=(
                --train-dataset "$DATASET_PATH"
                --iql-model     "${RUN_DIR}/iql_agent.pt"
                --bc-model      "${RUN_DIR}/bc_policy.pt"
                --obs-dim       "$OBS_DIM"
                --output-dir    "${RUN_DIR}/eval"
                --device        "$DEVICE"
            )
            if [ -f "$DATASET_TEST_PATH" ]; then
                EVAL_ARGS+=(--test-dataset "$DATASET_TEST_PATH")
            fi
            uv run python scripts/rl/eval_assignment.py "${EVAL_ARGS[@]}"

            if [ -f "$DATASET_DEVHOLD_PATH" ]; then
                uv run python scripts/rl/eval_assignment.py \
                    --train-dataset "$DATASET_PATH" \
                    --test-dataset  "$DATASET_DEVHOLD_PATH" \
                    --iql-model     "${RUN_DIR}/iql_agent.pt" \
                    --bc-model      "${RUN_DIR}/bc_policy.pt" \
                    --obs-dim       "$OBS_DIM" \
                    --output-dir    "${RUN_DIR}/eval_devholdout" \
                    --device        "$DEVICE"
            fi
        done
    done
done

# ──────────────────────────────────────────────────────────────────────
# Step4: 横断集約
# ──────────────────────────────────────────────────────────────────────
echo "[Step4] summary_metrics.json を横断集約中..."
uv run python scripts/rl/summarize_rl_results.py "$SWEEP_ROOT" \
    --output-csv "${SWEEP_ROOT}/summary.csv" \
    --output-md  "${SWEEP_ROOT}/summary.md"

# ──────────────────────────────────────────────────────────────────────
# Step5: 考察 (代表点について統計検定 + V(s) 駆動要因分析)
# ──────────────────────────────────────────────────────────────────────
if [ "$SKIP_ANALYSIS" = "1" ]; then
    echo "[Step5] SKIP_ANALYSIS=1 のためスキップ"
else
    echo "[Step5] 考察分析中 (λ=${ANALYSIS_LAMBDA}, γ=${ANALYSIS_GAMMA})..."
    for GRAN in $GRANULARITIES; do
        BASE_DIR="${SWEEP_ROOT}/${VARIANT}_${T0_WINDOW}m_${GRAN}"
        RUN_DIR="${BASE_DIR}/lam${ANALYSIS_LAMBDA}_gam${ANALYSIS_GAMMA}"
        DATASET_DIR="${BASE_DIR}/dataset"
        OUT_DIR="${ANALYSIS_ROOT}/${VARIANT}_${T0_WINDOW}m_${GRAN}"

        if [ ! -f "${RUN_DIR}/eval/pair_predictions.csv" ]; then
            echo "  [${GRAN}] スキップ: ${RUN_DIR}/eval/pair_predictions.csv が無い (λ/γ の組が sweep 対象外)"
            continue
        fi
        mkdir -p "$OUT_DIR"

        for TARGET in step_label step_label_dev monthly_accept; do
            uv run python scripts/rl/compare_methods_stats.py \
                "${RUN_DIR}/eval/pair_predictions.csv" \
                --target "$TARGET" --baseline score_irl_pi \
                --n-boot "$BOOT_N" \
                --output-dir "$OUT_DIR" \
                || echo "  [${GRAN}] ${TARGET}: 手法間検定をスキップ (対象列が退化している可能性)"
        done

        if [ "$GRAN" = "event" ]; then
            TEST_DS="${DATASET_DIR}/event_transitions_test.parquet"
            [ -f "$TEST_DS" ] || TEST_DS="${DATASET_DIR}/event_transitions.parquet"
        else
            TEST_DS="${DATASET_DIR}/transitions_test.parquet"
            [ -f "$TEST_DS" ] || TEST_DS="${DATASET_DIR}/transitions.parquet"
        fi

        uv run python scripts/rl/analyze_value_drivers.py \
            --dataset    "$TEST_DS" \
            --iql-model  "${RUN_DIR}/iql_agent.pt" \
            --obs-dim    "$OBS_DIM" \
            --output-dir "$OUT_DIR" \
            --n-repeats  "$VALUE_N_REPEATS" \
            --device     "$DEVICE" \
            || echo "  [${GRAN}] V(s) 駆動要因分析をスキップ (エラー発生)"
    done
fi

echo "========================================================"
echo " フルパイプライン完了"
echo "   軌跡キャッシュ : ${TRAJ_CACHE}"
echo "   各 run         : ${SWEEP_ROOT}/<variant>_<t0>m_<gran>/lam*_gam*/eval/"
echo "   集約結果       : ${SWEEP_ROOT}/summary.csv / summary.md"
echo "   考察分析       : ${ANALYSIS_ROOT}/<variant>_<t0>m_<gran>/"
echo "     ├─ methods_stats_<target>.csv/md  (手法間 bootstrap 検定)"
echo "     ├─ value_sensitivity.csv          (特徴グループ感度)"
echo "     ├─ dv_dx_spearman.csv             (ΔV↔Δ特徴 遷移相関)"
echo "     ├─ v_drop_profile.csv             (V急落遷移プロファイル)"
echo "     └─ value_drivers_summary.md       (考察の下書き)"
echo "========================================================"
