#!/bin/bash
# RL パイプライン実行スクリプト
#
# 以下の4ステップを順番に実行する:
#   Step 1: 軌跡抽出 (extract_mce_trajectories.py) → pkl キャッシュ
#   Step 2: RL データセット構築 (build_rl_dataset.py) → parquet
#   Step 3: オフライン RL 訓練 (train_offline_rl.py) → BC + IQL チェックポイント
#   Step 4: 評価 (eval_assignment.py) → eval_results.json
#
# 使い方:
#   bash scripts/rl/run_rl_pipeline.sh [t0_window] [variant] [gpu_id]
#
# 引数:
#   t0_window : 学習窓 (0-3 / 3-6 / 6-9 / 9-12, デフォルト: 0-3)
#   variant   : IRL モデルバリアント名 (lstm_baseline / lstm_hybrid_snapshot, デフォルト: lstm_baseline)
#   gpu_id    : CUDA デバイス番号 (デフォルト: 0, CPU 使用時は "cpu")
#
# 例:
#   bash scripts/rl/run_rl_pipeline.sh               # 0-3m, lstm_baseline, GPU 0
#   bash scripts/rl/run_rl_pipeline.sh 3-6 lstm_baseline cpu
#   bash scripts/rl/run_rl_pipeline.sh 0-3 lstm_hybrid_snapshot 1
#
# 環境変数による上書き:
#   REVIEWS              : CSV パス (default: data/combined_raw_main32.csv)
#   RAW_JSON_LIST_FILE   : raw JSON リストファイル (default: data/combined_raw_main32.raw_json_list.txt)
#   OUTBASE              : 出力ベースディレクトリ (default: outputs/main32_mce_volat_ep150)
#   SKIP_TRAJ_EXTRACT    : 1 にすると Step 1 をスキップ（pkl キャッシュ再利用）
#   SKIP_DATASET_BUILD   : 1 にすると Step 2 をスキップ（parquet 再利用）
#   BC_EPOCHS            : BC エポック数 (default: 50)
#   IQL_EPOCHS           : IQL エポック数 (default: 100)
#   REWARD_SHAPING       : R_θ シェイピング係数 λ (default: 0.3, r = step_label + λ·R_θ)
#   TEST_SPLIT_DATE      : test 分割の基準日 (default: TRAIN_END, 例 2022-01-01)
#                          feature_end >= この日付 の transition を transitions_test.parquet に分離

set -e
export PYTHONHASHSEED=0

# ── 引数 ───────────────────────────────────────────────────────────────
T0_WINDOW="${1:-0-3}"
VARIANT="${2:-lstm_baseline}"
GPU_ARG="${3:-0}"

if [ "$GPU_ARG" = "cpu" ]; then
    DEVICE="cpu"
    export CUDA_VISIBLE_DEVICES=""
else
    DEVICE="cuda"
    export CUDA_VISIBLE_DEVICES="$GPU_ARG"
fi

# ── パス設定 ───────────────────────────────────────────────────────────
REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"
OUTBASE="${OUTBASE:-outputs/main32_mce_volat_ep150}"
SKIP_TRAJ_EXTRACT="${SKIP_TRAJ_EXTRACT:-0}"
SKIP_DATASET_BUILD="${SKIP_DATASET_BUILD:-0}"
BC_EPOCHS="${BC_EPOCHS:-50}"
IQL_EPOCHS="${IQL_EPOCHS:-100}"
REWARD_SHAPING="${REWARD_SHAPING:-0.3}"
TEST_SPLIT_DATE="${TEST_SPLIT_DATE:-}"  # 既定は後段で TRAIN_END に設定

# T0 ウィンドウから train_start / train_end を導出
case "$T0_WINDOW" in
    0-3)  TRAIN_START="2019-01-01"; TRAIN_END="2022-01-01"; FW_START=0; FW_END=3 ;;
    3-6)  TRAIN_START="2019-01-01"; TRAIN_END="2022-01-01"; FW_START=3; FW_END=6 ;;
    6-9)  TRAIN_START="2019-01-01"; TRAIN_END="2022-01-01"; FW_START=6; FW_END=9 ;;
    9-12) TRAIN_START="2019-01-01"; TRAIN_END="2022-01-01"; FW_START=9; FW_END=12 ;;
    *) echo "ERROR: t0_window は 0-3 / 3-6 / 6-9 / 9-12 を指定してください" >&2; exit 1 ;;
esac

# モデルとキャッシュのパス
IRL_MODEL_DIR="${OUTBASE}/${VARIANT}/train_${T0_WINDOW}m"
IRL_MODEL="${IRL_MODEL_DIR}/mce_irl_model.pt"
IRL_CONFIG="${IRL_MODEL_DIR}/model_metadata.json"

TRAJ_CACHE_DIR="outputs/mce_irl_trajectory_cache"
TRAJ_CACHE="${TRAJ_CACHE_DIR}/mce_traj_${T0_WINDOW}.pkl"

RL_OUTPUT_DIR="outputs/rl/${VARIANT}_${T0_WINDOW}m"
DATASET_PATH="${RL_OUTPUT_DIR}/transitions.parquet"
DATASET_TEST_PATH="${RL_OUTPUT_DIR}/transitions_test.parquet"
EVAL_OUTPUT_DIR="${RL_OUTPUT_DIR}/eval"

# TEST_SPLIT_DATE が未指定なら TRAIN_END を使う
if [ -z "$TEST_SPLIT_DATE" ]; then
    TEST_SPLIT_DATE="$TRAIN_END"
fi

# state_dim / action_dim は model_metadata.json から取得
STATE_DIM=$(python3 -c "import json; d=json.load(open('${IRL_CONFIG}')); print(d['state_dim'])" 2>/dev/null || echo "32")
ACTION_DIM=$(python3 -c "import json; d=json.load(open('${IRL_CONFIG}')); print(d['action_dim'])" 2>/dev/null || echo "5")
OBS_DIM=$((STATE_DIM + 1))

echo "========================================================"
echo " RL Pipeline: variant=${VARIANT} t0=${T0_WINDOW}m"
echo "   IRL model : ${IRL_MODEL}"
echo "   state_dim : ${STATE_DIM}  action_dim : ${ACTION_DIM}  obs_dim : ${OBS_DIM}"
echo "   output    : ${RL_OUTPUT_DIR}"
echo "   device    : ${DEVICE}"
echo "   reward λ  : ${REWARD_SHAPING}  (r = step_label + λ·R_θ)"
echo "   test split: ${TEST_SPLIT_DATE}  (feature_end >= → test parquet)"
echo "========================================================"

# ファイル存在チェック
if [ ! -f "$IRL_MODEL" ]; then
    echo "ERROR: IRL モデルが見つかりません: $IRL_MODEL" >&2; exit 1
fi
if [ ! -f "$REVIEWS" ]; then
    echo "ERROR: レビューデータが見つかりません: $REVIEWS" >&2; exit 1
fi

mkdir -p "$TRAJ_CACHE_DIR" "$RL_OUTPUT_DIR"

# ──────────────────────────────────────────────────────────────────────
# Step 1: 軌跡抽出
# ──────────────────────────────────────────────────────────────────────
if [ "$SKIP_TRAJ_EXTRACT" = "1" ] && [ -f "$TRAJ_CACHE" ]; then
    echo "[Step 1] 軌跡キャッシュを再利用: $TRAJ_CACHE"
else
    if [ ! -f "$RAW_JSON_LIST_FILE" ]; then
        echo "ERROR: RAW_JSON_LIST_FILE が見つかりません: $RAW_JSON_LIST_FILE" >&2; exit 1
    fi
    # 姉妹スクリプト (run_mce_irl_variant_single.sh 等) と同じ配列パターン。
    # 非配列の unquoted 展開だと空白/glob 文字を含むパスで引数が壊れる。
    read -r -a RAW_JSON <<< "$(cat "$RAW_JSON_LIST_FILE")"

    echo "[Step 1] 軌跡を抽出中..."
    uv run python scripts/train/extract_mce_trajectories.py \
        --reviews "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --train-start "$TRAIN_START" \
        --train-end   "$TRAIN_END" \
        --future-window-start "$FW_START" \
        --future-window-end   "$FW_END" \
        --n-jobs -1 \
        --output "$TRAJ_CACHE"
    echo "[Step 1] 完了: $TRAJ_CACHE"
fi

# ──────────────────────────────────────────────────────────────────────
# Step 2: RL データセット構築
# ──────────────────────────────────────────────────────────────────────
if [ "$SKIP_DATASET_BUILD" = "1" ] && [ -f "$DATASET_PATH" ]; then
    echo "[Step 2] データセットを再利用: $DATASET_PATH"
else
    echo "[Step 2] RL データセットを構築中 (R_θ ラベル付け)..."
    uv run python scripts/rl/build_rl_dataset.py \
        --trajectories     "$TRAJ_CACHE" \
        --irl-model        "$IRL_MODEL" \
        --irl-config       "$IRL_CONFIG" \
        --output           "$DATASET_PATH" \
        --state-dim        "$STATE_DIM" \
        --action-dim       "$ACTION_DIM" \
        --reward-shaping   "$REWARD_SHAPING" \
        --test-split-date  "$TEST_SPLIT_DATE" \
        --device           "$DEVICE"
    echo "[Step 2] 完了: $DATASET_PATH"
fi

# ──────────────────────────────────────────────────────────────────────
# Step 3: オフライン RL 訓練 (BC → IQL)
# ──────────────────────────────────────────────────────────────────────
echo "[Step 3] オフライン RL を訓練中 (BC → IQL)..."
uv run python scripts/rl/train_offline_rl.py \
    --dataset    "$DATASET_PATH" \
    --output-dir "$RL_OUTPUT_DIR" \
    --obs-dim    "$OBS_DIM" \
    --n-actions  2 \
    --bc-epochs  "$BC_EPOCHS" \
    --iql-epochs "$IQL_EPOCHS" \
    --device     "$DEVICE"
echo "[Step 3] 完了: ${RL_OUTPUT_DIR}/iql_agent.pt"

# ──────────────────────────────────────────────────────────────────────
# Step 4: 評価
# ──────────────────────────────────────────────────────────────────────
echo "[Step 4] 政策を評価中 (IRL eval 互換出力)..."
EVAL_ARGS=(
    --train-dataset "$DATASET_PATH"
    --iql-model     "${RL_OUTPUT_DIR}/iql_agent.pt"
    --bc-model      "${RL_OUTPUT_DIR}/bc_policy.pt"
    --obs-dim       "$OBS_DIM"
    --output-dir    "$EVAL_OUTPUT_DIR"
    --device        "$DEVICE"
)
if [ -f "$DATASET_TEST_PATH" ]; then
    EVAL_ARGS+=(--test-dataset "$DATASET_TEST_PATH")
fi
uv run python scripts/rl/eval_assignment.py "${EVAL_ARGS[@]}"
echo "[Step 4] 完了: $EVAL_OUTPUT_DIR/"

echo "========================================================"
echo " 全ステップ完了"
echo "   チェックポイント : ${RL_OUTPUT_DIR}/"
echo "   評価結果         : ${EVAL_OUTPUT_DIR}/"
echo "      ├─ summary_metrics.json"
echo "      ├─ pair_predictions.csv"
echo "      └─ calibration_curve.csv"
echo "========================================================"
