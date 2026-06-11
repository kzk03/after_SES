#!/bin/bash
# [サーバ用] RF_Dir → KD pretrain → MCE-IRL warm-start を 4 train window で順次実行する。
#
# 概要:
#   1. 軌跡キャッシュを生成 (なければ)
#   2. 各 train window で kd_pretrain.py を走らせ kd_ckpts/train_X-Ym/irl_model.pt を生成
#   3. INIT_FROM_BASE に kd_ckpts/ を指定して run_mce_irl_variant_single.sh を呼び、
#      その既存パイプライン (train_mce_irl --init-from + 10 セル評価) に任せる
#
# 使い方:
#   bash scripts/variant/run_mce_irl_kd_warm_start.sh [variant_name] [outbase] [gpu_id] [save_importance]
#
# 例:
#   bash scripts/variant/run_mce_irl_kd_warm_start.sh lstm_baseline outputs/main32_mce_kd_warmstart 0
#
# 環境変数:
#   REVIEWS, RAW_JSON_LIST_FILE          : run_mce_irl_variant_single.sh と同じ規約
#   HUB_SCORES_CSV, COCHANGE_NEIGHBORS_CSV: co-change 用 path 特徴量 CSV
#   CACHE_TAG                             : 軌跡キャッシュの scope tag (default: main32)
#   KD_EPOCHS, KD_PATIENCE                : KD pretrain の epoch/patience (default: 30/5)
#   INIT_LR_SCALE                         : warm-start 後の MCE-IRL LR scale (default: 0.1)
#   ONLY_WINDOW                           : "0-3" 等を指定すると当該窓のみ実行 (smoke test 用)
#
set -e

VNAME="${1:-lstm_baseline}"
OUTBASE="${2:-outputs/main32_mce_kd_warmstart}"
GPU_ID="${3:-0}"
SAVE_IMPORTANCE="${4:-false}"

VTYPE=0
case "$VNAME" in
    lstm_baseline) VTYPE=0 ;;
    lstm_attention) VTYPE=1 ;;
    transformer) VTYPE=2 ;;
    lstm_two_tower) VTYPE=3 ;;
    *)
        echo "ERROR: unknown variant_name: $VNAME (use lstm_baseline/lstm_attention/transformer/lstm_two_tower)" >&2
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"

REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"
if [ ! -f "$REVIEWS" ]; then
    echo "ERROR: REVIEWS CSV not found: $REVIEWS" >&2
    exit 1
fi
if [ ! -f "$RAW_JSON_LIST_FILE" ]; then
    echo "ERROR: RAW_JSON_LIST_FILE not found: $RAW_JSON_LIST_FILE" >&2
    exit 1
fi
read -r -a RAW_JSON <<< "$(cat "$RAW_JSON_LIST_FILE")"

TRAIN_START="2019-01-01"
TRAIN_END="2022-01-01"
TRAIN_WINDOWS=("0-3" "3-6" "6-9" "9-12")
TRAIN_FS=(0 3 6 9)
TRAIN_FE=(3 6 9 12)

CACHE_TAG="${CACHE_TAG:-main32}"
CACHE_DIR="outputs/mce_irl_trajectory_cache/${CACHE_TAG}"
mkdir -p "$CACHE_DIR"

KD_CKPT_BASE="$OUTBASE/kd_ckpts"
mkdir -p "$KD_CKPT_BASE"

KD_EPOCHS="${KD_EPOCHS:-30}"
KD_PATIENCE="${KD_PATIENCE:-5}"
INIT_LR_SCALE="${INIT_LR_SCALE:-0.1}"
ONLY_WINDOW="${ONLY_WINDOW:-}"

EXTRA_PATH_FEAT_ARGS=()
if [ -n "${HUB_SCORES_CSV:-}" ] && [ -f "$HUB_SCORES_CSV" ]; then
    EXTRA_PATH_FEAT_ARGS+=(--hub-scores "$HUB_SCORES_CSV")
fi
if [ -n "${COCHANGE_NEIGHBORS_CSV:-}" ] && [ -f "$COCHANGE_NEIGHBORS_CSV" ]; then
    EXTRA_PATH_FEAT_ARGS+=(--cochange-neighbors "$COCHANGE_NEIGHBORS_CSV")
fi

echo ""
echo "============================================================"
echo "  [KD warm-start] variant=$VNAME (VTYPE=$VTYPE) GPU=$GPU_ID"
echo "  OUTBASE  : $OUTBASE"
echo "  KD ckpts : $KD_CKPT_BASE"
echo "  cache    : $CACHE_DIR"
echo "  ONLY_WIN : ${ONLY_WINDOW:-<all>}"
echo "============================================================"

# ── 軌跡キャッシュを事前生成 ──
for i in 0 1 2 3; do
    win="${TRAIN_WINDOWS[$i]}"
    if [ -n "$ONLY_WINDOW" ] && [ "$win" != "$ONLY_WINDOW" ]; then
        continue
    fi
    fs="${TRAIN_FS[$i]}"
    fe="${TRAIN_FE[$i]}"
    cache_path="$CACHE_DIR/mce_traj_${win}.pkl"
    if [ -f "$cache_path" ]; then
        echo "[cache ${win}m] skip (exists: $cache_path)"
    else
        echo "[cache ${win}m] extracting (future_window=${fs}-${fe}m)..."
        uv run python scripts/train/extract_mce_trajectories.py \
            --reviews "$REVIEWS" \
            --raw-json "${RAW_JSON[@]}" \
            --train-start "$TRAIN_START" \
            --train-end "$TRAIN_END" \
            --future-window-start "$fs" \
            --future-window-end "$fe" \
            --n-jobs -1 \
            --output "$cache_path" \
            "${EXTRA_PATH_FEAT_ARGS[@]}"
    fi
done

# ── KD pretrain を各窓で実行 ──
for i in 0 1 2 3; do
    win="${TRAIN_WINDOWS[$i]}"
    if [ -n "$ONLY_WINDOW" ] && [ "$win" != "$ONLY_WINDOW" ]; then
        continue
    fi
    fs="${TRAIN_FS[$i]}"
    fe="${TRAIN_FE[$i]}"
    kd_out="$KD_CKPT_BASE/train_${win}m"
    kd_ckpt="$kd_out/irl_model.pt"
    if [ -f "$kd_ckpt" ]; then
        echo "[KD ${win}m] skip (exists: $kd_ckpt)"
    else
        echo "[KD ${win}m] pretrain start..."
        uv run python scripts/train/kd_pretrain.py \
            --reviews "$REVIEWS" \
            --raw-json "${RAW_JSON[@]}" \
            --train-start "$TRAIN_START" \
            --train-end "$TRAIN_END" \
            --future-window-start "$fs" \
            --future-window-end "$fe" \
            --trajectories-cache "$CACHE_DIR/mce_traj_${win}.pkl" \
            --model-type "$VTYPE" \
            --epochs "$KD_EPOCHS" \
            --patience "$KD_PATIENCE" \
            --output "$kd_out" \
            "${EXTRA_PATH_FEAT_ARGS[@]}"
        echo "[KD ${win}m] done → $kd_ckpt"
    fi
done

# ── MCE-IRL warm-start + 評価を既存パイプラインに委譲 ──
# run_mce_irl_variant_single.sh は INIT_FROM_BASE/train_X-Ym/irl_model.pt から
# warm-start するので、KD ckpt ディレクトリをそのまま指定すればよい。
if [ -n "$ONLY_WINDOW" ]; then
    echo "[ONLY_WINDOW=$ONLY_WINDOW] 既存パイプラインを smoke 用に呼ぶには対象窓を明示する必要があるため、ここでは省略。"
    echo "  → 手動で次を実行:"
    echo "    INIT_FROM_BASE=$KD_CKPT_BASE INIT_LR_SCALE=$INIT_LR_SCALE \\"
    echo "    HUB_SCORES_CSV=${HUB_SCORES_CSV:-} COCHANGE_NEIGHBORS_CSV=${COCHANGE_NEIGHBORS_CSV:-} \\"
    echo "    CACHE_TAG=$CACHE_TAG \\"
    echo "    bash scripts/variant/run_mce_irl_variant_single.sh $VTYPE $VNAME $OUTBASE $GPU_ID $SAVE_IMPORTANCE"
    exit 0
fi

echo ""
echo "[MCE-IRL warm-start] 既存パイプラインを起動..."
INIT_FROM_BASE="$KD_CKPT_BASE" \
INIT_LR_SCALE="$INIT_LR_SCALE" \
HUB_SCORES_CSV="${HUB_SCORES_CSV:-}" \
COCHANGE_NEIGHBORS_CSV="${COCHANGE_NEIGHBORS_CSV:-}" \
CACHE_TAG="$CACHE_TAG" \
REVIEWS="$REVIEWS" \
RAW_JSON_LIST_FILE="$RAW_JSON_LIST_FILE" \
bash scripts/variant/run_mce_irl_variant_single.sh \
    "$VTYPE" "$VNAME" "$OUTBASE" "$GPU_ID" "$SAVE_IMPORTANCE"

echo ""
echo "[KD warm-start] 全完了"
