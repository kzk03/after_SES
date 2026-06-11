#!/bin/bash
# [サーバ用 / MCE-IRL 専用] 1つのバリアントに対して MCE-IRL 学習 + 評価10パターンを実行
#
# run_variant_single.sh の MCE-IRL 版。出力ディレクトリ・軌跡キャッシュ・モデル
# ファイル名はすべて MCE 系で別名を使い、既存 (Focal-supervised) の成果物と
# 衝突しないようにしている。
#
# 既存版との差分:
#   - 訓練スクリプト: scripts/train/train_mce_irl.py
#   - 評価スクリプト: scripts/analyze/eval/eval_mce_irl_path_prediction.py
#   - 軌跡キャッシュ: outputs/mce_irl_trajectory_cache/mce_traj_*.pkl
#   - モデルファイル: mce_irl_model.pt
#   - デフォルト OUTBASE: outputs/mce_irl_variant_comparison_server
#   - MT バリアント (3,4,5) は未対応（MCE-IRL は二値 action 設計のため）
#
# 使い方:
#   bash scripts/variant/run_mce_irl_variant_single.sh <variant_id> <variant_name> [outbase] [gpu_id] [save_importance]
#
# 例:
#   bash scripts/variant/run_mce_irl_variant_single.sh 0 lstm_baseline outputs/mce_irl_variant_comparison_server 0
#   bash scripts/variant/run_mce_irl_variant_single.sh 1 lstm_attention outputs/mce_irl_variant_comparison_server 0
#   bash scripts/variant/run_mce_irl_variant_single.sh 2 transformer    outputs/mce_irl_variant_comparison_server 0

set -e

# 再現性: dict/set 反復順を固定し、joblib worker 間でも一致させる
# (train_model.py / train_mce_irl.py の reviewer_dirs set 反復が原因で
#  同 train_start でも seed 固定下に AUC が ±1pt 揺れていた問題への対処)
export PYTHONHASHSEED=0

VTYPE="${1:?Usage: $0 <variant_id> <variant_name> [outbase] [gpu_id] [save_importance]}"
VNAME="${2:?Usage: $0 <variant_id> <variant_name> [outbase] [gpu_id] [save_importance]}"
OUTBASE="${3:-outputs/mce_irl_variant_comparison_server}"
GPU_ID="${4:-0}"
SAVE_IMPORTANCE="${5:-false}"

# MCE-IRL は variant 0/1/2/3 対応 (3 = LSTM Two-tower, 2026-05-15 追加)。
# multi-task ヘッド (variant 4/5) は持たない。
case "$VTYPE" in
    0|1|2|3) ;;
    *)
        echo "ERROR: MCE-IRL は variant 0/1/2/3 のみ対応 (指定: $VTYPE)" >&2
        exit 1
        ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# プロジェクト集合の差し替えは環境変数で行う:
#   REVIEWS              : 統合済み CSV
#                          (default: data/combined_raw_main32.csv — main32 スコープ)
#   RAW_JSON_LIST_FILE   : raw_json パスをスペース区切りで持つテキストファイル
#                          (default: data/combined_raw_main32.raw_json_list.txt)
# 例 (別スコープを使う場合):
#     REVIEWS=data/combined_raw_231.csv \
#     RAW_JSON_LIST_FILE=path/to/raw_json_list.txt \
#     bash scripts/variant/run_mce_irl_variant_single.sh 0 lstm_baseline outputs/foo 0
REVIEWS="${REVIEWS:-data/combined_raw_main32.csv}"
RAW_JSON_LIST_FILE="${RAW_JSON_LIST_FILE:-data/combined_raw_main32.raw_json_list.txt}"
if [ ! -f "$REVIEWS" ]; then
    echo "ERROR: REVIEWS CSV が見つからない: $REVIEWS" >&2
    exit 1
fi
if [ ! -f "$RAW_JSON_LIST_FILE" ]; then
    echo "ERROR: RAW_JSON_LIST_FILE が見つからない: $RAW_JSON_LIST_FILE" >&2
    exit 1
fi
read -r -a RAW_JSON <<< "$(cat "$RAW_JSON_LIST_FILE")"
TRAIN_START="2019-01-01"
TRAIN_END="2022-01-01"
EVAL_CUTOFF="2023-01-01"
EPOCHS="${EPOCHS:-150}"
PATIENCE="${PATIENCE:-15}"

TRAIN_WINDOWS=("0-3" "3-6" "6-9" "9-12")
TRAIN_FS=(0 3 6 9)
TRAIN_FE=(3 6 9 12)

# ── [SES 追加] horizon別「動的学習開始日」(②: 特徴量計測期間を全horizonで揃える) ──
# 後方クリップ (month_end+FE<=train_end) で長horizonほど最新month_endが早まり、
# 月次ステップ数が 0-3m=33 .. 9-12m=24 とズレる。等化方式は2通り:
#
# DYNAMIC_START=1 (Phase B, 後方延長): train_start を FS(h) ヶ月早め、全 horizon 33 ステップに揃える
#   train_start(h) = 2019-01-01 − FS(h)ヶ月  → 0-3m:2019-01 / 3-6m:2018-10 / 6-9m:2018-07 / 9-12m:2018-04
#   欠点: 古い 2018 年データが長 horizon に混ざり、prediction_time=2023 から見て分布乖離
#
# DYNAMIC_START=2 (Phase C, 前方削り): 短 horizon の train_start を後ろにずらし、全 horizon 24 ステップに揃える
#   train_start(h) = (train_end − FE(h)) − 24ヶ月  → 0-3m:2019-10 / 3-6m:2019-07 / 6-9m:2019-04 / 9-12m:2019-01
#   全 horizon の学習データが 2019-10 以降に揃い、古いデータ混入を回避（B の逆方向の等化）
#
# DYNAMIC_START=0 (既定) は従来通り全horizon 2019-01-01 (=Phase A 再現、ステップ数 33/30/27/24 とズレる)。
DYNAMIC_START="${DYNAMIC_START:-0}"
if [ "$DYNAMIC_START" = "1" ]; then
    TRAIN_STARTS=("2019-01-01" "2018-10-01" "2018-07-01" "2018-04-01")
elif [ "$DYNAMIC_START" = "2" ]; then
    TRAIN_STARTS=("2019-10-01" "2019-07-01" "2019-04-01" "2019-01-01")
else
    TRAIN_STARTS=("$TRAIN_START" "$TRAIN_START" "$TRAIN_START" "$TRAIN_START")
fi
# [SES] 評価IRLの月次履歴窓(月): B案=24(既定/論文準拠), A案=33(学習と窓長を揃える)
EVAL_DIR_HISTORY_MONTHS="${EVAL_DIR_HISTORY_MONTHS:-24}"
# [SES] device: server=cuda(既定), local=mps/cpu
DEVICE="${DEVICE:-cuda}"
echo "[SES] DYNAMIC_START=$DYNAMIC_START TRAIN_STARTS=(${TRAIN_STARTS[*]}) EVAL_DIR_HISTORY_MONTHS=$EVAL_DIR_HISTORY_MONTHS DEVICE=$DEVICE"

# MCE-IRL 専用キャッシュ (既存 outputs/trajectory_cache とは別物)
# CACHE_TAG で scope ごとに分離する (run_mce_pipeline.sh から渡される)。
# 未指定なら "default" として、旧10 repos デフォルトと scope 指定実行が混ざらないようにする。
CACHE_TAG="${CACHE_TAG:-default}"
CACHE_DIR="outputs/mce_irl_trajectory_cache/${CACHE_TAG}"
mkdir -p "$CACHE_DIR"

# Phase 2 (2026-05-15): co-change graph 由来の path 特徴量 CSV
#   HUB_SCORES_CSV / COCHANGE_NEIGHBORS_CSV を env で渡すと extract と eval に
#   --hub-scores / --cochange-neighbors を付与する。未指定なら従来通り (新 2 特徴量 0.0)。
EXTRA_PATH_FEAT_ARGS=()
if [ -n "${HUB_SCORES_CSV:-}" ] && [ -f "$HUB_SCORES_CSV" ]; then
    EXTRA_PATH_FEAT_ARGS+=(--hub-scores "$HUB_SCORES_CSV")
fi
if [ -n "${COCHANGE_NEIGHBORS_CSV:-}" ] && [ -f "$COCHANGE_NEIGHBORS_CSV" ]; then
    EXTRA_PATH_FEAT_ARGS+=(--cochange-neighbors "$COCHANGE_NEIGHBORS_CSV")
fi
if [ "${#EXTRA_PATH_FEAT_ARGS[@]}" -gt 0 ]; then
    echo "[Phase 2] path-feature CSVs: ${EXTRA_PATH_FEAT_ARGS[*]}"
fi

# warm-start 設定 (環境変数で制御):
#   INIT_FROM_BASE=outputs/variant_comparison_server/$VNAME を指定すると、
#   各窓で $INIT_FROM_BASE/train_${win}m/irl_model.pt から重みを load して
#   MCE NLL で fine-tune する (Focal-supervised → MCE-IRL warm-start)。
#   未設定なら cold start。
INIT_FROM_BASE="${INIT_FROM_BASE:-}"
INIT_LR_SCALE="${INIT_LR_SCALE:-0.1}"

# 評価1パターンを実行する関数
run_eval() {
    local model_path="$1"
    local train_win="$2"
    local eval_win="$3"
    local eval_fs="$4"
    local train_fs_val="$5"
    local eval_dir="$OUTBASE/$VNAME/train_${train_win}m/eval_${eval_win}m"
    mkdir -p "$eval_dir"

    if [ -f "$eval_dir/summary_metrics.json" ]; then
        echo "[MCE-IRL $VNAME train_${train_win}m -> eval_${eval_win}m] スキップ（評価済み）"
        return
    fi

    echo "[MCE-IRL $VNAME train_${train_win}m -> eval_${eval_win}m] 評価開始..."
    local importance_flag=""
    if [ "$SAVE_IMPORTANCE" = "true" ]; then
        importance_flag="--save-importance"
    fi
    uv run python scripts/analyze/eval/eval_mce_irl_path_prediction.py \
        --data "$REVIEWS" \
        --raw-json "${RAW_JSON[@]}" \
        --prediction-time "$EVAL_CUTOFF" \
        --delta-months 3 \
        --future-start-months "$eval_fs" \
        --rf-future-start-months "$train_fs_val" \
        --irl-dir-model "$model_path" \
        --rf-train-end "$TRAIN_END" \
        --dir-history-months "$EVAL_DIR_HISTORY_MONTHS" \
        --device "$DEVICE" \
        --n-jobs 4 \
        --output-dir "$eval_dir" \
        --calibrate \
        "${EXTRA_PATH_FEAT_ARGS[@]}" \
        $importance_flag
    echo "[MCE-IRL $VNAME train_${train_win}m -> eval_${eval_win}m] 評価完了"
}

# 評価並列数（同時にバックグラウンドで走る評価プロセス数）
EVAL_PARALLEL=4

echo ""
echo "============================================================"
echo "  [サーバ / MCE-IRL] バリアント $VTYPE: $VNAME (GPU=$GPU_ID)"
echo "  OUTBASE: $OUTBASE"
echo "  CACHE  : $CACHE_DIR"
echo "============================================================"

# ── 軌跡キャッシュを事前生成（窓ごとに 1 度だけ） ──
# extract_mce_trajectories.py は出力先がすでに存在すればスキップする。
# 必ず CACHE_DIR (outputs/mce_irl_trajectory_cache) 配下に書き、
# Focal-supervised の outputs/trajectory_cache/ とは混ざらないようにする。
for i in 0 1 2 3; do
    win="${TRAIN_WINDOWS[$i]}"
    fs="${TRAIN_FS[$i]}"
    fe="${TRAIN_FE[$i]}"
    ts="${TRAIN_STARTS[$i]}"
    cache_path="$CACHE_DIR/mce_traj_${win}_ts${ts}.pkl"
    if [ -f "$cache_path" ]; then
        echo "[MCE-IRL cache ${win}m] スキップ（キャッシュ済み: $cache_path）"
    else
        echo "[MCE-IRL cache ${win}m] 抽出開始 (future_window=${fs}-${fe}m)..."
        uv run python scripts/train/extract_mce_trajectories.py \
            --reviews "$REVIEWS" \
            --raw-json "${RAW_JSON[@]}" \
            --train-start "$ts" \
            --train-end "$TRAIN_END" \
            --future-window-start "$fs" \
            --future-window-end "$fe" \
            --n-jobs -1 \
            --output "$cache_path" \
            "${EXTRA_PATH_FEAT_ARGS[@]}"
        echo "[MCE-IRL cache ${win}m] 抽出完了 → $cache_path"
    fi
done

# ── 4訓練窓を逐次訓練（GPU 共有のため） ──
for i in 0 1 2 3; do
    win="${TRAIN_WINDOWS[$i]}"
    fs="${TRAIN_FS[$i]}"
    fe="${TRAIN_FE[$i]}"
    ts="${TRAIN_STARTS[$i]}"
    model_dir="$OUTBASE/$VNAME/train_${win}m"
    mkdir -p "$model_dir"

    if [ -f "$model_dir/mce_irl_model.pt" ]; then
        echo "[MCE-IRL $VNAME train_${win}m] 学習スキップ（学習済み）"
    else
        # warm-start: $INIT_FROM_BASE が指定されていて該当 checkpoint があれば load
        warm_args=()
        if [ -n "$INIT_FROM_BASE" ]; then
            init_ckpt="$INIT_FROM_BASE/train_${win}m/irl_model.pt"
            if [ -f "$init_ckpt" ]; then
                warm_args=(--init-from "$init_ckpt" --init-lr-scale "$INIT_LR_SCALE")
                echo "[MCE-IRL $VNAME train_${win}m] warm-start from $init_ckpt (LR×$INIT_LR_SCALE)"
            else
                echo "[MCE-IRL $VNAME train_${win}m] WARN: warm-start checkpoint 不在 ($init_ckpt) → cold start"
            fi
        fi
        echo "[MCE-IRL $VNAME train_${win}m] 学習開始 (future_window=${fs}-${fe}m, GPU=$GPU_ID)..."
        uv run python scripts/train/train_mce_irl.py \
            --directory-level \
            --model-type "$VTYPE" \
            --raw-json "${RAW_JSON[@]}" \
            --reviews "$REVIEWS" \
            --epochs "$EPOCHS" \
            --patience "$PATIENCE" \
            --train-start "$ts" \
            --train-end "$TRAIN_END" \
            --future-window-start "$fs" \
            --future-window-end "$fe" \
            --trajectories-cache "$CACHE_DIR/mce_traj_${win}_ts${ts}.pkl" \
            --output "$model_dir" \
            --skip-threshold \
            "${warm_args[@]}"
        echo "[MCE-IRL $VNAME train_${win}m] 学習完了"
    fi
done

# ── 評価: 最大 EVAL_PARALLEL 個ずつ並列実行 ──
running=0
for ti in 0 1 2 3; do
    train_win="${TRAIN_WINDOWS[$ti]}"
    model_path="$OUTBASE/$VNAME/train_${train_win}m/mce_irl_model.pt"

    if [ ! -f "$model_path" ]; then
        echo "[MCE-IRL $VNAME train_${train_win}m] モデルが見つからない、スキップ"
        continue
    fi

    for ei in $(seq $ti 3); do
        eval_win="${TRAIN_WINDOWS[$ei]}"
        eval_fs="${TRAIN_FS[$ei]}"
        train_fs_val="${TRAIN_FS[$ti]}"
        run_eval "$model_path" "$train_win" "$eval_win" "$eval_fs" "$train_fs_val" &
        running=$((running + 1))
        if [ "$running" -ge "$EVAL_PARALLEL" ]; then
            wait
            running=0
        fi
    done
done
if [ "$running" -gt 0 ]; then
    wait
fi
echo "[MCE-IRL $VNAME] 評価全パターン完了"

echo ""
echo "[MCE-IRL $VNAME] 全パターン完了"
