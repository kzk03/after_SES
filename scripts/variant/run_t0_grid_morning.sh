#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# [SES 追加分析] t0グリッド: IRL回収後のローカル一括集計（2026-07-06朝 用）
# ──────────────────────────────────────────────────────────────────────────
# GPUサーバから outputs/ を rsync 回収した後に、これ1本で
#   1) 系列別 headline 横断集計 (t0_compare.py, B定義)
#   2) グリッド統合表・図 (t0_compare_grid.py)
#   3) アーキタイプ分析 (t0_archetype.py, k=4) ※IRL必須なので回収後のみ動く
#   4) アーキタイプ横断集計 (t0_archetype_compare.py, 対応付け品質つき)
# を流す。IRL 未着の t0/系列は各ツールが自動 skip するので、部分回収でも安全。
#
# 使い方（リポジトリルートから）:
#   bash scripts/variant/run_t0_grid_morning.sh                 # 既定3系列
#   bash scripts/variant/run_t0_grid_morning.sh inter_fs36      # 1系列のみ
#   ARCHETYPE=0 bash scripts/variant/run_t0_grid_morning.sh     # 集計のみ(3,4を省略)
#
# 前段（手動）: rsync -avz --exclude='*.pkl' --progress \
#   socsel:/mnt/data1/kazuki-h/additional/outputs/ ~/projects/after_ses/outputs/
# ──────────────────────────────────────────────────────────────────────────
set -uo pipefail

SERIES=("${@:-}")
[ -z "${SERIES[0]:-}" ] && SERIES=(inter_fs36 align_fs36 inter_fs24)
T0S=(2022-01-01 2022-10-01 2023-01-01 2023-04-01 2024-01-01)
ARCHETYPE="${ARCHETYPE:-1}"
# アーキタイプを回す系列（本命のみ。増やすなら env で上書き）
ARCH_SERIES="${ARCH_SERIES:-inter_fs36 align_fs36}"

# 0) data/raw_json symlink（消えていることがある）
if [ ! -e data/raw_json ]; then
    echo "[setup] data/raw_json symlink を再作成"
    ln -s refetch_20260615/raw_json data/raw_json
fi

env_for () {  # 系列名 → t0_compare 系の環境変数
    case "$1" in
        inter_fs24) echo "INTER=1 T0_SWEEP_ROOT=outputs/t0_sweep_inter_fs24" ;;
        inter_fs36) echo "INTER=1 T0_SWEEP_ROOT=outputs/t0_sweep_inter_fs36" ;;
        inter_fs45) echo "INTER=1 T0_SWEEP_ROOT=outputs/t0_sweep_inter_fs45" ;;
        align_fs24) echo "ALIGN=1 T0_SWEEP_ROOT=outputs/t0_sweep_align_fs24" ;;
        align_fs36) echo "ALIGN=1 T0_SWEEP_ROOT=outputs/t0_sweep_align_fs36" ;;
        *) echo "ERROR: unknown series $1" >&2; return 1 ;;
    esac
}

echo "========== 1) 系列別 headline 横断集計 (B定義) =========="
for s in "${SERIES[@]}"; do
    e="$(env_for "$s")" || exit 1
    echo "--- $s ($e) ---"
    env $e uv run python scripts/analyze/study/t0_compare.py || echo "[warn] $s の集計に失敗（続行）"
done

echo "========== 2) グリッド統合 (表・図) =========="
uv run python scripts/analyze/study/t0_compare_grid.py || echo "[warn] グリッド統合に失敗（続行）"

if [ "$ARCHETYPE" = "1" ]; then
    echo "========== 3) アーキタイプ分析 (k=4, dev+pair) =========="
    for s in $ARCH_SERIES; do
        for t0 in "${T0S[@]}"; do
            case "$s" in
                inter_*) SWEEP_ROOT="outputs/t0_sweep_${s}" \
                         bash scripts/variant/run_t0_archetype_inter.sh "$t0" both 4 \
                         || echo "[warn] archetype $s $t0 失敗（続行）" ;;
                align_*) ALIGN=1 SWEEP_ROOT="outputs/t0_sweep_${s}" \
                         bash scripts/variant/run_t0_archetype.sh "$t0" both 4 \
                         || echo "[warn] archetype $s $t0 失敗（続行）" ;;
            esac
        done
    done

    echo "========== 4) アーキタイプ横断集計 (対応付け品質つき) =========="
    for s in $ARCH_SERIES align_fs24; do   # align_fs24 は既存出力の横断が未実行なので同時に
        for level in dev pair; do
            env "T0_SWEEP_ROOT=outputs/t0_sweep_${s}" \
                uv run python scripts/analyze/study/t0_archetype_compare.py --level "$level" --k 4 \
                || echo "[warn] archetype横断 $s $level 失敗（続行）"
        done
    done
fi

echo "========== 完了。結果の反映先 =========="
echo "  - writing/t0_grid_results.md        … §2 グリッド表 / §3 解釈 (inter_fs36 IRL で tail効果を確定)"
echo "  - writing/t0_archetype_discussion.md … §2 品質表・§6 に本命系列の原型を追記"
echo "  - writing/t0_sweep_inter_results.md  … §7 グリッドの結論を追記"
echo "  - 図: writing/fig/fig_t0_grid.pdf, fig_t0_archetype_*_{系列サフィックス}.pdf"
echo "確認ポイント: (1) inter_fs36 2022-01 の IRL が回復するか (tail vs 系列長の切り分け)"
echo "              (2) Ens>=max(IRL,RF) が inter_fs36 で保たれるか"
echo "              (3) archetype の match_margin<0 の点がどこに出るか"
