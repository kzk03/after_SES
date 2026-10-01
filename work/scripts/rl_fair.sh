#!/usr/bin/env bash
# 公平性を是正した PPO の再検証。逐次実行(並列だとCPU競合で遅くなる)。
#   共通条件は aosp_bl10 と完全一致 (seed42 / w_load 0.05 / w_gini 0.25 / cap5)
#   比較用に logged と balanced を各runに含め、同一run内で対にする
set -e
cd /Users/kazuki-h/projects/after_ses
COMMON="--episodes work/aosp_ep_7d.pkl --acceptance-model outputs/rl/contrib_model_aosp/contrib_model.joblib --state-dim 26 --n-test-episodes 85 --seed 42 --w-accept 1.0 --w-load 0.05 --w-gini 0.25 --load-cap 5 --background-history --stuck-when-no-response --policies logged,balanced"

echo "=== 改善4: 学習量 200k (3.0周) $(date +%H:%M) ==="
uv run python scripts/rl/train_assignment_rl.py $COMMON --total-timesteps 200000 --output-dir outputs/rl/assignment/aosp_fair200k
echo "=== 改善4 完了 $(date +%H:%M) ==="

echo "=== 改善4+3: 学習量 200k + 報酬整形 $(date +%H:%M) ==="
uv run python scripts/rl/train_assignment_rl.py $COMMON --total-timesteps 200000 --shaping-gamma 0.99 --output-dir outputs/rl/assignment/aosp_fair200k_shaped
echo "=== 改善4+3 完了 $(date +%H:%M) ==="
echo "全完了"
