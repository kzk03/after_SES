#!/usr/bin/env bash
# 貢献項の分散低減 (Rao-Blackwell 化) の検証。
#   学習時のみ貢献項を期待値 p_accept にする。評価は実現値なので比較は公平。
#   条件は aosp_fair200k と完全一致 (w_reject 0.3 も揃える) にして対比する。
set -e
cd /Users/kazuki-h/projects/after_ses
COMMON="--episodes work/aosp_ep_7d.pkl --acceptance-model outputs/rl/contrib_model_aosp/contrib_model.joblib --state-dim 26 --n-test-episodes 85 --seed 42 --w-accept 1.0 --w-load 0.05 --w-gini 0.25 --load-cap 5 --background-history --stuck-when-no-response --policies logged,balanced"

echo "=== 改善5: 分散低減のみ $(date +%H:%M) ==="
uv run python scripts/rl/train_assignment_rl.py $COMMON --total-timesteps 200000 --expected-accept-reward --output-dir outputs/rl/assignment/aosp_varred
echo "=== 改善5 完了 $(date +%H:%M) ==="
echo "全完了"
