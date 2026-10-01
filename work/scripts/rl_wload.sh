#!/usr/bin/env bash
# 改善6: 負荷項の SD を貢献項に匹敵させる (w_load 0.05 → 5.0)。
#   診断「信号が小さすぎて学べない」の当否を測る。
#   学習も評価も同じ w_load なので比較は公平。
set -e
cd /Users/kazuki-h/projects/after_ses
uv run python scripts/rl/train_assignment_rl.py \
  --episodes work/aosp_ep_7d.pkl \
  --acceptance-model outputs/rl/contrib_model_aosp/contrib_model.joblib \
  --state-dim 26 --n-test-episodes 85 --seed 42 \
  --w-accept 1.0 --w-reject 0.0 --w-load 5.0 --w-gini 0.25 --load-cap 5 \
  --background-history --stuck-when-no-response \
  --policies logged,balanced,least_load,balanced_cap \
  --total-timesteps 200000 --output-dir outputs/rl/assignment/aosp_wload5
echo "完了 $(date +%H:%M)"
