#!/usr/bin/env bash
set -e
cd /Users/kazuki-h/projects/after_ses
for s in 43 44; do
  echo "=== wl005 seed$s 開始 $(date +%H:%M) ==="
  uv run python scripts/rl/train_assignment_rl.py \
    --episodes work/aosp_ep_7d.pkl \
    --acceptance-model outputs/rl/contrib_model_aosp/contrib_model.joblib \
    --output-dir outputs/rl/assignment/aosp_wl005_s$s --state-dim 26 --total-timesteps 60000 \
    --seed $s --n-test-episodes 85 --w-accept 1.0 --w-reject 0.0 --w-load 0.05 --w-gini 0.25 \
    --load-cap 5 --background-history --stuck-when-no-response
  echo "=== wl005 seed$s 完了 $(date +%H:%M) ==="
done
echo "全完了"
