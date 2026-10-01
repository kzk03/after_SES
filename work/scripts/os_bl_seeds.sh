#!/usr/bin/env bash
# OpenStack の 13 方策比較を seed43/44 で回す (balanced の優位が seed 依存でないか確認)
set -e
cd /Users/kazuki-h/projects/after_ses
for s in 43 44; do
  echo "=== OpenStack bl seed$s 開始 $(date +%H:%M) ==="
  uv run python scripts/rl/train_assignment_rl.py \
    --episodes work/assign_ep.pkl \
    --acceptance-model outputs/rl/contrib_model_v3/contrib_model.joblib \
    --output-dir outputs/rl/assignment/os_bl10_s$s --state-dim 26 --total-timesteps 60000 \
    --seed $s --n-test-episodes 109 --w-accept 1.0 --w-reject 0.0 --w-load 0.2 --w-gini 0.25 \
    --load-cap 5 --background-history --stuck-when-no-response
  echo "=== OpenStack bl seed$s 完了 $(date +%H:%M) ==="
done
echo "全完了"
