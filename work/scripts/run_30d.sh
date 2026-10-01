#!/usr/bin/env bash
# エピソード長 30 日版のパイプラインを通す (60日版との比較用)
set -e
cd /Users/kazuki-h/projects/after_ses
echo "=== 1. 学習データ ==="
uv run python work/scripts/build_train.py --episodes work/assign_ep_30d.pkl --out work/contrib_train_30d.parquet
echo "=== 2. 貢献モデル ==="
uv run python work/scripts/train_contrib.py --episodes work/assign_ep_30d.pkl \
  --train work/contrib_train_30d.parquet --out-dir outputs/rl/contrib_model_30d
echo "=== 3. PPO ==="
uv run python scripts/rl/train_assignment_rl.py \
  --episodes work/assign_ep_30d.pkl \
  --acceptance-model outputs/rl/contrib_model_30d/contrib_model.joblib \
  --output-dir outputs/rl/assignment/new5_30d --state-dim 26 --total-timesteps 60000 \
  --seed 42 --n-test-episodes 190 --w-accept 1.0 --w-reject 0.0 --w-load 0.2 --w-gini 0.25 \
  --load-cap 5 --background-history --stuck-when-no-response --dump-step-log
echo "=== 完了 ==="
