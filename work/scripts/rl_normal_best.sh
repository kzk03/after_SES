#!/usr/bin/env bash
# 通常の重み設定でも「学習報酬の最良点」と「最終モデル」がtestで逆転するかを見る。
#   高w_load(改善6)では逆転した(+11.9→−0.9649 vs −172.0→−0.8749)。
#   これが極端な設定に固有か、通常設定でも起きるかで、
#   これまでの学習曲線の読み方(「37kで頭打ち」等)の意味が変わる。
set -e
cd /Users/kazuki-h/projects/after_ses
uv run python scripts/rl/train_assignment_rl.py \
  --episodes work/aosp_ep_7d.pkl \
  --acceptance-model outputs/rl/contrib_model_aosp/contrib_model.joblib \
  --state-dim 26 --n-test-episodes 85 --seed 42 \
  --w-accept 1.0 --w-reject 0.0 --w-load 0.05 --w-gini 0.25 --load-cap 5 \
  --background-history --stuck-when-no-response --save-best \
  --policies logged --total-timesteps 200000 \
  --output-dir outputs/rl/assignment/aosp_normal_best
echo "=== 学習と最終評価 完了 $(date +%H:%M) ==="
uv run python work/scripts/eval_best.py \
  --ppo outputs/rl/assignment/aosp_normal_best/ppo_best.zip --w-load 0.05 --label normal_best
echo "=== 最良モデル評価 完了 $(date +%H:%M) ==="
