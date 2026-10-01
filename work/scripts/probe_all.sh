#!/usr/bin/env bash
# 学習済み方策の「選んだ人の負荷の順位」を全 run で比較する。
# 改善5 の結果が出てから走らせる。
set -e
cd /Users/kazuki-h/projects/after_ses
uv run python work/scripts/policy_probe.py \
  --ppo outputs/rl/assignment/aosp_bl10/ppo_assignment.zip \
        outputs/rl/assignment/aosp_fair200k/ppo_assignment.zip \
        outputs/rl/assignment/aosp_fair200k_shaped/ppo_assignment.zip \
        outputs/rl/assignment/aosp_varred/ppo_assignment.zip \
  --labels 従来60k 改善4_200k 改善4+3_整形 改善5_分散低減
