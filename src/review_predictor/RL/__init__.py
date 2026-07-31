"""Review Assignment RL モジュール

IRL で学習済みの報酬関数 R_θ を利用して、
レビュー依頼割り当て政策を強化学習で最適化する。

フェーズ1: オフラインRL (BC → IQL)
フェーズ2: gymnasium env + SB3 オンラインRL (TQC)
フェーズ3: マルチエージェント評価 (pettingzoo)
"""
