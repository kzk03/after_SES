"""候補集合に対して順序不変な方策ネットワーク (柱2 用)

旧 env は候補ベクトルをフラット結合して `Discrete(n_candidates)` のインデックスで
選ばせていたため、候補リストの並び順が変わると同じ開発者が別物に見えた。
つまり方策が「並び順」を暗記する余地があった。

本モジュールは候補ごとに**共有**エンコーダとスコアラを当てることで、
候補の入れ替えに対してロジットも同じように入れ替わる (permutation-equivariant)
構造にする。これにより:
  - 並び順に依存しない
  - 候補数が変わっても同じ重みが使える
  - 学習した「良いレビュアーの条件」が候補位置に紐づかない

構成:
  candidates [B, N, F] ─┐
  task       [B, T] ────┴→ 共有MLP → e [B, N, H]
                            ↓ (有効候補のみで平均プーリング)
                          ctx [B, H]
  [e, ctx] → 共有MLP → e' [B, N, H] → 共有Linear(H,1) → logits [B, N]
"""

from __future__ import annotations

from typing import Dict

import gymnasium as gym
import torch as th
import torch.nn as nn
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CandidateSetExtractor(BaseFeaturesExtractor):
    """候補ごとの埋め込みを作る特徴抽出器。

    出力は [B, N*H] にフラット化した per-candidate 埋め込み。
    SharedScorer が [B, N, H] に戻して候補ごとにスコアを付ける。
    """

    def __init__(self, observation_space: gym.spaces.Dict, hidden: int = 64) -> None:
        n_cand, feat_dim = observation_space["candidates"].shape
        task_dim = observation_space["task"].shape[0]
        super().__init__(observation_space, features_dim=n_cand * hidden)

        self.n_cand = int(n_cand)
        self.hidden = int(hidden)

        self.encoder = nn.Sequential(
            nn.Linear(feat_dim + task_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # 候補プール全体の文脈を各候補に混ぜる (「相対的に誰が良いか」を見るため)
        self.with_context = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        cand = obs["candidates"].float()            # [B, N, F]
        task = obs["task"].float()                  # [B, T]
        mask = obs["mask"].float()                  # [B, N]

        b, n, _ = cand.shape
        x = th.cat([cand, task.unsqueeze(1).expand(b, n, -1)], dim=-1)
        e = self.encoder(x)                         # [B, N, H]

        # 有効候補だけで平均プーリング (パディングを混ぜない)
        m = mask.unsqueeze(-1)
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0).unsqueeze(-1)
        ctx = (e * m).sum(dim=1, keepdim=True) / denom   # [B, 1, H]

        e2 = self.with_context(th.cat([e, ctx.expand_as(e)], dim=-1))
        return e2.reshape(b, n * self.hidden)


class SharedScorer(nn.Module):
    """per-candidate 埋め込みから候補ごとのロジットを出す共有ヘッド。

    全候補に同じ Linear(H, 1) を当てるので順序不変性が保たれる。
    (通常の Linear(N*H, N) だと候補位置ごとに別の重みになってしまう)
    """

    def __init__(self, n_cand: int, hidden: int) -> None:
        super().__init__()
        self.n_cand = n_cand
        self.hidden = hidden
        self.score = nn.Linear(hidden, 1)

    def forward(self, latent: th.Tensor) -> th.Tensor:
        b = latent.shape[0]
        e = latent.view(b, self.n_cand, self.hidden)
        return self.score(e).squeeze(-1)            # [B, N]


class SetMaskablePolicy(MaskableActorCriticPolicy):
    """MaskablePPO 用の順序不変方策。

    action_net だけを SharedScorer に差し替える。net_arch は
    pi=[] (=恒等) にして per-candidate 埋め込みをそのまま渡し、
    vf 側は通常の MLP で状態価値を出す。
    """

    def __init__(self, *args, hidden: int = 64, **kwargs) -> None:
        kwargs.setdefault("features_extractor_class", CandidateSetExtractor)
        kwargs.setdefault("features_extractor_kwargs", {"hidden": hidden})
        # pi=[] にすると mlp_extractor の policy 側が恒等になり、
        # 特徴抽出器の出力 (per-candidate 埋め込み) が latent_pi にそのまま来る
        kwargs.setdefault("net_arch", {"pi": [], "vf": [64, 64]})
        self._hidden = hidden
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        n_cand = int(self.observation_space["candidates"].shape[0])
        # super() が作った Linear(N*H, N) を共有スコアラに置き換える
        self.action_net = SharedScorer(n_cand, self._hidden)
        # optimizer は置き換え後のパラメータで作り直す
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )
