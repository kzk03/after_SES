"""IRL 報酬関数ラッパー

MCEIRLSystem の学習済みネットワークを RL 環境から呼び出すためのインターフェース。
predict_continuation_probability_monthly() の dict 版ではなく、
network.forward() を直接呼び出すことでバッチ処理に対応する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem


class IRLRewardFn:
    """MCEIRLSystem の学習済み報酬関数 R_θ をラップする。

    報酬シグナル: R(s,1) - R(s,0)  (継続行動と非継続行動の reward 差分)
    継続確率:    π(a=1|s) = softmax(R(s,·))[1]  (Boltzmann 方策)
    """

    def __init__(self, irl_system: "MCEIRLSystem") -> None:
        self.system = irl_system
        self.device = irl_system.device

    @torch.no_grad()
    def batch_reward(
        self,
        state_seqs: torch.Tensor,
        action_seqs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """バッチの最終ステップでの報酬と継続確率を返す。

        Args:
            state_seqs:  [B, L, state_dim]
            action_seqs: [B, L, action_dim]
            lengths:     [B] — 各系列の有効長

        Returns:
            reward: [B]  R(s,1) - R(s,0)
            pi_a1:  [B]  π(a=1|s)
        """
        self.system.network.eval()
        logits = self.system.network(
            state_seqs.to(self.device),
            action_seqs.to(self.device),
            lengths.to(self.device),
        )  # [B, 2]
        pi = F.softmax(logits, dim=-1)
        reward = logits[:, 1] - logits[:, 0]
        return reward, pi[:, 1]

    @torch.no_grad()
    def single_reward(
        self,
        state_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> tuple[float, float]:
        """単一系列の報酬と継続確率を返す（環境のステップ計算用）。

        Args:
            state_seq:  [L, state_dim]
            action_seq: [L, action_dim]

        Returns:
            reward: float  R(s,1) - R(s,0)
            pi_a1:  float  π(a=1|s)
        """
        state_b = state_seq.unsqueeze(0).to(self.device)
        action_b = action_seq.unsqueeze(0).to(self.device)
        length = torch.tensor([state_seq.shape[0]], dtype=torch.long, device=self.device)
        reward, pi = self.batch_reward(state_b, action_b, length)
        return float(reward[0].item()), float(pi[0].item())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: dict,
        device: str = "cpu",
    ) -> "IRLRewardFn":
        """チェックポイントから MCEIRLSystem をロードして IRLRewardFn を返す。"""
        import torch
        from review_predictor.IRL.model.mce_irl_predictor import MCEIRLSystem

        config = dict(config)
        config.setdefault("model_type", 0)
        system = MCEIRLSystem(config)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "network_state_dict" in state_dict:
            system.network.load_state_dict(state_dict["network_state_dict"])
        else:
            system.network.load_state_dict(state_dict)
        system.network.to(system.device)
        return cls(system)
