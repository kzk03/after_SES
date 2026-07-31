"""Implicit Q-Learning (IQL)

Kostrikov et al. 2021 "Offline Reinforcement Learning with Implicit Q-Learning"
の自前 PyTorch 実装（d4rl / SB3 非依存, ~200行）。

アルゴリズム概要:
  1. V(s): expectile regression で offline 悲観性を表現
     L_V(ψ) = E[L_τ^2(Q_θ(s,a) - V_ψ(s))]
     L_τ^2(u) = |τ - 1{u<0}| u^2  (expectile 損失, τ∈(0.5, 1.0))

  2. Q(s,a): Bellman バックアップ（V を次状態に使う）
     L_Q(θ) = E[(r + γ V_ψ(s') - Q_θ(s,a))^2]

  3. π(a|s): advantage-weighted Behavioral Cloning
     L_π(φ) = -E[exp(β(Q_θ(s,a) - V_ψ(s))) log π_φ(a|s)]
     温度パラメータ β で exploitation 度合いを調整

offline RL の特性上、環境サンプリングは不要で dataset のみで訓練する。
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# ネットワーク定義
# ─────────────────────────────────────────────

class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(_MLP):
    """V(s): スカラー値を返す。"""
    def __init__(self, obs_dim: int, hidden: int = 256, dropout: float = 0.05):
        super().__init__(obs_dim, 1, hidden, dropout)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)  # [B]


class QNetwork(_MLP):
    """Q(s, a): 離散行動版 — 全行動の Q 値を同時に返す。"""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, dropout: float = 0.05):
        super().__init__(obs_dim, n_actions, hidden, dropout)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # [B, n_actions]


class PolicyNetwork(_MLP):
    """π(a|s): 行動ロジットを返す。"""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256, dropout: float = 0.05):
        super().__init__(obs_dim, n_actions, hidden, dropout)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # [B, n_actions]


# ─────────────────────────────────────────────
# IQL エージェント
# ─────────────────────────────────────────────

class IQLAgent:
    """Implicit Q-Learning オフライン RL エージェント。

    使い方:
        agent = IQLAgent(obs_dim=obs_dim, n_actions=2)
        stats = agent.train(dataset, epochs=100)
        action = agent.act(obs)
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.9,
        beta: float = 3.0,
        adv_clip: float = 10.0,
        lr: float = 3e-4,
        target_update_freq: int = 100,
        polyak: float = 0.005,
        exp_clip_log: float = 6.0,
        divergence_threshold: float = 1e4,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            obs_dim: 観測次元
            n_actions: 離散行動数
            hidden_dim: MLP 隠れ層次元
            gamma: 割引率
            tau: expectile パラメータ（0.5 < τ < 1.0 で悲観的）
            beta: advantage-weighted BC の温度パラメータ
            adv_clip: advantage のクリップ上限
            lr: 学習率
            target_update_freq: target ネットワークの hard copy 頻度
                （polyak=0 のときのみ使用。旧挙動の再現用）
            polyak: target ネットワークの soft update 係数。>0 なら毎ステップ
                q_target ← (1-polyak)·q_target + polyak·q_net で更新する。
                hard copy (100ステップごと全置換) は V↔Q の相互ブートストラップを
                増幅して V を発散させたため、既定を soft update にしている。
            exp_clip_log: advantage-weighted BC の重み exp(β·adv) の指数部の上限。
                adv_clip だけでは β 倍後に exp が overflow し、π が NaN になる。
            divergence_threshold: この値を超える epoch 平均 V loss を発散とみなし、
                学習を打ち切る（全 epoch 走ってから壊れていると気づくのを防ぐ）
            device: torch デバイス文字列
        """
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.adv_clip = adv_clip
        self.lr = lr
        self.device = torch.device(device)
        self.target_update_freq = target_update_freq
        self.polyak = polyak
        self.exp_clip_log = exp_clip_log
        self.divergence_threshold = divergence_threshold

        self.v_net = ValueNetwork(obs_dim, hidden_dim).to(self.device)
        self.q_net = QNetwork(obs_dim, n_actions, hidden_dim).to(self.device)
        self.q_target = copy.deepcopy(self.q_net).to(self.device)
        self.policy = PolicyNetwork(obs_dim, n_actions, hidden_dim).to(self.device)

        self.v_opt = optim.AdamW(self.v_net.parameters(), lr=lr)
        self.q_opt = optim.AdamW(self.q_net.parameters(), lr=lr)
        self.pi_opt = optim.AdamW(self.policy.parameters(), lr=lr)

        self._train_step = 0

    # ------------------------------------------------------------------
    # 損失計算
    # ------------------------------------------------------------------

    def _expectile_loss(self, diff: torch.Tensor) -> torch.Tensor:
        """L_τ^2(u) = |τ - 1{u<0}| u^2"""
        weight = torch.where(diff < 0, self.tau, 1.0 - self.tau)
        return (weight * diff.pow(2)).mean()

    def _q_sa_target(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """Q_target(s, a) を勾配なしで計算する ([B])。

        _v_loss と _pi_loss の両方で使うため _update で 1 回だけ計算して
        共有する (q_target は _update 末尾でしか同期されないため等価)。
        """
        with torch.no_grad():
            q_vals = self.q_target(states)               # [B, A]
            return q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

    def _v_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_sa: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q_sa is None:
            q_sa = self._q_sa_target(states, actions)
        v = self.v_net(states)                           # [B]
        return self._expectile_loss(q_sa - v)

    def _q_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            v_next = self.v_net(next_states)             # [B]
            target_q = rewards + self.gamma * (1.0 - dones) * v_next
        q_vals = self.q_net(states)                      # [B, A]
        q_sa = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
        return F.mse_loss(q_sa, target_q)

    def _pi_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        q_sa: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q_sa is None:
            q_sa = self._q_sa_target(states, actions)
        with torch.no_grad():
            # v は直前の V 更新を反映した最新値を使う (IQL のセマンティクス)
            v = self.v_net(states)                       # [B]
            adv = (q_sa - v).clamp(max=self.adv_clip)
            # exp は指数部で頭打ちにする。adv_clip だけだと β 倍で overflow し
            # weight が inf → 損失が NaN → 政策が丸ごと壊れる
            weight = torch.exp((self.beta * adv).clamp(max=self.exp_clip_log))
            weight = weight / weight.mean().clamp(min=1e-6)

        logits = self.policy(states)                     # [B, A]
        log_pi = F.log_softmax(logits, dim=-1)
        log_pi_a = log_pi.gather(1, actions.unsqueeze(1)).squeeze(1)
        return -(weight * log_pi_a).mean()

    # ------------------------------------------------------------------
    # 訓練ステップ
    # ------------------------------------------------------------------

    def _update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Dict[str, float]:
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # q_target(s, a) は V 更新と Policy 更新の両方で使う。
        # q_target は本ステップ末尾まで変化しないため 1 回だけ計算して共有する。
        q_sa_target = self._q_sa_target(states, actions)

        # 1. V 更新
        self.v_opt.zero_grad()
        lv = self._v_loss(states, actions, q_sa=q_sa_target)
        lv.backward()
        nn.utils.clip_grad_norm_(self.v_net.parameters(), 1.0)
        self.v_opt.step()

        # 2. Q 更新
        self.q_opt.zero_grad()
        lq = self._q_loss(states, actions, rewards, next_states, dones)
        lq.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_opt.step()

        # 3. Policy 更新
        self.pi_opt.zero_grad()
        lpi = self._pi_loss(states, actions, q_sa=q_sa_target)
        lpi.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.pi_opt.step()

        self._train_step += 1
        if self.polyak > 0.0:
            with torch.no_grad():
                for p_t, p in zip(self.q_target.parameters(), self.q_net.parameters()):
                    p_t.mul_(1.0 - self.polyak).add_(p, alpha=self.polyak)
        elif self._train_step % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return {
            "v_loss": float(lv.item()),
            "q_loss": float(lq.item()),
            "pi_loss": float(lpi.item()),
        }

    # ------------------------------------------------------------------
    # 訓練ループ
    # ------------------------------------------------------------------

    def train(
        self,
        dataset,
        epochs: int = 100,
        batch_size: int = 256,
    ) -> Dict[str, Any]:
        """IQL 訓練ループ。

        Args:
            dataset: OfflineRLDataset インスタンス
            epochs: エポック数
            batch_size: ミニバッチサイズ

        Returns:
            訓練統計 dict
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        v_losses: List[float] = []
        q_losses: List[float] = []
        pi_losses: List[float] = []

        logger.info("IQL 訓練開始 (epochs=%d, dataset=%d)", epochs, len(dataset))

        diverged = False
        for epoch in range(epochs):
            epoch_v, epoch_q, epoch_pi, n_b = 0.0, 0.0, 0.0, 0
            for batch in loader:
                stats = self._update(*batch)
                epoch_v += stats["v_loss"]
                epoch_q += stats["q_loss"]
                epoch_pi += stats["pi_loss"]
                n_b += 1

            v_losses.append(epoch_v / max(n_b, 1))
            q_losses.append(epoch_q / max(n_b, 1))
            pi_losses.append(epoch_pi / max(n_b, 1))

            if epoch % 10 == 0:
                logger.info(
                    "IQL epoch %d: V=%.4f Q=%.4f π=%.4f",
                    epoch, v_losses[-1], q_losses[-1], pi_losses[-1],
                )

            # 発散検知: 残り epoch を走らせても壊れた値関数が育つだけなので打ち切る
            latest = [v_losses[-1], q_losses[-1], pi_losses[-1]]
            if any(not math.isfinite(x) for x in latest):
                logger.error("IQL 発散を検知 (epoch %d, 非有限な loss): V=%s Q=%s π=%s → 打ち切り",
                             epoch, v_losses[-1], q_losses[-1], pi_losses[-1])
                diverged = True
            elif v_losses[-1] > self.divergence_threshold:
                logger.error("IQL 発散を検知 (epoch %d): V loss %.3e > 閾値 %.3e → 打ち切り",
                             epoch, v_losses[-1], self.divergence_threshold)
                diverged = True
            if diverged:
                break

        logger.info("IQL 訓練完了 (steps=%d, epochs=%d/%d, diverged=%s)",
                    self._train_step, len(v_losses), epochs, diverged)
        return {
            "v_losses": v_losses,
            "q_losses": q_losses,
            "pi_losses": pi_losses,
            "diverged": diverged,
            "epochs_run": len(v_losses),
        }

    # ------------------------------------------------------------------
    # 推論
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """numpy 観測から行動を選択する。"""
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            logits = self.policy(obs_t)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())

    # ------------------------------------------------------------------
    # チェックポイント
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """全コンストラクタ引数 + 学習状態を保存する。

        推論のみの利用ではネットワーク state_dict と config だけで足りるが、
        学習再開 (resume) にも耐えるよう _train_step / q_target / optimizer
        状態も含めて完全に round-trip させる。
        """
        torch.save(
            {
                "v_net": self.v_net.state_dict(),
                "q_net": self.q_net.state_dict(),
                "q_target": self.q_target.state_dict(),
                "policy": self.policy.state_dict(),
                "v_opt": self.v_opt.state_dict(),
                "q_opt": self.q_opt.state_dict(),
                "pi_opt": self.pi_opt.state_dict(),
                "train_step": self._train_step,
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "config": {
                    "gamma": self.gamma,
                    "tau": self.tau,
                    "beta": self.beta,
                    "adv_clip": self.adv_clip,
                    "lr": self.lr,
                    "target_update_freq": self.target_update_freq,
                    "polyak": self.polyak,
                    "exp_clip_log": self.exp_clip_log,
                    "divergence_threshold": self.divergence_threshold,
                    "hidden_dim": self.v_net.net[0].out_features,
                },
            },
            path,
        )
        logger.info("IQL チェックポイントを保存: %s", path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "IQLAgent":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        cfg = ckpt.get("config", {})
        agent = cls(
            obs_dim=ckpt["obs_dim"],
            n_actions=ckpt["n_actions"],
            hidden_dim=cfg.get("hidden_dim", 256),
            gamma=cfg.get("gamma", 0.99),
            tau=cfg.get("tau", 0.9),
            beta=cfg.get("beta", 3.0),
            adv_clip=cfg.get("adv_clip", 100.0),
            lr=cfg.get("lr", 3e-4),
            target_update_freq=cfg.get("target_update_freq", 100),
            # 旧チェックポイント (これらのキーが無い) は hard copy 時代のものなので
            # polyak=0 = 旧挙動で復元し、当時の学習条件を取り違えないようにする
            polyak=cfg.get("polyak", 0.0),
            exp_clip_log=cfg.get("exp_clip_log", float("inf")),
            divergence_threshold=cfg.get("divergence_threshold", float("inf")),
            device=device,
        )
        agent.v_net.load_state_dict(ckpt["v_net"])
        agent.q_net.load_state_dict(ckpt["q_net"])
        # 旧フォーマット (q_target 未保存) では q_net のコピーで初期化する
        agent.q_target.load_state_dict(ckpt.get("q_target", ckpt["q_net"]))
        agent.policy.load_state_dict(ckpt["policy"])
        # optimizer / step カウンタは存在すれば復元 (学習再開用、旧フォーマット互換)
        if "v_opt" in ckpt:
            agent.v_opt.load_state_dict(ckpt["v_opt"])
            agent.q_opt.load_state_dict(ckpt["q_opt"])
            agent.pi_opt.load_state_dict(ckpt["pi_opt"])
        agent._train_step = int(ckpt.get("train_step", 0))
        return agent

    def load_bc_weights(self, bc_policy) -> None:
        """BC policy の重みを IQL 政策ネットワークの warm-start に使う。"""
        if hasattr(bc_policy, "net") and hasattr(self.policy, "net"):
            try:
                self.policy.load_state_dict(bc_policy.state_dict())
                logger.info("BC weights → IQL policy に転写完了")
            except Exception as e:
                logger.warning("BC 重み転写失敗（アーキテクチャ不一致の可能性）: %s", e)
