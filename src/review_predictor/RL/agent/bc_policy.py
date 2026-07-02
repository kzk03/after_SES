"""Behavioral Cloning (BC) 政策

歴史上の割り当て（expert action = 実際に割り当てられた開発者の行動ラベル）を
模倣する cross-entropy 最小化による政策学習。

BC は offline RL の warm-start として使い、Phase1b の IQL 初期値を提供する。
ネットワーク: シンプルな 3 層 MLP（obs_dim → hidden → hidden → n_actions）
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class BCPolicy(nn.Module):
    """Behavioral Cloning 政策ネットワーク (MLP)。

    入力: 観測ベクトル [obs_dim]
    出力: 行動ロジット [n_actions]
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """行動ロジット [B, n_actions] を返す。"""
        return self.net(obs)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """numpy 観測から行動を選択する（推論時）。"""
        device = next(self.parameters()).device
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            logits = self.forward(obs_t)
            if deterministic:
                return int(logits.argmax(dim=-1).item())
            probs = F.softmax(logits, dim=-1)
            return int(torch.multinomial(probs, 1).item())


class BCTrainer:
    """Behavioral Cloning の訓練ループ。"""

    def __init__(
        self,
        policy: BCPolicy,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.policy = policy.to(device)
        self.device = torch.device(device)
        self.optimizer = optim.AdamW(
            policy.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def train(
        self,
        dataset,
        epochs: int = 50,
        batch_size: int = 256,
        val_ratio: float = 0.1,
        patience: int = 5,
    ) -> Dict[str, Any]:
        """BC 訓練ループ。

        Args:
            dataset: OfflineRLDataset インスタンス
            epochs: 最大エポック数
            batch_size: ミニバッチサイズ
            val_ratio: 検証セット割合
            patience: 早期終了 patience

        Returns:
            訓練統計 dict
        """
        from torch.utils.data import DataLoader, Subset

        n = len(dataset)
        indices = list(range(n))
        random.shuffle(indices)
        # val_n は train_idx が空にならないよう n-1 でクリップする
        # (n=1 の場合は val_n=0 となり、検証セットなしで全件を学習に使う)
        val_n = min(max(1, int(n * val_ratio)), max(0, n - 1))
        train_idx = indices[val_n:]
        val_idx = indices[:val_n]

        train_loader = DataLoader(
            Subset(dataset, train_idx), batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), batch_size=batch_size, shuffle=False
        )

        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        patience_counter = 0
        train_losses: List[float] = []
        val_losses: List[float] = []

        logger.info("BC 訓練開始 (epochs=%d, n_train=%d, n_val=%d)", epochs, len(train_idx), len(val_idx))

        for epoch in range(epochs):
            self.policy.train()
            epoch_loss = 0.0
            n_batches = 0
            for states, actions, *_ in train_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                logits = self.policy(states)
                loss = F.cross_entropy(logits, actions)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train)

            self.policy.eval()
            val_loss = 0.0
            n_val_b = 0
            with torch.no_grad():
                for states, actions, *_ in val_loader:
                    states = states.to(self.device)
                    actions = actions.to(self.device)
                    logits = self.policy(states)
                    val_loss += F.cross_entropy(logits, actions).item()
                    n_val_b += 1
            avg_val = val_loss / max(n_val_b, 1)
            val_losses.append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = copy.deepcopy(self.policy.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or patience_counter >= patience:
                logger.info(
                    "BC epoch %d: train=%.4f val=%.4f patience=%d/%d",
                    epoch, avg_train, avg_val, patience_counter, patience,
                )

            if patience_counter >= patience:
                logger.info("BC 早期終了: epoch=%d, best_val=%.4f", epoch, best_val_loss)
                break

        if best_state is not None:
            self.policy.load_state_dict(best_state)

        logger.info("BC 訓練完了: best_val_loss=%.4f", best_val_loss)
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "obs_dim": self.policy.obs_dim,
                "n_actions": self.policy.n_actions,
                "hidden_dim": self.policy.net[0].out_features,
                "dropout": self.policy.dropout,
            },
            path,
        )
        logger.info("BC policy を保存: %s", path)

    @classmethod
    def load_policy(cls, path: str, device: str = "cpu") -> BCPolicy:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        policy = BCPolicy(
            obs_dim=ckpt["obs_dim"],
            n_actions=ckpt["n_actions"],
            hidden_dim=ckpt.get("hidden_dim", 256),
            dropout=ckpt.get("dropout", 0.1),
        )
        policy.load_state_dict(ckpt["policy_state_dict"])
        return policy.to(device)
