"""
IRLモデルのネットワークバリアント

6つのバリアントを提供:
  V0: LSTM（ベースライン、既存RetentionIRLNetworkと同等）
  V1: LSTM + Temporal Attention
  V2: Transformer
  V3: LSTM + Multi-task（4ヘッド）
  V4: LSTM + Attention + Multi-task
  V5: Transformer + Multi-task
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════
#  共通コンポーネント
# ═══════════════════════════════════════════════════════════════

class BaseIRLNetwork(nn.Module):
    """全バリアント共通の state/action エンコーダと reward predictor"""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode(self, state: torch.Tensor, action: torch.Tensor):
        """state/action をエンコードして連結。

        Args:
            state:  [B, L, state_dim]
            action: [B, L, action_dim]

        Returns:
            combined: [B, L, hidden_dim]
        """
        B, L, _ = state.shape
        s_enc = self.state_encoder(state.view(-1, state.shape[-1])).view(B, L, -1)
        a_enc = self.action_encoder(action.view(-1, action.shape[-1])).view(B, L, -1)
        return torch.cat([s_enc, a_enc], dim=-1)  # [B, L, hidden_dim]


class LSTMBackbone(nn.Module):
    """2層 LSTM + LayerNorm"""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, combined: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            combined: [B, L, hidden_dim]
            lengths:  [B]

        Returns:
            lstm_out: [B, L, hidden_dim]  全ステップ出力（LayerNorm済み）
            last_hidden: [B, hidden_dim]  各系列の最終ステップ
        """
        B = combined.size(0)
        lengths_cpu = lengths.cpu()
        sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
        _, unsort_idx = sorted_idx.sort()

        packed = nn.utils.rnn.pack_padded_sequence(
            combined[sorted_idx], sorted_lengths,
            batch_first=True, enforce_sorted=True,
        )
        out_packed, _ = self.lstm(packed)
        out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=combined.size(1),
        )
        lstm_out = out_sorted[unsort_idx]
        lstm_out = self.norm(lstm_out)

        # 最終ステップを取得
        last_hidden = torch.zeros(B, lstm_out.size(-1), device=combined.device)
        for i in range(B):
            last_hidden[i] = lstm_out[i, lengths[i].item() - 1, :]

        return lstm_out, last_hidden


class GRUBackbone(nn.Module):
    """2層 GRU + LayerNorm（SES査読B-3対応: RNN系列モデル比較用）。

    LSTMBackbone とインタフェース互換（セル状態を持たない点のみ異なる）。
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(
            hidden_dim, hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, combined: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            combined: [B, L, hidden_dim]
            lengths:  [B]

        Returns:
            gru_out: [B, L, hidden_dim]  全ステップ出力（LayerNorm済み）
            last_hidden: [B, hidden_dim]  各系列の最終ステップ
        """
        B = combined.size(0)
        lengths_cpu = lengths.cpu()
        sorted_lengths, sorted_idx = lengths_cpu.sort(descending=True)
        _, unsort_idx = sorted_idx.sort()

        packed = nn.utils.rnn.pack_padded_sequence(
            combined[sorted_idx], sorted_lengths,
            batch_first=True, enforce_sorted=True,
        )
        out_packed, _ = self.gru(packed)
        out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=combined.size(1),
        )
        gru_out = out_sorted[unsort_idx]
        gru_out = self.norm(gru_out)

        # 最終ステップを取得
        last_hidden = torch.zeros(B, gru_out.size(-1), device=combined.device)
        for i in range(B):
            last_hidden[i] = gru_out[i, lengths[i].item() - 1, :]

        return gru_out, last_hidden


class TemporalAttention(nn.Module):
    """LSTM 出力に対する Temporal Attention。

    最終 hidden state を query として、全タイムステップの出力を
    key/value とした attention weighted sum を計算する。
    どの時期が重要かが可視化可能（論文向き）。
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = math.sqrt(hidden_dim)

    def forward(self, lstm_out: torch.Tensor, last_hidden: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out:    [B, L, hidden_dim]
            last_hidden: [B, hidden_dim]
            lengths:     [B]

        Returns:
            context: [B, hidden_dim]  attention 重み付き和
        """
        query = self.query_proj(last_hidden).unsqueeze(1)   # [B, 1, D]
        keys = self.key_proj(lstm_out)                       # [B, L, D]
        scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / self.scale  # [B, L]

        # padding マスク
        max_len = lstm_out.size(1)
        mask = torch.arange(max_len, device=lstm_out.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float('-inf'))

        weights = torch.softmax(scores, dim=-1)              # [B, L]
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)  # [B, D]
        return context


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerBackbone(nn.Module):
    """TransformerEncoder + CLS トークン"""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 2,
                 nhead: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, combined: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            combined: [B, L, hidden_dim]
            lengths:  [B]

        Returns:
            all_out: [B, L, hidden_dim]  各ステップの出力（CLS除く）
            cls_out: [B, hidden_dim]     CLS トークンの出力
        """
        B, L, D = combined.shape

        # CLS トークンを先頭に追加
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, combined], dim=1)  # [B, 1+L, D]
        x = self.pos_encoder(x)

        # padding マスク（CLS=False, 有効ステップ=False, パディング=True）
        max_len_with_cls = L + 1
        mask = torch.zeros(B, max_len_with_cls, dtype=torch.bool, device=combined.device)
        for i in range(B):
            actual = lengths[i].item() + 1  # +1 for CLS
            mask[i, actual:] = True

        out = self.encoder(x, src_key_padding_mask=mask)
        out = self.norm(out)

        cls_out = out[:, 0, :]     # [B, D]
        all_out = out[:, 1:, :]    # [B, L, D]
        return all_out, cls_out


class ContinuationHead(nn.Module):
    """1つの継続予測ヘッド"""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiTaskHead(nn.Module):
    """4つの ContinuationHead（将来窓 0-3m, 3-6m, 6-9m, 9-12m）"""

    NUM_HEADS = 4

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.heads = nn.ModuleList([
            ContinuationHead(hidden_dim, dropout) for _ in range(self.NUM_HEADS)
        ])

    def forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Args:
            x: [B, D] or [B*L, D]

        Returns:
            {0: [B,1], 1: [B,1], 2: [B,1], 3: [B,1]}
        """
        return {i: head(x) for i, head in enumerate(self.heads)}


# ═══════════════════════════════════════════════════════════════
#  V0: LSTM ベースライン
# ═══════════════════════════════════════════════════════════════

class IRLNetworkV0(BaseIRLNetwork):
    """LSTM ベースライン（既存 RetentionIRLNetwork と同等）"""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(state_dim, action_dim, hidden_dim, dropout)
        self.backbone = LSTMBackbone(hidden_dim, dropout)
        self.continuation_predictor = ContinuationHead(hidden_dim, dropout)

    def forward(self, state, action, lengths):
        combined = self._encode(state, action)
        _, last_hidden = self.backbone(combined, lengths)
        reward = self.reward_predictor(last_hidden)
        cont = self.continuation_predictor(last_hidden)
        return reward, cont

    def forward_all_steps(self, state, action, lengths, return_reward=False):
        combined = self._encode(state, action)
        lstm_out, _ = self.backbone(combined, lengths)
        B, L, D = lstm_out.shape
        flat = lstm_out.reshape(-1, D)
        cont = self.continuation_predictor(flat).squeeze(-1).view(B, L)
        if return_reward:
            rew = self.reward_predictor(flat).squeeze(-1).view(B, L)
            return rew, cont
        return cont


# ═══════════════════════════════════════════════════════════════
#  V1: LSTM + Attention
# ═══════════════════════════════════════════════════════════════

class IRLNetworkV1(BaseIRLNetwork):
    """LSTM + Temporal Attention"""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(state_dim, action_dim, hidden_dim, dropout)
        self.backbone = LSTMBackbone(hidden_dim, dropout)
        self.attention = TemporalAttention(hidden_dim)
        self.continuation_predictor = ContinuationHead(hidden_dim, dropout)

    def forward(self, state, action, lengths):
        combined = self._encode(state, action)
        lstm_out, last_hidden = self.backbone(combined, lengths)
        context = self.attention(lstm_out, last_hidden, lengths)
        reward = self.reward_predictor(context)
        cont = self.continuation_predictor(context)
        return reward, cont

    def forward_all_steps(self, state, action, lengths, return_reward=False):
        combined = self._encode(state, action)
        lstm_out, _ = self.backbone(combined, lengths)
        B, L, D = lstm_out.shape
        flat = lstm_out.reshape(-1, D)
        cont = self.continuation_predictor(flat).squeeze(-1).view(B, L)
        if return_reward:
            rew = self.reward_predictor(flat).squeeze(-1).view(B, L)
            return rew, cont
        return cont


# ═══════════════════════════════════════════════════════════════
#  V2: Transformer
# ═══════════════════════════════════════════════════════════════

class IRLNetworkV2(BaseIRLNetwork):
    """Transformer (CLS トークン集約)"""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(state_dim, action_dim, hidden_dim, dropout)
        self.backbone = TransformerBackbone(hidden_dim, dropout=dropout)
        self.continuation_predictor = ContinuationHead(hidden_dim, dropout)

    def forward(self, state, action, lengths):
        combined = self._encode(state, action)
        _, cls_out = self.backbone(combined, lengths)
        reward = self.reward_predictor(cls_out)
        cont = self.continuation_predictor(cls_out)
        return reward, cont

    def forward_all_steps(self, state, action, lengths, return_reward=False):
        combined = self._encode(state, action)
        all_out, _ = self.backbone(combined, lengths)
        B, L, D = all_out.shape
        flat = all_out.reshape(-1, D)
        cont = self.continuation_predictor(flat).squeeze(-1).view(B, L)
        if return_reward:
            rew = self.reward_predictor(flat).squeeze(-1).view(B, L)
            return rew, cont
        return cont


# ═══════════════════════════════════════════════════════════════
#  V3: LSTM + Multi-task
# ═══════════════════════════════════════════════════════════════

class IRLNetworkV3(BaseIRLNetwork):
    """LSTM + Multi-task（4ヘッド）"""

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(state_dim, action_dim, hidden_dim, dropout)
        self.backbone = LSTMBackbone(hidden_dim, dropout)
        self.multitask_head = MultiTaskHead(hidden_dim, dropout)

    def forward(self, state, action, lengths):
        combined = self._encode(state, action)
        _, last_hidden = self.backbone(combined, lengths)
        reward = self.reward_predictor(last_hidden)
        cont_dict = self.multitask_head(last_hidden)
        return reward, cont_dict

    def forward_all_steps(self, state, action, lengths, return_reward=False):
        combined = self._encode(state, action)
        lstm_out, _ = self.backbone(combined, lengths)
        B, L, D = lstm_out.shape
        flat = lstm_out.reshape(-1, D)
        cont_dict = {
            i: head(flat).squeeze(-1).view(B, L)
            for i, head in enumerate(self.multitask_head.heads)
        }
        if return_reward:
            rew = self.reward_predictor(flat).squeeze(-1).view(B, L)
            return rew, cont_dict
        return cont_dict


# ═══════════════════════════════════════════════════════════════
#  V6: LSTM + Snapshot MLP (Hybrid: IRL の時系列 + RF_dir 風の snapshot)
# ═══════════════════════════════════════════════════════════════

class IRLNetworkV6_Hybrid(BaseIRLNetwork):
    """LSTM (時系列符号化) と snapshot MLP (累積特徴の neural tabular branch) を
    並列に持ち、両者の hidden を連結して継続予測する単一モデル。

    - LSTM branch: 既存 V0 と同じく [B, L, hidden] の系列を出力し最終 hidden を取る
    - MLP branch:  各サンプルの最終時刻の raw state (state_dim) を MLP に通す
                   → RF_dir の inductive bias を neural net で再現する位置付け
    - 出力: 両 hidden を連結し reward / continuation head に入力
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.1, tab_dim: int = 64):
        super().__init__(state_dim, action_dim, hidden_dim, dropout)
        self.backbone = LSTMBackbone(hidden_dim, dropout)

        # snapshot 用の tabular branch (≒ neural RF_dir)
        self.tab_dim = tab_dim
        self.snapshot_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, tab_dim),
            nn.LayerNorm(tab_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 結合次元の reward/continuation 予測器（BaseIRLNetwork の同名属性を上書き）
        combined_dim = hidden_dim + tab_dim
        self.reward_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.continuation_predictor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _snapshot(state: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """各系列の最終有効ステップにおける raw state を抽出。

        Args:
            state:   [B, L, state_dim]
            lengths: [B]

        Returns:
            snapshot: [B, state_dim]
        """
        B = state.size(0)
        idx = (lengths.to(state.device).long() - 1).clamp(min=0)
        return state[torch.arange(B, device=state.device), idx, :]

    def forward(self, state, action, lengths):
        combined = self._encode(state, action)
        _, last_hidden = self.backbone(combined, lengths)       # [B, hidden]

        snapshot = self._snapshot(state, lengths)               # [B, state_dim]
        h_tab = self.snapshot_mlp(snapshot)                     # [B, tab_dim]

        h_joint = torch.cat([last_hidden, h_tab], dim=-1)       # [B, hidden+tab]
        reward = self.reward_predictor(h_joint)
        cont = self.continuation_predictor(h_joint)
        return reward, cont

    def forward_all_steps(self, state, action, lengths, return_reward=False):
        combined = self._encode(state, action)
        lstm_out, _ = self.backbone(combined, lengths)          # [B, L, hidden]
        B, L, D = lstm_out.shape

        # 各ステップの raw state を MLP に通して per-step tab branch を作る
        h_tab_all = self.snapshot_mlp(
            state.reshape(-1, state.size(-1))
        ).view(B, L, -1)                                        # [B, L, tab]

        h_joint = torch.cat([lstm_out, h_tab_all], dim=-1)      # [B, L, hidden+tab]
        flat = h_joint.reshape(-1, h_joint.size(-1))

        cont = self.continuation_predictor(flat).squeeze(-1).view(B, L)
        if return_reward:
            rew = self.reward_predictor(flat).squeeze(-1).view(B, L)
            return rew, cont
        return cont


# ═══════════════════════════════════════════════════════════════
#  V7: LSTM + Archetype Auxiliary Classifier
# ═══════════════════════════════════════════════════════════════

class IRLNetworkV7_ArchetypeAux(BaseIRLNetwork):
    """LSTM ベースラインに開発者 archetype 4 クラス分類の補助ヘッドを併設したモデル。

    主タスク: 継続予測（V0 と同じ）。
    補助タスク: 開発者を Archetypal Analysis の 4 タイプ (A1〜A4) に分類する。
    LSTM hidden state を 2 タスクで共有し、archetype 区別性のある表現を学習させる狙い。

    学習側で archetype ラベルを mce_irl_predictor 経由で渡し、
    主損失 + λ * 補助損失を最適化する（mce_irl_predictor.py 側で合成）。
    """

    NUM_ARCHETYPES = 4

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__(state_dim, action_dim, hidden_dim, dropout)
        self.backbone = LSTMBackbone(hidden_dim, dropout)
        self.continuation_predictor = ContinuationHead(hidden_dim, dropout)
        self.archetype_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.NUM_ARCHETYPES),
        )

    def forward(self, state, action, lengths):
        combined = self._encode(state, action)
        _, last_hidden = self.backbone(combined, lengths)
        reward = self.reward_predictor(last_hidden)
        cont = self.continuation_predictor(last_hidden)
        return reward, cont

    def forward_with_archetype(self, state, action, lengths):
        """主出力に加えて archetype ロジット (B, 4) を返す（学習時に使用）。"""
        combined = self._encode(state, action)
        _, last_hidden = self.backbone(combined, lengths)
        reward = self.reward_predictor(last_hidden)
        cont = self.continuation_predictor(last_hidden)
        arche_logits = self.archetype_head(last_hidden)
        return reward, cont, arche_logits

    def forward_all_steps(self, state, action, lengths, return_reward=False):
        combined = self._encode(state, action)
        lstm_out, _ = self.backbone(combined, lengths)
        B, L, D = lstm_out.shape
        flat = lstm_out.reshape(-1, D)
        cont = self.continuation_predictor(flat).squeeze(-1).view(B, L)
        if return_reward:
            rew = self.reward_predictor(flat).squeeze(-1).view(B, L)
            return rew, cont
        return cont


# ═══════════════════════════════════════════════════════════════
#  ファクトリ
# ═══════════════════════════════════════════════════════════════

VARIANT_REGISTRY = {
    0: IRLNetworkV0,
    1: IRLNetworkV1,
    2: IRLNetworkV2,
    3: IRLNetworkV3,
    6: IRLNetworkV6_Hybrid,
    7: IRLNetworkV7_ArchetypeAux,
}

VARIANT_NAMES = {
    0: "lstm_baseline",
    1: "lstm_attention",
    2: "transformer",
    3: "lstm_multitask",
    6: "lstm_hybrid_snapshot",
    7: "lstm_archetype_aux",
}


def create_network(variant: int, state_dim: int, action_dim: int,
                   hidden_dim: int = 128, dropout: float = 0.1) -> BaseIRLNetwork:
    """バリアント番号からネットワークを生成"""
    if variant not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(VARIANT_REGISTRY.keys())}")
    cls = VARIANT_REGISTRY[variant]
    return cls(state_dim, action_dim, hidden_dim, dropout)


def is_multitask(variant: int) -> bool:
    """Multi-task バリアントかどうか（V3 のみ：4 horizon multi-head）"""
    return variant in {3}


def is_archetype_aux(variant: int) -> bool:
    """Archetype 補助タスクバリアントかどうか（V7 のみ）"""
    return variant == 7
