"""PATRON multi-head network (spec architecture, CPU-friendly).

Input (B, 96, 12)
  -> Conv1d projection 12 -> 32
  -> 3 residual CNN blocks: 32/k3, 64/k5, 64/k7
  -> BiLSTM 64 per direction (128)
  -> Multi-head self-attention, 4 heads, d=128
  -> FFN 128 -> 128 -> 64
  -> attention pooling over 96 steps -> shared vector 64
  -> four heads: pattern(12), barrier(3), quantile(6*3*3), momentum(5)

The quantile head is parameterised so quantiles can never cross:
``q10 = median - softplus(a)``, ``q90 = median + softplus(b)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h = self.dropout(h)
        return F.gelu(h + self.residual(x))


class AttentionPooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        scores = torch.matmul(self.query.expand(x.size(0), -1, -1), x.transpose(1, 2)) * self.scale
        weights = torch.softmax(scores, dim=-1)  # (B, 1, L)
        return torch.bmm(weights, x).squeeze(1)  # (B, D)


@dataclass
class PatronOutputs:
    pattern: torch.Tensor
    barrier: torch.Tensor
    quantile: torch.Tensor
    momentum: torch.Tensor


class PatronNet(nn.Module):
    def __init__(
        self,
        feature_count: int = 12,
        sequence_length: int = 96,
        horizon: int = 6,
        n_range_fields: int = 3,
        n_quantiles: int = 3,
        n_patterns: int = 12,
        n_barrier: int = 3,
        n_momentum: int = 5,
        projection_dim: int = 32,
        cnn_channels: tuple[int, ...] = (32, 64, 64),
        cnn_kernels: tuple[int, ...] = (3, 5, 7),
        lstm_hidden: int = 64,
        lstm_layers: int = 1,
        attention_heads: int = 4,
        attention_dim: int = 128,
        shared_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        if len(cnn_channels) != len(cnn_kernels):
            raise ValueError("cnn_channels and cnn_kernels must have equal length")
        if len(cnn_channels) != 3:
            raise ValueError("PATRON v1 expects exactly 3 CNN blocks")
        if attention_dim != 2 * lstm_hidden:
            raise ValueError("attention_dim must equal 2 * lstm_hidden")
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.n_range_fields = n_range_fields
        self.n_quantiles = n_quantiles

        self.input_projection = nn.Conv1d(feature_count, projection_dim, 1)
        blocks = []
        in_ch = projection_dim
        for out_ch, kernel in zip(cnn_channels, cnn_kernels):
            blocks.append(ConvBlock(in_ch, out_ch, kernel, dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        self.bilstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attn_norm = nn.LayerNorm(attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, attention_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim, shared_dim),
            nn.GELU(),
        )
        self.ffn_residual = nn.Linear(attention_dim, shared_dim)
        self.pool = AttentionPooling(shared_dim)
        self.shared = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pattern_head = nn.Linear(shared_dim, n_patterns)
        self.barrier_head = nn.Linear(shared_dim, n_barrier)
        self.quantile_head = nn.Linear(
            shared_dim, horizon * n_range_fields * n_quantiles
        )
        self.momentum_head = nn.Linear(shared_dim, n_momentum)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3 or x.size(1) != self.sequence_length or x.size(2) != 12:
            raise ValueError(
                f"expected (B, {self.sequence_length}, 12), got {tuple(x.shape)}"
            )
        h = self.input_projection(x.transpose(1, 2))  # (B, 32, L)
        h = self.cnn(h)
        h = h.transpose(1, 2)  # (B, L, C)

        lstm_out, _ = self.bilstm(h)  # (B, L, 128)
        normed = self.attn_norm(lstm_out)
        attn_out, _ = self.attention(normed, normed, normed, need_weights=False)
        attn_out = attn_out + lstm_out  # residual

        ffn_out = self.ffn(attn_out) + self.ffn_residual(attn_out)  # (B, L, 64)
        context = self.pool(ffn_out)  # (B, 64)
        shared = self.shared(context)

        q_raw = self.quantile_head(shared).view(
            -1, self.horizon, self.n_range_fields, self.n_quantiles
        )
        median = q_raw[..., 0]
        lower = median - F.softplus(q_raw[..., 1])
        upper = median + F.softplus(q_raw[..., 2])
        quantiles = torch.stack([lower, median, upper], dim=-1)

        return {
            "pattern": self.pattern_head(shared),
            "barrier": self.barrier_head(shared),
            "quantile": quantiles,
            "momentum": self.momentum_head(shared),
        }


def build_model(cfg) -> PatronNet:
    return PatronNet(
        feature_count=cfg.feature_count,
        sequence_length=cfg.sequence_length,
        horizon=cfg.horizon,
        n_range_fields=len(cfg.forecast.range.fields),
        n_quantiles=len(cfg.quantiles),
        n_patterns=len(cfg.pattern_tags),
        n_barrier=3,
        n_momentum=len(cfg.momentum_classes),
        projection_dim=cfg.model.input_projection_dim,
        cnn_channels=tuple(cfg.model.cnn_channels),
        cnn_kernels=tuple(cfg.model.cnn_kernels),
        lstm_hidden=cfg.model.bilstm_hidden_per_direction,
        lstm_layers=cfg.model.bilstm_layers,
        attention_heads=cfg.model.attention_heads,
        attention_dim=cfg.model.attention_dim,
        shared_dim=cfg.model.shared_dim,
        dropout=cfg.model.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
