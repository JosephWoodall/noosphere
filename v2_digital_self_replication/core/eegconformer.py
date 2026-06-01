"""
EEGConformer: Convolutional Transformer for EEG signal classification.
Song et al. (2022). doi:10.1109/TNNLS.2022.3150699

Architecture:
  Conv module:  temporal conv → depthwise spatial → pointwise → temporal pool
  Transformer:  multi-head self-attention over pooled time steps
  Head:         global average pool → linear classifier

Input:  (B, n_channels, T)  — same convention as EEGNet
Output: (B, n_classes) logits

Trained end-to-end from scratch on each fold's training data (no pretrained
checkpoint), making it a fair within-subject comparison with EEGNet.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop(attn_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class EEGConformer(nn.Module):
    """
    EEGConformer for within-subject EEG classification.

    Parameters follow Song et al. (2022) Table I for small-scale BCI datasets:
      n_filters=40, D=2, pool1=8, pool2=8, n_heads=4, n_layers=2.
    """

    def __init__(
        self,
        n_classes:  int,
        n_channels: int = 21,
        T:          int = 256,
        n_filters:  int = 40,
        D:          int = 2,
        pool1:      int = 8,
        pool2:      int = 8,
        n_heads:    int = 4,
        n_layers:   int = 2,
        dropout:    float = 0.5,
    ):
        super().__init__()

        # ── Conv module ───────────────────────────────────────────────────────
        # Temporal convolution: extract oscillatory features across time
        self.temporal = nn.Sequential(
            nn.Conv2d(1, n_filters, (1, 25), padding=(0, 12), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        # Depthwise spatial: mix across EEG channels
        self.spatial = nn.Sequential(
            nn.Conv2d(n_filters, n_filters * D, (n_channels, 1),
                      groups=n_filters, bias=False),
            nn.BatchNorm2d(n_filters * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(dropout),
        )
        # Pointwise: compress channel dimension
        self.pointwise = nn.Sequential(
            nn.Conv2d(n_filters * D, n_filters, (1, 1), bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(dropout),
        )

        # Compute sequence length after conv+pool
        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, T)
            t_out = self.pointwise(self.spatial(self.temporal(probe)))
            seq_len = t_out.shape[-1]   # time steps remaining
            d_model = t_out.shape[1]    # = n_filters

        self.seq_len = seq_len
        self.d_model = d_model

        # Learnable positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # ── Transformer module ────────────────────────────────────────────────
        self.transformer = nn.Sequential(
            *[_TransformerLayer(d_model, n_heads, dropout)
              for _ in range(n_layers)]
        )

        # ── Classification head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) or (B, 1, C, T)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)                          # (B, 1, C, T)

        x = self.temporal(x)                            # (B, F, C, T)
        x = self.spatial(x)                             # (B, F*D, 1, T//pool1)
        x = self.pointwise(x)                           # (B, F, 1, T//pool1//pool2)
        x = x.squeeze(2).permute(0, 2, 1)              # (B, seq_len, d_model)

        x = x + self.pos_emb                            # positional encoding
        x = self.transformer(x)                         # (B, seq_len, d_model)
        x = x.mean(dim=1)                               # global avg pool over time
        return self.head(x)                             # (B, n_classes)
