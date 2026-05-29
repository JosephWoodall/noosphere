"""
EEGNet: Compact CNN baseline for EEG-based BCI (Lawhern et al., 2018).
doi:10.1088/1741-2552/aace8c

Input:  (B, n_channels, T)  — channels-last NOT used; standard channels×time
Output: (B, n_classes)      logits

Architecture exactly follows the paper:
  Block 1: F1 temporal filters → Depthwise spatial (D*F1 filters) → ELU → AvgPool
  Block 2: Separable conv (depthwise + pointwise) → ELU → AvgPool
  Classify: flatten → Linear

Used as a modern neural BCI baseline in the LOSO evaluation harness.
EEGNet has no pretrained checkpoint — it is trained end-to-end from scratch
on each fold's training data, making it a fair within-subject comparison.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(
        self,
        n_classes:  int,
        n_channels: int = 21,
        T:          int = 256,
        F1:         int = 8,
        D:          int = 2,
        dropout:    float = 0.25,
    ):
        super().__init__()
        F2 = F1 * D

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, T // 2), padding=(0, T // 4), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),  # depthwise
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        # Block 2: separable = depthwise + pointwise
        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, (1, 1),  bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        # Compute flattened size once
        with torch.no_grad():
            probe = torch.zeros(1, 1, n_channels, T)
            flat  = self.block2(self.block1(probe)).flatten(1).shape[1]
        self.classifier = nn.Linear(flat, n_classes)
        self._flat = flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) or (B, 1, C, T)."""
        if x.dim() == 3:
            x = x.unsqueeze(1)          # → (B, 1, C, T)
        return self.classifier(self.block2(self.block1(x)).flatten(1))
