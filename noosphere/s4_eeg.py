"""
noosphere/s4_eeg.py
===================
Optimized RS-S4 Architecture for CPU Execution
"""

import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class S4DLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model, self.d_state = d_model, 8
        self.A_log = nn.Parameter(torch.log(torch.arange(1, 9).float()).unsqueeze(0).expand(d_model, -1).clone())
        self.B = nn.Parameter(torch.sqrt(2 * torch.arange(8) + 1).unsqueeze(0).expand(d_model, -1).clone())
        self.C = nn.Parameter(torch.randn(d_model, 8) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))
        self.log_dt = nn.Parameter(torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))

    def forward(self, u: torch.Tensor):
        L = u.shape[1]
        dt, A = torch.exp(self.log_dt).unsqueeze(-1), -torch.exp(self.A_log)
        A_bar = torch.exp(dt * A)
        B_bar = (A_bar - 1.0) / (A + 1e-8) * self.B
        k = (torch.exp(torch.log(A_bar.clamp(min=1e-8)).unsqueeze(1) * torch.arange(L, device=u.device).view(1, L, 1)) * (self.C * B_bar).unsqueeze(1)).sum(-1)
        u_t, n_fft = u.transpose(1, 2), 2**math.ceil(math.log2(2*L))
        y = torch.fft.irfft(torch.fft.rfft(k, n=n_fft) * torch.fft.rfft(u_t, n=n_fft), n=n_fft)[..., :L]
        return (y + u_t * self.D.unsqueeze(-1)).transpose(1, 2)

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, (1, 15), padding=(0, 7), bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels // 2, (1, 31), padding=(0, 15), bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        return F.elu(self.bn(x))

class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=1)
        
    def forward(self, x):
        attn = torch.sigmoid(self.conv(x))
        return x * attn

class S4EEGEncoder(nn.Module):
    def __init__(self, in_channels=21, d_model=32, n_actions=8, do_norm=True, **kwargs):
        super().__init__()
        self.do_norm = do_norm
        self.inception = InceptionBlock(1, d_model)
        self.spat_conv = nn.Conv2d(d_model, d_model, (in_channels, 1), bias=False)
        self.attn = SpatialAttention(d_model)
        self.bn = nn.BatchNorm2d(d_model)
        self.s = S4DLayer(d_model)
        # Combine Adaptive Pooling and Global Average Pooling
        self.pool1 = nn.AdaptiveAvgPool1d(4)
        self.pool2 = nn.AdaptiveMaxPool1d(4)
        self.out = nn.Sequential(
            nn.Linear(d_model * 8, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor, mask=None) -> dict:
        if self.do_norm:
            x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True).clamp(min=1e-6))

        x = x.unsqueeze(1)
        x = self.inception(x)
        x = F.elu(self.bn(self.spat_conv(x)))
        x = x.squeeze(2)
        x = self.attn(x)
        x = x.transpose(1, 2)

        x = self.s(x)
        x_t = x.transpose(1, 2)
        # Rich pooling: Mean + Max
        feat = torch.cat([self.pool1(x_t), self.pool2(x_t)], dim=1).flatten(1)
        logits = self.out(feat)
        return {"alpha": logits, "intent_probs": F.softmax(logits, dim=-1), "embed": feat}

class DirichletEDLLoss(nn.Module):
    def __init__(self, **kwargs): super().__init__()
    def forward(self, alpha, target, step): return F.cross_entropy(alpha, target.argmax(-1))
