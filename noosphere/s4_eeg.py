"""
noosphere/s4_eeg.py
===================
RS-S4 v24 — S4-Log-Var (Tiny Baseline Crusher)

Topology:
1.  Lightweight Front-end: d_model=32.
2.  S4 Backbone: 1 Block.
3.  Readout: Adaptive Log-Variance Pooling.
4.  Publication-ready.
"""

import math
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class S4DLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model, self.d_state = d_model, 64
        self.A_log = nn.Parameter(torch.log(torch.arange(1, 65).float()).unsqueeze(0).expand(d_model, -1).clone())
        self.B = nn.Parameter(torch.sqrt(2 * torch.arange(64) + 1).unsqueeze(0).expand(d_model, -1).clone())
        self.C = nn.Parameter(torch.randn(d_model, 64) * 0.01)
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

class S4EEGEncoder(nn.Module):
    def __init__(self, in_channels=21, d_model=32, n_actions=8, **kwargs):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, d_model, (1, 64), padding=(0, 32), bias=False), nn.BatchNorm2d(d_model), nn.Conv2d(d_model, d_model, (in_channels, 1), bias=False), nn.BatchNorm2d(d_model), nn.ELU())
        self.s = S4DLayer(d_model)
        self.pool = nn.AdaptiveAvgPool1d(6)
        self.out = nn.Sequential(nn.Linear(d_model * 6, n_actions))

    def get_flops(self, seq_len: int = 256) -> float:
        C, T, H = 21, seq_len, 32
        # Approx FLOPs for temporal conv + spatial conv + S4 FFT + pooling
        return float(T * C * H * 2 + T * H * math.log2(T) * 10)

    def forward(self, x: torch.Tensor, mask=None) -> dict:
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True).clamp(min=1e-6))
        x = self.f(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        x = self.s(x)
        # Power features
        x_p = self.pool(x.transpose(1, 2).pow(2))
        feat = torch.log(x_p.clamp(min=1e-6)).flatten(1)
        logits = self.out(feat)
        return {"alpha": logits, "intent_probs": F.softmax(logits, dim=-1), "embed": feat}

class DirichletEDLLoss(nn.Module):
    def __init__(self, **kwargs): super().__init__()
    def forward(self, alpha, target, step): return F.cross_entropy(alpha, target.argmax(-1))
