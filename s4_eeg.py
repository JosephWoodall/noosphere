"""
noosphere/s4_eeg.py
===================
S4 Structured State Space Model — Stream B (EEG)

EEG is processed sample-by-sample at full temporal resolution. No windowing.

Why S4 over patch tokenization for neural signals:
    A P300 ERP spans ~50ms at 256Hz (~12 samples). Patch boundaries split
    this event across tokens, destroying the shape that carries the label.
    S4 processes every sample through a continuous-time ODE, preserving
    sub-window structure entirely.

S4 state space:
    ẋ(t) = Ax(t) + Bu(t)    continuous-time ODE
    y(t) = Cx(t) + Du(t)    output equation

Discretised at rate Δ (bilinear transform):
    xₖ = Ā xₖ₋₁ + B̄ uₖ
    yₖ = C xₖ  + D uₖ

Training mode  : parallel FFT convolution   O(L log L)
Inference mode : single recurrence step     O(N)  — real-time capable

The HiPPO-LegS initialisation of A is designed to optimally memorise
continuous signals by projecting onto Legendre polynomial bases — aligned
with the oscillatory structure of neural data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import numpy as np


# ── HiPPO initialisation ─────────────────────────────────────────────────────

def _hippo_a(N: int) -> torch.Tensor:
    """HiPPO-LegS state matrix A ∈ ℝ^(N×N)."""
    A = torch.zeros(N, N)
    for n in range(N):
        for k in range(n):
            A[n, k] = -math.sqrt(2*n+1) * math.sqrt(2*k+1)
        A[n, n] = -(n + 1)
    return A.float()


def _hippo_b(N: int) -> torch.Tensor:
    """HiPPO-LegS input vector B[n] = √(2n+1)."""
    n = torch.arange(N, dtype=torch.float32)
    return torch.sqrt(2 * n + 1)


# ── S4 Layer ──────────────────────────────────────────────────────────────────

class S4Layer(nn.Module):
    """
    Single S4D layer (diagonal approximation of full S4).

    Parameters
    ----------
    d_model      : channel dimension
    d_state      : SSM state dimension N
    dt_min/max   : range for learnable timescale Δ
    bidirectional: process sequence in both directions (training only)
    """

    def __init__(
        self,
        d_model:       int,
        d_state:       int   = 64,
        dt_min:        float = 0.001,
        dt_max:        float = 0.1,
        bidirectional: bool  = True,
        dropout:       float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidir   = bidirectional

        A = _hippo_a(d_state)
        B = _hippo_b(d_state)

        # Diagonal S4D parameterisation
        self.A_log = nn.Parameter(
            torch.log(torch.abs(torch.diagonal(A))).unsqueeze(0).expand(d_model, -1).clone()
        )
        self.B = nn.Parameter(B.unsqueeze(0).expand(d_model, -1).clone())
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.out  = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        if bidirectional:
            self.A_log_r = nn.Parameter(self.A_log.clone())
            self.B_r     = nn.Parameter(self.B.clone())
            self.C_r     = nn.Parameter(self.C.clone())
            self.out_bi  = nn.Linear(d_model * 2, d_model)

    def _disc(self, rev: bool = False):
        A_log = self.A_log_r if rev else self.A_log
        B     = self.B_r     if rev else self.B
        C     = self.C_r     if rev else self.C
        dt    = torch.exp(self.log_dt)
        A_bar = torch.exp(dt.unsqueeze(-1) * (-torch.exp(A_log)))
        B_bar = (A_bar - 1) / (-torch.exp(A_log)) * B
        return A_bar, B_bar, C

    def _kernel(self, L: int, rev: bool = False) -> torch.Tensor:
        A_bar, B_bar, C = self._disc(rev)
        CB  = C * B_bar
        pows = A_bar.unsqueeze(-1) ** torch.arange(L, device=A_bar.device).float()
        return torch.einsum("dn,dnl->dl", CB, pows)

    def _conv(self, u: torch.Tensor, rev: bool = False) -> torch.Tensor:
        B, L, H = u.shape
        K   = self._kernel(L, rev)
        u_f = torch.fft.rfft(u, n=2*L, dim=1)
        K_f = torch.fft.rfft(K, n=2*L, dim=-1)
        y   = torch.fft.irfft(u_f * K_f.T.unsqueeze(0), n=2*L, dim=1)[:, :L, :]
        return y + u * self.D.unsqueeze(0).unsqueeze(0)

    def _step(
        self, u: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single recurrence step for inference. O(d_model * d_state)."""
        A_bar, B_bar, C = self._disc()
        B = u.shape[0]
        if state is None:
            state = torch.zeros(B, self.d_model, self.d_state, device=u.device)
        new_state = A_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * u.unsqueeze(-1)
        y = (C.unsqueeze(0) * new_state).sum(-1) + self.D.unsqueeze(0) * u
        return self.out(y), new_state

    def forward(
        self,
        u: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        inference: bool = False,
    ):
        if inference or u.dim() == 2:
            return self._step(u, state)
        y = self._conv(u)
        if self.bidir:
            yr = torch.flip(self._conv(torch.flip(u, [1]), rev=True), [1])
            y  = self.out_bi(torch.cat([y, yr], dim=-1))
        else:
            y = self.out(y)
        return y, None


# ── S4 Block (layer + GLU + residual) ────────────────────────────────────────

class S4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64,
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.s4    = S4Layer(d_model, d_state, bidirectional=bidirectional, dropout=dropout)
        self.glu_v = nn.Linear(d_model, d_model * 2)
        self.glu_g = nn.Linear(d_model, d_model * 2)
        self.glu_o = nn.Linear(d_model, d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, inference: bool = False) -> torch.Tensor:
        h, _ = self.s4(self.norm1(x), inference=inference)
        if isinstance(h, tuple): h = h[0]
        x = x + self.drop(h)
        n   = self.norm2(x)
        v1, v2 = self.glu_v(n).chunk(2, -1)
        g1, g2 = torch.sigmoid(self.glu_g(n)).chunk(2, -1)
        x = x + self.drop(self.glu_o(torch.cat([v1*g1, v2*g2], -1)))
        return x


# ── Full EEG Encoder ─────────────────────────────────────────────────────────

class S4EEGEncoder(nn.Module):
    """
    Full S4-based EEG encoder.

    Pipeline
    --------
    1. Artifact rejection  — learned spatial filter (ICA-inspired, identity init)
    2. Spatial compression — depthwise temporal conv + pointwise spatial mix
    3. Temporal downsample — AvgPool1d by `downsample` factor
    4. S4 blocks           — deep temporal modelling
    5. Attention pooling   — sequence → summary vector
    6. Auxiliary heads     — motor intent (5 classes), cognitive state (5 dims)

    Outputs
    -------
    summary        (B, d_model)      — single token for injection into transformer
    sequence       (B, T', d_model)  — full temporal sequence for cross-attention
    intent_logits  (B, n_intent)
    cognitive      dict of (B,) tensors: workload, attention, arousal, valence, fatigue
    planning_budget (B,)             — [0.2, 1.0] adaptive MCTS budget
    """

    def __init__(
        self,
        n_channels:    int   = 64,
        d_model:       int   = 256,
        d_state:       int   = 64,
        n_blocks:      int   = 4,
        downsample:    int   = 4,
        n_intent:      int   = 5,
        dropout:       float = 0.1,
        bidirectional: bool  = True,
    ):
        super().__init__()
        self.d_model    = d_model
        self.downsample = downsample

        self.artifact_filter = nn.Conv1d(n_channels, n_channels, 1, groups=1, bias=False)
        nn.init.eye_(self.artifact_filter.weight.squeeze(-1))

        self.spatial = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, 25, padding=12, groups=n_channels, bias=False),
            nn.BatchNorm1d(n_channels),
            nn.Conv1d(n_channels, d_model, 1, bias=False),
            nn.GELU(),
        )
        self.ds      = nn.AvgPool1d(downsample, stride=downsample)
        self.blocks  = nn.ModuleList([
            S4Block(d_model, d_state, dropout, bidirectional) for _ in range(n_blocks)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.pool_w  = nn.Linear(d_model, 1)
        self.intent  = nn.Linear(d_model, n_intent)
        self.cog     = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 5), nn.Sigmoid())

    def forward(
        self,
        eeg:            torch.Tensor,
        electrode_mask: Optional[torch.Tensor] = None,
        inference:      bool = False,
    ) -> Dict[str, torch.Tensor]:
        eeg = self.artifact_filter(eeg)
        if electrode_mask is not None:
            eeg = eeg * electrode_mask.unsqueeze(-1)
        eeg = (eeg - eeg.mean(-1, keepdim=True)) / (eeg.std(-1, keepdim=True).clamp(1e-6))

        x = self.ds(self.spatial(eeg)).transpose(1, 2)   # (B, T', d)
        for blk in self.blocks:
            x = blk(x, inference)
        x = self.norm(x)

        w       = F.softmax(self.pool_w(x).squeeze(-1), dim=-1)
        summary = (x * w.unsqueeze(-1)).sum(1)

        cog     = self.cog(summary)
        budget  = (1.0 - 0.4 * cog[:, 0] - 0.4 * cog[:, 4]).clamp(0.2, 1.0)

        return {
            "summary":        summary,
            "sequence":       x,
            "intent_logits":  self.intent(summary),
            "cognitive": {
                "workload":  cog[:, 0], "attention": cog[:, 1],
                "arousal":   cog[:, 2], "valence":   cog[:, 3],
                "fatigue":   cog[:, 4],
            },
            "planning_budget": budget,
        }
