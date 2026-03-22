"""
noosphere/s4_eeg.py
===================
S4 Structured State Space Model — Stream B (EEG)

EEG is processed sample-by-sample at full temporal resolution. No windowing.

Changes in v1.3.0
-----------------
1. BatchNorm → GroupNorm in spatial encoder.
   BatchNorm fails silently at batch_size=1 (inference on a single segment).
   GroupNorm normalises within each sample — works at any batch size.

2. Stable S4 kernel via log-space power computation.
   A_bar.unsqueeze(-1) ** arange(L) accumulates floating-point error for
   large L because powers of values near 1.0 lose mantissa bits progressively.
   Replaced with exp(arange(L) * log(A_bar)) which stays in log space.

3. Continuous xyz head directly on the S4 encoder.
   The 5-class discrete intent head is too coarse for precise coordinate
   prediction. A new `xyz_head` produces a direct 3D coordinate estimate
   from the summary embedding. Gradient flows back through the full S4 stack
   — the encoder learns to represent signals useful for continuous spatial
   intent, not just class separation.
   Output: `continuous_xyz` (B, 3)  — arm tip target in metres

4. Confidence head: scalar uncertainty estimate [0,1] for the xyz prediction.
   Used by ActBridge confidence gate and TemporalSmoother alpha schedule.
   Low confidence → smoother damps prediction → arm moves conservatively.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── HiPPO initialisation ─────────────────────────────────────────────────────


def _hippo_a(N: int) -> torch.Tensor:
    """HiPPO-LegS state matrix A ∈ ℝ^(N×N)."""
    A = torch.zeros(N, N)
    for n in range(N):
        for k in range(n):
            A[n, k] = -math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1)
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
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidir = bidirectional

        A = _hippo_a(d_state)
        B = _hippo_b(d_state)

        self.A_log = nn.Parameter(
            torch.log(torch.abs(torch.diagonal(A)))
            .unsqueeze(0)
            .expand(d_model, -1)
            .clone()
        )
        self.B = nn.Parameter(B.unsqueeze(0).expand(d_model, -1).clone())
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(
            dt_min
        )
        self.log_dt = nn.Parameter(log_dt)

        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

        if bidirectional:
            self.A_log_r = nn.Parameter(self.A_log.clone())
            self.B_r = nn.Parameter(self.B.clone())
            self.C_r = nn.Parameter(self.C.clone())
            self.out_bi = nn.Linear(d_model * 2, d_model)

    def _disc(self, rev: bool = False):
        A_log = self.A_log_r if rev else self.A_log
        B = self.B_r if rev else self.B
        C = self.C_r if rev else self.C
        dt = torch.exp(self.log_dt)
        A_bar = torch.exp(dt.unsqueeze(-1) * (-torch.exp(A_log)))
        B_bar = (A_bar - 1) / (-torch.exp(A_log)) * B
        return A_bar, B_bar, C

    def _kernel(self, L: int, rev: bool = False) -> torch.Tensor:
        """
        Compute convolution kernel of length L.
        Uses log-space exponentiation for numerical stability:
            A_bar^k = exp(k * log(A_bar))
        This avoids progressive mantissa loss when computing A_bar^L
        for large L (e.g. L=256 at 256Hz sampling rate).
        """
        A_bar, B_bar, C = self._disc(rev)
        CB = C * B_bar  # (d_model, d_state)
        # Stable: exp(k * log(A_bar)) rather than A_bar ** k
        log_A = torch.log(A_bar.clamp(min=1e-30))  # (d_model, d_state)
        k = torch.arange(L, device=A_bar.device, dtype=A_bar.dtype)  # (L,)
        pows = torch.exp(k.view(1, 1, L) * log_A.unsqueeze(-1))  # (d, N, L)
        return torch.einsum("dn,dnl->dl", CB, pows)  # (d, L)

    def _conv(self, u: torch.Tensor, rev: bool = False) -> torch.Tensor:
        B, L, H = u.shape
        K = self._kernel(L, rev)
        u_f = torch.fft.rfft(u, n=2 * L, dim=1)
        K_f = torch.fft.rfft(K, n=2 * L, dim=-1)
        y = torch.fft.irfft(u_f * K_f.T.unsqueeze(0), n=2 * L, dim=1)[:, :L, :]
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
            y = self.out_bi(torch.cat([y, yr], dim=-1))
        else:
            y = self.out(y)
        return y, None


# ── S4 Block (layer + GLU + residual) ────────────────────────────────────────


class S4Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.s4 = S4Layer(
            d_model, d_state, bidirectional=bidirectional, dropout=dropout
        )
        self.glu_v = nn.Linear(d_model, d_model * 2)
        self.glu_g = nn.Linear(d_model, d_model * 2)
        self.glu_o = nn.Linear(d_model * 2, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, inference: bool = False) -> torch.Tensor:
        # S4 sub-layer
        h, _ = self.s4(self.norm1(x), inference=inference)
        # h is always a tensor here — _step and _conv both return tensors, not tuples
        x = x + self.drop(h)
        # GLU sub-layer
        n = self.norm2(x)
        v1, v2 = self.glu_v(n).chunk(2, -1)
        g1, g2 = torch.sigmoid(self.glu_g(n)).chunk(2, -1)
        x = x + self.drop(self.glu_o(torch.cat([v1 * g1, v2 * g2], -1)))
        return x


# ── Full EEG Encoder ─────────────────────────────────────────────────────────


class S4EEGEncoder(nn.Module):
    """
    Full S4-based EEG encoder.

    Pipeline
    --------
    1. Artifact rejection  — learned spatial filter (identity init)
    2. Spatial compression — depthwise temporal conv + pointwise mix
                             Uses GroupNorm (not BatchNorm) so inference
                             at batch_size=1 works correctly.
    3. Temporal downsample — AvgPool1d
    4. S4 blocks           — deep temporal modelling
    5. Attention pooling   — sequence → summary vector
    6. Output heads:
        intent_logits   (B, n_intent)  — coarse discrete class (5)
        continuous_xyz  (B, 3)         — direct continuous coordinate prediction
        confidence      (B,)           — prediction certainty [0,1]
        cognitive       dict of (B,) scalars
        planning_budget (B,)

    continuous_xyz is the precision improvement over discrete intent_logits.
    Gradient flows back through the full S4 stack from coordinate supervision,
    so the encoder learns signal representations optimised for spatial precision,
    not just class separation.

    confidence is used by:
        - TemporalSmoother: high uncertainty → higher α (more damping)
        - ActBridge: low confidence → hold rather than act
    """

    def __init__(
        self,
        n_channels: int = 3,
        d_model: int = 256,
        d_state: int = 64,
        n_blocks: int = 4,
        downsample: int = 4,
        n_intent: int = 5,
        dropout: float = 0.1,
        bidirectional: bool = True,
        max_reach: float = 0.70,
    ):
        super().__init__()
        self.d_model = d_model
        self.downsample = downsample
        self.max_reach = max_reach

        # Learned spatial filter (identity init — no information destroyed at start)
        self.artifact_filter = nn.Conv1d(n_channels, n_channels, 1, bias=False)
        nn.init.eye_(self.artifact_filter.weight.squeeze(-1))

        # Spatial encoder.
        # GroupNorm(groups=n_channels) normalises each channel independently,
        # equivalent to InstanceNorm for 1-D conv — works at any batch size.
        self.spatial = nn.Sequential(
            nn.Conv1d(
                n_channels, n_channels, 25, padding=12, groups=n_channels, bias=False
            ),
            nn.GroupNorm(n_channels, n_channels),
            nn.Conv1d(n_channels, d_model, 1, bias=False),
            nn.GELU(),
        )
        self.ds = nn.AvgPool1d(downsample, stride=downsample)
        self.blocks = nn.ModuleList(
            [S4Block(d_model, d_state, dropout, bidirectional) for _ in range(n_blocks)]
        )
        self.norm = nn.LayerNorm(d_model)

        # Attention pooling: sequence → fixed-size summary
        self.pool_w = nn.Linear(d_model, 1)

        # ── Output heads ──────────────────────────────────────────────────────

        # Coarse discrete intent (5 classes) — fast, low-resolution
        self.intent = nn.Linear(d_model, n_intent)

        # Continuous 3D coordinate prediction — precise, high-resolution
        # Two-layer MLP with bottleneck; tanh output scaled to arm reach
        self.xyz_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3),
            nn.Tanh(),
        )

        # Confidence head: scalar uncertainty [0, 1]
        # Trained implicitly — high spread in xyz predictions → low confidence
        # Also accepts direct supervision if ground-truth is available.
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Cognitive state: 5 dims all in [0,1]
        self.cog = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 5),
            nn.Sigmoid(),
        )

    def forward(
        self,
        eeg: torch.Tensor,
        electrode_mask: Optional[torch.Tensor] = None,
        inference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Artifact filter + electrode masking
        eeg = self.artifact_filter(eeg)
        if electrode_mask is not None:
            eeg = eeg * electrode_mask.unsqueeze(-1)

        # Per-sample normalisation (not per-batch, so it works at any batch size)
        eeg = (eeg - eeg.mean(-1, keepdim=True)) / (
            eeg.std(-1, keepdim=True).clamp(1e-6)
        )

        # Spatial encode + downsample
        x = self.ds(self.spatial(eeg)).transpose(1, 2)  # (B, T', d_model)

        # S4 temporal modelling
        for blk in self.blocks:
            x = blk(x, inference)
        x = self.norm(x)

        # Attention-weighted pooling → (B, d_model)
        w = F.softmax(self.pool_w(x).squeeze(-1), dim=-1)
        summary = (x * w.unsqueeze(-1)).sum(1)

        # Continuous coordinate: tanh output scaled to arm reach
        # xyz is in [-max_reach, +max_reach] per axis
        xyz = self.xyz_head(summary) * self.max_reach  # (B, 3)

        # Confidence and cognitive state
        conf = self.conf_head(summary).squeeze(-1)  # (B,)
        cog = self.cog(summary)  # (B, 5)

        # Planning budget: reduce MCTS sims when tired or overloaded
        budget = (1.0 - 0.4 * cog[:, 0] - 0.4 * cog[:, 4]).clamp(0.2, 1.0)

        return {
            "summary": summary,
            "sequence": x,
            "intent_logits": self.intent(summary),
            "continuous_xyz": xyz,
            "confidence": conf,
            "cognitive": {
                "workload": cog[:, 0],
                "attention": cog[:, 1],
                "arousal": cog[:, 2],
                "valence": cog[:, 3],
                "fatigue": cog[:, 4],
            },
            "planning_budget": budget,
        }
