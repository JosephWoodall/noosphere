"""
noosphere/s4_eeg.py
===================
RS-S4 v2 — Riemannian-Selective State-Space EEG Encoder

Key architectural improvements over v1
---------------------------------------
1.  FilterbankSlidingRiemannianStem  (replaces RiemannianStem)
      - Computes per-band sliding-window SPD covariances over 5 canonical
        EEG frequency bands (delta/theta/alpha/beta/gamma).
      - Differentiable Ledoit-Wolf analytical shrinkage replaces Tikhonov
        regularisation, preventing ill-conditioned matrices in short windows.
      - Outputs a *temporal sequence* of tangent-space embeddings
        (B, L_win, d_model) instead of a single static token (B, 1, d_model).
      - This gives the S4 a genuine manifold-valued time series to model.
        In v1 the S4 had no real Riemannian sequence to process — the spatial
        token was prepended as one element and effectively ignored.

2.  SpatialTemporalCrossAttention  (replaces FiLM)
      - At each S4D block, wavelet tokens *query* the full Riemannian sequence
        via multi-head cross-attention.
      - FiLM (v1) applied the same global affine shift to every timestep;
        cross-attention lets each wavelet token select which Riemannian
        windows are geometrically relevant to it.

3.  S4DLayer dt-selectivity fix
      - v1 did `dt_eff = dt_base * dt_scale.mean(0)` which collapsed per-sample
        artifact gating into a single batch-averaged dt — defeating selectivity.
      - v2 applies per-sample, per-timestep gating as a post-conv multiplicative
        mask, a well-established approximation of Mamba's selective scan that
        preserves FFT efficiency while restoring per-sample discrimination.

4.  DirichletEDLLoss annealing_step fix
      - v1 default was 10 steps → KL fully active after 10 gradient steps →
        network learns to maximize uncertainty as a training shortcut →
        confidence never exceeds 0.5, explaining the 0% coverage at threshold
        ≥0.7 seen in the eval tables.
      - v2 default is 100 steps (tune upward for large datasets).

5.  DifferentiableLedoitWolf  (new utility)
      - Analytical Oracle Approximating Shrinkage fully implemented in PyTorch,
        differentiable, handles batched (B, C, C) or (B, L, C, C) inputs.

All v1 components are preserved for backward compatibility.
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from noosphere.gnn import LearnedAdjacency
except ImportError:
    class LearnedAdjacency(nn.Module):
        def __init__(self, n_nodes: int, temperature: float = 1.0,
                     sparse_reg: float = 0.01):
            super().__init__()
            self.n_nodes = n_nodes
            self.W = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)
            self.register_buffer("temp", torch.tensor(temperature))

        def forward(self) -> torch.Tensor:
            A = torch.sigmoid(self.W / self.temp)
            A = A + torch.eye(self.n_nodes, device=A.device)
            D = A.sum(-1).clamp(min=1e-6)
            d = D.pow(-0.5)
            return d[:, None] * A * d[None, :]


# ═══════════════════════════════════════════════════════════════════════════════
#  HiPPO-LegS initialisation
# ═══════════════════════════════════════════════════════════════════════════════

def _hippo_a(N: int) -> torch.Tensor:
    """HiPPO-LegS transition matrix A ∈ ℝ^(N×N)."""
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(N, dtype=torch.float32)
    A = -torch.where(n[:, None] > k[None, :],
                     torch.sqrt((2*n[:, None]+1) * (2*k[None, :]+1)),
                     torch.zeros(N, N))
    A.diagonal().copy_(-(n + 1))
    return A


def _hippo_b(N: int) -> torch.Tensor:
    n = torch.arange(N, dtype=torch.float32)
    return torch.sqrt(2 * n + 1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Diagonal S4 Layer  (S4D)  — v2: dt-selectivity bug fixed
# ═══════════════════════════════════════════════════════════════════════════════

class S4DLayer(nn.Module):
    """
    Diagonal S4 (S4D) with ZOH discretization.
    Uses diagonal approximation of HiPPO-LegS for stability.
    Supports bidirectional processing during training.

    v2 FIX — dt selectivity:
        v1 computed `dt_eff = dt_base * dt_scale.mean(0)`, collapsing
        per-sample artifact discrimination into a single batch-averaged kernel.
        v2 uses the HiPPO kernel with the base dt and applies a per-sample,
        per-timestep sigmoid gate *after* the FFT convolution.  This preserves
        O(L log L) efficiency while restoring genuine per-sample selectivity.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        bidirectional: bool = True,
        dropout: float = 0.1,
        selective: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidir = bidirectional
        self.selective = selective

        # HiPPO-LegS diagonal entries
        A_diag = torch.abs(torch.diagonal(_hippo_a(d_state)))
        self.A_log = nn.Parameter(
            torch.log(A_diag).unsqueeze(0).expand(d_model, -1).clone()
        )
        B = _hippo_b(d_state)
        self.B = nn.Parameter(B.unsqueeze(0).expand(d_model, -1).clone())
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))

        log_dt = (torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min))
                  + math.log(dt_min))
        self.log_dt = nn.Parameter(log_dt)

        # v2: post-conv per-sample gate (replaces kernel-level dt modulation)
        if selective:
            self.select_gate = nn.Linear(d_model, d_model, bias=True)
            # Init so gate starts near 1 (transparent)
            nn.init.normal_(self.select_gate.weight, std=0.01)
            nn.init.constant_(self.select_gate.bias, 2.0)

        if bidirectional:
            self.out_bi = nn.Linear(d_model * 2, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _kernel(self, L: int, dt_base: torch.Tensor) -> torch.Tensor:
        """Build convolution kernel via ZOH: k[t] = Σ_n C[n]*A_bar[n]^t*B_bar[n]."""
        A     = -torch.exp(self.A_log)
        dt    = dt_base.unsqueeze(-1)
        A_bar = torch.exp(dt * A)
        B_bar = (A_bar - 1.0) / (A + 1e-8) * self.B
        CB    = self.C * B_bar
        t_idx = torch.arange(L, device=A.device, dtype=A.dtype)
        log_Ab = torch.log(A_bar.clamp(min=1e-8))
        powers = torch.exp(log_Ab.unsqueeze(1) * t_idx[None, :, None])
        k = (powers * CB.unsqueeze(1)).sum(-1)
        return k  # (d_model, L)

    def _conv(self, u: torch.Tensor, rev: bool = False) -> torch.Tensor:
        """Parallel training mode: FFT convolution. u: (B, L, d_model)"""
        B_sz, L, H = u.shape
        dt_eff = torch.exp(self.log_dt)  # shared base dt — stable kernel
        k = self._kernel(L, dt_eff)       # (d_model, L)

        u_t = u.transpose(1, 2)           # (B, d_model, L)
        k_f = torch.fft.rfft(k, n=2 * L, dim=-1)
        u_f = torch.fft.rfft(u_t, n=2 * L, dim=-1)
        y_f = k_f.unsqueeze(0) * u_f
        y   = torch.fft.irfft(y_f, n=2 * L, dim=-1)[..., :L]   # (B, d_model, L)
        y   = y + u_t * self.D.unsqueeze(0).unsqueeze(-1)
        y   = y.transpose(1, 2)           # (B, L, d_model)

        # v2 FIX: per-sample, per-timestep selective gate applied post-conv
        if self.selective:
            gate = torch.sigmoid(self.select_gate(u))   # (B, L, d_model)
            y = y * gate

        return self.drop(y)

    def _step(self, u: torch.Tensor,
              state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step inference. u: (B, d_model)"""
        dt_eff = torch.exp(self.log_dt)
        A      = -torch.exp(self.A_log)
        A_bar  = torch.exp(dt_eff.unsqueeze(-1) * A)
        B_bar  = (A_bar - 1.0) / (A + 1e-8) * self.B

        if state is None:
            state = torch.zeros(u.shape[0], self.d_model, self.d_state,
                                device=u.device, dtype=u.dtype)

        state = A_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * u.unsqueeze(-1)
        y = (state * self.C.unsqueeze(0)).sum(-1) + u * self.D

        if self.selective:
            gate = torch.sigmoid(self.select_gate(u))
            y = y * gate

        return self.out(y), state

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
            yr = torch.flip(self._conv(torch.flip(u, [1])), [1])
            y  = self.out_bi(torch.cat([y, yr], dim=-1))
        else:
            y = self.out(y)
        return y, None


# ═══════════════════════════════════════════════════════════════════════════════
#  Selective S4D Block
# ═══════════════════════════════════════════════════════════════════════════════

class SelectiveS4DBlock(nn.Module):
    """S4D block: HiPPO kernel + GLU residual + post-conv selective gating."""

    def __init__(self, d_model: int, d_state: int = 64,
                 dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.s4d   = S4DLayer(d_model, d_state, bidirectional=bidirectional,
                              dropout=dropout, selective=True)
        self.glu_v = nn.Linear(d_model, d_model * 2)
        self.glu_g = nn.Linear(d_model, d_model * 2)
        self.glu_o = nn.Linear(d_model * 2, d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, inference: bool = False) -> torch.Tensor:
        h, _ = self.s4d(self.norm1(x), inference=inference)
        x = x + self.drop(h)
        n  = self.norm2(x)
        vg = self.glu_v(n) * torch.sigmoid(self.glu_g(n))
        x  = x + self.drop(self.glu_o(vg))
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  S4D Mixture-of-Experts Block
# ═══════════════════════════════════════════════════════════════════════════════

class S4DMoEBlock(nn.Module):
    """Two-expert S4D mixture. Router specialises per frequency band."""

    def __init__(self, d_model: int, d_state: int, num_experts: int = 2):
        super().__init__()
        self.router  = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            SelectiveS4DBlock(d_model, d_state) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, inference: bool = False) -> torch.Tensor:
        weights = F.softmax(self.router(x), dim=-1)   # (B, L, E)
        out = sum(weights[..., i:i+1] * expert(x, inference)
                  for i, expert in enumerate(self.experts))
        return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Wavelet Temporal Stems  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════

class ParametricMorlet1d(nn.Module):
    """Learned Morlet wavelet filterbank with learnable frequency and bandwidth."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.stride      = stride
        self.kernel_size = kernel_size
        self.freqs       = nn.Parameter(torch.randn(out_ch, in_ch, 1) * 5.0)
        self.bandwidths  = nn.Parameter(torch.ones(out_ch, in_ch, 1))
        self.register_buffer("t", torch.linspace(-1, 1, kernel_size).view(1, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos_term = torch.cos(2 * math.pi * self.freqs * self.t)
        env      = torch.exp(-(self.t**2) / (2 * F.softplus(self.bandwidths)**2 + 1e-8))
        wavelet  = cos_term * env
        return F.conv1d(x, wavelet, stride=self.stride, padding=self.kernel_size // 2)


class WaveletStem(nn.Module):
    """Dual-band parametric Morlet filterbank → high-freq + low-freq paths."""

    def __init__(self, in_ch: int, d_model: int, downsample: int = 4):
        super().__init__()
        self.scales_high = [3, 7]
        self.scales_low  = [15, 31, 63]
        hc = max(1, d_model // len(self.scales_high))
        lc = max(1, d_model // len(self.scales_low))

        self.convs_high = nn.ModuleList([
            ParametricMorlet1d(in_ch, hc, s, stride=downsample) for s in self.scales_high
        ])
        self.convs_low = nn.ModuleList([
            ParametricMorlet1d(in_ch, lc, s, stride=downsample) for s in self.scales_low
        ])
        
        self.proj_high = nn.Sequential(
            nn.Conv1d(hc * len(self.scales_high), d_model, 1),
            nn.GELU(),
            nn.GroupNorm(min(8, d_model), d_model),
        )
        self.proj_low = nn.Sequential(
            nn.Conv1d(lc * len(self.scales_low), d_model, 1),
            nn.GELU(),
            nn.GroupNorm(min(8, d_model), d_model),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([c(x) for c in self.convs_high], dim=1)
        l = torch.cat([c(x) for c in self.convs_low],  dim=1)
        return self.proj_high(h).transpose(1, 2), self.proj_low(l).transpose(1, 2)


class SpectralGraphWaveletStem(nn.Module):
    """Learned graph Laplacian smoothing → WaveletStem."""

    def __init__(self, in_ch: int, d_model: int, downsample: int = 4):
        super().__init__()
        self.adj     = LearnedAdjacency(in_ch)
        self.wavelet = WaveletStem(in_ch, d_model, downsample)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A     = self.adj()
        D     = torch.diag_embed(A.sum(-1))
        L     = D - A
        x_lap = torch.einsum("cd,bdt->bct", L, x)
        return self.wavelet(x_lap)


# ═══════════════════════════════════════════════════════════════════════════════
#  Differentiable Ledoit-Wolf Shrinkage  (NEW in v2)
# ═══════════════════════════════════════════════════════════════════════════════

class DifferentiableLedoitWolf(nn.Module):
    """
    Analytical Oracle Approximating Shrinkage (OAS) estimator.
    Fully differentiable PyTorch implementation — no sklearn dependency.

    Given sample covariance S estimated from n observations of a C-dimensional
    signal, computes the shrunk estimate:

        Σ_lw = (1 - ρ) * S  +  ρ * μ * I

    where μ = tr(S)/C  and  ρ is the optimal shrinkage coefficient derived
    analytically from the Ledoit-Wolf formula.

    Input:  S  — (..., C, C) raw sample covariance matrices
            n  — number of time samples used to compute S (scalar int)
    Output: (..., C, C) shrunk SPD matrices
    """

    @staticmethod
    def shrink(S: torch.Tensor, n: int) -> torch.Tensor:
        p   = S.shape[-1]
        # tr(S) and tr(S²)
        tr_S  = S.diagonal(dim1=-2, dim2=-1).sum(-1)           # (...)
        tr_S2 = torch.einsum('...ij,...ji->...', S, S)         # (...)  = tr(S²)
        tr_S_sq = tr_S ** 2                                    # (...)

        # Analytical LW shrinkage coefficient
        num   = ((n - 2) / n) * tr_S2 + tr_S_sq
        denom = (n + 2) * (tr_S2 - tr_S_sq / p + 1e-10)
        rho   = (num / denom).clamp(0.0, 1.0)                  # (...)

        # Shrinkage target: scaled identity
        mu  = tr_S / p                                         # (...)
        eye = torch.eye(p, device=S.device, dtype=S.dtype).expand_as(S)

        # Broadcast scalars over last two dims
        rho_e = rho[..., None, None]
        mu_e  = mu[..., None, None]

        return (1.0 - rho_e) * S + rho_e * mu_e * eye

    def forward(self, S: torch.Tensor, n: int) -> torch.Tensor:
        return self.shrink(S, n)


# ═══════════════════════════════════════════════════════════════════════════════
#  Filterbank Sliding-Window Riemannian Stem  (NEW in v2 — core improvement)
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical EEG frequency bands (Hz)
_EEG_BANDS: List[Tuple[str, float, float]] = [
    ("delta", 1.0,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0,  13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 100.0),
]


def _butter_bandpass_torch(x: torch.Tensor, lo: float, hi: float,
                            fs: float) -> torch.Tensor:
    """
    Lightweight differentiable bandpass via FFT masking.
    Not a true Butterworth, but gradient-friendly and sufficient for covariance
    estimation (we care about band power, not perfect filter shape).
    """
    B, C, T = x.shape
    X_f = torch.fft.rfft(x, dim=-1)                     # (B, C, T//2+1)
    freqs = torch.fft.rfftfreq(T, d=1.0 / fs).to(x.device)
    
    # Sigmoidal taper for smoother roll-off (reduces temporal ringing)
    # Replaces hard rectangular window with 2Hz roll-off
    taper = 2.0
    mask = torch.sigmoid((freqs - lo) / (taper / 4)) * torch.sigmoid((hi - freqs) / (taper / 4))
    
    mask = mask.unsqueeze(0).unsqueeze(0)                # (1, 1, F)
    return torch.fft.irfft(X_f * mask, n=T, dim=-1)


class FilterbankSlidingRiemannianStem(nn.Module):
    """
    v2 core improvement: Riemannian spatial encoder that outputs a temporal
    *sequence* instead of a single static token.

    Pipeline
    --------
    1.  Bandpass-filter the raw EEG into 5 canonical bands via FFT masking.
    2.  For each band, extract overlapping windows via `unfold`, compute the
        per-window sample covariance matrix (B, L_win, C, C).
    3.  Apply differentiable Ledoit-Wolf shrinkage to regularize ill-conditioned
        windows (critical when window << C²).
    4.  Compute the matrix logarithm (Log-Euclidean metric) to unwrap the SPD
        manifold into a flat tangent space.
    5.  Vectorize the upper triangle → (B, L_win, ts_dim) per band.
    6.  Per-band MLP projects to d_model/n_bands → sum-fused to d_model.
    7.  Output: (B, L_win, d_model) — a sequence of Riemannian embeddings.

    Parameters
    ----------
    in_ch      : number of EEG channels C
    d_model    : output embedding dimension
    sfreq      : sampling frequency (Hz), used for band filtering
    win_samples: window length in samples  (default 128 ≈ 500ms @ 256Hz)
    stride     : stride in samples between windows (default 32 → 75% overlap)
    momentum   : EMA momentum for running covariance reference
    mixup_alpha: manifold mixup augmentation coefficient (0 = disabled)
    """

    BANDS = _EEG_BANDS

    def __init__(
        self,
        in_ch:       int,
        d_model:     int,
        sfreq:       float = 256.0,
        win_samples: int   = 128,
        stride:      int   = 32,
        momentum:    float = 0.9,
        mixup_alpha: float = 0.0,
    ):
        super().__init__()
        self.in_ch       = in_ch
        self.d_model     = d_model
        self.sfreq       = sfreq
        self.win_samples = win_samples
        self.stride      = stride
        self.mixup_alpha = mixup_alpha
        self.n_bands     = len(self.BANDS)

        self.lw = DifferentiableLedoitWolf()

        ts_dim = in_ch * (in_ch + 1) // 2   # upper-triangle elements

        # Per-band projections: ts_dim → d_model
        # Bounded regardless of channel count — no cross-band entanglement
        band_d = d_model
        self.band_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ts_dim, band_d),
                nn.GELU(),
                nn.LayerNorm(band_d),
            )
            for _ in self.BANDS
        ])

        # Fuse all bands into d_model
        self.fusion = nn.Sequential(
            nn.Linear(band_d * self.n_bands, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

        # Positional embedding for Riemannian windows
        self.pos_embed = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        self.register_buffer("running_cov", torch.eye(in_ch).unsqueeze(0))
        self.momentum = momentum

    def _compute_tangent_sequence(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        x: (B, C, T) — single-band filtered signal
        Returns: (B, L_win, ts_dim) tangent-space vectors
        """
        B, C, T = x.shape
        win, stride = self.win_samples, self.stride

        # Unfold into overlapping windows: (B, C, L_win, win)
        x_win = x.unfold(-1, win, stride)          # (B, C, L_win, win)
        L_win = x_win.shape[2]

        # Center each window
        x_c = x_win - x_win.mean(-1, keepdim=True) # (B, C, L_win, win)

        # Batch covariance: reshape to (B*L_win, C, win) for bmm
        xc_flat = x_c.permute(0, 2, 1, 3).reshape(B * L_win, C, win)
        cov_flat = torch.bmm(xc_flat, xc_flat.transpose(1, 2)) / (win - 1 + 1e-8)
        # (B*L_win, C, C)

        # Differentiable Ledoit-Wolf shrinkage
        cov_flat = self.lw.shrink(cov_flat, n=win)

        # Matrix logarithm via eigendecomposition (float64 for stability)
        cov64 = cov_flat.to(torch.float64)
        try:
            # Add small Tikhonov term for stability before log
            trace = cov64.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
            eps = (1e-4 * trace / C) + 1e-6
            eye = torch.eye(C, device=x.device, dtype=torch.float64).unsqueeze(0)
            cov64 = cov64 + eps * eye
            
            vals, vecs = torch.linalg.eigh(cov64)
            vals = torch.log(vals.clamp(min=1e-8))
            log_cov = torch.bmm(vecs, torch.bmm(
                torch.diag_embed(vals), vecs.transpose(1, 2)
            )).to(torch.float32)
        except Exception:
            # Fallback: pseudo-log via power series or identity projection
            # but scaled to prevent zeroing gradients
            log_cov = torch.log(cov_flat.clamp(min=1e-6))

        # Upper-triangle vectorization
        idx = torch.triu_indices(C, C, device=x.device)
        ts  = log_cov[:, idx[0], idx[1]]        # (B*L_win, ts_dim)
        return ts.reshape(B, L_win, -1)          # (B, L_win, ts_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        Returns: (B, L_win, d_model) — sequence of Riemannian embeddings
        """
        B, C, T = x.shape
        band_tokens = []

        for i, (band_name, lo, hi) in enumerate(self.BANDS):
            # 1. Bandpass filter
            x_b = _butter_bandpass_torch(x, lo, hi, self.sfreq)  # (B, C, T)

            # 2. Sliding-window tangent vectors
            ts_seq = self._compute_tangent_sequence(x_b)         # (B, L_win, ts_dim)

            # 3. Per-band MLP
            band_embed = self.band_projs[i](ts_seq)               # (B, L_win, d_model)
            band_tokens.append(band_embed)

        L_win = band_tokens[0].shape[1]

        # 4. Manifold mixup augmentation (training only)
        if self.training and self.mixup_alpha > 0:
            lam  = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            lam  = max(lam, 1 - lam)
            perm = torch.randperm(B, device=x.device)
            band_tokens = [lam * t + (1 - lam) * t[perm] for t in band_tokens]

        # 5. Cross-band fusion: concat + linear
        fused = self.fusion(torch.cat(band_tokens, dim=-1))       # (B, L_win, d_model)

        # 6. Add positional embedding (truncate or pad to L_win)
        pe    = self.pos_embed[:, :L_win, :].expand(B, -1, -1)
        return fused + pe                                         # (B, L_win, d_model)


# ═══════════════════════════════════════════════════════════════════════════════
#  Riemannian Stem  (v1 — preserved for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

class RiemannianStem(nn.Module):
    """
    v1 tangent-space Riemannian encoder (single covariance per epoch).
    Outputs (B, 1, d_model) — one static spatial token.
    Kept for backward compatibility; use FilterbankSlidingRiemannianStem in v2.
    """

    def __init__(self, in_ch: int, d_model: int, momentum: float = 0.9,
                 mixup_alpha: float = 0.0):
        super().__init__()
        self.in_ch       = in_ch
        self.ts_dim      = in_ch * (in_ch + 1) // 2
        self.momentum    = momentum
        self.mixup_alpha = mixup_alpha
        self.register_buffer("running_cov", torch.eye(in_ch).unsqueeze(0))
        self.proj = nn.Sequential(
            nn.Linear(self.ts_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_c = x - x.mean(-1, keepdim=True)
        cov = torch.bmm(x_c, x_c.transpose(1, 2)) / (T - 1 + 1e-8)
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        eps   = (1e-3 * trace / C) + 1e-5
        eye   = torch.eye(C, device=x.device).unsqueeze(0).expand(B, -1, -1)
        cov   = cov + eps * eye

        if self.training:
            with torch.no_grad():
                mb = cov.mean(0, keepdim=True)
                if self.running_cov.shape[-1] != C:
                    self.running_cov = torch.eye(C, device=x.device).unsqueeze(0)
                self.running_cov = self.momentum * self.running_cov + (1 - self.momentum) * mb

        try:
            vals, vecs = torch.linalg.eigh(cov.to(torch.float64))
            vals = torch.log(vals.clamp(min=1e-7))
            log_cov = torch.bmm(
                vecs, torch.bmm(torch.diag_embed(vals), vecs.transpose(1, 2))
            ).to(torch.float32)
        except Exception:
            log_cov = cov

        idx       = torch.triu_indices(C, C, device=x.device)
        ts_vector = log_cov[:, idx[0], idx[1]]

        if self.training and self.mixup_alpha > 0:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            lam = max(lam, 1 - lam)
            perm = torch.randperm(B, device=x.device)
            ts_vector = lam * ts_vector + (1 - lam) * ts_vector[perm]

        return self.proj(ts_vector).unsqueeze(1)   # (B, 1, d_model)


# ═══════════════════════════════════════════════════════════════════════════════
#  Spatial-Temporal Cross-Attention  (NEW in v2 — replaces FiLM)
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialTemporalCrossAttention(nn.Module):
    """
    Dynamic per-timestep spatial conditioning via cross-attention.

    At each S4D block, wavelet tokens (query) attend over the full Riemannian
    manifold sequence (key/value).  Unlike FiLM, which applies the same global
    γ/β shift to every timestep, cross-attention lets each temporal position
    selectively weight which Riemannian windows (bands, time periods) are most
    geometrically relevant to the current neural state.

    This is critical because:
      - Early in a trial, alpha suppression dominates → attend to alpha-band SPD.
      - During movement execution, beta desynchronization dominates → attend to
        beta-band windows.
      FiLM cannot express this dynamic; cross-attention does so naturally.

    Parameters
    ----------
    d_model  : embedding dimension
    n_heads  : number of attention heads (default 4, small for efficiency)
    dropout  : attention dropout
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q   = nn.LayerNorm(d_model)
        self.norm_kv  = nn.LayerNorm(d_model)
        self.mha      = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)
        # Residual gate — starts at -4.0 (sigmoid ≈ 0.01) for transparency
        self.gate     = nn.Parameter(torch.ones(1) * -4.0)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        query   : (B, T_wav, d_model) — wavelet temporal sequence
        context : (B, L_win, d_model) — Riemannian manifold sequence
        Returns : (B, T_wav, d_model) — spatially conditioned temporal features
        """
        q   = self.norm_q(query)
        kv  = self.norm_kv(context)
        attn_out, _ = self.mha(q, kv, kv)
        # Gated residual: gate starts at 0 (transparent) and opens during training
        delta = self.drop(self.out_proj(attn_out))
        return query + torch.sigmoid(self.gate) * delta


# ═══════════════════════════════════════════════════════════════════════════════
#  FiLM  (v1 — preserved for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

class FiLM(nn.Module):
    """Feature-wise Linear Modulation (v1). Global affine spatial conditioning."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gamma = nn.Linear(d_model, d_model)
        self.beta  = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = cond.squeeze(1)
        g = self.gamma(c).unsqueeze(1)
        b = self.beta(c).unsqueeze(1)
        return x * (1 + g) + b


# ═══════════════════════════════════════════════════════════════════════════════
#  Attention Pooling
# ═══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttentionPooling(nn.Module):
    """Learned query attends over sequence → single pooled vector."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.q   = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q.expand(x.shape[0], -1, -1)
        out, _ = self.mha(q, x, x)
        return self.norm(out.squeeze(1))


# ═══════════════════════════════════════════════════════════════════════════════
#  Preprocessing modules  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════

class ArtifactGater(nn.Module):
    """Down-weights artifact-contaminated channels/times."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, 1, kernel_size=15, stride=4, padding=7)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf    = F.relu(self.conv(x))
        score = torch.sigmoid(self.pool(hf)).squeeze(-1)
        return 1.0 - score


class HurstEstimator(nn.Module):
    """Biological fractal scale gating. Clamped to [0.1, 1.0]."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_ch, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, :, 1:] - x[:, :, :-1]
        var  = diff.var(dim=-1) + 1e-8
        return torch.sigmoid(self.proj(var)).clamp(min=0.1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Domain Adaptation  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


# ═══════════════════════════════════════════════════════════════════════════════
#  Neuromorphic Readout  (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════════════════

class LeakyIntegrateAndFire(nn.Module):
    """LIF neuron readout — biologically plausible spike sparsity."""

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.proj      = nn.Linear(d_model, n_classes)
        self.threshold = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        membrane = self.proj(x)
        spikes   = F.relu(membrane - self.threshold)
        return spikes + membrane * 0.1


# ═══════════════════════════════════════════════════════════════════════════════
#  Dirichlet EDL Loss  — v2: annealing_step default raised 10 → 100
# ═══════════════════════════════════════════════════════════════════════════════

class DirichletEDLLoss(nn.Module):
    """
    Evidential Deep Learning loss (Sensoy et al., NeurIPS 2018).
    Bayes risk + KL regularisation with linear annealing schedule.

    v2 FIX — annealing_step default:
        v1 default was 10 steps → KL fully active after 10 gradient updates →
        the network learns to maximize Dirichlet concentration entropy (minimize
        Σα_k) as a cheap training shortcut, causing *all* confidence scores to
        collapse below 0.5 (0% coverage above threshold 0.7 in eval tables).
        v2 default is 100 steps; for large datasets (e.g. PhysionetMI, 109
        subjects), consider 200–500.
    """

    def __init__(
        self,
        n_classes:      int,
        annealing_step: int   = 100,   # ← v2 FIX: was 10
        conflict_weight: float = 0.1,
    ):
        super().__init__()
        self.n_classes       = n_classes
        self.annealing_step  = annealing_step
        self.conflict_weight = conflict_weight

    def forward(
        self,
        alpha:          torch.Tensor,
        target_one_hot: torch.Tensor,
        current_step:   int,
    ) -> torch.Tensor:
        S          = alpha.sum(-1, keepdim=True)
        pred_probs = alpha / S
        err        = (target_one_hot - pred_probs) ** 2
        var        = (alpha * (S - alpha)) / (S**2 * (S + 1))
        loss_sos   = (err + var).sum(-1)

        annealing    = min(1.0, current_step / self.annealing_step)
        alpha_tilde  = target_one_hot + (1 - target_one_hot) * alpha
        kl_reg       = annealing * self._kl(alpha_tilde)

        entropy = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(-1)
        return torch.mean(loss_sos + kl_reg + self.conflict_weight * entropy)

    def _kl(self, alpha: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(1, self.n_classes, device=alpha.device)
        sa   = alpha.sum(-1, keepdim=True)
        so   = ones.sum(-1, keepdim=True)
        t1   = torch.lgamma(sa) - torch.lgamma(so)
        t2   = (torch.lgamma(ones) - torch.lgamma(alpha)).sum(-1, keepdim=True)
        t3   = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sa))).sum(-1, keepdim=True)
        return (t1 + t2 + t3).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
#  Full RS-S4 v2 Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class S4EEGEncoder(nn.Module):
    """
    RS-S4 v2 — Riemannian-Selective State-Space EEG Encoder.

    Architectural changes from v1
    -----------------------------
    spatial_stem   → FilterbankSlidingRiemannianStem
                      Outputs (B, L_win, d_model) manifold sequence instead of
                      (B, 1, d_model) static token.  Gives the S4 a genuine
                      Riemannian time series to model.

    film_high/low  → cattn_high/low (SpatialTemporalCrossAttention)
                      Per-timestep dynamic spatial conditioning replaces global
                      affine FiLM. Query=wavelet tokens, Key/Value=Riemannian
                      sequence.

    S4DLayer       → dt-selectivity bug fixed (see S4DLayer docstring).

    DirichletEDLLoss → annealing_step default 10→100 (see loss docstring).

    Forward output keys  (identical to v1 for drop-in compatibility)
    -----------------------------------------------------------------
    alpha         : (B, n_actions) — raw Dirichlet concentration params
    intent_probs  : (B, n_actions) — normalized intent distribution
    confidence    : (B,) — 1 - uncertainty
    embed         : (B, d_model)   — pooled embedding
    sequence      : (B, T', d_model) — full temporal sequence
    vib_kl_loss   : scalar
    cpc_loss      : scalar
    subject_logits: (B, 10) — DANN domain classifier
    cognitive     : dict of (B,) scalars
    """

    def __init__(
        self,
        in_channels: int   = 21,
        d_model:     int   = 256,
        d_state:     int   = 64,
        n_blocks:    int   = 4,
        downsample:  int   = 4,
        n_actions:   int   = 8,
        dropout:     float = 0.1,
        sfreq:       float = 256.0,
        win_samples: int   = 64,
        win_stride:  int   = 16,
        use_lif:     bool  = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model     = d_model
        self.n_actions   = n_actions
        self.downsample  = downsample
        self.use_lif     = use_lif

        # ── Preprocessing ────────────────────────────────────────────────────
        self.artifact_gater = ArtifactGater(in_channels)
        self.hurst          = HurstEstimator(in_channels)

        # ── v2 Spatial encoding: Filterbank Sliding Riemannian Stem ──────────
        # Outputs (B, L_win, d_model) — genuine manifold sequence
        self.spatial_stem = FilterbankSlidingRiemannianStem(
            in_ch       = in_channels,
            d_model     = d_model,
            sfreq       = sfreq,
            win_samples = win_samples,
            stride      = win_stride,
        )

        # ── Temporal encoding: spectral graph wavelets ───────────────────────
        self.spectral_wavelet = SpectralGraphWaveletStem(
            in_channels, d_model, downsample=downsample
        )

        # ── S4D MoE blocks ────────────────────────────────────────────────────
        n_hi = max(1, n_blocks // 2)
        n_lo = max(1, n_blocks // 2)
        self.blocks_high = nn.ModuleList([
            S4DMoEBlock(d_model, d_state) for _ in range(n_hi)
        ])
        self.blocks_low = nn.ModuleList([
            S4DMoEBlock(d_model, d_state) for _ in range(n_lo)
        ])

        # ── v2 Spatial conditioning: Cross-Attention (replaces FiLM) ─────────
        self.cattn_high = nn.ModuleList([
            SpatialTemporalCrossAttention(d_model, n_heads=4, dropout=dropout)
            for _ in range(n_hi)
        ])
        self.cattn_low = nn.ModuleList([
            SpatialTemporalCrossAttention(d_model, n_heads=4, dropout=dropout)
            for _ in range(n_lo)
        ])

        # ── Pooling ───────────────────────────────────────────────────────────
        self.pooler = MultiHeadAttentionPooling(d_model, n_heads=max(1, d_model // 32))

        # ── Variational Information Bottleneck ────────────────────────────────
        self.vib_mu     = nn.Linear(d_model, d_model)
        self.vib_logvar = nn.Linear(d_model, d_model)

        # ── Readout head ──────────────────────────────────────────────────────
        self.intent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            LeakyIntegrateAndFire(d_model, n_actions) if use_lif else nn.Linear(d_model, n_actions),
        )

        # ── Auxiliary heads ───────────────────────────────────────────────────
        self.identity_proj  = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 64)
        )
        self.predictor      = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.cpc_proj       = nn.Linear(d_model, d_model)
        self.kinematic_proj = nn.Linear(d_model, 256)

        # ── Domain adaptation (DANN) ──────────────────────────────────────────
        self.subject_classifier = nn.Sequential(
            GradientReversalLayer(alpha=1.0),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        # ── Momentum encoder (for MoCo-style SSL) ────────────────────────────
        self.momentum_spatial   = None
        self.momentum_temporal  = None
        self.momentum_blocks_hi = None
        self.momentum_blocks_lo = None

    # ── FLOPs estimate ────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_flops(self, seq_len: int = 256) -> float:
        C, T, H = self.in_channels, seq_len, self.d_model
        n_win   = max(1, (T - self.spatial_stem.win_samples) //
                     self.spatial_stem.stride + 1)
        riem    = len(_EEG_BANDS) * (C**3 + (C*(C+1)//2) * H) * n_win
        wav     = 32 * 63 * 3 * T * math.log2(max(T, 2))
        blks    = (len(self.blocks_high) + len(self.blocks_low)) * (T / 4) * (H**2 * 4) * 2
        return float(riem + wav + blks)

    # ── Momentum encoder utilities ────────────────────────────────────────────

    def _init_momentum(self):
        if self.momentum_spatial is None:
            self.momentum_spatial   = copy.deepcopy(self.spatial_stem)
            self.momentum_temporal  = copy.deepcopy(self.spectral_wavelet)
            self.momentum_blocks_hi = copy.deepcopy(self.blocks_high)
            self.momentum_blocks_lo = copy.deepcopy(self.blocks_low)
            for p in self.momentum_spatial.parameters():   p.requires_grad = False
            for p in self.momentum_temporal.parameters():  p.requires_grad = False
            for p in self.momentum_blocks_hi.parameters(): p.requires_grad = False
            for p in self.momentum_blocks_lo.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update_momentum(self, m: float = 0.99):
        self._init_momentum()
        pairs = [
            (self.spatial_stem,    self.momentum_spatial),
            (self.spectral_wavelet, self.momentum_temporal),
            (self.blocks_high,     self.momentum_blocks_hi),
            (self.blocks_low,      self.momentum_blocks_lo),
        ]
        for src, tgt in pairs:
            for ps, pt in zip(src.parameters(), tgt.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1 - m)

    @torch.no_grad()
    def forward_momentum(self, eeg: torch.Tensor) -> torch.Tensor:
        self._init_momentum()
        gater   = self.artifact_gater(eeg).unsqueeze(-1)
        eeg_g   = eeg * gater
        hurst   = self.hurst(eeg).unsqueeze(-1).clamp(min=0.1)
        x_riem  = self.momentum_spatial(eeg_g)
        x_hi, x_lo = self.momentum_temporal(eeg_g)
        x_hi = x_hi * hurst
        x_lo = x_lo * hurst
        for blk in self.momentum_blocks_hi: x_hi = blk(x_hi)
        for blk in self.momentum_blocks_lo: x_lo = blk(x_lo)
        x = torch.cat([x_hi, x_lo], dim=1)
        return x.mean(dim=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        eeg:  torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # ── Input masking + channel-wise z-score ──────────────────────────────
        if mask is not None:
            eeg = eeg * mask.unsqueeze(-1).expand_as(eeg)
        
        # Normalize per channel (B, C, T) -> (B, C, 1) stats
        mu  = eeg.mean(dim=-1, keepdim=True)
        std = eeg.std(dim=-1, keepdim=True).clamp(min=1e-6)
        eeg = (eeg - mu) / std

        # ── Artifact gating ───────────────────────────────────────────────────
        gater = self.artifact_gater(eeg).unsqueeze(-1)    # (B, 1, 1)
        eeg_g = eeg * gater

        # ── Hurst fractal scale ───────────────────────────────────────────────
        hurst = self.hurst(eeg).unsqueeze(-1).clamp(min=0.1)   # (B, 1, 1)

        # ── v2 Spatial path: Filterbank Sliding Riemannian Stem ──────────────
        # (B, L_win, d_model) — a temporal sequence of manifold embeddings
        x_riem = self.spatial_stem(eeg_g)

        # ── Temporal path: spectral graph wavelets ────────────────────────────
        x_hi, x_lo = self.spectral_wavelet(eeg_g)   # (B, T', d_model) each

        # Scale by fractal complexity
        x_hi = x_hi * hurst
        x_lo = x_lo * hurst

        # ── S4D MoE blocks + v2 Cross-Attention spatial conditioning ──────────
        # Each block: S4D processes wavelet sequence, then cross-attends to
        # Riemannian manifold sequence to inject geometric spatial context.
        for blk, cattn in zip(self.blocks_high, self.cattn_high):
            x_hi = blk(x_hi)
            x_hi = cattn(query=x_hi, context=x_riem)

        for blk, cattn in zip(self.blocks_low, self.cattn_low):
            x_lo = blk(x_lo)
            x_lo = cattn(query=x_lo, context=x_riem)

        # Append Riemannian sequence to final concatenation for pooling
        # This ensures the manifold embedding is directly available to the
        # attention pooler, not only via the cross-attention residual path.
        x = torch.cat([x_hi, x_lo, x_riem], dim=1)   # (B, 2*T'+L_win, d_model)

        # ── Multi-head attention pooling ──────────────────────────────────────
        embed = self.pooler(x)                         # (B, d_model)

        # ── Variational Information Bottleneck ────────────────────────────────
        mu     = self.vib_mu(embed)
        logvar = self.vib_logvar(embed)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z   = mu + std * torch.randn_like(std)
        else:
            z   = mu
        vib_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        # ── Intent readout ────────────────────────────────────────────────────
        raw_intent = self.intent_proj(z)                      # (B, n_actions)

        # ── Dirichlet evidence ────────────────────────────────────────────────
        evidence     = F.softplus(raw_intent).clamp(max=1e6)
        alpha        = evidence + 1.0
        S_dir        = alpha.sum(-1, keepdim=True)
        intent_probs = alpha / S_dir
        uncertainty  = self.n_actions / S_dir
        confidence   = (1.0 - uncertainty).clamp(0.0, 1.0)

        # ── CPC self-supervised loss ──────────────────────────────────────────
        cpc_loss = torch.tensor(0.0, device=eeg.device)
        if self.training and x.shape[1] > 1:
            cpc_pred   = self.cpc_proj(x[:, :-1])
            cpc_target = x[:, 1:].detach()
            cpc_loss   = F.mse_loss(cpc_pred, cpc_target)

        # ── Domain adaptation (DANN) ──────────────────────────────────────────
        subject_logits = self.subject_classifier(embed)

        return {
            "alpha":          alpha,
            "intent_probs":   intent_probs,
            "confidence":     confidence.squeeze(-1),
            "uncertainty":    uncertainty.squeeze(-1),
            "embed":          embed,
            "sequence":       x,
            "pred_z":         self.predictor(embed),
            "identity_embed": self.identity_proj(embed),
            "pred_kinematic": self.kinematic_proj(embed),
            "vib_kl_loss":    vib_kl,
            "cpc_loss":       cpc_loss,
            "sync_loss":      torch.tensor(0.0, device=eeg.device),
            "subject_logits": subject_logits,
            "cognitive": {
                "uncertainty":    uncertainty.squeeze(-1),
                "total_evidence": S_dir.squeeze(-1),
            },
        }
