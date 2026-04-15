"""
noosphere/s4_eeg.py
===================
S4 EEG Encoder — Restored + Extended (joseph-woodall-branch)

Architecture (all novel components kept, four bugs fixed):

Core SSM
--------
- HiPPO-LegS initialization restored (mathematically principled state basis)
- Bidirectional diagonal S4 (S4D) with ZOH discretization restored
- Selective gating (Mamba-style dt modulation) layered ON TOP of the working S4D kernel
- Conv-based parallel scan for training (O(L log L)); step-mode for inference

Spatial Path
------------
- RiemannianStem: tangent-space log-covariance projection (SOTA for MI-EEG)
- SpectralGraphWaveletStem: learned graph Laplacian + parametric Morlet wavelets

Temporal Path
-------------
- Dual high/low Morlet wavelet branches
- SelectiveS4DBlock: HiPPO S4D + selective dt-gating + GLU residual
- FiLM conditioning from spatial branch

Readout
-------
- MultiHeadAttentionPooling (no info bottleneck vs single linear)
- Variational Information Bottleneck (mu only at inference; no noise; KL returned)
- LeakyIntegrateAndFire neuromorphic output
- Dirichlet EDL: alpha, intent_probs returned (training uses log_softmax(alpha))
- FIXED: intent_proj now has nn.Linear as first layer (Bug 1 fix)

Domain Adaptation
-----------------
- GradientReversalLayer + subject_classifier (DANN, Ganin et al. 2016)

Self-Supervised
---------------
- CPC predictive coding loss
- Contrastive SSL via intent_probs cosine similarity

Fixes applied vs degraded version
----------------------------------
1. intent_proj: nn.Linear(d_model, d_model) restored as first layer
2. alpha output is now raw evidence (pre-normalization) so that
   F.log_softmax(alpha) used in the benchmark is correct
3. HurstEstimator gating clamped [0.1, 1.0] to prevent zeroing out temporal path
4. VIB KL loss + d_state properly exposed; symplectic scan removed
"""

import copy
import math
from typing import Dict, Optional, Tuple

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


# ── HiPPO-LegS initialisation ─────────────────────────────────────────────────

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


# ── Diagonal S4 Layer (S4D) ────────────────────────────────────────────────────

class S4DLayer(nn.Module):
    """
    Diagonal S4 (S4D) with ZOH discretization.
    Uses diagonal approximation of HiPPO-LegS for efficiency and stability.
    Supports bidirectional processing (forward + reversed) during training.
    dt is optionally modulated by input content (Mamba-style selectivity).
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

        # HiPPO-LegS diagonal entries (real part only for S4D)
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

        # Mamba-style input-dependent dt modulation
        if selective:
            self.dt_proj = nn.Linear(d_model, d_model, bias=True)
            self.liquid_tau = nn.Parameter(torch.ones(d_model))

        if bidirectional:
            self.out_bi = nn.Linear(d_model * 2, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def _kernel(self, L: int, dt_base: torch.Tensor) -> torch.Tensor:
        """Build convolution kernel via ZOH: k[t] = sum_n C[n]*A_bar[n]^t*B_bar[n]."""
        A     = -torch.exp(self.A_log)                         # (d_model, d_state)
        dt    = dt_base.unsqueeze(-1)                          # (d_model, 1)
        A_bar = torch.exp(dt * A)                              # (d_model, d_state)
        B_bar = (A_bar - 1.0) / (A + 1e-8) * self.B           # ZOH: (d_model, d_state)
        CB    = self.C * B_bar                                 # (d_model, d_state)

        # Stable log-space power: powers[h, t, n] = A_bar[h,n]^t
        t_idx  = torch.arange(L, device=A.device, dtype=A.dtype)   # (L,)
        log_Ab = torch.log(A_bar.clamp(min=1e-8))                   # (d_model, d_state)
        powers = torch.exp(log_Ab.unsqueeze(1) * t_idx[None, :, None])  # (d_model, L, d_state)

        # k[h, t] = sum_n CB[h,n] * powers[h,t,n]
        k = (powers * CB.unsqueeze(1)).sum(-1)                 # (d_model, L)
        return k

    def _conv(self, u: torch.Tensor, rev: bool = False) -> torch.Tensor:
        """Parallel training mode: FFT convolution. u: (B, L, d_model)"""
        B_sz, L, H = u.shape

        # Compute base dt (optionally input-modulated)
        dt_base = torch.exp(self.log_dt)  # (d_model,)
        if self.selective:
            # Content-dependent dt: sigmoid-gated modulation
            dt_gate = torch.sigmoid(
                self.dt_proj(u) * self.liquid_tau
            )  # (B, L, d_model)
            # Use mean across sequence as a stable per-sample dt scale
            dt_scale = dt_gate.mean(dim=1)       # (B, d_model)
            dt_eff = dt_base * dt_scale.mean(0)  # (d_model,) — mean over batch
        else:
            dt_eff = dt_base

        k = self._kernel(L, dt_eff)  # (d_model, L)

        u_t = u.transpose(1, 2)  # (B, d_model, L)
        k_f = torch.fft.rfft(k, n=2 * L, dim=-1)
        u_f = torch.fft.rfft(u_t, n=2 * L, dim=-1)
        y_f = k_f.unsqueeze(0) * u_f            # (B, d_model, 2L//2+1)
        y   = torch.fft.irfft(y_f, n=2 * L, dim=-1)[..., :L]  # (B, d_model, L)
        y   = y + u_t * self.D.unsqueeze(0).unsqueeze(-1)
        return self.drop(y.transpose(1, 2))  # (B, L, d_model)

    def _step(self, u: torch.Tensor,
              state: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-step inference. u: (B, d_model)"""
        dt_eff = torch.exp(self.log_dt)
        A = -torch.exp(self.A_log)
        A_bar = torch.exp(dt_eff.unsqueeze(-1) * A)  # (d_model, d_state)
        B_bar = (A_bar - 1.0) / (A + 1e-8) * self.B

        if state is None:
            state = torch.zeros(u.shape[0], self.d_model, self.d_state,
                                device=u.device, dtype=u.dtype)

        state = A_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * u.unsqueeze(-1)
        y = (state * self.C.unsqueeze(0)).sum(-1) + u * self.D
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


# ── Selective S4D Block (HiPPO kernel + GLU residual) ─────────────────────────

class SelectiveS4DBlock(nn.Module):
    """
    S4D block with:
    - HiPPO-LegS initialized diagonal SSM kernel
    - ZOH discretization (stable, no symplectic approximation)
    - Bidirectional (training), unidirectional (inference)
    - Mamba-style selective dt gating
    - GLU gate sub-layer + residual
    """

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
        n     = self.norm2(x)
        vg    = self.glu_v(n) * torch.sigmoid(self.glu_g(n))  # gated
        x     = x + self.drop(self.glu_o(vg))
        return x


# ── S4D MoE Block (Mixture-of-Experts over S4D blocks) ────────────────────────

class S4DMoEBlock(nn.Module):
    """Two-expert S4D mixture. Router learns to specialize per frequency band."""

    def __init__(self, d_model: int, d_state: int, num_experts: int = 2):
        super().__init__()
        self.router  = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            SelectiveS4DBlock(d_model, d_state) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor, inference: bool = False) -> torch.Tensor:
        weights = F.softmax(self.router(x), dim=-1)  # (B, L, E)
        out = sum(weights[..., i:i+1] * expert(x, inference)
                  for i, expert in enumerate(self.experts))
        return out


# ── Wavelet Spatial-Temporal Stems ────────────────────────────────────────────

class ParametricMorlet1d(nn.Module):
    """Learned Morlet wavelet filterbank — learnable frequency and bandwidth."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.freqs      = nn.Parameter(torch.randn(out_ch, in_ch, 1) * 5.0)
        self.bandwidths = nn.Parameter(torch.ones(out_ch, in_ch, 1))
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
            ParametricMorlet1d(in_ch, hc, s, stride=downsample)
            for s in self.scales_high
        ])
        self.convs_low = nn.ModuleList([
            ParametricMorlet1d(in_ch, lc, s, stride=downsample)
            for s in self.scales_low
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
    """Learned graph Laplacian smoothing → WaveletStem. Captures inter-electrode topology."""

    def __init__(self, in_ch: int, d_model: int, downsample: int = 4):
        super().__init__()
        self.adj     = LearnedAdjacency(in_ch)
        self.wavelet = WaveletStem(in_ch, d_model, downsample)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        A = self.adj()
        D = torch.diag_embed(A.sum(-1))
        L = D - A
        # Laplacian smoothing: mix spatially adjacent channels
        x_lap = torch.einsum("cd,bdt->bct", L, x)
        return self.wavelet(x_lap)


# ── Riemannian Stem ────────────────────────────────────────────────────────────

class RiemannianStem(nn.Module):
    """
    Tangent-space Riemannian geometry encoder.
    Computes log-Euclidean covariance → upper-triangular vectorization → MLP.
    SOTA spatial encoder for MI-EEG (Barachant et al., 2012).
    """

    def __init__(self, in_ch: int, d_model: int, momentum: float = 0.9,
                 mixup_alpha: float = 0.2):
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

        # Tikhonov regularisation proportional to trace
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
            log_cov = torch.bmm(vecs, torch.bmm(torch.diag_embed(vals),
                                                  vecs.transpose(1, 2))).to(torch.float32)
        except Exception:
            log_cov = cov  # fallback

        idx       = torch.triu_indices(C, C, device=x.device)
        ts_vector = log_cov[:, idx[0], idx[1]]

        if self.training and self.mixup_alpha > 0:
            lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            lam = max(lam, 1 - lam)
            perm = torch.randperm(B, device=x.device)
            ts_vector = lam * ts_vector + (1 - lam) * ts_vector[perm]

        return self.proj(ts_vector).unsqueeze(1)  # (B, 1, d_model)


# ── Attention Pooling ──────────────────────────────────────────────────────────

class MultiHeadAttentionPooling(nn.Module):
    """Learned query attends over sequence, aggregates to single vector."""

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.q    = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.mha  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q.expand(x.shape[0], -1, -1)
        out, _ = self.mha(q, x, x)
        return self.norm(out.squeeze(1))


# ── Conditioning ──────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    """Feature-wise Linear Modulation: scale+shift temporal features by spatial context."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gamma = nn.Linear(d_model, d_model)
        self.beta  = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        c = cond.squeeze(1)                      # (B, d_model)
        g = self.gamma(c).unsqueeze(1)           # (B, 1, d_model)
        b = self.beta(c).unsqueeze(1)            # (B, 1, d_model)
        return x * (1 + g) + b


# ── Artifact Gater ────────────────────────────────────────────────────────────

class ArtifactGater(nn.Module):
    """Learns to down-weight artifact-contaminated channels/times."""

    def __init__(self, in_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, 1, kernel_size=15, stride=4, padding=7)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf    = F.relu(self.conv(x))
        score = torch.sigmoid(self.pool(hf)).squeeze(-1)
        return 1.0 - score   # high HF energy → low gate → artifact suppressed


# ── Hurst Estimator ────────────────────────────────────────────────────────────

class HurstEstimator(nn.Module):
    """
    Biological fractal scale gating.
    FIXED: clamped to [0.1, 1.0] so the temporal path can never be zeroed out.
    """

    def __init__(self, in_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_ch, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, :, 1:] - x[:, :, :-1]
        var  = diff.var(dim=-1) + 1e-8
        # clamp to prevent zeroing temporal path
        return torch.sigmoid(self.proj(var)).clamp(min=0.1)


# ── Domain Adaptation ─────────────────────────────────────────────────────────

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


# ── Neuromorphic Readout ───────────────────────────────────────────────────────

class LeakyIntegrateAndFire(nn.Module):
    """LIF neuron readout — biologically plausible, adds spike sparsity."""

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.proj      = nn.Linear(d_model, n_classes)
        self.threshold = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        membrane = self.proj(x)
        spikes   = F.relu(membrane - self.threshold)
        return spikes + membrane * 0.1  # soft spiking: spikes + leak


# ── Evidential Loss ────────────────────────────────────────────────────────────

class DirichletEDLLoss(nn.Module):
    """
    Evidential Deep Learning loss for Dirichlet-parametrized uncertainty.
    Bayes risk + KL regularisation (Sensoy et al., NeurIPS 2018).
    """

    def __init__(self, n_classes: int, annealing_step: int = 10,
                 conflict_weight: float = 0.1):
        super().__init__()
        self.n_classes      = n_classes
        self.annealing_step = annealing_step
        self.conflict_weight = conflict_weight

    def forward(self, alpha: torch.Tensor, target_one_hot: torch.Tensor,
                current_step: int) -> torch.Tensor:
        S           = alpha.sum(-1, keepdim=True)
        pred_probs  = alpha / S
        err         = (target_one_hot - pred_probs) ** 2
        var         = (alpha * (S - alpha)) / (S**2 * (S + 1))
        loss_sos    = (err + var).sum(-1)
        annealing   = min(1.0, current_step / self.annealing_step)
        alpha_tilde = target_one_hot + (1 - target_one_hot) * alpha
        kl_reg      = annealing * self._kl(alpha_tilde)
        entropy     = -(pred_probs * torch.log(pred_probs + 1e-8)).sum(-1)
        return torch.mean(loss_sos + kl_reg + self.conflict_weight * entropy)

    def _kl(self, alpha: torch.Tensor) -> torch.Tensor:
        ones = torch.ones(1, self.n_classes, device=alpha.device)
        sa   = alpha.sum(-1, keepdim=True)
        so   = ones.sum(-1, keepdim=True)
        t1   = torch.lgamma(sa) - torch.lgamma(so)
        t2   = (torch.lgamma(ones) - torch.lgamma(alpha)).sum(-1, keepdim=True)
        t3   = ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sa))).sum(-1, keepdim=True)
        return (t1 + t2 + t3).squeeze(-1)


# ── Full S4 EEG Encoder ────────────────────────────────────────────────────────

class S4EEGEncoder(nn.Module):
    """
    Full S4-based EEG encoder for Motor Imagery BCI.

    Forward output keys
    -------------------
    alpha        : (B, n_actions) — raw Dirichlet concentration params
                   Training: F.log_softmax(alpha, -1) → NLL loss  ← benchmark uses this
    intent_probs : (B, n_actions) — normalized intent distribution
    confidence   : (B,) — 1 - uncertainty
    embed        : (B, d_model) — pooled embedding
    sequence     : (B, T', d_model) — full temporal sequence
    vib_kl_loss  : scalar — KL term for VIB (weight ~0.01 in total loss)
    cpc_loss     : scalar — predictive coding auxiliary loss
    subject_logits: (B, 10) — domain classifier for adversarial adaptation
    cognitive    : dict of (B,) scalars
    """

    def __init__(
        self,
        in_channels: int = 21,
        d_model:     int = 256,
        d_state:     int = 64,
        n_blocks:    int = 4,
        downsample:  int = 4,
        n_actions:   int = 8,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model     = d_model
        self.n_actions   = n_actions
        self.downsample  = downsample

        # ── Preprocessing ────────────────────────────────────────────────────
        self.artifact_gater = ArtifactGater(in_channels)
        self.hurst          = HurstEstimator(in_channels)

        # ── Spatial encoding (Riemannian geometry) ───────────────────────────
        self.spatial_stem = RiemannianStem(in_channels, d_model)

        # ── Temporal encoding (spectral graph wavelets) ──────────────────────
        self.spectral_wavelet = SpectralGraphWaveletStem(in_channels, d_model,
                                                          downsample=downsample)

        # ── S4D MoE blocks for high and low frequency paths ──────────────────
        n_hi = max(1, n_blocks // 2)
        n_lo = max(1, n_blocks // 2)
        self.blocks_high = nn.ModuleList([
            S4DMoEBlock(d_model, d_state) for _ in range(n_hi)
        ])
        self.blocks_low = nn.ModuleList([
            S4DMoEBlock(d_model, d_state) for _ in range(n_lo)
        ])

        # FiLM: spatial features condition temporal processing
        self.film_high = nn.ModuleList([FiLM(d_model) for _ in range(n_hi)])
        self.film_low  = nn.ModuleList([FiLM(d_model) for _ in range(n_lo)])

        # ── Pooling ───────────────────────────────────────────────────────────
        self.pooler = MultiHeadAttentionPooling(d_model, n_heads=max(1, d_model // 32))

        # ── Variational Information Bottleneck ────────────────────────────────
        self.vib_mu     = nn.Linear(d_model, d_model)
        self.vib_logvar = nn.Linear(d_model, d_model)

        # ── Readout head (BUG 1 FIX: nn.Linear restored as first layer) ──────
        self.intent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),   # ← Bug 1 fix: was missing
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            LeakyIntegrateAndFire(d_model, n_actions),
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
            nn.Linear(64, 10),   # 10 dummy subject classes for regularisation
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
        riem    = C**3 + (C*(C+1)//2) * H
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
        gater    = self.artifact_gater(eeg).unsqueeze(-1)
        eeg_g    = eeg * gater
        hurst    = self.hurst(eeg).unsqueeze(-1).clamp(min=0.1)
        x_s      = self.momentum_spatial(eeg_g)
        x_hi, x_lo = self.momentum_temporal(eeg_g)
        x_hi     = torch.cat([x_s, x_hi * hurst], dim=1)
        x_lo     = torch.cat([x_s, x_lo * hurst], dim=1)
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
        # ── Input masking + normalization ─────────────────────────────────────
        if mask is not None:
            eeg = eeg * mask.unsqueeze(-1).expand_as(eeg)

        # Per-sample z-score (works at any batch size, including B=1)
        eeg = (eeg - eeg.mean(-1, keepdim=True)) / (
            eeg.std(-1, keepdim=True).clamp(min=1e-6)
        )

        # ── Artifact gating ───────────────────────────────────────────────────
        gater   = self.artifact_gater(eeg).unsqueeze(-1)    # (B, 1, 1)
        eeg_g   = eeg * gater

        # ── Hurst fractal scale (BUG 3 FIX: clamped min=0.1) ─────────────────
        hurst   = self.hurst(eeg).unsqueeze(-1).clamp(min=0.1)  # (B, 1, 1)

        # ── Spatial path (Riemannian) ─────────────────────────────────────────
        x_spatial = self.spatial_stem(eeg_g)                # (B, 1, d_model)

        # ── Temporal path (spectral graph wavelets) ───────────────────────────
        x_hi, x_lo = self.spectral_wavelet(eeg_g)           # (B, T', d_model) each

        # Scale by fractal complexity
        x_hi = x_hi * hurst
        x_lo = x_lo * hurst

        # Prepend spatial token to temporal sequences
        x_hi = torch.cat([x_spatial, x_hi], dim=1)          # (B, 1+T', d_model)
        x_lo = torch.cat([x_spatial, x_lo], dim=1)

        # ── S4D MoE blocks + FiLM spatial conditioning ────────────────────────
        for blk, film in zip(self.blocks_high, self.film_high):
            x_hi = blk(x_hi)
            x_hi = film(x_hi, x_spatial)

        for blk, film in zip(self.blocks_low, self.film_low):
            x_lo = blk(x_lo)
            x_lo = film(x_lo, x_spatial)

        x = torch.cat([x_hi, x_lo], dim=1)                  # (B, 2*(1+T'), d_model)

        # ── Multi-head attention pooling ──────────────────────────────────────
        embed = self.pooler(x)                               # (B, d_model)

        # ── Variational Information Bottleneck ────────────────────────────────
        mu     = self.vib_mu(embed)
        logvar = self.vib_logvar(embed)

        if self.training:
            std    = torch.exp(0.5 * logvar)
            z      = mu + std * torch.randn_like(std)
        else:
            z      = mu   # deterministic at eval — no noise

        vib_kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        # ── Intent readout (BUG 1 FIX: Linear layer present) ─────────────────
        raw_intent = self.intent_proj(z)                     # (B, n_actions)

        # ── Dirichlet evidence (BUG 2 FIX: alpha is raw pre-normalized) ──────
        # evidence must be positive; F.softplus is smoother than relu
        evidence   = F.softplus(raw_intent).clamp(max=1e6)
        alpha      = evidence + 1.0                          # Dirichlet params ≥ 1

        S           = alpha.sum(-1, keepdim=True)            # total evidence
        intent_probs = alpha / S                             # normalized probs
        uncertainty  = self.n_actions / S                   # epistemic uncertainty
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
            # Primary classification outputs
            "alpha":          alpha,               # raw Dirichlet params (for log_softmax training)
            "intent_probs":   intent_probs,        # normalized probabilities
            "confidence":     confidence.squeeze(-1),
            "uncertainty":    uncertainty.squeeze(-1),
            # Embeddings
            "embed":          embed,
            "sequence":       x,
            "pred_z":         self.predictor(embed),
            "identity_embed": self.identity_proj(embed),
            "pred_kinematic": self.kinematic_proj(embed),
            # Auxiliary losses (weight in training loop as needed)
            "vib_kl_loss":    vib_kl,
            "cpc_loss":       cpc_loss,
            "sync_loss":      torch.tensor(0.0, device=eeg.device),
            # Domain adaptation
            "subject_logits": subject_logits,
            # Cognitive state proxy
            "cognitive": {
                "uncertainty":    uncertainty.squeeze(-1),
                "total_evidence": S.squeeze(-1),
            },
        }
