"""
noosphere/s4_eeg.py
===================
State-Space Model for EEG with Evidential Intent Decoding

Features:
- Proprioceptive Blindness Fixed: `xyz_head` stripped out. S4 is now strictly 
  an intent decoder, passing raw temporal embeddings up to the Fusion layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from typing import Dict, Optional, Tuple

import numpy as np
try:
    from torch_geometric.nn import DenseGCNConv
except ImportError:
    DenseGCNConv = None

# ── [DirichletEDLLoss and S4Block remain unchanged] ──
class DirichletEDLLoss(nn.Module):
    def __init__(self, n_classes: int, annealing_step: int = 10):
        super().__init__()
        self.n_classes = n_classes
        self.annealing_step = annealing_step

    def forward(self, alpha: torch.Tensor, target_one_hot: torch.Tensor, current_step: int) -> torch.Tensor:
        S = torch.sum(alpha, dim=-1, keepdim=True)
        pred_probs = alpha / S
        err = (target_one_hot - pred_probs) ** 2
        var = (alpha * (S - alpha)) / (S ** 2 * (S + 1))
        loss_sos = torch.sum(err + var, dim=-1)
        annealing_coef = min(1.0, current_step / self.annealing_step)
        alpha_tilde = target_one_hot + (1 - target_one_hot) * alpha
        kl_reg = annealing_coef * self._kl_divergence(alpha_tilde)
        return torch.mean(loss_sos + kl_reg)

    def _kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        ones = torch.ones([1, self.n_classes], device=alpha.device)
        sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        sum_ones = torch.sum(ones, dim=-1, keepdim=True)
        
        # More stable KL for Dirichlet
        term1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_ones)
        term2 = torch.sum(torch.lgamma(ones) - torch.lgamma(alpha), dim=-1, keepdim=True)
        term3 = torch.sum((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha)), dim=-1, keepdim=True)
        
        return (term1 + term2 + term3).squeeze(-1)

class S4DKernel(nn.Module):
    """Efficient Diagonal State-Space Kernel (S4D)."""
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Log-space initialization for stability
        # A is diagonal, complex-valued for S4D-Lin initialization
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(d_state).repeat(d_model, 1) / d_state)
        self.C = nn.Parameter(torch.randn(d_model, d_state, 2) / math.sqrt(d_state)) # Complex C

    def forward(self, L: int) -> torch.Tensor:
        """Returns the convolution kernel of length L."""
        # [S4D Cauchy Kernel Formulation]
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H, N)
        C = self.C[..., 0] + 1j * self.C[..., 1]           # (H, N)

        t = torch.arange(L, device=A.device)
        kernel = (C.unsqueeze(-1) * torch.exp(A.unsqueeze(-1) * t)).sum(dim=1).real # (H, L)
        return kernel

class S4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.kernel = S4DKernel(d_model, d_state)
        self.dropout = nn.Dropout(0.2)

        self.output_linear = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1) # Gated Linear Unit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, H)"""
        residual = x
        x = self.norm(x)
        B, L, H = x.shape

        # FFT Convolution with S4D Kernel
        x = x.transpose(1, 2) # (B, H, L)
        k = self.kernel(L)     # (H, L)

        x_fft = torch.fft.rfft(x, n=2*L)
        k_fft = torch.fft.rfft(k, n=2*L)
        y = torch.fft.irfft(x_fft * k_fft, n=2*L)[..., :L]

        y = y.transpose(1, 2) # (B, L, H)
        y = self.dropout(self.output_linear(y))

        return residual + y
class SpectralStem(nn.Module):
    def __init__(self, in_channels: int, d_model: int, hop_length: int = 4):
        super().__init__()
        self.hop_length = hop_length
        self.scales = [64, 128, 256] 
        self.bin_ranges = [(1, 10), (2, 20), (4, 40)] 
        
        # Robust region definition
        if in_channels >= 21:
            self.regions = {
                "motor":    slice(0, 9),
                "parietal": slice(9, 16),
                "frontal":  slice(16, 21)
            }
        elif in_channels >= 3:
            # Split into 3 equal-ish parts
            step = max(1, in_channels // 3)
            self.regions = {
                "r1": slice(0, step),
                "r2": slice(step, 2 * step),
                "r3": slice(2 * step, in_channels)
            }
        else:
            self.regions = {"all": slice(0, in_channels)}
        
        self.spatial_dim = 32
        self.spatial_mixers = nn.ModuleDict({
            r: nn.Conv1d(max(1, len(range(*s.indices(in_channels)))), self.spatial_dim, kernel_size=1, bias=False)
            for r, s in self.regions.items()
        })
        
        n_regions = len(self.regions)
        total_bins = sum(r[1] - r[0] for r in self.bin_ranges)
        self.freq_weight = nn.Parameter(torch.ones(1, 1, total_bins * n_regions * 2, 1)) 
        
        self.proj = nn.Sequential(
            nn.Linear(self.spatial_dim * total_bins * n_regions * 2, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        for s in self.scales:
            self.register_buffer(f"window_{s}", torch.hann_window(s))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        regional_feats = []
        for r, slice_obj in self.regions.items():
            xr = x[:, slice_obj, :]
            if xr.shape[1] == 0: continue
            xr_centered = xr - xr.mean(dim=-1, keepdim=True)
            # Use small eps for stability in covariance
            cov = torch.bmm(xr_centered, xr_centered.transpose(1, 2)) / (T - 1 + 1e-8)
            adj = F.softmax(cov / (T ** 0.5 + 1e-8), dim=-1) 
            xr_gcn = torch.bmm(adj, xr) + xr 
            xr_mixed = self.spatial_mixers[r](xr_gcn)
            regional_feats.append(xr_mixed)
            
        x_mixed = torch.cat(regional_feats, dim=1) 
        B_mix, C_mix, T_mix = x_mixed.shape
        x_flat = x_mixed.reshape(B_mix * C_mix, T_mix)
        
        all_stfts = []
        for s, (b_start, b_end) in zip(self.scales, self.bin_ranges):
            window = getattr(self, f"window_{s}")
            stft_complex = torch.stft(x_flat, n_fft=s, hop_length=self.hop_length, 
                                     window=window, return_complex=True, center=True, 
                                     pad_mode='reflect')
            # Extract bins
            s_real = stft_complex.real.reshape(B_mix, C_mix, -1, stft_complex.shape[-1])[:, :, b_start:b_end, :]
            s_imag = stft_complex.imag.reshape(B_mix, C_mix, -1, stft_complex.shape[-1])[:, :, b_start:b_end, :]
            all_stfts.append(torch.cat([s_real, s_imag], dim=2))
            
        min_t = min(s.shape[-1] for s in all_stfts)
        all_stfts = [s[..., :min_t] for s in all_stfts]
        stft_merged = torch.cat(all_stfts, dim=2) # (B_mix, C_mix, F_total, T_min)
        
        # Frequency weighting
        B_s, C_s, F_s, T_s = stft_merged.shape
        # Weight per regional channel
        stft_merged = stft_merged.reshape(B_mix, len(self.regions), self.spatial_dim, F_s, T_s)
        fw = self.freq_weight.reshape(1, len(self.regions), 1, F_s, 1)
        stft_merged = (stft_merged * fw).reshape(B_mix, C_mix, F_s, T_s)
        
        stft_flat = stft_merged.reshape(B_mix, -1, min_t).transpose(1, 2)
        return self.proj(stft_flat).transpose(1, 2)

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model))
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        attn_out, _ = self.mha(self.q.expand(x.shape[0], -1, -1), x, x)
        return self.norm(attn_out.squeeze(1))

class RiemannianStem(nn.Module):
    """
    Projects EEG covariance into Tangent Space.
    Provides a stable, subject-invariant spatial starting point.
    """
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.in_channels = in_channels
        # Tangent space dimension for C channels is C*(C+1)/2
        self.ts_dim = in_channels * (in_channels + 1) // 2
        
        self.proj = nn.Sequential(
            nn.Linear(self.ts_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T) raw EEG
        Returns: (B, 1, d_model) spatial embedding
        """
        B, C, T = x.shape
        # 1. Compute Batch Covariance: (B, C, C)
        x = x - x.mean(dim=-1, keepdim=True)
        cov = torch.bmm(x, x.transpose(1, 2)) / (T - 1)
        
        # 2. Robust Shrinkage: add trace-relative identity to ensure positive definiteness
        # Especially important for low-density arrays or noisy signals
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        # Combine trace-relative eps with a small fixed floor to handle zero-padded channels
        eps = (1e-4 * trace / C) + 1e-6
        eye = torch.eye(C, device=x.device).unsqueeze(0)
        cov = cov + eps * eye
        
        # 3. Tangent Space Projection (Log-Euclidean) in Double Precision
        try:
            from torch.linalg import eigh
            # Cast to float64 for stable decomposition
            vals, vecs = eigh(cov.to(torch.float64))
            vals = torch.log(vals.clamp(min=1e-9))
            log_cov = torch.bmm(vecs, torch.bmm(torch.diag_embed(vals), vecs.transpose(1, 2)))
            log_cov = log_cov.to(torch.float32)
        except Exception:
            # Fallback to zero log-covariance if eigh fails
            log_cov = torch.zeros_like(cov)
        
        # 4. Extract upper triangle for vectorization
        idx = torch.triu_indices(C, C, device=x.device)
        ts_vector = log_cov[:, idx[0], idx[1]]
        
        # 5. Project to d_model
        return self.proj(ts_vector).unsqueeze(1) # (B, 1, d_model)

class SpatialAttention(nn.Module):
    """Learned channel-wise importance weighting for high-density arrays."""
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channels, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        weights = self.attn(x.mean(dim=-1)).unsqueeze(-1)
        return x * weights

class S4EEGEncoder(nn.Module):
    def __init__(self, in_channels: int = 21, d_model: int = 256, d_state: int = 128, n_blocks: int = 6, downsample: int = 4, n_actions: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_actions = n_actions
        
        # 1. Spatial Domain: Adaptive Riemannian + Local Attention
        self.spatial_attn = SpatialAttention(in_channels, d_model)
        self.spatial_stem = RiemannianStem(in_channels, d_model)
        
        # 2. Temporal Domain: Multi-Scale Feature Extraction
        if in_channels < 10:
            self.temporal_stem = nn.Sequential(
                nn.Conv1d(in_channels, d_model, kernel_size=15, padding=7, stride=2),
                nn.GELU(),
                nn.GroupNorm(min(8, d_model), d_model),
                nn.Conv1d(d_model, d_model, kernel_size=15, padding=7, stride=2),
                nn.GELU()
            )
        else:
            self.temporal_stem = SpectralStem(in_channels, d_model, hop_length=downsample)
        
        # 3. Processing Blocks: Deeper S4D chain
        self.blocks = nn.ModuleList([S4Block(d_model, d_state) for _ in range(n_blocks)])
        self.pooler = MultiHeadAttentionPooling(d_model, n_heads=8)
        
        # Head Refinement
        self.intent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2),
            nn.Linear(d_model, n_actions)
        )
        self.identity_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 64)
        )
        self.predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.momentum_spatial = None
        self.momentum_temporal = None
        self.momentum_blocks = None

    @torch.no_grad()
    def get_flops(self, seq_len: int = 256) -> float:
        C, T, H = self.in_channels, seq_len, self.d_model
        # Riemannian: O(C^3) for eigh + O(C^2) for projection
        riem_flops = C**3 + (C*(C+1)//2)*H
        # Temporal: Approx from previous SpectralStem
        temp_flops = 32 * 63 * 3 * T * math.log2(T) 
        # Blocks: 4 * L * H^2
        block_flops = len(self.blocks) * (T/4) * (H**2 * 4)
        return float(riem_flops + temp_flops + block_flops)

    def init_momentum_encoder(self):
        if self.momentum_spatial is None:
            self.momentum_spatial = copy.deepcopy(self.spatial_stem)
            self.momentum_temporal = copy.deepcopy(self.temporal_stem)
            self.momentum_blocks = copy.deepcopy(self.blocks)
            for p in self.momentum_spatial.parameters(): p.requires_grad = False
            for p in self.momentum_temporal.parameters(): p.requires_grad = False
            for p in self.momentum_blocks.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update_momentum(self, m: float = 0.99):
        if self.momentum_spatial is None: self.init_momentum_encoder()
        for p, p_m in zip(self.spatial_stem.parameters(), self.momentum_spatial.parameters()): p_m.data.mul_(m).add_(p.data, alpha=1 - m)
        for p, p_m in zip(self.temporal_stem.parameters(), self.momentum_temporal.parameters()): p_m.data.mul_(m).add_(p.data, alpha=1 - m)
        for p, p_m in zip(self.blocks.parameters(), self.momentum_blocks.parameters()): p_m.data.mul_(m).add_(p.data, alpha=1 - m)

    @torch.no_grad()
    def forward_momentum(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.momentum_spatial is None: self.init_momentum_encoder()
        x_s = self.momentum_spatial(eeg)
        x_t = self.momentum_temporal(eeg).transpose(1, 2)
        x = torch.cat([x_s, x_t], dim=1)
        for block in self.momentum_blocks: x = block(x)
        return x.mean(dim=1)

    def forward(self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = eeg.shape[0]

        if mask is not None:
            mask_t = mask.unsqueeze(-1).expand_as(eeg)
            eeg = eeg * mask_t

        # 0. Spatial Attention
        eeg_weighted = self.spatial_attn(eeg)

        # 1. Spatial Pathway
        x_spatial = self.spatial_stem(eeg_weighted) # (B, 1, d_model)

        # 2. Temporal Pathway
        x_temporal = self.temporal_stem(eeg_weighted) # (B, d_model, T') or (B, T', d_model)
        if x_temporal.shape[1] == self.d_model:
            x_temporal = x_temporal.transpose(1, 2) # ensure (B, T', d_model)

        # 3. Fusion: Prepend spatial context to temporal sequence
        x = torch.cat([x_spatial, x_temporal], dim=1)

        # 4. S4 processing
        for block in self.blocks:
            x = block(x)

        # 5. Summary State
        current_state = self.pooler(x)

        # 6. Evidential Decoding
        evidence = F.softplus(self.intent_proj(current_state))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        intent_probs = alpha / S
        uncertainty = self.n_actions / S
        confidence = 1.0 - uncertainty

        # Identity Manifold Projection
        identity_embed = self.identity_proj(current_state)

        return {
            "sequence":    x,
            "embed":       current_state,
            "intent_probs": intent_probs,
            "confidence":  confidence.squeeze(-1),
            "alpha":       alpha,
            "uncertainty": uncertainty.squeeze(-1),
            "identity_embed": identity_embed,
            "pred_z":      self.predictor(current_state),
            "cognitive": {
                "uncertainty":    uncertainty.squeeze(-1),
                "total_evidence": S.squeeze(-1),
            },
        }