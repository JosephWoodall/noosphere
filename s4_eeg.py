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
        S = torch.sum(alpha, dim=-1, keepdim=True)
        beta_alpha = torch.exp(torch.lgamma(S) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True))
        beta_ones = torch.exp(torch.lgamma(torch.tensor(self.n_classes, dtype=torch.float32)) - 
                              self.n_classes * torch.lgamma(torch.tensor(1.0)))
        kl = beta_ones / (beta_alpha + 1e-8) + torch.sum((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S)), dim=-1, keepdim=True)
        return kl.squeeze(-1)

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

class SpectralStem(nn.Module):
    def __init__(self, in_channels: int, d_model: int, hop_length: int = 4):
        super().__init__()
        self.hop_length = hop_length
        self.scales = [64, 128, 256] 
        self.bin_ranges = [(1, 10), (2, 20), (4, 40)] 
        
        # Define region indices
        self.regions = {
            "motor":    slice(0, 9),
            "parietal": slice(9, 16),
            "frontal":  slice(16, 21)
        }
        
        self.spatial_dim = 32
        self.spatial_mixers = nn.ModuleDict({
            r: nn.Conv1d(len(range(*s.indices(in_channels))), self.spatial_dim, kernel_size=1, bias=False)
            for r, s in self.regions.items()
        })
        
        total_bins = sum(r[1] - r[0] for r in self.bin_ranges)
        # 2x for Real + Imaginary components
        self.freq_weight = nn.Parameter(torch.ones(1, 1, total_bins * 3 * 2, 1)) 
        
        self.proj = nn.Sequential(
            nn.Linear(self.spatial_dim * total_bins * 3 * 2, d_model),
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
            xr_centered = xr - xr.mean(dim=-1, keepdim=True)
            cov = torch.bmm(xr_centered, xr_centered.transpose(1, 2)) / (T - 1)
            adj = F.softmax(cov / (T ** 0.5), dim=-1) 
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
            # Concatenate Real and Imag parts instead of taking .abs()
            stft_real = stft_complex.real.reshape(B_mix, C_mix, -1, stft_complex.shape[-1])[:, :, b_start:b_end, :]
            stft_imag = stft_complex.imag.reshape(B_mix, C_mix, -1, stft_complex.shape[-1])[:, :, b_start:b_end, :]
            all_stfts.append(torch.cat([stft_real, stft_imag], dim=2))
            
        min_t = min(s.shape[-1] for s in all_stfts)
        all_stfts = [s[..., :min_t] for s in all_stfts]
        
        stft_merged = torch.cat(all_stfts, dim=2) 
        
        # Proper broadcasting for (B, 3*spatial_dim, total_bins*2, min_t)
        B_s, C_s, F_s, T_s = stft_merged.shape
        stft_merged = stft_merged.reshape(B_s, 3, self.spatial_dim, F_s, T_s)
        fw = self.freq_weight.reshape(1, 3, 1, F_s, 1)
        stft_merged = (stft_merged * fw).reshape(B_s, C_s, F_s, T_s)
        
        stft_flat = stft_merged.reshape(B_mix, -1, min_t).transpose(1, 2)
        return self.proj(stft_flat).transpose(1, 2)

class S4EEGEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, d_model: int = 256, d_state: int = 64, n_blocks: int = 4, downsample: int = 4, n_actions: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_actions = n_actions
        
        self.stem = SpectralStem(in_channels, d_model, hop_length=downsample)
        self.blocks = nn.ModuleList([S4Block(d_model, d_state) for _ in range(n_blocks)])
        self.pooler = MultiHeadAttentionPooling(d_model, n_heads=8)
        
        self.intent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, n_actions)
        )
        
        self.predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    @torch.no_grad()
    def get_flops(self, seq_len: int = 256) -> float:
        """Estimate FLOPs for a single forward pass."""
        B, C, T = 1, sum(len(range(*s.indices(self.in_channels))) for s in self.stem.regions.values()), seq_len
        # GCN: O(C^2 * T) + O(C^2) per region
        gcn_flops = sum((len(range(*s.indices(self.in_channels)))**2 * T) for s in self.stem.regions.values())
        # Spatial Mixer: O(C * C_mix * T)
        mix_flops = sum(len(range(*s.indices(self.in_channels))) * self.stem.spatial_dim * T for s in self.stem.regions.values())
        # STFT: O(C_mix * T * log T) roughly
        stft_flops = self.stem.spatial_dim * 3 * T * math.log2(T)
        # S4Blocks: 4 * (Linear + Conv + Linear)
        block_flops = len(self.blocks) * (T/4) * (self.d_model**2 * 4 + self.d_model * 6)
        return float(gcn_flops + mix_flops + stft_flops + block_flops)
        self.momentum_stem = None
        self.momentum_blocks = None

    def init_momentum_encoder(self):
        if self.momentum_stem is None:
            self.momentum_stem = copy.deepcopy(self.stem)
            self.momentum_blocks = copy.deepcopy(self.blocks)
            for p in self.momentum_stem.parameters(): p.requires_grad = False
            for p in self.momentum_blocks.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update_momentum(self, m: float = 0.99):
        if self.momentum_stem is None: self.init_momentum_encoder()
        for p, p_m in zip(self.stem.parameters(), self.momentum_stem.parameters()): p_m.data.mul_(m).add_(p.data, alpha=1 - m)
        for p, p_m in zip(self.blocks.parameters(), self.momentum_blocks.parameters()): p_m.data.mul_(m).add_(p.data, alpha=1 - m)

    @torch.no_grad()
    def forward_momentum(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.momentum_stem is None: self.init_momentum_encoder()
        x = self.momentum_stem(eeg)
        x = x.transpose(1, 2)
        for block in self.momentum_blocks: x = block(x)
        return x.mean(dim=1)

    def forward(self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = eeg.shape[0]

        if mask is not None:
            mask_t = mask.unsqueeze(-1).expand_as(eeg)
            eeg = eeg * mask_t

        # SpectralStem: (B, C, T) -> (B, d_model, T_prime)
        x = self.stem(eeg)

        # S4 blocks expect (B, T, d_model)
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)

        # Use Attention Pooling instead of just last time-step
        current_state = self.pooler(x)

        # Evidential intent decoding (Dirichlet)
        evidence = F.softplus(self.intent_proj(current_state))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        intent_probs = alpha / S
        uncertainty = self.n_actions / S
        confidence = 1.0 - uncertainty

        return {
            "sequence":    x,                            # (B, T', d_model)
            "embed":       current_state,                 # (B, d_model)
            "topological": None,                          # placeholder; no TDA in this build
            "intent_probs": intent_probs,                 # (B, n_actions)
            "confidence":  confidence.squeeze(-1),        # (B,)
            "alpha":       alpha,                         # (B, n_actions)
            "uncertainty": uncertainty.squeeze(-1),       # (B,)
            "pred_z":      self.predictor(current_state), # (B, d_model)
            "cognitive": {
                "uncertainty":    uncertainty.squeeze(-1),
                "total_evidence": S.squeeze(-1),
            },
        }