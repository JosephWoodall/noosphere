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

class S4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model * 2, d_model * 2, kernel_size=3, padding=1, groups=d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = F.silu(self.conv(x))
        x = x.transpose(1, 2)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * torch.sigmoid(x2) 
        x = self.dropout(self.out_proj(x))
        return x + residual

class SpectralStem(nn.Module):
    def __init__(self, in_channels: int, d_model: int, n_fft: int = 128, hop_length: int = 4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Hard biological constraint: ERD exists ONLY in 8-32Hz.
        self.bin_start = 4
        self.bin_end = 16
        freq_bins = self.bin_end - self.bin_start
        
        # factorized Spatial Contrast Equivalent to Common Spatial Patterns (CSP)
        # Prevent combinatorial memory explosion on dense arrays (e.g. 128 channels).
        # CSP typically uses 4-6 components; 48 is extremely generous.
        self.spatial_dim = min(in_channels * 3, 48)
        self.spatial_mixer = nn.Conv1d(in_channels, self.spatial_dim, kernel_size=1, bias=False)
        self.spatial_norm = nn.LayerNorm(self.spatial_dim)
        
        self.proj = nn.Sequential(
            nn.Linear(self.spatial_dim * freq_bins, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        # [CRITICAL METADATA INTERVENTION]
        # CSP relies on phase coherence. Spatial filtering MUST happen in time-domain 
        # BEFORE Fourier magnitude extraction destroys relative phase between C3 and C4.
        
        
        # [DYNAMIC GRAPH CONVOLUTION]
        # Construct the adjacency matrix dynamically using raw sequence covariance
        # This allows the model to learn the geodesic topology of the scalp for any array density
        x_centered = x - x.mean(dim=-1, keepdim=True)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (T - 1)
        adj = F.softmax(cov, dim=-1) # (B, C, C) Normalized adjacency mapping
        
        # Native message propagation across the electrode graph
        x_gcn = torch.bmm(adj, x)
        
        # Time-domain spatial filter: (B, C, T) -> (B, spatial_dim, T)
        x_mixed = self.spatial_mixer(x_gcn)
        
        B_mix, C_mix, T_mix = x_mixed.shape
        x_flat = x_mixed.reshape(B_mix * C_mix, T_mix)
        
        window = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.hop_length, 
                          window=window, return_complex=True, center=True, pad_mode='reflect').abs()
        
        F_bins_total = stft.shape[1]
        T_prime = stft.shape[2]
        
        # Reshape to (B, C_mix, F_bins_total, T_prime)
        stft_mix = stft.reshape(B_mix, C_mix, F_bins_total, T_prime)
        
        # Slicing the ERD band
        stft_erd = stft_mix[:, :, self.bin_start:self.bin_end, :]
        
        # Flatten for channel/frequency entanglement -> (B, C_mix * F_erd, T_prime)
        F_erd = stft_erd.shape[2]
        stft_erd_flat = stft_erd.reshape(B_mix, C_mix * F_erd, T_prime).transpose(1, 2)
        
        out = self.proj(stft_erd_flat)
        return out.transpose(1, 2)

class S4EEGEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, d_model: int = 256, d_state: int = 64, n_blocks: int = 4, downsample: int = 4, n_actions: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        
        # Explicit STFT spectral transform mapping ERD/ERS directly
        self.stem = SpectralStem(in_channels, d_model, n_fft=128, hop_length=downsample)

        self.blocks = nn.ModuleList([S4Block(d_model, d_state) for _ in range(n_blocks)])
        
        # Explicit Spatial topology is handled dynamically in the stem
        self.intent_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_actions))
        self.spatial_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        
        self.predictor = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
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

    def forward(self, x):
        """
        x: (B, C, T)
        Returns: Dict with spatial, temporal, topological embeddings
        """
        B, C, T = x.shape
        
        # Spatial processing (C -> d_model)
        spatial = self.spatial_proj(x)  # (B, d_model, T)
        
        # Temporal processing with S4
        temp = spatial
        for layer in self.s4_layers:
            temp, _ = layer(temp)  # (B, d_model, T)
            
        # Initialize topo_embed to None by default <--- ADD THIS LINE
        topo_embed = None 
            
        # Get topological embeddings if GNN is available
        if hasattr(self, 'gnn') and hasattr(self, 'edge_index'):
            if self.edge_index is not None:
                try:
                    # GNN expects (Batch*Time, Channels, Features)
                    # For now we'll just pass a dummy feature since our GNN is simple
                    x_reshaped = x.transpose(1, 2).reshape(B*T, C, 1)
                    topo_embed = self.gnn(x_reshaped, self.edge_index)
                    topo_embed = topo_embed.reshape(B, T, -1).transpose(1, 2)
                except Exception as e:
                    print(f"Warning: GNN embedding failed: {e}")
                    topo_embed = torch.zeros(B, getattr(self, 'topo_dim', 32), T, device=x.device)
                    
        # Combine features
        # Simple addition for now, could be more complex
        out = spatial + temp
        
        return {
            "spatial": spatial,
            "temporal": temp,
            "topological": topo_embed,
            "combined": out
        }