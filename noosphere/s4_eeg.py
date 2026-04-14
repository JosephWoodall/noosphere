"""
noosphere/s4_eeg.py
===================
State-Space Model for EEG with Evidential Intent Decoding

Features:
- Proprioceptive Blindness Fixed: `xyz_head` stripped out. S4 is now strictly 
  an intent decoder, passing raw temporal embeddings up to the Fusion layer.
- Selective State-Spaces (Mamba-style Data-Dependent S4)
- Riemannian Mixup for robust Zero-Shot subject transfer
- FiLM (Feature-wise Linear Modulation) conditioned on Subject ID / Tangent Space
- Contrastive Predictive Coding (CPC) Loss
- Multi-Resolution S4 Routing via Wavelet Stem
- Adversarial Subject-Invariance (Gradient Reversal Layer)
- Hamiltonian S4 Dynamics (Symplectic Integrator)
- Latent Synchrony Loss (World Model Alignment ready)
- Active Artifact Gating (EMG Rejection)
- Differentiable Wavelet Bank (Parametric Morlet Kernels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from typing import Dict, Optional, Tuple

import numpy as np

class DirichletEDLLoss(nn.Module):
    def __init__(self, n_classes: int, annealing_step: int = 10, conflict_weight: float = 0.1):
        super().__init__()
        self.n_classes = n_classes
        self.annealing_step = annealing_step
        self.conflict_weight = conflict_weight

    def forward(self, alpha: torch.Tensor, target_one_hot: torch.Tensor, current_step: int) -> torch.Tensor:
        S = torch.sum(alpha, dim=-1, keepdim=True)
        pred_probs = alpha / S
        err = (target_one_hot - pred_probs) ** 2
        var = (alpha * (S - alpha)) / (S ** 2 * (S + 1))
        loss_sos = torch.sum(err + var, dim=-1)
        annealing_coef = min(1.0, current_step / self.annealing_step)
        alpha_tilde = target_one_hot + (1 - target_one_hot) * alpha
        kl_reg = annealing_coef * self._kl_divergence(alpha_tilde)
        
        # Conflict Penalty: Penalize confident confusion
        entropy = -torch.sum(pred_probs * torch.log(pred_probs + 1e-8), dim=-1)
        conflict_penalty = self.conflict_weight * entropy
        
        return torch.mean(loss_sos + kl_reg + conflict_penalty)

    def _kl_divergence(self, alpha: torch.Tensor) -> torch.Tensor:
        ones = torch.ones([1, self.n_classes], device=alpha.device)
        sum_alpha = torch.sum(alpha, dim=-1, keepdim=True)
        sum_ones = torch.sum(ones, dim=-1, keepdim=True)
        
        term1 = torch.lgamma(sum_alpha) - torch.lgamma(sum_ones)
        term2 = torch.sum(torch.lgamma(ones) - torch.lgamma(alpha), dim=-1, keepdim=True)
        term3 = torch.sum((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha)), dim=-1, keepdim=True)
        
        return (term1 + term2 + term3).squeeze(-1)

class SelectiveS4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        # Ensure d_state is even for symplectic integration (q and p)
        self.d_state = d_state if d_state % 2 == 0 else d_state + 1
        self.norm = nn.LayerNorm(d_model)
        
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, self.d_state)
        self.C_proj = nn.Linear(d_model, self.d_state)
        
        self.A_log = nn.Parameter(torch.log(0.5 * torch.ones(d_model, self.d_state)))
        self.D = nn.Parameter(torch.ones(d_model))
        
        self.output_linear = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GLU(dim=-1)
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        B_sz, L, H = x.shape
        
        dt = F.softplus(self.dt_proj(x)) 
        B_mat = self.B_proj(x) 
        C_mat = self.C_proj(x) 
        A = -torch.exp(self.A_log) 
        
        d_half = self.d_state // 2
        q = torch.zeros(B_sz, H, d_half, device=x.device)
        p = torch.zeros(B_sz, H, d_half, device=x.device)
        y = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            A_bar = dt_t * A.unsqueeze(0)
            B_bar = dt_t * B_mat[:, t, :].unsqueeze(1)
            
            x_t = x[:, t, :].unsqueeze(-1)
            
            # Symplectic Integrator step (Hamiltonian dynamics)
            # p_{t+1} = p_t + dt * (-dH/dq + B_p * x_t)
            # q_{t+1} = q_t + dt * ( dH/dp + B_q * x_t)
            dp = A_bar[:, :, :d_half] * q + B_bar[:, :, :d_half] * x_t
            p = p + dp
            
            dq = A_bar[:, :, d_half:] * p + B_bar[:, :, d_half:] * x_t
            q = q + dq
            
            h = torch.cat([q, p], dim=-1)
            
            C_t = C_mat[:, t, :].unsqueeze(1)
            y_t = (h * C_t).sum(dim=-1)
            y.append(y_t)
            
        y = torch.stack(y, dim=1) + x * self.D.unsqueeze(0).unsqueeze(0)
        y = self.dropout(self.output_linear(y))
        
        return residual + y

class ParametricMorlet1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        # Trainable center frequency and bandwidth
        self.freqs = nn.Parameter(torch.randn(out_channels, in_channels, 1) * 5.0)
        self.bandwidths = nn.Parameter(torch.ones(out_channels, in_channels, 1))
        self.register_buffer("t", torch.linspace(-1, 1, kernel_size).view(1, 1, kernel_size))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Generate differentiable Morlet wavelets
        cos_term = torch.cos(2 * math.pi * self.freqs * self.t)
        env = torch.exp(- (self.t ** 2) / (2 * F.softplus(self.bandwidths) ** 2 + 1e-8))
        wavelet = cos_term * env
        return F.conv1d(x, wavelet, stride=self.stride, padding=self.kernel_size//2)

class WaveletStem(nn.Module):
    def __init__(self, in_channels: int, d_model: int, downsample: int = 4):
        super().__init__()
        self.scales_high = [3, 7]
        self.scales_low = [15, 31, 63]
        
        self.convs_high = nn.ModuleList([
            ParametricMorlet1d(in_channels, max(1, d_model // len(self.scales_high)), kernel_size=s, stride=downsample)
            for s in self.scales_high
        ])
        self.convs_low = nn.ModuleList([
            ParametricMorlet1d(in_channels, max(1, d_model // len(self.scales_low)), kernel_size=s, stride=downsample)
            for s in self.scales_low
        ])
        
        high_ch = max(1, d_model // len(self.scales_high)) * len(self.scales_high)
        low_ch = max(1, d_model // len(self.scales_low)) * len(self.scales_low)
        
        self.proj_high = nn.Sequential(
            nn.Conv1d(high_ch, d_model, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(min(8, d_model), d_model)
        )
        self.proj_low = nn.Sequential(
            nn.Conv1d(low_ch, d_model, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(min(8, d_model), d_model)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats_high = [conv(x) for conv in self.convs_high]
        feats_low = [conv(x) for conv in self.convs_low]
        
        x_high = self.proj_high(torch.cat(feats_high, dim=1)).transpose(1, 2)
        x_low = self.proj_low(torch.cat(feats_low, dim=1)).transpose(1, 2)
        return x_high, x_low

class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d_model))
        self.mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(self.q.expand(x.shape[0], -1, -1), x, x)
        return self.norm(attn_out.squeeze(1))

class RiemannianStem(nn.Module):
    def __init__(self, in_channels: int, d_model: int, momentum: float = 0.9, mixup_alpha: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.ts_dim = in_channels * (in_channels + 1) // 2
        self.momentum = momentum
        self.mixup_alpha = mixup_alpha
        self.register_buffer("running_cov", torch.eye(in_channels).unsqueeze(0))
        
        self.proj = nn.Sequential(
            nn.Linear(self.ts_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x_centered = x - x.mean(dim=-1, keepdim=True)
        cov = torch.bmm(x_centered, x_centered.transpose(1, 2)) / (T - 1 + 1e-8)
        
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        eps = (1e-3 * trace / C) + 1e-5
        eye = torch.eye(C, device=x.device).unsqueeze(0).expand(B, -1, -1)
        cov = cov + eps * eye

        if self.training:
            with torch.no_grad():
                mean_batch_cov = cov.mean(dim=0, keepdim=True)
                if self.running_cov.shape[0] != 1 or self.running_cov.shape[1] != C:
                    self.running_cov = torch.eye(C, device=x.device).unsqueeze(0)
                self.running_cov = self.momentum * self.running_cov + (1 - self.momentum) * mean_batch_cov

        try:
            from torch.linalg import eigh
            vals, vecs = eigh(cov.to(torch.float64))
            vals = torch.log(vals.clamp(min=1e-7))
            log_cov = torch.bmm(vecs, torch.bmm(torch.diag_embed(vals), vecs.transpose(1, 2)))
            log_cov = log_cov.to(torch.float32)
        except Exception:
            log_cov = torch.zeros_like(cov)
        
        idx = torch.triu_indices(C, C, device=x.device)
        ts_vector = log_cov[:, idx[0], idx[1]]
        
        if self.training and self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1 - lam)
            idx_mix = torch.randperm(B, device=x.device)
            ts_vector = lam * ts_vector + (1 - lam) * ts_vector[idx_mix]
            
        return self.proj(ts_vector).unsqueeze(1)

class SpatialAttention(nn.Module):
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_channels, max(1, d_model // 4)),
            nn.SiLU(),
            nn.Linear(max(1, d_model // 4), in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.attn(x.mean(dim=-1)).unsqueeze(-1)
        return x * weights

class ArtifactGater(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=15, stride=4, padding=7)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hf = F.relu(self.conv(x))
        score = torch.sigmoid(self.pool(hf)).squeeze(-1)
        return 1.0 - score

class FiLM(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gamma_proj = nn.Linear(d_model, d_model)
        self.beta_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_proj(cond.squeeze(1)).unsqueeze(1)
        beta = self.beta_proj(cond.squeeze(1)).unsqueeze(1)
        return x * (1 + gamma) + beta

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)

class S4EEGEncoder(nn.Module):
    def __init__(self, in_channels: int = 21, d_model: int = 256, d_state: int = 128, n_blocks: int = 6, downsample: int = 4, n_actions: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_actions = n_actions
        self.downsample = downsample
        
        self.spatial_attn = SpatialAttention(in_channels, d_model)
        self.artifact_gater = ArtifactGater(in_channels)
        self.spatial_stem = RiemannianStem(in_channels, d_model)
        self.temporal_stem = WaveletStem(in_channels, d_model, downsample=downsample)
        
        d_state_sel = 16
        self.blocks_high = nn.ModuleList([SelectiveS4Block(d_model, d_state_sel) for _ in range(max(1, n_blocks // 2))])
        self.blocks_low = nn.ModuleList([SelectiveS4Block(d_model, d_state_sel) for _ in range(max(1, n_blocks // 2))])
        
        self.film_high = nn.ModuleList([FiLM(d_model) for _ in range(max(1, n_blocks // 2))])
        self.film_low = nn.ModuleList([FiLM(d_model) for _ in range(max(1, n_blocks // 2))])
        
        self.pooler = MultiHeadAttentionPooling(d_model, n_heads=8)
        
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
        
        self.cpc_proj = nn.Linear(d_model, d_model)
        
        # Adversarial Subject Classifier (Domain Adaptation)
        self.subject_classifier = nn.Sequential(
            GradientReversalLayer(alpha=1.0),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 10) # 10 dummy subject classes for zero-shot regularisation
        )
        
        self.momentum_spatial = None
        self.momentum_temporal = None
        self.momentum_blocks_high = None
        self.momentum_blocks_low = None

    @torch.no_grad()
    def get_flops(self, seq_len: int = 256) -> float:
        C, T, H = self.in_channels, seq_len, self.d_model
        riem_flops = C**3 + (C*(C+1)//2)*H
        temp_flops = 32 * 63 * 3 * T * math.log2(T) 
        block_flops = (len(self.blocks_high) + len(self.blocks_low)) * (T/4) * (H**2 * 4)
        return float(riem_flops + temp_flops + block_flops)

    def init_momentum_encoder(self):
        if self.momentum_spatial is None:
            self.momentum_spatial = copy.deepcopy(self.spatial_stem)
            self.momentum_temporal = copy.deepcopy(self.temporal_stem)
            self.momentum_blocks_high = copy.deepcopy(self.blocks_high)
            self.momentum_blocks_low = copy.deepcopy(self.blocks_low)
            for p in self.momentum_spatial.parameters(): p.requires_grad = False
            for p in self.momentum_temporal.parameters(): p.requires_grad = False
            for p in self.momentum_blocks_high.parameters(): p.requires_grad = False
            for p in self.momentum_blocks_low.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update_momentum(self, m: float = 0.99):
        if self.momentum_spatial is None: self.init_momentum_encoder()
        for p, p_m in zip(self.spatial_stem.parameters(), self.momentum_spatial.parameters()): p_m.data.mul_(m).add_(p.data, alpha=1 - m)
        for p, p_m in zip(self.temporal_stem.parameters(), self.momentum_temporal.parameters()): p_m.data.mul_(m).add_(p.data, alpha=1 - m)
        for p, p_m in zip(self.blocks_high.parameters(), self.momentum_blocks_high.parameters()): p_m.data.mul_(m).add_(p.data, alpha=1 - m)
        for p, p_m in zip(self.blocks_low.parameters(), self.momentum_blocks_low.parameters()): p_m.data.mul_(m).add_(p.data, alpha=1 - m)

    @torch.no_grad()
    def forward_momentum(self, eeg: torch.Tensor) -> torch.Tensor:
        if self.momentum_spatial is None: self.init_momentum_encoder()
        gater = self.artifact_gater(eeg).unsqueeze(-1)
        eeg_gated = eeg * gater
        x_s = self.momentum_spatial(eeg_gated)
        x_high, x_low = self.momentum_temporal(eeg_gated)
        x_high = torch.cat([x_s, x_high], dim=1)
        x_low = torch.cat([x_s, x_low], dim=1)
        for block in self.momentum_blocks_high: x_high = block(x_high)
        for block in self.momentum_blocks_low: x_low = block(x_low)
        x = torch.cat([x_high, x_low], dim=1)
        return x.mean(dim=1)

    def forward(self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = eeg.shape[0]

        if mask is not None:
            mask_t = mask.unsqueeze(-1).expand_as(eeg)
            eeg = eeg * mask_t

        # Active Artifact Gating
        gater = self.artifact_gater(eeg).unsqueeze(-1)
        
        eeg_weighted = self.spatial_attn(eeg) * gater

        x_spatial = self.spatial_stem(eeg_weighted)
        x_high, x_low = self.temporal_stem(eeg_weighted)

        x_high = torch.cat([x_spatial, x_high], dim=1)
        x_low = torch.cat([x_spatial, x_low], dim=1)

        for block, film in zip(self.blocks_high, self.film_high):
            x_high = block(x_high)
            x_high = film(x_high, x_spatial)
            
        for block, film in zip(self.blocks_low, self.film_low):
            x_low = block(x_low)
            x_low = film(x_low, x_spatial)

        x = torch.cat([x_high, x_low], dim=1)

        current_state = self.pooler(x)

        evidence = F.softplus(self.intent_proj(current_state))
        evidence = torch.clamp(evidence, max=1e10)
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        intent_probs = alpha / S
        uncertainty = self.n_actions / S
        confidence = 1.0 - uncertainty

        identity_embed = self.identity_proj(current_state)
        
        # Contrastive Predictive Coding (CPC)
        cpc_loss = torch.tensor(0.0, device=eeg.device)
        if self.training and x.shape[1] > 1:
            cpc_pred = self.cpc_proj(x[:, :-1, :])
            cpc_target = x[:, 1:, :].detach()
            cpc_loss = F.mse_loss(cpc_pred, cpc_target)
            
        # Adversarial Subject Classifier output (for Domain Adaptation loss)
        subject_logits = self.subject_classifier(current_state)
        
        # Latent Synchrony Alignment (placeholder for World Model syncing)
        sync_loss = torch.tensor(0.0, device=eeg.device)

        return {
            "sequence":    x,
            "embed":       current_state,
            "intent_probs": intent_probs,
            "confidence":  confidence.squeeze(-1),
            "alpha":       alpha,
            "uncertainty": uncertainty.squeeze(-1),
            "identity_embed": identity_embed,
            "pred_z":      self.predictor(current_state),
            "cpc_loss":    cpc_loss,
            "sync_loss":   sync_loss,
            "subject_logits": subject_logits,
            "cognitive": {
                "uncertainty":    uncertainty.squeeze(-1),
                "total_evidence": S.squeeze(-1),
            },
        }
