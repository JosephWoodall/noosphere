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
    from ripser import ripser
except ImportError:
    ripser = None

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = F.silu(self.conv(x))
        x = x.transpose(1, 2)
        x1, x2 = x.chunk(2, dim=-1)
        x = x1 * torch.sigmoid(x2) 
        x = self.out_proj(x)
        return x + residual

class TopologyExtractor(nn.Module):
    def __init__(self, d_model: int, max_points: int = 64):
        super().__init__()
        self.max_points = max_points
        self.proj = nn.Linear(2, d_model)
        
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        B, T, d = seq.shape
        device = seq.device
        
        betti_features = torch.zeros(B, 2, device=device)
        
        if ripser is None:
            return self.proj(betti_features)
            
        if T > self.max_points:
            seq_np = seq[:, -self.max_points:, :].detach().cpu().numpy()
        else:
            seq_np = seq.detach().cpu().numpy()
            
        for b in range(B):
            try:
                dgns = ripser(seq_np[b], maxdim=1)['dgms']
                
                h0_lifetimes = dgns[0][:-1, 1] - dgns[0][:-1, 0] if len(dgns[0]) > 1 else np.array([0.0])
                betti_0_sum = np.sum(h0_lifetimes)
                
                h1_lifetimes = dgns[1][:, 1] - dgns[1][:, 0] if len(dgns[1]) > 0 else np.array([0.0])
                betti_1_sum = np.sum(h1_lifetimes)
                
                betti_features[b, 0] = float(betti_0_sum)
                betti_features[b, 1] = float(betti_1_sum)
            except Exception:
                pass
                
        return self.proj(betti_features)

class S4EEGEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, d_model: int = 256, d_state: int = 64, n_blocks: int = 4, downsample: int = 4, n_actions: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_actions = n_actions
        
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, d_model // 2, kernel_size=15, stride=2, padding=7),
            nn.GELU(),
            nn.GroupNorm(8, d_model // 2),
            nn.Conv1d(d_model // 2, d_model, kernel_size=7, stride=downsample // 2, padding=3),
            nn.GELU(),
            nn.GroupNorm(16, d_model)
        )

        self.blocks = nn.ModuleList([S4Block(d_model, d_state) for _ in range(n_blocks)])
        
        self.tda = TopologyExtractor(d_model)
        
        # Only the Discrete Cognitive head remains. Spatial head moved to perception.py
        self.intent_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, n_actions))
        
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
        return x[:, -1, :]

    def forward(self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = eeg.shape[0]
        
        if mask is not None:
            mask_t = mask.unsqueeze(-1).expand_as(eeg)
            eeg = eeg * mask_t

        x = self.stem(eeg)
        x = x.transpose(1, 2)
        for block in self.blocks: x = block(x)
            
        current_state = x[:, -1, :] 

        evidence = F.softplus(self.intent_proj(current_state))
        alpha = evidence + 1.0
        S = torch.sum(alpha, dim=-1, keepdim=True)
        intent_probs = alpha / S
        
        uncertainty = self.n_actions / S
        confidence = 1.0 - uncertainty
        
        topo_embed = self.tda(x)

        return {
            "sequence": x,                 
            "embed": current_state,        
            "topological": topo_embed,
            "intent_probs": intent_probs,  
            "confidence": confidence.squeeze(-1),      
            "alpha": alpha,                
            "uncertainty": uncertainty.squeeze(-1),
            "pred_z": self.predictor(current_state),
            "cognitive": {
                "uncertainty": uncertainty.squeeze(-1),
                "total_evidence": S.squeeze(-1)
            }
        }