"""
noosphere/learning.py
=====================
Multi-Mode Learning System with SIGReg

Changes:
1. Replaced NT-Xent and EEGReconstruction with LeWM's SIGRegLoss.
2. Enforces N(0, I) on embeddings to prevent representation collapse.
3. Closes the position error loop with actual tip feedback via Huber loss.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# ── Learning signal types ─────────────────────────────────────────────────────

class LearningSignal:
    REWARD         = 0x01
    ANOMALY        = 0x03
    CORRECTION     = 0x04

# ── Supervised: coordinate regression ────────────────────────────────────────

class SupervisedCoordinateLoss(nn.Module):
    def __init__(self, max_reach: float = 0.70, reach_penalty: float = 0.5):
        super().__init__()
        self.max_reach     = max_reach
        self.reach_penalty = reach_penalty

    def forward(self, pred_xyz: torch.Tensor, true_xyz: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        coord_loss = F.mse_loss(pred_xyz, true_xyz)
        dist       = pred_xyz.norm(dim=-1)
        excess     = F.relu(dist - self.max_reach)
        reach_loss = self.reach_penalty * excess.pow(2).mean()
        total      = coord_loss + reach_loss
        return total, {
            "supervised/coord_mse":      coord_loss.item(),
            "supervised/reach_penalty":  reach_loss.item(),
        }

# ── Position error feedback loss ──────────────────────────────────────────────

class PositionErrorLoss(nn.Module):
    def __init__(self, delta: float = 0.03):
        super().__init__()
        self.delta = delta

    def forward(self, predicted_xyz: torch.Tensor, actual_tip: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = F.huber_loss(predicted_xyz, actual_tip, delta=self.delta)
        error_m = (predicted_xyz - actual_tip).norm(dim=-1).mean()
        return loss, {
            "position_error/huber":   loss.item(),
            "position_error/mean_m":  error_m.item(),
        }

# ── Sketched-Isotropic-Gaussian Regularizer (SIGReg) ──────────────────────────

class SIGRegLoss(nn.Module):
    """
    Sketched-Isotropic-Gaussian Regularizer (from LeWM).
    Enforces the latent embeddings to follow a standard Gaussian N(0, I).
    Mathematically prevents representation collapse without contrastive pairs or EMAs.
    """
    def __init__(self, lambda_mu: float = 1.0):
        super().__init__()
        self.lambda_mu = lambda_mu

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        # z shape: (B, d)
        B, d = z.shape
        mu = z.mean(dim=0)
        z_centered = z - mu
        
        # Empirical covariance matrix
        cov = (z_centered.T @ z_centered) / (B - 1 + 1e-8)
        
        # Objective: (1/d) * || Cov - I ||_F^2 + lambda * || mu ||_2^2
        loss_cov = F.mse_loss(cov, torch.eye(d, device=z.device))
        loss_mu = self.lambda_mu * torch.norm(mu, p=2) ** 2
        
        loss = loss_cov + loss_mu
        return loss, {
            "unsupervised/sigreg_loss": loss.item(),
            "unsupervised/cov_mse": loss_cov.item(),
            "unsupervised/mu_norm": loss_mu.item()
        }

# ── π-StepNFT ─────────────────────────────────────────────────────────────────

class StepNFTPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.state_dim = state_dim; self.action_dim = action_dim
        self.flow_net     = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.SiLU(),
        )
        self.mu_head      = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        h = self.flow_net(state)
        return torch.distributions.Normal(self.mu_head(h),
                                           self.log_std_head(h).clamp(-4, 2).exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(state).log_prob(action).sum(-1)

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        d = self.forward(state)
        return d.mean if deterministic else d.rsample()

class StepNFTLoss(nn.Module):
    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pos_log_probs: torch.Tensor, neg_log_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        H  = pos_log_probs.shape[1]
        w  = torch.linspace(1.0/H, 1.0, H, device=pos_log_probs.device)
        loss = -self.beta * ((pos_log_probs - neg_log_probs) * w.unsqueeze(0)).mean()
        return loss, {
            "stepnft/loss":          loss.item(),
            "stepnft/pos_logp_mean": pos_log_probs.mean().item(),
            "stepnft/neg_logp_mean": neg_log_probs.mean().item(),
        }

# ── Unified learning manager ──────────────────────────────────────────────────

@dataclass
class LearningConfig:
    mode:                 str   = "all"
    supervised_weight:    float = 1.0
    unsupervised_weight:  float = 0.5   # Increased weight for SIGReg
    rl_weight:            float = 1.0
    position_error_weight:float = 1.5   
    sigreg_lambda_mu:     float = 1.0
    stepnft_beta:         float = 1.0
    reach_penalty:        float = 0.5
    max_reach:            float = 0.70
    huber_delta:          float = 0.05  
    d_model:              int   = 256
    n_channels:           int   = 3

class LearningManager:
    """Coordinates all learning modes using SIGReg for unsupervised representation."""
    def __init__(self, cfg: LearningConfig = LearningConfig()):
        self.cfg      = cfg
        self.sup      = SupervisedCoordinateLoss(cfg.max_reach, cfg.reach_penalty)
        self.pos_err  = PositionErrorLoss(delta=cfg.huber_delta * 0.6)
        self.sigreg   = SIGRegLoss(lambda_mu=cfg.sigreg_lambda_mu)
        self.nft      = StepNFTLoss(cfg.stepnft_beta)
        self._pending_corrections: List[Dict] = []

    def compute_supervised_loss(self, pred_xyz: torch.Tensor, true_xyz: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.sup(pred_xyz, true_xyz)
        return self.cfg.supervised_weight * loss, m

    def compute_position_error_loss(self, predicted_xyz: torch.Tensor, actual_tip: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.pos_err(predicted_xyz, actual_tip)
        return self.cfg.position_error_weight * loss, m

    def compute_unsupervised_loss(self, eeg: torch.Tensor, encoder_fn) -> Tuple[torch.Tensor, Dict]:
        """SIGReg forward pass. No augmentations or paired views required."""
        z = encoder_fn(eeg) # (B, d)
        loss, m = self.sigreg(z)
        return self.cfg.unsupervised_weight * loss, m

    def compute_rl_loss(self, pos_log_probs: torch.Tensor, neg_log_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.nft(pos_log_probs, neg_log_probs)
        return self.cfg.rl_weight * loss, m

    def queue_correction(self, embedding: np.ndarray, actual_tip: np.ndarray):
        self._pending_corrections.append({"embedding": embedding, "actual_tip": actual_tip})

    def drain_corrections(self) -> List[Dict]:
        out = self._pending_corrections.copy()
        self._pending_corrections.clear()
        return out