"""
noosphere/learning.py
=====================
Multi-Mode Learning System

Features:
- Purged dead RL classes (StepNFTPolicy/Loss).
- Exclusively manages representation (SIGReg) and physical calibration (Supervised/Huber).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from dataclasses import dataclass

class LearningSignal:
    REWARD         = 0x01
    ANOMALY        = 0x03
    CORRECTION     = 0x04

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

class SIGRegLoss(nn.Module):
    def __init__(self, lambda_mu: float = 1.0):
        super().__init__()
        self.lambda_mu = lambda_mu

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, d = z.shape
        mu = z.mean(dim=0)
        z_centered = z - mu
        cov = (z_centered.T @ z_centered) / (B - 1 + 1e-8)
        
        loss_cov = F.mse_loss(cov, torch.eye(d, device=z.device))
        loss_mu = self.lambda_mu * torch.norm(mu, p=2) ** 2
        
        loss = loss_cov + loss_mu
        return loss, {
            "unsupervised/sigreg_loss": loss.item(),
            "unsupervised/cov_mse": loss_cov.item(),
            "unsupervised/mu_norm": loss_mu.item()
        }

class SpatialTopologyLoss(nn.Module):
    def __init__(self, d_model: int, n_channels: int = 3):
        super().__init__()
        self.adj_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_channels * n_channels)
        )
        
    def forward(self, topo_embed: torch.Tensor, raw_eeg: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, C, T = raw_eeg.shape
        # Compute empirical spatial cross-correlation
        eeg_centered = raw_eeg - raw_eeg.mean(dim=-1, keepdim=True)
        eeg_std = raw_eeg.std(dim=-1, keepdim=True) + 1e-8
        eeg_norm = eeg_centered / eeg_std
        true_adj = torch.bmm(eeg_norm, eeg_norm.transpose(1, 2)) / (T - 1)
        
        # Predict spatial adjacency from the topological embedding
        pred_adj = self.adj_predictor(topo_embed).view(B, C, C)
        pred_adj = (pred_adj + pred_adj.transpose(1, 2)) / 2.0
        
        loss = F.mse_loss(pred_adj, true_adj)
        return loss, {"unsupervised/spatial_topo_loss": loss.item()}

@dataclass
class LearningConfig:
    mode:                 str   = "all"
    supervised_weight:    float = 1.0
    unsupervised_weight:  float = 0.5   
    position_error_weight:float = 1.5   
    sigreg_lambda_mu:     float = 1.0
    reach_penalty:        float = 0.5
    max_reach:            float = 0.70
    huber_delta:          float = 0.05  
    d_model:              int   = 256
    n_channels:           int   = 3

class LearningManager:
    def __init__(self, cfg: LearningConfig = LearningConfig()):
        self.cfg      = cfg
        self.sup      = SupervisedCoordinateLoss(cfg.max_reach, cfg.reach_penalty)
        self.pos_err  = PositionErrorLoss(delta=cfg.huber_delta * 0.6)
        self.sigreg   = SIGRegLoss(lambda_mu=cfg.sigreg_lambda_mu)
        self._pending_corrections: List[Dict] = []

    def compute_supervised_loss(self, pred_xyz: torch.Tensor, true_xyz: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.sup(pred_xyz, true_xyz)
        return self.cfg.supervised_weight * loss, m

    def compute_position_error_loss(self, predicted_xyz: torch.Tensor, actual_tip: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.pos_err(predicted_xyz, actual_tip)
        return self.cfg.position_error_weight * loss, m

    def compute_unsupervised_loss(self, eeg: torch.Tensor, encoder_fn) -> Tuple[torch.Tensor, Dict]:
        z = encoder_fn(eeg) 
        loss, m = self.sigreg(z)
        return self.cfg.unsupervised_weight * loss, m

    def queue_correction(self, embedding: np.ndarray, actual_tip: np.ndarray):
        self._pending_corrections.append({"embedding": embedding, "actual_tip": actual_tip})

    def drain_corrections(self) -> List[Dict]:
        out = self._pending_corrections.copy()
        self._pending_corrections.clear()
        return out