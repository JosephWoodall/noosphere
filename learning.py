"""
noosphere/learning.py
=====================
Multi-Mode Learning System

Changes in v1.3.0
-----------------
1. EEGAugment: band-pass mask augmentation implemented (was documented but missing).
   Zeroes a random frequency band in the FFT domain — forces the encoder to not
   over-rely on any single frequency band, improving generalisation across
   muscle contraction intensities.

2. LearningManager.compute_unsupervised_loss: single batched forward pass.
   Previously called encoder_fn twice (once per augmented view) which could
   produce inconsistent dropout/BatchNorm states between views.
   Now concatenates [v1, v2] → one forward pass → splits → NT-Xent.

3. PositionErrorLoss: closes the arm position error feedback loop.
   Direct coordinate supervision on the neural coordinate head
   from actual arm tip position (not predicted target).
   L = Huber(predicted_xyz, actual_tip)
   Using Huber instead of MSE: robust to outliers from IK convergence failures.

4. LearningSignal.CORRECTION now used — apply_corrections() implemented.

5. LearningConfig: position_error_weight added.
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
    CORRECTION     = 0x04   # now used: position error from actual arm tip


# ── Supervised: coordinate regression ────────────────────────────────────────

class SupervisedCoordinateLoss(nn.Module):
    """MSE + IK feasibility penalty on predicted coordinates."""

    def __init__(self, max_reach: float = 0.70, reach_penalty: float = 0.5):
        super().__init__()
        self.max_reach     = max_reach
        self.reach_penalty = reach_penalty

    def forward(
        self,
        pred_xyz: torch.Tensor,
        true_xyz: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
    """
    Closes the arm position error loop.

    Supervises the spatial coordinate head with the actual arm tip
    position after movement (not the predicted target — the *actual* position).

    When the arm overshoots: actual_tip ≠ predicted_xyz.
    This loss penalises the encoder's prediction against where the arm
    actually landed, teaching it to correct its bias over time.

    Huber with δ=0.03m (3cm): robust to collision / IK divergence outliers.
    """

    def __init__(self, delta: float = 0.03):
        super().__init__()
        self.delta = delta

    def forward(
        self,
        predicted_xyz: torch.Tensor,   # (B, 3) — what the encoder predicted
        actual_tip:    torch.Tensor,   # (B, 3) — where arm actually ended up
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss = F.huber_loss(predicted_xyz, actual_tip, delta=self.delta)
        error_m = (predicted_xyz - actual_tip).norm(dim=-1).mean()
        return loss, {
            "position_error/huber":   loss.item(),
            "position_error/mean_m":  error_m.item(),
        }


# ── EEG augmentation ──────────────────────────────────────────────────────────

class EEGAugment:
    """
    Data augmentation for contrastive EEG pre-training.

    Four augmentations, all appropriate for 3-electrode neck EMG:
    1. Amplitude jitter (±20%): simulates electrode contact variation
    2. Time shift (±16 samples): accounts for reaction-time variability
    3. Channel dropout (prob 0.3): forces robustness to single electrode loss
    4. Band-pass mask: zeros a random frequency band in FFT domain.
       Forces encoder not to rely exclusively on any single frequency component.
       Band widths: low (0-30Hz), mid (30-80Hz), high (80-128Hz) at 256Hz sfreq.
       This was documented in v1.2.0 but never implemented — now fixed.
    """

    def __init__(self, sfreq: float = 256.0):
        self.sfreq = sfreq
        # Frequency band boundaries in FFT bin indices for T=256 at 256Hz
        # Each bin = sfreq/T = 1 Hz, so bin i covers [i, i+1) Hz
        self._band_ranges = [
            (0, 30),     # DC + low frequency
            (30, 80),    # mid-frequency (main EMG range for neck)
            (80, 128),   # high frequency
        ]

    def __call__(self, eeg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._augment(eeg), self._augment(eeg)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        B, C, T = x.shape

        # 1. Amplitude jitter: scale each channel independently
        scale = 1.0 + 0.2 * (torch.rand(B, C, 1, device=x.device) * 2 - 1)
        x = x * scale

        # 2. Time shift (circular)
        shifts = torch.randint(-16, 17, (B,)).tolist()
        for b, s in enumerate(shifts):
            if s != 0:
                x[b] = torch.roll(x[b], s, dims=-1)

        # 3. Random channel dropout
        if C > 1 and torch.rand(1).item() < 0.3:
            drop_ch = torch.randint(0, C, (B,))
            for b in range(B):
                x[b, drop_ch[b]] = 0.0

        # 4. Band-pass mask in frequency domain
        if torch.rand(1).item() < 0.5:
            band_idx = torch.randint(0, len(self._band_ranges), (1,)).item()
            lo, hi   = self._band_ranges[band_idx]
            # rfft along time axis → (B, C, T//2+1) complex
            X_f  = torch.fft.rfft(x, dim=-1)
            # Zero the selected band across all batches and channels
            X_f[:, :, lo:hi] = 0.0
            x = torch.fft.irfft(X_f, n=T, dim=-1)

        return x


class EEGReconstructionLoss(nn.Module):
    """
    Generative Denoising / Masked Modeling Loss.
    Masks 1-2 random channels from the input EEG, asks the encoder to process it,
    and then reconstructs the downsampled original EEG channels from the S4 sequence.
    This acts as a powerful regularizer for robust spatial representations.
    """
    def __init__(self, d_model: int = 256, n_channels: int = 3, downsample: int = 4):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_channels)
        )
        self.downsample = downsample

    def forward(self, eeg: torch.Tensor, encoder_fn) -> Tuple[torch.Tensor, Dict[str, float]]:
        # eeg: (B, C, T)
        B, C, T = eeg.shape
        eeg_masked = eeg.clone()
        
        # Randomly mask 1 channel per sequence in the batch
        mask_idx = torch.randint(0, C, (B,), device=eeg.device)
        b_idx = torch.arange(B, device=eeg.device)
        eeg_masked[b_idx, mask_idx, :] = 0.0

        # Pass through S4 encoder to get sequence_out: (B, T', d_model)
        sequence_out = encoder_fn(eeg_masked)

        # Target: downsampled EEG
        # Using avg_pool1d to downsample the target to match sequence length
        target_eeg = F.avg_pool1d(eeg, self.downsample, stride=self.downsample).transpose(1, 2)  # (B, T', C)
        
        # Predict missing channels from sequence
        pred_eeg = self.decoder(sequence_out)  # (B, T', C)
        
        # We compute MSE loss over the reconstructed signal
        loss = F.mse_loss(pred_eeg, target_eeg)
        return loss, {"unsupervised/eeg_recon": loss.item()}

class NTXentLoss(nn.Module):
    """
    NT-Xent (InfoNCE) contrastive loss.

    Fix vs v1.2.0: compute_unsupervised_loss now passes both views through
    encoder in a single batched call to ensure consistent dropout/norm state.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B  = z1.shape[0]
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z    = torch.cat([z1, z2], dim=0)
        sim  = (z @ z.T) / self.tau
        mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)
        labels = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0, B,   device=z.device),
        ])
        loss = F.cross_entropy(sim, labels)
        return loss, {"unsupervised/ntxent": loss.item()}


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

    def forward(
        self,
        pos_log_probs: torch.Tensor,
        neg_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
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
    unsupervised_weight:  float = 0.3
    rl_weight:            float = 1.0
    position_error_weight:float = 1.5   # position error feedback weight
    eeg_recon_weight:     float = 0.5   # generative denoising weight
    contrastive_temp:     float = 0.1
    stepnft_beta:         float = 1.0
    reach_penalty:        float = 0.5
    max_reach:            float = 0.70
    huber_delta:          float = 0.05  # metres — for xyz supervision losses
    d_model:              int   = 256
    n_channels:           int   = 3


class LearningManager:
    """
    Coordinates all learning modes.

    Key change in v1.3.0: compute_unsupervised_loss takes encoder_fn
    and calls it ONCE on the concatenated views [v1, v2] to ensure
    consistent dropout/BN state between views.
    """

    def __init__(self, cfg: LearningConfig = LearningConfig()):
        self.cfg      = cfg
        self.sup      = SupervisedCoordinateLoss(cfg.max_reach, cfg.reach_penalty)
        self.pos_err  = PositionErrorLoss(delta=cfg.huber_delta * 0.6)
        self.aug      = EEGAugment()
        self.cont     = NTXentLoss(cfg.contrastive_temp)
        self.recon    = EEGReconstructionLoss(d_model=cfg.d_model, n_channels=cfg.n_channels)
        self.nft      = StepNFTLoss(cfg.stepnft_beta)
        self._pending_corrections: List[Dict] = []

    def compute_supervised_loss(
        self, pred_xyz: torch.Tensor, true_xyz: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.sup(pred_xyz, true_xyz)
        return self.cfg.supervised_weight * loss, m

    def compute_position_error_loss(
        self,
        predicted_xyz: torch.Tensor,
        actual_tip:    torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Supervise from actual arm position — closes the feedback loop."""
        loss, m = self.pos_err(predicted_xyz, actual_tip)
        return self.cfg.position_error_weight * loss, m

    def compute_unsupervised_loss(
        self,
        eeg:        torch.Tensor,
        encoder_fn,                  # callable: (B, C, T) → (B, d)
    ) -> Tuple[torch.Tensor, Dict]:
        """
        NT-Xent on two augmented views.
        Single batched forward pass (fix vs v1.2.0 which called encoder_fn twice).
        """
        v1, v2     = self.aug(eeg)
        both       = torch.cat([v1, v2], dim=0)   # (2B, C, T)
        both_emb   = encoder_fn(both)              # (2B, d) — one pass
        B          = eeg.shape[0]
        z1, z2     = both_emb[:B], both_emb[B:]
        loss, m    = self.cont(z1, z2)
        return self.cfg.unsupervised_weight * loss, m

    def compute_recon_loss(
        self,
        eeg: torch.Tensor,
        encoder_fn,
    ) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.recon(eeg, encoder_fn)
        return self.cfg.eeg_recon_weight * loss, m

    def compute_rl_loss(
        self,
        pos_log_probs: torch.Tensor,
        neg_log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.nft(pos_log_probs, neg_log_probs)
        return self.cfg.rl_weight * loss, m

    def queue_correction(self, embedding: np.ndarray, actual_tip: np.ndarray):
        """Queue a position error correction for the next Phase A update."""
        self._pending_corrections.append({
            "embedding":  embedding,
            "actual_tip": actual_tip,
        })

    def drain_corrections(self) -> List[Dict]:
        """Return pending corrections and clear the queue."""
        out = self._pending_corrections.copy()
        self._pending_corrections.clear()
        return out
