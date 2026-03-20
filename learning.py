"""
noosphere/learning.py
=====================
Multi-Mode Learning System

Supports all three paradigms for the intent→movement improvement loop:

    Supervised    — labeled (EEG_features, kinematic_target) pairs
                    Loss: MSE on predicted vs actual coordinates
                    When: initial bootstrap from expert demonstrations

    Unsupervised  — no labels; contrastive self-supervised on EEG segments
                    Loss: NT-Xent (InfoNCE) on augmented EEG pairs
                    When: continuous pre-training on unlabeled EEG stream

    Reinforcement — reward from successful reach without collision
                    Method: π-StepNFT (step-wise negative-aware fine-tuning)
                    When: after world model is functional (post-warmup)

π-StepNFT integration (arxiv 2603.02083):
──────────────────────────────────────────
The paper proposes a critic-free alternative to actor-critic RL that:
    1. Requires only a single forward pass per optimization step
    2. Eliminates the auxiliary value network (saves memory + compute)
    3. Uses step-wise likelihood ratios as implicit critics
    4. Generalizes better OOD by avoiding multimodal feature overfitting

For noosphere, this maps as:
    - "flow steps" → world model imagination rollout steps
    - "negative samples" → imagined trajectories with collision / IK failure
    - "positive samples" → imagined trajectories that reach target cleanly
    - The policy is fine-tuned to increase likelihood of positive step sequences

This is particularly suited to the apparatus control task because:
    - Action space is continuous and wide (6 DOF × angle range)
    - Reward is sparse (reach target without hitting anything)
    - Maintaining a separate critic doubles memory on edge hardware

Learning signal types:
    REWARD         = 0x01   RL scalar reward
    SUPERVISED_XYZ = 0x02   labeled 3D coordinate target
    ANOMALY        = 0x03   anomaly score (unsupervised signal strength)
    CORRECTION     = 0x04   human correction (override signal)
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
    SUPERVISED_XYZ = 0x02
    ANOMALY        = 0x03
    CORRECTION     = 0x04


# ── Supervised: coordinate regression ────────────────────────────────────────

class SupervisedCoordinateLoss(nn.Module):
    """
    MSE loss between predicted coordinates and labeled kinematic targets.
    Applied when labeled (EEG, xyz) pairs are available.

    Also computes IK feasibility penalty: if predicted coordinate is
    outside arm reach, add an L2 penalty proportional to the excess.
    """

    def __init__(self, max_reach: float = 0.70, reach_penalty: float = 0.5):
        super().__init__()
        self.max_reach     = max_reach
        self.reach_penalty = reach_penalty

    def forward(
        self,
        pred_xyz: torch.Tensor,   # (B, 3)
        true_xyz: torch.Tensor,   # (B, 3)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        coord_loss = F.mse_loss(pred_xyz, true_xyz)

        # Penalize predictions outside arm reach
        dist = pred_xyz.norm(dim=-1)   # (B,)
        excess = F.relu(dist - self.max_reach)
        reach_loss = self.reach_penalty * excess.pow(2).mean()

        total = coord_loss + reach_loss
        return total, {
            "supervised/coord_mse": coord_loss.item(),
            "supervised/reach_penalty": reach_loss.item(),
        }


# ── Unsupervised: contrastive EEG pre-training ────────────────────────────────

class EEGAugment:
    """
    Data augmentation for contrastive EEG pre-training.
    Two views of the same segment → same representation.
    Two different segments → different representations.

    Augmentations appropriate for 3-electrode neck EEG:
        - Amplitude jitter (±20%)
        - Time shift (±16 samples)
        - Channel dropout (random electrode zeroed)
        - Band-pass mask (zero one frequency band)
    """

    def __init__(self, sfreq: float = 256.0):
        self.sfreq = sfreq

    def __call__(self, eeg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns two augmented views of eeg: (B, 3, T)."""
        return self._augment(eeg), self._augment(eeg)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        B, C, T = x.shape

        # Amplitude jitter
        scale = 1.0 + 0.2 * (torch.rand(B, C, 1, device=x.device) * 2 - 1)
        x = x * scale

        # Time shift
        shift = torch.randint(-16, 17, (B,)).tolist()
        for b, s in enumerate(shift):
            if s != 0:
                x[b] = torch.roll(x[b], s, dims=-1)

        # Random channel dropout (for 3-ch, drop 1 with prob 0.3)
        if C > 1 and torch.rand(1).item() < 0.3:
            ch = torch.randint(0, C, (B,))
            for b in range(B):
                x[b, ch[b]] = 0.0

        return x


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) for contrastive learning.
    InfoNCE loss that maximizes similarity of positive pairs (same segment, different augmentations)
    and minimizes similarity of negative pairs (different segments).

    Temperature τ controls sharpness: lower τ → harder negatives → richer representations.
    Typical τ = 0.1 for EEG signals.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        z1, z2: (B, d) — embeddings of two views.
        """
        B = z1.shape[0]
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # All pairs: (2B, 2B) similarity matrix
        z    = torch.cat([z1, z2], dim=0)   # (2B, d)
        sim  = (z @ z.T) / self.tau          # (2B, 2B)

        # Mask self-similarity
        mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, -1e9)

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(B, 2*B, device=z.device),
            torch.arange(0, B,   device=z.device),
        ])

        loss = F.cross_entropy(sim, labels)
        return loss, {"unsupervised/ntxent": loss.item()}


# ── π-StepNFT: step-wise negative-aware fine-tuning ──────────────────────────

class StepNFTPolicy(nn.Module):
    """
    π-StepNFT policy (arxiv 2603.02083).

    Critic-free, likelihood-based RL for embodied control.
    Replaces the actor-critic with a single network that is fine-tuned
    using step-wise negative/positive trajectory labeling.

    Key insight: instead of learning a value function, use the ratio of
    log-probabilities between successful and failed step sequences as
    an implicit advantage estimate.

    For noosphere apparatus control:
        Positive trajectory: imagined rollout that reaches target, IK converges,
                             no obstacle collision.
        Negative trajectory: imagined rollout that misses target OR collides.

    The policy is updated to increase log P(positive steps) - log P(negative steps)
    at each position in the trajectory.

    This requires:
        1. A flow/policy model that assigns step-wise likelihoods
        2. A way to label trajectories as positive/negative
        3. A single forward pass per update (no value network)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim

        # Flow policy network (conditions each action step on state)
        self.flow_net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.SiLU(),
        )
        self.mu_head      = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        h = self.flow_net(state)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(-4, 2)
        return torch.distributions.Normal(mu, log_std.exp())

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Step-wise log probability log π(aₜ | sₜ)."""
        return self.forward(state).log_prob(action).sum(-1)

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.forward(state)
        return dist.mean if deterministic else dist.rsample()


class StepNFTLoss(nn.Module):
    """
    Step-wise Negative-aware Fine-Tuning loss.

    For each step t in a trajectory:
        L_t = -β · [log π(aₜ|sₜ)⁺ - log π(aₜ|sₜ)⁻]

    Where ⁺ denotes a positive (successful) trajectory and ⁻ negative.
    β is a step-wise weighting that increases toward the end of the trajectory
    (later steps matter more for reach success).

    Implementation:
        pos_log_probs: (B, H) — log probs for successful imagined trajectories
        neg_log_probs: (B, H) — log probs for failed imagined trajectories
        step_weights:  (H,)   — linear ramp [1/H, 2/H, ..., 1.0]
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        pos_log_probs: torch.Tensor,   # (B, H)
        neg_log_probs: torch.Tensor,   # (B, H)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        H = pos_log_probs.shape[1]
        # Step weights: linearly increasing toward horizon
        w = torch.linspace(1.0/H, 1.0, H, device=pos_log_probs.device)
        step_diff = pos_log_probs - neg_log_probs   # (B, H)
        loss = -self.beta * (step_diff * w.unsqueeze(0)).mean()
        return loss, {
            "stepnft/loss":         loss.item(),
            "stepnft/pos_logp_mean": pos_log_probs.mean().item(),
            "stepnft/neg_logp_mean": neg_log_probs.mean().item(),
        }


# ── Unified learning manager ──────────────────────────────────────────────────

@dataclass
class LearningConfig:
    mode:              str   = "all"    # "supervised" | "unsupervised" | "rl" | "all"
    supervised_weight: float = 1.0
    unsupervised_weight:float= 0.3
    rl_weight:         float = 1.0
    contrastive_temp:  float = 0.1
    stepnft_beta:      float = 1.0
    reach_penalty:     float = 0.5
    max_reach:         float = 0.70


class LearningManager:
    """
    Coordinates all three learning modes.

    On each training step, computes whichever losses are available given
    the current batch and combines them into a single backward pass.
    """

    def __init__(self, cfg: LearningConfig = LearningConfig()):
        self.cfg  = cfg
        self.sup  = SupervisedCoordinateLoss(cfg.max_reach, cfg.reach_penalty)
        self.aug  = EEGAugment()
        self.cont = NTXentLoss(cfg.contrastive_temp)
        self.nft  = StepNFTLoss(cfg.stepnft_beta)
        self._metrics: List[Dict] = []

    def compute_supervised_loss(
        self,
        pred_xyz: torch.Tensor,
        true_xyz: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.sup(pred_xyz, true_xyz)
        return self.cfg.supervised_weight * loss, m

    def compute_unsupervised_loss(
        self,
        eeg:         torch.Tensor,      # (B, 3, T)
        encoder_fn,                     # callable: eeg → embedding (B, d)
    ) -> Tuple[torch.Tensor, Dict]:
        v1, v2   = self.aug(eeg)
        z1, z2   = encoder_fn(v1), encoder_fn(v2)
        loss, m  = self.cont(z1, z2)
        return self.cfg.unsupervised_weight * loss, m

    def compute_rl_loss(
        self,
        pos_log_probs: torch.Tensor,    # (B, H)
        neg_log_probs: torch.Tensor,    # (B, H)
    ) -> Tuple[torch.Tensor, Dict]:
        loss, m = self.nft(pos_log_probs, neg_log_probs)
        return self.cfg.rl_weight * loss, m
