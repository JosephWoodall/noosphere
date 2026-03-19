"""
noosphere/rssm.py
=================
Recurrent State Space Model (RSSM)

The latent dynamics core. Maintains a structured world state sₜ = (hₜ, zₜ):

    hₜ  — deterministic GRU state      captures long-range temporal dependencies
    zₜ  — stochastic discrete latent   captures uncertainty and multimodal futures

Two forward modes:
    observe_step   uses real observations → trains the posterior q(zₜ | hₜ, oₜ)
    imagine_step   no observations       → runs the prior p(ẑₜ | hₜ) for planning

The KL loss KL(q ‖ p) forces the prior to anticipate the posterior — this is
what enables the model to imagine plausible futures without seeing real data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, kl_divergence
from typing import Tuple, Dict


class StraightThroughOneHot(nn.Module):
    """
    Discrete categorical latents via straight-through estimator.
    Forward:  argmax  (discrete — no gradient)
    Backward: softmax (continuous — gradient flows)
    """
    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        probs = F.softmax(logits, dim=-1)
        hard  = F.one_hot(probs.argmax(dim=-1), self.n_classes).float()
        hard  = hard + probs - probs.detach()
        return hard, probs


class RSSM(nn.Module):
    """
    Recurrent State Space Model.

    Parameters
    ----------
    embed_dim    : observation embedding dimension (from perception)
    action_dim   : action embedding dimension
    det_dim      : GRU hidden state size
    stoch_cats   : number of categorical variables (default 32)
    stoch_classes: classes per categorical variable (default 32)
    hidden_dim   : MLP hidden size
    """

    def __init__(
        self,
        embed_dim:     int = 512,
        action_dim:    int = 64,
        det_dim:       int = 512,
        stoch_cats:    int = 32,
        stoch_classes: int = 32,
        hidden_dim:    int = 512,
    ):
        super().__init__()
        self.det_dim      = det_dim
        self.stoch_cats   = stoch_cats
        self.stoch_classes= stoch_classes
        self.stoch_dim    = stoch_cats * stoch_classes

        self.gru_input_proj = nn.Linear(self.stoch_dim + action_dim, det_dim)
        self.gru            = nn.GRUCell(det_dim, det_dim)

        self.prior_mlp = nn.Sequential(
            nn.Linear(det_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, stoch_cats * stoch_classes),
        )
        self.posterior_mlp = nn.Sequential(
            nn.Linear(det_dim + embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, stoch_cats * stoch_classes),
        )
        self.st = StraightThroughOneHot(stoch_classes)

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        return {
            "h": torch.zeros(batch_size, self.det_dim,   device=device),
            "z": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Planning step — no observation. Returns (h_next, z_next, prior_probs)."""
        x      = self.gru_input_proj(torch.cat([z, a], dim=-1))
        h_next = self.gru(x, h)
        logits = self.prior_mlp(h_next).view(-1, self.stoch_cats, self.stoch_classes)
        z_next, prior_probs = self.st(logits)
        return h_next, z_next.view(-1, self.stoch_dim), prior_probs

    def observe_step(
        self,
        h:         torch.Tensor,
        z:         torch.Tensor,
        a:         torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training step — uses real obs. Returns (h_next, z_next, prior_probs, posterior_probs)."""
        x      = self.gru_input_proj(torch.cat([z, a], dim=-1))
        h_next = self.gru(x, h)

        prior_logits = self.prior_mlp(h_next).view(-1, self.stoch_cats, self.stoch_classes)
        _, prior_probs = self.st(prior_logits)

        post_logits  = self.posterior_mlp(torch.cat([h_next, obs_embed], dim=-1))
        post_logits  = post_logits.view(-1, self.stoch_cats, self.stoch_classes)
        z_next, posterior_probs = self.st(post_logits)

        return h_next, z_next.view(-1, self.stoch_dim), prior_probs, posterior_probs

    def kl_loss(
        self,
        prior_probs:     torch.Tensor,
        posterior_probs: torch.Tensor,
        free_nats: float = 1.0,
        balance:   float = 0.8,
    ) -> torch.Tensor:
        """
        Balanced KL (DreamerV3):
            L = 0.8 · KL(sg(q) ‖ p)  +  0.2 · KL(q ‖ sg(p))
        free_nats clips minimum to avoid posterior collapse.
        """
        p = OneHotCategorical(probs=prior_probs)
        q = OneHotCategorical(probs=posterior_probs)
        kl = (
            balance       * kl_divergence(OneHotCategorical(probs=posterior_probs.detach()), p).sum(-1) +
            (1 - balance) * kl_divergence(q, OneHotCategorical(probs=prior_probs.detach())).sum(-1)
        )
        return kl.clamp(min=free_nats).mean()

    @property
    def state_dim(self) -> int:
        return self.det_dim + self.stoch_dim


class ConsequenceModel(nn.Module):
    """
    Predicts consequences from latent state sₜ:
        reward      — scalar regression
        termination — binary classification
        value       — N-step return estimate (critic baseline)
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        def _head(out):
            return nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, out),
            )
        self.reward_head      = _head(1)
        self.termination_head = _head(1)
        self.value_head       = _head(1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "reward":      self.reward_head(state).squeeze(-1),
            "termination": torch.sigmoid(self.termination_head(state).squeeze(-1)),
            "value":       self.value_head(state).squeeze(-1),
        }


class ObservationDecoder(nn.Module):
    """Decodes latent state back to observation embedding space for reconstruction loss."""
    def __init__(self, state_dim: int, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)
