"""
Action-Conditioned Transition Model: h_{t+1} = T(h_t, a_t)

h_t  — ZOH-SSM latent pooled from the last 64 timesteps of window t  (B, d_model)
a_t  — 6-DOF decoded motor command at time t, tanh-bounded ∈ [-1,1]^6  (B, d_dof)

Trained with a self-consistency loss during supervised fine-tuning:
  the planned state T(h, decoder(h).mu) must still decode to the same intent.

This turns the ZOH-SSM + decoder pair into a genuine world model:
given the brain's current dynamical state and its decoded motor intent,
predict the latent state the brain will be in after executing that intent.

Reference: model-based closed-loop BCI — Shenoy & Carmena (2014),
  doi:10.1016/j.neuron.2014.06.020.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionConditionedTransition(nn.Module):
    """
    Residual MLP transition function.

    Architecture: (h ⊕ a) → d_hidden → d_hidden → d_model, with skip h→h.
    LayerNorm + GELU throughout; output re-normalised with LayerNorm so latent
    magnitudes stay compatible with the pre-trained decoder.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_dof: int = 6,
        d_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_dof = d_dof

        self.net = nn.Sequential(
            nn.Linear(d_model + d_dof, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_model),
        )
        # Residual projection: keeps gradients stable when net output is small
        self.skip = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        # Initialise near-identity: small net weights, identity skip
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.eye_(self.skip.weight)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        h : (B, d_model)  current pooled latent
        a : (B, d_dof)    current decoded DOF command, ∈ [-1, 1]
        → h_next : (B, d_model)  predicted next latent
        """
        x = torch.cat([h, a], dim=-1)
        return self.norm(self.skip(h) + self.net(x))


def transition_self_consistency_loss(
    transition: ActionConditionedTransition,
    decoder,
    h: torch.Tensor,
    weight: float = 0.1,
) -> torch.Tensor:
    """
    Self-consistency loss: T(h, decoder(h).mu) must decode to the same intent.

    Stops gradient through the target mu so the transition model learns to
    map toward the decoder manifold, not to pull the decoder with it.

    weight: scales the loss relative to the primary supervised loss.
    """
    with torch.no_grad():
        a = decoder(h).mu          # (B, 6) — current decoded action, no grad
    h_plan = transition(h, a)      # (B, d_model)
    mu_plan = decoder(h_plan).mu   # (B, 6)
    return weight * nn.functional.mse_loss(mu_plan, a)
