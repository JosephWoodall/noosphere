"""
Continuous intent decoder: SSM hidden state → motor command + uncertainty + ERN probability.

Output space:
  mu    ∈ [-1, 1]^n_dof   — continuous normalized joint command
  sigma ∈ (0, ∞)^n_dof   — aleatoric uncertainty per DOF
  ern   ∈ (0, 1)          — probability of error-related negativity (safety gate input)

High sigma → Kalman filter trusts prediction less, relies more on momentum.
High ern   → SafetyGate halts movement for ern_halt_duration seconds.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MotorIntent(NamedTuple):
    mu: torch.Tensor        # (B, n_dof), values in [-1, 1]
    sigma: torch.Tensor     # (B, n_dof), values > 0
    ern_prob: torch.Tensor  # (B, 1), values in (0, 1)


class IntentDecoder(nn.Module):
    """
    Two-layer MLP from SSM hidden state to (mu, sigma, ern_prob).

    mu is tanh-bounded to keep commands in hardware-safe range.
    sigma uses softplus so it stays positive and gradients never vanish.
    ern_prob is sigmoid — treated as a binary classifier output.
    """

    def __init__(self, d_model: int = 128, n_dof: int = 6, d_hidden: int = 64):
        super().__init__()
        self.n_dof = n_dof
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.mu_head = nn.Linear(d_hidden, n_dof)
        self.sigma_head = nn.Linear(d_hidden, n_dof)
        self.ern_head = nn.Linear(d_hidden, 1)

        nn.init.zeros_(self.mu_head.bias)
        # Initialize sigma head to output small but non-trivial uncertainty
        nn.init.constant_(self.sigma_head.bias, -1.0)

    def forward(self, h: torch.Tensor) -> MotorIntent:
        """
        h: (B, d_model) or (B, T, d_model).
        If T dimension present, uses only the last timestep (causal inference).
        """
        if h.dim() == 3:
            h = h[:, -1, :]  # (B, d_model)

        x = self.norm(h)
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))

        mu = torch.tanh(self.mu_head(x))                    # bounded [-1, 1]
        sigma = F.softplus(self.sigma_head(x)) + 1e-4       # positive
        ern_prob = torch.sigmoid(self.ern_head(x))           # probability

        return MotorIntent(mu=mu, sigma=sigma, ern_prob=ern_prob)


class IntentLoss(nn.Module):
    """
    Training objective combining:
      - NLL under diagonal Gaussian: (mu - target)^2 / (2*sigma^2) + log(sigma)
      - ERN classification: BCE(ern_prob, ern_label)
      - Uncertainty regularizer: penalize sigma collapsing to zero
    """

    def __init__(self, ern_weight: float = 1.0, sigma_reg: float = 0.01):
        super().__init__()
        self.ern_weight = ern_weight
        self.sigma_reg = sigma_reg

    def forward(
        self,
        intent: MotorIntent,
        target_mu: torch.Tensor,
        ern_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Gaussian NLL (equivalent to MSE when sigma is fixed, better when learned)
        nll = (
            (intent.mu - target_mu) ** 2 / (2 * intent.sigma ** 2)
            + torch.log(intent.sigma)
        ).mean()

        # Optional ERN supervision
        ern_loss = torch.tensor(0.0, device=nll.device)
        if ern_label is not None:
            ern_loss = F.binary_cross_entropy(
                intent.ern_prob.squeeze(-1), ern_label.float()
            )

        # Entropy regularizer — keep sigma from collapsing to zero
        sigma_reg = -torch.log(intent.sigma).mean() * self.sigma_reg

        return nll + self.ern_weight * ern_loss + sigma_reg
