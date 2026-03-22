"""
noosphere/rssm.py
=================
Recurrent State Space Model (RSSM)

Latent state sₜ = (hₜ, zₜ):
    hₜ  — deterministic GRU state      long-range temporal memory
    zₜ  — stochastic discrete latent   uncertainty + multimodal futures

Two forward modes:
    observe_step   uses real observations → trains posterior q(zₜ | hₜ, oₜ)
    imagine_step   no observations       → runs prior p(ẑₜ | hₜ) for planning

Changes in v1.3.1
-----------------
- kl_loss: probs clamped to [1e-6, 1] before OneHotCategorical construction.
  Without this, values near 0 produce log(0) = -inf inside the KL computation,
  which propagates as NaN through the entire training loss.
- reset_episode moved here (was only on PhysicsAugmentedRSSM wrapper) so bare
  RSSM can also be used as a standalone planner simulator cleanly.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, kl_divergence


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
        hard = F.one_hot(probs.argmax(dim=-1), self.n_classes).float()
        hard = hard + probs - probs.detach()
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
        embed_dim: int = 512,
        action_dim: int = 64,
        det_dim: int = 512,
        stoch_cats: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.det_dim = det_dim
        self.stoch_cats = stoch_cats
        self.stoch_classes = stoch_classes
        self.stoch_dim = stoch_cats * stoch_classes

        self.gru_input_proj = nn.Linear(self.stoch_dim + action_dim, det_dim)
        self.gru = nn.GRUCell(det_dim, det_dim)

        self.prior_mlp = nn.Sequential(
            nn.Linear(det_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_cats * stoch_classes),
        )
        self.posterior_mlp = nn.Sequential(
            nn.Linear(det_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_cats * stoch_classes),
        )
        self.st = StraightThroughOneHot(stoch_classes)

        # Episode-level recurrent state (used by PhysicsAugmentedRSSM wrapper)
        self._ep_h: Dict[int, torch.Tensor] = {}

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            "h": torch.zeros(batch_size, self.det_dim, device=device),
            "z": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def reset_episode(self):
        self._ep_h.clear()

    def imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Planning step — no observation. Returns (h_next, z_next, prior_probs)."""
        x = self.gru_input_proj(torch.cat([z, a], dim=-1))
        h_next = self.gru(x, h)
        logits = self.prior_mlp(h_next).view(-1, self.stoch_cats, self.stoch_classes)
        z_next, prior_probs = self.st(logits)
        return h_next, z_next.view(-1, self.stoch_dim), prior_probs

    def observe_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training step — uses real obs. Returns (h_next, z_next, prior_probs, posterior_probs)."""
        x = self.gru_input_proj(torch.cat([z, a], dim=-1))
        h_next = self.gru(x, h)

        prior_logits = self.prior_mlp(h_next).view(
            -1, self.stoch_cats, self.stoch_classes
        )
        _, prior_probs = self.st(prior_logits)

        post_logits = self.posterior_mlp(torch.cat([h_next, obs_embed], dim=-1))
        post_logits = post_logits.view(-1, self.stoch_cats, self.stoch_classes)
        z_next, posterior_probs = self.st(post_logits)

        return h_next, z_next.view(-1, self.stoch_dim), prior_probs, posterior_probs

    def kl_loss(
        self,
        prior_probs: torch.Tensor,
        posterior_probs: torch.Tensor,
        free_nats: float = 1.0,
        balance: float = 0.8,
    ) -> torch.Tensor:
        """
        Balanced KL (DreamerV3):
            L = 0.8 · KL(sg(q) ‖ p)  +  0.2 · KL(q ‖ sg(p))

        Bug fix v1.3.1: clamp probs to [1e-6, 1] before constructing
        OneHotCategorical. Without this, probs that are exactly 0 (common
        after straight-through discretisation) produce log(0) = -inf in the
        KL computation → NaN propagates through the entire training loss.
        """
        EPS = 1e-6
        p_clamped = prior_probs.clamp(min=EPS)
        p_clamped = p_clamped / p_clamped.sum(-1, keepdim=True)
        # Replace any remaining NaNs (e.g. from all-zero rows) with uniform
        p_clamped = torch.where(
            torch.isnan(p_clamped),
            torch.ones_like(p_clamped) / p_clamped.shape[-1],
            p_clamped,
        )

        q_clamped = posterior_probs.clamp(min=EPS)
        q_clamped = q_clamped / q_clamped.sum(-1, keepdim=True)
        # Replace any remaining NaNs with uniform
        q_clamped = torch.where(
            torch.isnan(q_clamped),
            torch.ones_like(q_clamped) / q_clamped.shape[-1],
            q_clamped,
        )

        kl = balance * kl_divergence(
            OneHotCategorical(probs=q_clamped.detach()),
            OneHotCategorical(probs=p_clamped),
        ).sum(-1) + (1 - balance) * kl_divergence(
            OneHotCategorical(probs=q_clamped),
            OneHotCategorical(probs=p_clamped.detach()),
        ).sum(-1)
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
                nn.Linear(state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out),
            )

        self.reward_head = _head(1)
        self.termination_head = _head(1)
        self.value_head = _head(1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "reward": self.reward_head(state).squeeze(-1),
            "termination": torch.sigmoid(self.termination_head(state).squeeze(-1)),
            "value": self.value_head(state).squeeze(-1),
        }

    def min_value(self, state: torch.Tensor) -> torch.Tensor:
        """Compatibility shim — ConsequenceModel has one value head."""
        return self.value_head(state).squeeze(-1)


class ObservationDecoder(nn.Module):
    """Decodes latent state back to observation embedding for reconstruction loss."""

    def __init__(self, state_dim: int, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.decoder(state)


# ── Digital consequence model (v1.5.0) ───────────────────────────────────────


class DigitalConsequenceHead(nn.Module):
    """
    Extends ConsequenceModel to predict digital state changes.

    The world model now predicts not just reward but the *structured
    digital state* that will result from an action. This gives it a
    richer, more accurate model of reality for digital domains.

    Predicted outputs:
        exit_code_logit  (B, 3)   — {success, error, timeout} classification
        stdout_len_pred  (B,)     — predicted stdout length (log-normalised)
        state_change_pred(B,)     — predicted magnitude of state change [0,1]
        n_files_pred     (B,)     — predicted files affected
        process_spawned  (B,)     — probability a new process was spawned

    Supervised by ShellExecutor output features so the world model learns
    that "ls" → short stdout, no state change, no process; whereas
    "make_build" → long stdout, files modified, processes spawned.
    """

    def __init__(
        self, state_dim: int, hidden_dim: int = 256, digital_state_dim: int = 64
    ):
        super().__init__()
        self.digital_state_dim = digital_state_dim

        # Exit code classifier: success / error / timeout
        self.exit_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )
        # Predicted stdout length (log-normalised [0,1])
        self.stdout_len_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Predicted magnitude of digital state change
        self.state_change_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        # Predicted next digital state (full vector regression)
        self.next_state_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, digital_state_dim),
        )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "exit_logits": self.exit_head(state),  # (B, 3)
            "stdout_len": self.stdout_len_head(state).squeeze(-1),  # (B,)
            "state_change": self.state_change_head(state).squeeze(-1),  # (B,)
            "next_digital": self.next_state_head(state),  # (B, D)
        }

    def loss(
        self,
        preds: Dict[str, torch.Tensor],
        actual_exit: torch.Tensor,  # (B,) long — 0=success, 1=error, 2=timeout
        actual_stdout_len: torch.Tensor,  # (B,) float normalised
        actual_state_change: torch.Tensor,  # (B,) float
        actual_next_state: Optional[torch.Tensor] = None,  # (B, D)
    ) -> torch.Tensor:
        """Supervised loss from ShellExecutor output."""
        import torch.nn.functional as F

        L = F.cross_entropy(preds["exit_logits"], actual_exit.long())
        L += F.mse_loss(preds["stdout_len"], actual_stdout_len)
        L += F.mse_loss(preds["state_change"], actual_state_change)
        if actual_next_state is not None:
            L += F.mse_loss(preds["next_digital"], actual_next_state)
        return L


class EnhancedConsequenceModel(ConsequenceModel):
    """
    ConsequenceModel extended with DigitalConsequenceHead.

    Backward compatible — forward() returns same keys as ConsequenceModel
    plus digital prediction keys when digital_mode=True.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        digital_state_dim: int = 64,
        digital_mode: bool = True,
    ):
        super().__init__(state_dim, hidden_dim)
        self.digital_mode = digital_mode
        if digital_mode:
            self.digital = DigitalConsequenceHead(
                state_dim, hidden_dim, digital_state_dim
            )

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward(state)
        if self.digital_mode:
            out.update(self.digital(state))
        return out

    def digital_loss(
        self,
        state,
        actual_exit,
        actual_stdout_len,
        actual_state_change,
        actual_next_state=None,
    ):
        if not self.digital_mode:
            return torch.tensor(0.0, device=state.device)
        preds = self.digital(state)
        return self.digital.loss(
            preds,
            actual_exit,
            actual_stdout_len,
            actual_state_change,
            actual_next_state,
        )
