"""
noosphere/rssm.py
=================
Recurrent State Space Model (RSSM)

Latent state sₜ = (hₜ, zₜ):
    hₜ  — deterministic Mamba state    long-range temporal memory (y, hidden)
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


class SinkhornKnopp(nn.Module):
    """
    Numerical stable Iterative Optimal Transport in log-space.
    Prevents NaNs and overflows common in standard Sinkhorn.
    """

    def __init__(self, num_iters: int = 10, epsilon: float = 0.05):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        # Log-space iteration: log(P) = log(K) + f + g
        # cost_matrix: (B, M, N)
        B, M, N = cost_matrix.shape
        log_K = -cost_matrix / self.epsilon
        
        f = torch.zeros(B, M, 1, device=cost_matrix.device)
        g = torch.zeros(B, 1, N, device=cost_matrix.device)

        for _ in range(self.num_iters):
            # Row normalization
            f = -torch.logsumexp(log_K + g, dim=2, keepdim=True)
            # Column normalization
            g = -torch.logsumexp(log_K + f, dim=1, keepdim=True)

        return torch.exp(log_K + f + g)


class HNMWorldModel(nn.Module):
    """
    Holonomic Neural Manifold (HNM) World Model.
    Replaces Optimal Transport with Parallel Transport on a Fiber Bundle.
    The world state is a (position, phase) pair on a curved manifold.
    Intent is decoded as the geometric phase shift (holonomy).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 64,
        det_dim: int = 512,
        stoch_cats: int = 16,    # Fiber dimension n for SO(n)
        stoch_classes: int = 16, # Ignored
        hidden_dim: int = 512,
        d_state: int = 16,
    ):
        super().__init__()
        self.det_dim = det_dim
        self.n = stoch_cats      # Dimension of the holonomy matrix (SO(n))
        self.d_state = d_state
        self.stoch_dim = self.n * self.n  # The z state is the flattened rotation matrix

        # Projection from Holonomy to Embedding space to drive recurrence
        self.phi_to_embed = nn.Sequential(
            nn.Linear(self.stoch_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
        )

        # Base Manifold Recurrence (Mamba)
        self.cell = ModeSelectiveSSMCell(
            embed_dim + action_dim, det_dim, self.stoch_dim, d_state=d_state
        )

        # Gauge Field Projectors (Generates skew-symmetric connection matrices)
        self.post_gauge_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, (self.n * (self.n - 1)) // 2),
        )
        self.prior_gauge_mlp = nn.Sequential(
            nn.Linear(det_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, (self.n * (self.n - 1)) // 2),
        )

        self._ep_h: Dict[int, torch.Tensor] = {}

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        # Identity matrix for initial holonomy
        I = torch.eye(self.n, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        return {
            "h": torch.zeros(
                batch_size, self.det_dim * (self.d_state + 1), device=device
            ),
            "z": I.reshape(batch_size, -1),
        }

    def _unpack_h(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = h[:, : self.det_dim]
        h_ssm = h[:, self.det_dim :].view(-1, self.det_dim, self.d_state)
        return y, h_ssm

    def _pack_h(self, y: torch.Tensor, h_ssm: torch.Tensor) -> torch.Tensor:
        return torch.cat([y, h_ssm.reshape(y.shape[0], -1)], dim=-1)

    def _vec_to_skew(self, v: torch.Tensor) -> torch.Tensor:
        """Map a vector of parameters to a skew-symmetric matrix (Lie Algebra so(n))."""
        B = v.shape[0]
        M = torch.zeros(B, self.n, self.n, device=v.device, dtype=v.dtype)
        indices = torch.triu_indices(self.n, self.n, offset=1)
        M[:, indices[0], indices[1]] = v
        M[:, indices[1], indices[0]] = -v
        return M

    def observe_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y, h_ssm = self._unpack_h(h)
        Phi = z.view(-1, self.n, self.n).to(torch.float32)

        # 1. Compute Connection (Gauge Field) from observation
        A_params = self.post_gauge_mlp(obs_embed)
        A = self._vec_to_skew(A_params).to(torch.float32)

        # 2. Parallel Transport (Matrix Exponential) - stable in float32
        Phi_next = torch.matmul(torch.matrix_exp(A), Phi)
        z_next = Phi_next.reshape(-1, self.stoch_dim).to(obs_embed.dtype)

        # 3. Base Manifold recurrence
        # Drive base by the Resonant Phase projected to embedding space
        res_embed = self.phi_to_embed(z_next)
        u = torch.cat([res_embed, a], dim=-1)
        y_next, h_ssm_next = self.cell(u, z_next, h_ssm)
        h_next = self._pack_h(y_next, h_ssm_next)

        # 4. Prior Prediction (Gauge Field from base state)
        A_prior_params = self.prior_gauge_mlp(y_next)
        
        return h_next, z_next, A_prior_params, A_params

    def imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, h_ssm = self._unpack_h(h)
        Phi = z.view(-1, self.n, self.n).to(torch.float32)

        # 1. Compute Prior Connection from base manifold state
        A_params = self.prior_gauge_mlp(y)
        A = self._vec_to_skew(A_params).to(torch.float32)

        # 2. Parallel Transport along prior geodesic
        Phi_next = torch.matmul(torch.matrix_exp(A), Phi)
        z_next = Phi_next.reshape(-1, self.stoch_dim).to(y.dtype)

        # 3. Recurrent Update
        res_embed = self.phi_to_embed(z_next)
        u = torch.cat([res_embed, a], dim=-1)
        y_next, h_ssm_next = self.cell(u, z_next, h_ssm)
        h_next = self._pack_h(y_next, h_ssm_next)

        return h_next, z_next, A_params

    def kl_loss(
        self,
        prior_probs: torch.Tensor,
        posterior_probs: torch.Tensor,
        free_nats: float = 1.0,
        balance: float = 0.8,
    ) -> torch.Tensor:
        """
        Holonomic Curvature Loss: replaces KL.
        Computes the Frobenius distance between the Prior and Posterior 
        Gauge field parameters (the connections in the Lie Algebra).
        """
        diff = balance * F.mse_loss(posterior_probs.detach(), prior_probs, reduction="none") + \
               (1 - balance) * F.mse_loss(posterior_probs, prior_probs.detach(), reduction="none")
        return diff.sum(-1).clamp(min=free_nats).mean()

    @property
    def state_dim(self) -> int:
        return self.det_dim * (self.d_state + 1) + self.stoch_dim


class SKARWorldModel(nn.Module):
    """
    Sinkhorn-Knopp Attractor Resonance (SKAR) World Model.
    Replaces discrete sampling with an iterative Optimal Transport alignment
    between data and a learnable codebook of cognitive attractors.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 64,
        det_dim: int = 512,
        stoch_cats: int = 32,  # Used as K (num attractors)
        stoch_classes: int = 32,  # Ignored (compat)
        hidden_dim: int = 512,
        d_state: int = 16,
    ):
        super().__init__()
        self.det_dim = det_dim
        self.d_state = d_state
        self.K = stoch_cats  # Number of attractors
        self.stoch_dim = embed_dim  # The stoch latent is a resonant embedding

        # Learnable Attractor Codebook (initialized with small variance)
        self.attractors = nn.Parameter(torch.randn(self.K, embed_dim) * 0.01)

        # Iterative Solver (Log-space)
        self.sinkhorn = SinkhornKnopp(num_iters=15, epsilon=0.05)

        # Recurrent core (Mamba)
        self.cell = ModeSelectiveSSMCell(
            embed_dim + action_dim, det_dim, embed_dim, d_state=d_state
        )

        # Prior Cost Matrix predictor (predicts alignment to attractors)
        self.prior_cost_mlp = nn.Sequential(
            nn.Linear(det_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.K),
        )

        self._ep_h: Dict[int, torch.Tensor] = {}

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return {
            "h": torch.zeros(
                batch_size, self.det_dim * (self.d_state + 1), device=device
            ),
            "z": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def _unpack_h(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y = h[:, : self.det_dim]
        h_ssm = h[:, self.det_dim :].view(-1, self.det_dim, self.d_state)
        return y, h_ssm

    def _pack_h(self, y: torch.Tensor, h_ssm: torch.Tensor) -> torch.Tensor:
        return torch.cat([y, h_ssm.reshape(y.shape[0], -1)], dim=-1)

    def reset_episode(self):
        self._ep_h.clear()

    def _get_cost(self, x: torch.Tensor) -> torch.Tensor:
        # Cosine distance cost matrix: (B, 1, K)
        x_norm = F.normalize(x, dim=-1)
        a_norm = F.normalize(self.attractors, dim=-1)
        dist = 1.0 - torch.matmul(x_norm, a_norm.T)  # (B, K)
        return dist.unsqueeze(1)  # (B, 1, K)

    def observe_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y, h_ssm = self._unpack_h(h)

        # 1. Posterior Resonance (Optimal Transport)
        cost_post = self._get_cost(obs_embed)
        plan_post = self.sinkhorn(cost_post)  # (B, 1, K)
        probs_post = plan_post.squeeze(1)  # (B, K) alignment probabilities
        z_next = torch.matmul(probs_post, self.attractors)  # Resonant projection

        # 2. Recurrent Update
        u = torch.cat([z_next, a], dim=-1)
        y_next, h_ssm_next = self.cell(u, z_next, h_ssm)
        h_next = self._pack_h(y_next, h_ssm_next)

        # 3. Prior Cost Prediction
        logits_prior = self.prior_cost_mlp(y_next)  # (B, K)
        probs_prior = F.softmax(-logits_prior, dim=-1)  # Softmin cost as prob

        return h_next, z_next, probs_prior, probs_post

    def imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y, h_ssm = self._unpack_h(h)

        # 1. Prior Resonance
        logits_prior = self.prior_cost_mlp(y)
        probs_prior = F.softmax(-logits_prior, dim=-1)
        z_next = torch.matmul(probs_prior, self.attractors)

        # 2. Recurrent Update
        u = torch.cat([z_next, a], dim=-1)
        y_next, h_ssm_next = self.cell(u, z_next, h_ssm)
        h_next = self._pack_h(y_next, h_ssm_next)

        return h_next, z_next, probs_prior

    def kl_loss(
        self,
        prior_probs: torch.Tensor,
        posterior_probs: torch.Tensor,
        free_nats: float = 1.0,
        balance: float = 0.8,
    ) -> torch.Tensor:
        EPS = 1e-9
        p = prior_probs.clamp(min=EPS)
        q = posterior_probs.clamp(min=EPS)
        
        # Cross-entropy based KL for stability
        # KL(q||p) = sum(q * (log q - log p))
        kl_qp = (q * (torch.log(q) - torch.log(p))).sum(-1)
        kl_pq = (p * (torch.log(p) - torch.log(q))).sum(-1)
        
        kl = balance * kl_qp + (1 - balance) * kl_pq
             
        return kl.clamp(min=free_nats).mean()

    @property
    def state_dim(self) -> int:
        return self.det_dim * (self.d_state + 1) + self.stoch_dim


class ModeSelectiveSSMCell(nn.Module):
    """
    Unique Mamba-style recurrent cell for World Models.
    Transition dynamics (dt, B, C) are generated from the discrete
    stochastic latent state (the 'mode'), creating a piecewise-continuous
    dynamical system governed by cognitive intents.
    """

    def __init__(
        self, input_dim: int, det_dim: int, stoch_dim: int, d_state: int = 16
    ):
        super().__init__()
        self.d_model = det_dim
        self.d_state = d_state

        # Mode-selective dynamics projections
        self.dt_proj = nn.Linear(stoch_dim, det_dim)
        self.B_proj = nn.Linear(stoch_dim, d_state)
        self.C_proj = nn.Linear(stoch_dim, d_state)

        # Structured A matrix initialization
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1).float())
            .unsqueeze(0)
            .expand(det_dim, -1)
            .clone()
        )
        self.D = nn.Parameter(torch.ones(det_dim))

        # Input/Output projections with Normalization
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, det_dim, bias=False),
            nn.LayerNorm(det_dim),
            nn.SiLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(det_dim, det_dim, bias=False),
            nn.LayerNorm(det_dim),
        )

    def forward(
        self, u_t: torch.Tensor, z_t: torch.Tensor, h_ssm_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(u_t)

        # Generate SSM parameters from discrete mode z_t
        dt = F.softplus(self.dt_proj(z_t))
        B = self.B_proj(z_t)
        C = self.C_proj(z_t)
        A = -torch.exp(self.A_log)

        # Discretize via zero-order hold
        dt = dt.unsqueeze(-1)
        A_bar = torch.exp(dt * A)
        B_bar = (A_bar - 1.0) / (A + 1e-8) * B.unsqueeze(1)

        # Update hidden state: (B, D, N)
        h_ssm_next = A_bar * h_ssm_prev + B_bar * x.unsqueeze(-1)

        # Compute output: (B, D)
        y = (h_ssm_next * C.unsqueeze(1)).sum(-1)
        y = self.out_proj(y + x * self.D)
        return y, h_ssm_next


class RSSM(nn.Module):
    """
    Recurrent State Space Model with Mode-Selective Mamba core.

    Parameters
    ----------
    embed_dim    : observation embedding dimension
    action_dim   : action embedding dimension
    det_dim      : continuous state size (d_model)
    stoch_cats   : number of categorical variables
    stoch_classes: classes per categorical variable
    hidden_dim   : MLP hidden size
    d_state      : SSM internal state dimension (default 16)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        action_dim: int = 64,
        det_dim: int = 512,
        stoch_cats: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 512,
        d_state: int = 16,
    ):
        super().__init__()
        self.det_dim = det_dim
        self.d_state = d_state
        self.stoch_cats = stoch_cats
        self.stoch_classes = stoch_classes
        self.stoch_dim = stoch_cats * stoch_classes

        # Unique Mamba-style core
        self.cell = ModeSelectiveSSMCell(
            self.stoch_dim + action_dim, det_dim, self.stoch_dim, d_state=d_state
        )

        # DreamerV3 style normalized MLPs
        self.prior_mlp = nn.Sequential(
            nn.Linear(det_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_cats * stoch_classes),
        )
        self.posterior_mlp = nn.Sequential(
            nn.Linear(det_dim + embed_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_cats * stoch_classes),
        )
        self.st = StraightThroughOneHot(stoch_classes)

        # Episode-level recurrent state
        self._ep_h: Dict[int, torch.Tensor] = {}

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        # h is returned as a flat tensor to maintain backward compatibility,
        # but it contains both the output y and the flattened hidden SSM state.
        return {
            "h": torch.zeros(
                batch_size, self.det_dim * (self.d_state + 1), device=device
            ),
            "z": torch.zeros(batch_size, self.stoch_dim, device=device),
        }

    def _unpack_h(
        self, h: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = h[:, : self.det_dim]
        h_ssm = h[:, self.det_dim :].view(-1, self.det_dim, self.d_state)
        return y, h_ssm

    def _pack_h(self, y: torch.Tensor, h_ssm: torch.Tensor) -> torch.Tensor:
        return torch.cat([y, h_ssm.reshape(y.shape[0], -1)], dim=-1)

    def reset_episode(self):
        self._ep_h.clear()

    def imagine_step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Planning step — no observation. Returns (h_next, z_next, prior_probs)."""
        y, h_ssm = self._unpack_h(h)
        u = torch.cat([z, a], dim=-1)

        y_next, h_ssm_next = self.cell(u, z, h_ssm)
        h_next = self._pack_h(y_next, h_ssm_next)

        logits = self.prior_mlp(y_next).view(
            -1, self.stoch_cats, self.stoch_classes
        )
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
        y, h_ssm = self._unpack_h(h)
        u = torch.cat([z, a], dim=-1)

        y_next, h_ssm_next = self.cell(u, z, h_ssm)
        h_next = self._pack_h(y_next, h_ssm_next)

        prior_logits = self.prior_mlp(y_next).view(
            -1, self.stoch_cats, self.stoch_classes
        )
        _, prior_probs = self.st(prior_logits)

        post_logits = self.posterior_mlp(torch.cat([y_next, obs_embed], dim=-1))
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
        return self.det_dim * (self.d_state + 1) + self.stoch_dim


class ConsequenceModel(nn.Module):
    """
    Evaluates the imagined latent state to verify physical/digital safety 
    and calculate optimal path-planning rewards.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.termination_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.net(state)
        return {
            "reward": self.reward_head(h),
            "value": self.value_head(h),
            "termination": torch.sigmoid(self.termination_head(h))
        }


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
