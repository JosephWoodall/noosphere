"""
Latency-Minimising MPC Planner with Monte Carlo Guidance for closed-loop BCI.

Algorithm (hybrid CEM + gradient descent):

  Phase 1 — Monte Carlo CEM (global, uncertainty-aware):
    For k in range(mc_rollouts):
      Sample action sequence a^k ~ N(mu_decoder, sigma_decoder)  ← uses brain's own uncertainty
      Roll out h_1..H = T(h, a^k_0), T(h_1, a^k_1), ...          (no gradient, fast)
      Compute J^k = Σ γ^t · ||decode(h_t).mu - dof_goal||²
    Keep elite fraction (top-25% lowest cost)
    Set init_actions = mean of elite sequences

  Phase 2 — Gradient descent (local refinement from MC warm-start):
    Optimise J w.r.t. action sequence, starting from init_actions
    n_iters gradient steps with Adam

This hybrid is strictly better than either pure MC (noisy) or pure gradient
(local minimum near a potentially bad initialisation).  The MC phase uses
sigma to probe the action space proportional to neural uncertainty — high
sigma means broader exploration, exactly when we need it most.

Cost function J encodes latency preference via discount γ < 1:
  earlier convergence to dof_goal is rewarded more than late convergence.

Fast inference mode: self_condition() — single forward pass, no gradient.
  h_plan = T(h, decoder(h).mu)   Used in real-time BCI loop (<5 ms CPU).

References
----------
Williams et al. (2017) MPPI: Information-theoretic MPC.
  doi:10.1109/ICRA.2017.7989202
Chua et al. (2018) PETS: Deep reinforcement learning in a probabilistic ensemble.
  doi:10.48550/arXiv.1805.12114
Shenoy & Carmena (2014) Combining decoder design and neural adaptation in BCI.
  doi:10.1016/j.neuron.2014.06.020
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatencyPlanner(nn.Module):
    """
    Hybrid CEM + gradient-descent MPC planner.

    Parameters
    ----------
    transition    : ActionConditionedTransition — differentiable world model
    decoder       : IntentDecoder — maps latent → MotorIntent(mu, sigma)
    horizon       : planning horizon H (rollout steps)
    n_iters       : gradient-descent iterations per plan() call
    lr            : Adam lr for action optimisation
    gamma         : discount factor for latency (γ < 1 → prefer early convergence)
    lambda_smooth : smoothness penalty weight on consecutive actions
    lambda_reg    : L2 action regulariser
    mc_rollouts   : number of MC candidate sequences (0 = pure gradient, no CEM)
    mc_elite_frac : fraction of MC rollouts kept as elite for warm-starting
    """

    def __init__(
        self,
        transition: nn.Module,
        decoder: nn.Module,
        horizon: int = 3,
        n_iters: int = 12,
        lr: float = 0.05,
        gamma: float = 0.8,
        lambda_smooth: float = 0.01,
        lambda_reg: float = 0.001,
        mc_rollouts: int = 32,
        mc_elite_frac: float = 0.25,
    ):
        super().__init__()
        # Stored as lists to avoid registering as submodules — their parameters
        # belong to DigitalTwin's optimizer, not the planner's.
        self._transition = [transition]
        self._decoder = [decoder]

        self.horizon = horizon
        self.n_iters = n_iters
        self.lr = lr
        self.gamma = gamma
        self.lambda_smooth = lambda_smooth
        self.lambda_reg = lambda_reg
        self.mc_rollouts = mc_rollouts
        self.mc_elite_frac = mc_elite_frac

    @property
    def transition(self) -> nn.Module:
        return self._transition[0]

    @property
    def decoder(self) -> nn.Module:
        return self._decoder[0]

    @torch.no_grad()
    def self_condition(self, h: torch.Tensor) -> torch.Tensor:
        """
        Zero-latency fast path for real-time BCI inference.

        Decodes the current intent and applies one transition step.
        No gradient computation, no parameter search — O(1) forward passes.

        h : (B, d_model)  → h_plan : (B, d_model)
        """
        a = self.decoder(h).mu          # (B, d_dof), tanh-bounded ∈ [-1, 1]
        return self.transition(h, a)

    @torch.no_grad()
    def _mc_phase(
        self,
        h0: torch.Tensor,               # (1, d_model)
        mu0: torch.Tensor,              # (1, d_dof)  — decoder mean
        sigma0: torch.Tensor,           # (1, d_dof)  — decoder std
        dof_goal: torch.Tensor,         # (1, d_dof)
    ) -> torch.Tensor:
        """
        CEM Monte Carlo phase.  Returns elite mean over action sequences.

        Sampling distribution: N(mu0, sigma0) — uses the brain's own decoded
        uncertainty as the exploration radius.  High sigma (ambiguous neural
        state) → wide exploration.  Low sigma (clear intent) → tight search.

        Returns init_actions : (H, d_dof) elite mean, clamped to [-1, 1].
        """
        H = self.horizon
        K = self.mc_rollouts
        n_elite = max(1, int(K * self.mc_elite_frac))
        d_dof = mu0.shape[-1]

        # Sample K candidate sequences: (K, H, d_dof)
        noise = torch.randn(K, H, d_dof, device=h0.device)
        # Scale noise by sigma (broadcast mu/sigma to (K, H, d_dof))
        actions = mu0.unsqueeze(0) + sigma0.unsqueeze(0) * noise
        actions = actions.clamp(-1.0, 1.0)

        # Roll out all K sequences, compute cost
        costs = torch.zeros(K, device=h0.device)
        for k in range(K):
            h = h0.clone()
            for t in range(H):
                h = self.transition(h, actions[k, t : t + 1])
                mu_pred = self.decoder(h).mu
                costs[k] += (self.gamma ** t) * F.mse_loss(mu_pred, dof_goal)

        # Select elite (lowest cost) and return their mean
        elite_idx = costs.argsort()[:n_elite]
        elite_actions = actions[elite_idx]               # (n_elite, H, d_dof)
        return elite_actions.mean(0).clamp(-1.0, 1.0)   # (H, d_dof)

    def plan(
        self,
        h0: torch.Tensor,
        dof_goal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Full planning: CEM warm-start → gradient refinement.

        h0       : (1, d_model)  current latent (single sample)
        dof_goal : (1, d_dof)    target DOF configuration; if None, uses
                                 self-conditioning (decoder(h0).mu as goal)

        Returns
        -------
        best_action   : (1, d_dof)   first action in the optimal sequence
        h_predicted   : (1, d_model) predicted state after first action
        final_loss    : float        trajectory cost at convergence
        """
        H = self.horizon

        with torch.no_grad():
            intent0 = self.decoder(h0)
            mu0 = intent0.mu          # (1, d_dof)
            sigma0 = intent0.sigma    # (1, d_dof)
            if dof_goal is None:
                dof_goal = mu0        # self-conditioning mode

        # ── Phase 1: CEM Monte Carlo warm-start ──────────────────────────────
        if self.mc_rollouts > 0:
            init_actions = self._mc_phase(h0, mu0, sigma0, dof_goal)
        else:
            init_actions = mu0.repeat(H, 1)   # (H, d_dof)

        # ── Phase 2: gradient refinement from MC warm-start ──────────────────
        # torch.enable_grad() ensures plan() works even when called inside a
        # torch.no_grad() inference block (e.g. twin.eval() context).
        with torch.enable_grad():
            actions = nn.Parameter(init_actions.clone().requires_grad_(True))
            opt = torch.optim.Adam([actions], lr=self.lr)

            final_loss = float("nan")
            for _ in range(self.n_iters):
                opt.zero_grad()
                h = h0
                J = torch.zeros(1, device=h0.device)

                for k in range(H):
                    a_k = torch.tanh(actions[k : k + 1])   # enforce [-1, 1]
                    h = self.transition(h, a_k)
                    mu_k = self.decoder(h).mu
                    J = J + (self.gamma ** k) * F.mse_loss(mu_k, dof_goal)

                if H > 1:
                    J = J + self.lambda_smooth * F.mse_loss(
                        actions[1:], actions[:-1].detach()
                    )
                J = J + self.lambda_reg * (actions ** 2).mean()

                J.backward()
                nn.utils.clip_grad_norm_([actions], max_norm=1.0)
                opt.step()
                final_loss = J.item()

        with torch.no_grad():
            a0 = torch.tanh(actions[0:1]).detach()   # (1, d_dof)
            h1 = self.transition(h0, a0)             # (1, d_model)

        return a0, h1, final_loss

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Convenience alias for self_condition. Composable in nn.Sequential."""
        return self.self_condition(h)
