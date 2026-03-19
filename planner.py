"""
noosphere/planner.py
====================
Planning — MCTS over Latent Space + Actor-Critic

MCTS
----
Operates entirely inside the world model's imagination. No real environment
calls during planning. UCB1 with actor prior (AlphaZero-style).

    Select → Expand (via world model) → Evaluate (value head + horizon rollout)
    → Backup → return action with highest visit count

Actor-Critic
------------
Policy πθ and value Vφ trained on imagined trajectories via TD(λ).
Gradients update only actor/critic — world model is frozen.

    TD(λ): Gₜ = rₜ + γ[(1-λ)Vₜ₊₁ + λGₜ₊₁]    λ=0.95
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


# ── Action encoder ────────────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    def __init__(self, n_actions: int, action_dim: int = 64, continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        if continuous:
            self.proj  = nn.Sequential(nn.Linear(n_actions, action_dim), nn.SiLU())
        else:
            self.embed = nn.Embedding(n_actions, action_dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.proj(a) if self.continuous else self.embed(a)


# ── Actor / Critic ────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """πθ(a | sₜ) — discrete categorical or continuous Gaussian policy."""
    def __init__(self, state_dim: int, n_actions: int,
                 hidden: int = 256, continuous: bool = False):
        super().__init__()
        self.continuous = continuous
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.SiLU(),
        )
        if continuous:
            self.mu      = nn.Linear(hidden, n_actions)
            self.log_std = nn.Linear(hidden, n_actions)
        else:
            self.logits  = nn.Linear(hidden, n_actions)

    def forward(self, s: torch.Tensor) -> torch.distributions.Distribution:
        x = self.trunk(s)
        if self.continuous:
            return torch.distributions.Normal(self.mu(x), self.log_std(x).clamp(-4,2).exp())
        return torch.distributions.Categorical(logits=self.logits(x))

    def act(self, s: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        d = self.forward(s)
        if deterministic:
            return d.mean if self.continuous else d.logits.argmax(-1)
        return d.sample()


class Critic(nn.Module):
    """Vφ(sₜ) — clipped double-Q value function."""
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        def _v():
            return nn.Sequential(
                nn.Linear(state_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),    nn.SiLU(),
                nn.Linear(hidden, 1),
            )
        self.v1 = _v(); self.v2 = _v()

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.v1(s).squeeze(-1), self.v2(s).squeeze(-1)

    def min_value(self, s: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.forward(s))


# ── MCTS node ─────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    h: torch.Tensor
    z: torch.Tensor
    parent:     Optional["MCTSNode"] = None
    action:     Optional[int]        = None
    prior:      float                = 1.0
    visits:     int                  = 0
    value_sum:  float                = 0.0
    children:   Dict[int,"MCTSNode"] = field(default_factory=dict)

    @property
    def Q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def ucb(self, parent_visits: int, c: float = 1.25) -> float:
        return self.Q + c * self.prior * math.sqrt(parent_visits) / (1 + self.visits)

    def is_leaf(self) -> bool:
        return not self.children


# ── MCTS planner ──────────────────────────────────────────────────────────────

class MCTSPlanner:
    """
    AlphaZero-style MCTS over world-model latent space.

    Parameters
    ----------
    rssm             : RSSM (or PhysicsAugmentedRSSM) — used as simulator
    consequence      : ConsequenceModel
    actor            : Actor — provides root prior
    action_encoder   : ActionEncoder
    n_actions        : action space size
    n_simulations    : tree search budget
    horizon          : rollout depth for leaf evaluation
    """
    def __init__(self, rssm, consequence, actor: Actor,
                 action_encoder: ActionEncoder, n_actions: int,
                 n_simulations: int = 50, horizon: int = 15,
                 gamma: float = 0.99, c_puct: float = 1.25,
                 device: torch.device = torch.device("cpu")):
        self.rssm   = rssm;  self.cons = consequence
        self.actor  = actor; self.ae   = action_encoder
        self.N = n_actions; self.nsim = n_simulations
        self.H = horizon;   self.g    = gamma
        self.c = c_puct;    self.dev  = device

    @torch.no_grad()
    def search(self, h: torch.Tensor, z: torch.Tensor) -> int:
        root = MCTSNode(h=h, z=z)
        self._expand(root)
        for _ in range(self.nsim):
            node, path = self._select(root)
            v = self._evaluate(node)
            self._backup(path, v)
        return max(root.children, key=lambda a: root.children[a].visits)

    def _select(self, node: MCTSNode):
        path = [node]
        while not node.is_leaf():
            node = node.children[max(node.children, key=lambda a: node.children[a].ucb(node.visits, self.c))]
            path.append(node)
        return node, path

    def _expand(self, node: MCTSNode):
        s     = torch.cat([node.h, node.z], -1)
        probs = self.actor.forward(s)
        prior = probs.probs.squeeze(0).cpu().numpy() \
                if isinstance(probs, torch.distributions.Categorical) \
                else np.ones(self.N)/self.N
        for a in range(self.N):
            h2, z2, _ = self.rssm.imagine_step(node.h, node.z,
                            self.ae(torch.tensor([a], device=self.dev)))
            node.children[a] = MCTSNode(h=h2, z=z2, parent=node,
                                         action=a, prior=float(prior[a]))

    def _evaluate(self, node: MCTSNode) -> float:
        h, z = node.h, node.z
        R, disc = 0.0, 1.0
        for _ in range(self.H):
            s    = torch.cat([h,z],-1)
            c    = self.cons(s)
            R   += disc * c["reward"].item()
            disc *= self.g
            if c["termination"].item() > 0.5: break
            a    = self.actor.act(s, deterministic=True)
            if a.dim() == 0: a = a.unsqueeze(0)
            h, z, _ = self.rssm.imagine_step(h, z, self.ae(a))
        return R + disc * self.cons(torch.cat([h,z],-1))["value"].item()

    def _backup(self, path: List[MCTSNode], v: float):
        for node in reversed(path):
            node.visits    += 1
            node.value_sum += v
            v *= self.g


# ── Imagination buffer (actor-critic training) ───────────────────────────────

class ImaginationBuffer:
    """Stores imagined trajectories for TD(λ) actor-critic training."""
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.g = gamma; self.l = lam
        self.states: List[torch.Tensor]   = []
        self.actions: List[torch.Tensor]  = []
        self.rewards: List[torch.Tensor]  = []
        self.values: List[torch.Tensor]   = []
        self.log_probs: List[torch.Tensor]= []
        self.dones: List[torch.Tensor]    = []

    def add(self, s, a, r, v, lp, d):
        self.states.append(s); self.actions.append(a); self.rewards.append(r)
        self.values.append(v); self.log_probs.append(lp); self.dones.append(d)

    def lambda_returns(self) -> torch.Tensor:
        T = len(self.rewards)
        G = torch.zeros(T)
        vs = torch.stack(self.values).detach()
        rs = torch.stack(self.rewards).detach()
        ds = torch.stack(self.dones).detach()
        Gn = vs[-1]
        for t in reversed(range(T)):
            nd = 1.0 - ds[t].float()
            Gn = rs[t] + self.g*nd*((1-self.l)*vs[t] + self.l*Gn)
            G[t] = Gn
        return G

    def clear(self):
        for lst in [self.states,self.actions,self.rewards,self.values,
                    self.log_probs,self.dones]: lst.clear()
