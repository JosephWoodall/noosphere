"""
noosphere/planner.py
====================
MCTS Planner, Actor, Critic, ImaginationBuffer

Changes in v1.3.1
-----------------
1. ImaginationBuffer.lambda_returns: builds G tensor on the same device as
   rewards, not always on CPU. Eliminates silent host↔device copy each AC step.

2. ImaginationBuffer: `clear()` now also resets the buffer after lambda_returns
   so it can be reused across AC update calls without re-instantiation.

3. Actor.entropy(): added convenience method for use in actor loss.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Action encoder ────────────────────────────────────────────────────────────

class ActionEncoder(nn.Module):
    """Embeds integer actions as continuous vectors."""
    def __init__(self, n_actions: int, action_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_actions, action_dim)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.emb(a.long())


# ── Actor ─────────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """
    Categorical policy.
    Returns a Categorical distribution over n_actions.
    """
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),    nn.SiLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.net(state))

    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.forward(state)
        return dist.mode if deterministic else dist.sample()

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state).entropy()


# ── Critic ────────────────────────────────────────────────────────────────────

class Critic(nn.Module):
    """
    Clipped double-Q critic.
    Two independent value heads; min is used for conservative estimates.
    """
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        def _v():
            return nn.Sequential(
                nn.Linear(state_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, hidden),    nn.SiLU(),
                nn.Linear(hidden, 1),
            )
        self.v1 = _v()
        self.v2 = _v()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.v1(state).squeeze(-1), self.v2(state).squeeze(-1)

    def min_value(self, state: torch.Tensor) -> torch.Tensor:
        v1, v2 = self.forward(state)
        return torch.min(v1, v2)


# ── MCTS node ─────────────────────────────────────────────────────────────────

@dataclass
class MCTSNode:
    h: torch.Tensor
    z: torch.Tensor
    parent:    Optional["MCTSNode"] = None
    action:    Optional[int]        = None
    prior:     float                = 1.0
    visits:    int                  = 0
    value_sum: float                = 0.0
    children:  Dict[int, "MCTSNode"] = field(default_factory=dict)

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
    AlphaZero-style MCTS in world-model latent space.
    The world model is the simulator — no real environment calls during search.
    """
    def __init__(self, rssm, consequence, actor: Actor,
                 action_encoder: ActionEncoder, n_actions: int,
                 n_simulations: int = 50, horizon: int = 15,
                 gamma: float = 0.99, c_puct: float = 1.25,
                 device: torch.device = torch.device("cpu")):
        self.rssm  = rssm;  self.cons = consequence
        self.actor = actor; self.ae   = action_encoder
        self.N     = n_actions; self.nsim = n_simulations
        self.H     = horizon;   self.g    = gamma
        self.c     = c_puct;    self.dev  = device

    @property
    def n_simulations(self):
        return self.nsim

    @n_simulations.setter
    def n_simulations(self, v: int):
        self.nsim = v

    @torch.no_grad()
    def search(self, h: torch.Tensor, z: torch.Tensor) -> int:
        root = MCTSNode(h=h, z=z)
        self._expand(root)
        for _ in range(self.nsim):
            node, path = self._select(root)
            if node.is_leaf() and node.visits > 0:
                self._expand(node)
                if node.children:
                    node = next(iter(node.children.values()))
                    path.append(node)
            v = self._evaluate(node)
            self._backup(path, v)
        return max(root.children, key=lambda a: root.children[a].visits)

    def _select(self, node: MCTSNode):
        path = [node]
        while not node.is_leaf():
            node = node.children[
                max(node.children, key=lambda a: node.children[a].ucb(node.visits, self.c))
            ]
            path.append(node)
        return node, path

    def _expand(self, node: MCTSNode):
        s     = torch.cat([node.h, node.z], -1)
        dist  = self.actor.forward(s)
        prior = dist.probs.squeeze(0).cpu().numpy()
        for a in range(self.N):
            h2, z2, _ = self.rssm.imagine_step(
                node.h, node.z, self.ae(torch.tensor([a], device=self.dev))
            )
            node.children[a] = MCTSNode(
                h=h2, z=z2, parent=node, action=a, prior=float(prior[a])
            )

    def _evaluate(self, node: MCTSNode) -> float:
        h, z = node.h, node.z
        R, disc = 0.0, 1.0
        for _ in range(self.H):
            s = torch.cat([h, z], -1)
            c = self.cons(s)
            R += disc * c["reward"].item()
            disc *= self.g
            if c["termination"].item() > 0.5:
                break
            a = self.actor.act(s, deterministic=True)
            if a.dim() == 0:
                a = a.unsqueeze(0)
            h, z, _ = self.rssm.imagine_step(h, z, self.ae(a))
        return R + disc * self.cons(torch.cat([h, z], -1))["value"].item()

    def _backup(self, path: List[MCTSNode], v: float):
        for node in reversed(path):
            node.visits    += 1
            node.value_sum += v
            v *= self.g


# ── Imagination buffer ────────────────────────────────────────────────────────

class ImaginationBuffer:
    """
    Stores imagined trajectories for TD(λ) actor-critic training.

    Fix v1.3.1: lambda_returns builds G on the device of the rewards tensor,
    not always on CPU. This eliminates a host↔device copy each AC update.
    """
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.g = gamma; self.l = lam
        self.states:    List[torch.Tensor] = []
        self.actions:   List[torch.Tensor] = []
        self.rewards:   List[torch.Tensor] = []
        self.values:    List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.dones:     List[torch.Tensor] = []

    def add(self, s, a, r, v, lp, d):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.values.append(v)
        self.log_probs.append(lp)
        self.dones.append(d)

    def lambda_returns(self) -> torch.Tensor:
        T  = len(self.rewards)
        vs = torch.stack(self.values).detach()
        rs = torch.stack(self.rewards).detach()
        ds = torch.stack(self.dones).detach()
        # Build G on same device as rs — no CPU detour
        G  = torch.zeros(T, device=rs.device)
        Gn = vs[-1]
        for t in reversed(range(T)):
            nd  = 1.0 - ds[t].float()
            Gn  = rs[t] + self.g * nd * ((1 - self.l) * vs[t] + self.l * Gn)
            G[t] = Gn if Gn.numel() == 1 else Gn.mean()
        return G

    def clear(self):
        for lst in [self.states, self.actions, self.rewards,
                    self.values, self.log_probs, self.dones]:
            lst.clear()
