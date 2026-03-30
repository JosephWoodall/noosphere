"""
noosphere/planner.py
====================
Actor, Critic, and PUCT MCTS Planner

Features:
- PUCT Algorithm: The MCTSPlanner uses the Behavioral Cloning Actor to provide
  the structural prior P(s,a) for node expansion. The AI cannot search outside 
  the boundaries of how you personally operate.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional

class ActionEncoder(nn.Module):
    def __init__(self, n_actions: int, action_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_actions, action_dim)
    def forward(self, action: torch.Tensor) -> torch.Tensor:
        return self.emb(action)

class Actor(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, n_actions)
        )
    def forward(self, state: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.net(state)
        return torch.distributions.Categorical(logits=logits)

class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        def make_q():
            return nn.Sequential(
                nn.Linear(state_dim, hidden_dim), nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, 1)
            )
        self.q1, self.q2 = make_q(), make_q()

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state)

    def min_value(self, state: torch.Tensor) -> torch.Tensor:
        v1, v2 = self.forward(state)
        return torch.min(v1, v2)

class ImaginationBuffer:
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma; self.lam = lam
        self.clear()
    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []
    def add(self, s, a, r, v, lp, d):
        self.states.append(s); self.actions.append(a); self.rewards.append(r)
        self.values.append(v); self.log_probs.append(lp); self.dones.append(d)
    def lambda_returns(self) -> torch.Tensor:
        R = self.values[-1]
        returns = []
        for r, v, d in zip(reversed(self.rewards), reversed(self.values), reversed(self.dones)):
            R = r + self.gamma * (1.0 - d) * ((1.0 - self.lam) * v + self.lam * R)
            returns.insert(0, R)
        return torch.stack(returns)

class MCTSNode:
    def __init__(self, h: torch.Tensor, z: torch.Tensor, prior: float = 1.0):
        self.h = h; self.z = z
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.reward = 0.0
        self.termination = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

class MCTSPlanner:
    def __init__(self, rssm, consequence, actor, action_enc, n_actions, n_simulations: int = 30, horizon: int = 10, gamma: float = 0.99, c_puct: float = 1.5, device=None):
        self.rssm = rssm
        self.consequence = consequence
        self.actor = actor
        self.action_enc = action_enc
        self.n_actions = n_actions
        self.n_simulations = n_simulations
        self.horizon = horizon
        self.gamma = gamma
        self.c_puct = c_puct
        self.device = device
        self.episodic_bonus = 0.0

    @torch.no_grad()
    def search(self, root_h: torch.Tensor, root_z: torch.Tensor) -> Tuple[int, torch.Tensor]:
        root = MCTSNode(root_h, root_z)
        
        # Inject the Digital Twin prior at the root
        state = torch.cat([root_h, root_z], -1)
        root_probs = self.actor(state).probs.squeeze(0).cpu().numpy()
        
        for action in range(self.n_actions):
            root.children[action] = MCTSNode(None, None, prior=root_probs[action])

        for _ in range(self.n_simulations):
            node = root
            search_path = [node]
            action_path = []
            
            # Selection (PUCT)
            for _ in range(self.horizon):
                if not node.children: break
                
                best_score = -float('inf')
                best_action = -1
                
                for a, child in node.children.items():
                    u = self.c_puct * child.prior * math.sqrt(node.visits + 1) / (1 + child.visits)
                    q = child.value
                    score = q + u
                    if score > best_score:
                        best_score = score
                        best_action = a
                        
                action_path.append(best_action)
                node = node.children[best_action]
                search_path.append(node)

            # Expansion & Evaluation
            parent = search_path[-2]
            action = action_path[-1]
            a_emb = self.action_enc(torch.tensor([action], device=self.device))
            h2, z2, _ = self.rssm.imagine_step(parent.h, parent.z, a_emb)
            
            s2 = torch.cat([h2, z2], -1)
            cons = self.consequence(s2)
            
            node.h = h2
            node.z = z2
            node.reward = cons["reward"].item()
            node.termination = cons["termination"].item()
            
            value = cons["value"].item()
            
            if node.termination < 0.5:
                child_probs = self.actor(s2).probs.squeeze(0).cpu().numpy()
                for a in range(self.n_actions):
                    node.children[a] = MCTSNode(None, None, prior=child_probs[a])

            # Backpropagation
            for i, n in enumerate(reversed(search_path)):
                n.visits += 1
                n.value_sum += value
                value = n.reward + self.gamma * (1.0 - n.termination) * value

        # Action selection based on visit counts
        visits = torch.tensor([root.children[a].visits for a in range(self.n_actions)], dtype=torch.float32, device=self.device)
        probs = visits / visits.sum()
        best_action = int(probs.argmax().item())
        
        return best_action, probs