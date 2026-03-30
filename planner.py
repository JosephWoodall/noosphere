"""
noosphere/planner.py
====================
Actor, Critic, and PUCT MCTS Planner

Features:
- Dual-Control Action Encoder: Fuses the discrete action and continuous spatial intent.
- Dual-Head Actor: Predicts BOTH discrete macro-intents and continuous physical trajectories.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class ActionEncoder(nn.Module):
    def __init__(self, n_actions: int, action_dim: int):
        super().__init__()
        # Reserve 3 dimensions for the continuous spatial vector (xyz)
        self.emb = nn.Embedding(n_actions, action_dim - 3)
        
    def forward(self, discrete_action: torch.Tensor, continuous_action: torch.Tensor) -> torch.Tensor:
        # Fuse digital command with physical intent into a unified dimension
        discrete_emb = self.emb(discrete_action)
        return torch.cat([discrete_emb, continuous_action], dim=-1)

class Actor(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU()
        )
        self.discrete_head = nn.Linear(hidden_dim, n_actions)
        self.continuous_head = nn.Linear(hidden_dim, 3) # Physical trajectory
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        h = self.net(state)
        return torch.distributions.Categorical(logits=self.discrete_head(h)), self.continuous_head(h)

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

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: return self.q1(state), self.q2(state)
    def min_value(self, state: torch.Tensor) -> torch.Tensor:
        v1, v2 = self.forward(state)
        return torch.min(v1, v2)

class ImaginationBuffer:
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma; self.lam = lam; self.clear()
    def clear(self):
        self.states, self.discrete_actions, self.continuous_actions, self.rewards = [], [], [], []
        self.values, self.log_probs, self.dones = [], [], []
    def add(self, s, a_d, a_c, r, v, lp, d):
        self.states.append(s); self.discrete_actions.append(a_d); self.continuous_actions.append(a_c)
        self.rewards.append(r); self.values.append(v); self.log_probs.append(lp); self.dones.append(d)
    def lambda_returns(self) -> torch.Tensor:
        R = self.values[-1]
        returns = []
        for r, v, d in zip(reversed(self.rewards), reversed(self.values), reversed(self.dones)):
            R = r + self.gamma * (1.0 - d) * ((1.0 - self.lam) * v + self.lam * R)
            returns.insert(0, R)
        return torch.stack(returns)

class MCTSNode:
    def __init__(self, h: Optional[torch.Tensor], z: Optional[torch.Tensor], prior: float = 1.0):
        self.h = h; self.z = z; self.prior = prior
        self.visits = 0; self.value_sum = 0.0
        self.children: Dict[int, MCTSNode] = {}
        self.reward = 0.0; self.termination = 0.0
        self.continuous_action = torch.zeros(3)

    @property
    def value(self) -> float: return self.value_sum / self.visits if self.visits > 0 else 0.0

class MCTSPlanner:
    def __init__(self, rssm, consequence, actor, action_enc, n_actions, n_simulations: int = 30, horizon: int = 10, gamma: float = 0.99, c_puct: float = 1.5, device=None):
        self.rssm = rssm; self.consequence = consequence; self.actor = actor
        self.action_enc = action_enc; self.n_actions = n_actions
        self.n_simulations = n_simulations; self.horizon = horizon
        self.gamma = gamma; self.c_puct = c_puct; self.device = device

    @torch.no_grad()
    def search(self, root_h: torch.Tensor, root_z: torch.Tensor, bci_discrete: Optional[torch.Tensor] = None, bci_continuous: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor, torch.Tensor]:
        root = MCTSNode(root_h, root_z)
        state = torch.cat([root_h, root_z], -1)
        
        actor_dist, actor_cont = self.actor(state)
        actor_probs = actor_dist.probs.squeeze(0).cpu().numpy()
        actor_cont = actor_cont.squeeze(0)
        
        root_probs = bci_discrete.cpu().numpy() if bci_discrete is not None else actor_probs
        root_cont = bci_continuous if bci_continuous is not None else actor_cont
        
        for action in range(self.n_actions):
            root.children[action] = MCTSNode(None, None, prior=root_probs[action])
            root.children[action].continuous_action = root_cont

        for _ in range(self.n_simulations):
            node = root; search_path = [node]; action_path = []
            
            for _ in range(self.horizon):
                if not node.children: break
                best_score = -float('inf'); best_action = -1
                for a, child in node.children.items():
                    u = self.c_puct * child.prior * math.sqrt(node.visits + 1) / (1 + child.visits)
                    score = child.value + u
                    if score > best_score:
                        best_score = score; best_action = a
                action_path.append(best_action); node = node.children[best_action]; search_path.append(node)

            parent = search_path[-2]
            action = action_path[-1]
            
            # Unroll with fused action representation
            a_emb = self.action_enc(torch.tensor([action], device=self.device), node.continuous_action.unsqueeze(0))
            h2, z2, _ = self.rssm.imagine_step(parent.h, parent.z, a_emb)
            
            s2 = torch.cat([h2, z2], -1)
            cons = self.consequence(s2)
            
            node.h = h2; node.z = z2
            node.termination = cons["termination"].item()
            
            base_reward = cons.get("reward", torch.zeros(1)).item()
            intent_penalty = 0.0
            if bci_discrete is not None:
                intent_penalty = -5.0 * (1.0 - bci_discrete[action].item())
                
            node.reward = base_reward + intent_penalty
            value = cons.get("value", torch.zeros(1)).item()
            
            if node.termination < 0.5:
                child_dist, child_cont = self.actor(s2)
                child_probs = child_dist.probs.squeeze(0).cpu().numpy()
                child_cont = child_cont.squeeze(0)
                for a in range(self.n_actions):
                    node.children[a] = MCTSNode(None, None, prior=child_probs[a])
                    node.children[a].continuous_action = child_cont

            for i, n in enumerate(reversed(search_path)):
                n.visits += 1; n.value_sum += value
                value = n.reward + self.gamma * (1.0 - n.termination) * value

        visits = torch.tensor([root.children[a].visits for a in range(self.n_actions)], dtype=torch.float32, device=self.device)
        probs = visits / visits.sum()
        best_action = int(probs.argmax().item())
        
        return best_action, probs, root_cont