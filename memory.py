"""
noosphere/memory.py
===================
Memory Systems

Three complementary stores:

    SequenceReplayBuffer  —  stores full episode trajectories for RSSM training
                             via TBPTT (sequences, not i.i.d. transitions)

    EpisodicMemory        —  long-term key-value store; retrieval by cosine
                             similarity; provides context from analogous past
                             situations during inference

    WorkingMemory         —  rolling short-term buffer of recent
                             (state, action, reward) tuples
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import random


class SequenceReplayBuffer:
    """
    Stores complete episode trajectories. Samples fixed-length contiguous
    sequences to preserve temporal coherence for RSSM training.
    """

    def __init__(self, max_episodes: int = 1000, seq_len: int = 50):
        self.max_episodes = max_episodes
        self.seq_len      = seq_len
        self.episodes: deque = deque(maxlen=max_episodes)
        self._ep = self._new_ep()

    @staticmethod
    def _new_ep() -> Dict[str, List]:
        return {k: [] for k in ["rgb","depth","eeg","structured","kinematics",
                                 "actions","rewards","dones"]}

    def add_step(self, obs: Dict[str, Any], action: int,
                 reward: float, done: bool):
        for k in ["rgb","depth","eeg","structured","kinematics"]:
            self._ep[k].append(obs.get(k))
        self._ep["actions"].append(action)
        self._ep["rewards"].append(reward)
        self._ep["dones"].append(done)
        if done:
            self._commit()

    def _commit(self):
        ep = self._ep
        if len(ep["actions"]) < self.seq_len:
            self._ep = self._new_ep(); return
        committed = {
            "actions": np.array(ep["actions"], dtype=np.int64),
            "rewards": np.array(ep["rewards"], dtype=np.float32),
            "dones":   np.array(ep["dones"],   dtype=np.float32),
        }
        for m in ["rgb","depth","eeg","structured","kinematics"]:
            vals = ep[m]
            if any(v is not None for v in vals):
                try:
                    ref   = next(v for v in vals if v is not None)
                    filled= [v if v is not None else np.zeros_like(ref) for v in vals]
                    committed[m] = np.stack(filled)
                except Exception:
                    pass
        self.episodes.append(committed)
        self._ep = self._new_ep()

    def sample(self, B: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if len(self.episodes) < B:
            return {}
        seqs = []
        for _ in range(B):
            ep  = random.choice(self.episodes)
            T   = len(ep["actions"])
            s   = np.random.randint(0, max(1, T - self.seq_len))
            seqs.append({k: v[s:s+self.seq_len] if isinstance(v, np.ndarray) else v
                         for k, v in ep.items()})
        batch = {}
        for k in ["actions","rewards","dones"]:
            try:
                batch[k] = torch.tensor(np.stack([s[k] for s in seqs]), device=device)
            except Exception: pass
        for m in ["rgb","depth","eeg","structured","kinematics"]:
            try:
                arrs = [s[m] for s in seqs if m in s]
                if len(arrs) == B:
                    batch[m] = torch.tensor(np.stack(arrs), dtype=torch.float32, device=device)
            except Exception: pass
        return batch

    def __len__(self): return len(self.episodes)

    @property
    def total_steps(self) -> int:
        return sum(len(ep["actions"]) for ep in self.episodes)


class EpisodicMemory(nn.Module):
    """
    Circular key-value memory with cosine-similarity retrieval.

    key   = compressed latent state  (what was the situation?)
    value = episode outcome summary  (what happened?)
    """
    def __init__(self, key_dim: int = 256, value_dim: int = 64,
                 capacity: int = 10_000, n_retrieve: int = 5):
        super().__init__()
        self.capacity   = capacity
        self.n_retrieve = n_retrieve
        self.key_proj   = nn.Sequential(
            nn.Linear(key_dim, key_dim), nn.SiLU(), nn.Linear(key_dim, key_dim))
        self.key_norm   = nn.LayerNorm(key_dim)
        self.register_buffer("keys",   torch.zeros(capacity, key_dim))
        self.register_buffer("values", torch.zeros(capacity, value_dim))
        self.register_buffer("_ptr",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("_full",  torch.zeros(1, dtype=torch.bool))

    def write(self, state: torch.Tensor, value: torch.Tensor):
        k   = self.key_norm(self.key_proj(state.detach()))
        ptr = int(self._ptr.item())
        self.keys[ptr]   = k.squeeze(0)
        self.values[ptr] = value.squeeze(0)
        ptr = (ptr + 1) % self.capacity
        self._ptr[0] = ptr
        if ptr == 0: self._full[0] = True

    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.capacity if self._full else int(self._ptr.item())
        if n == 0:
            dev = query.device
            return (torch.zeros(self.n_retrieve, self.values.shape[-1], device=dev),
                    torch.zeros(self.n_retrieve, device=dev))
        q    = F.normalize(self.key_norm(self.key_proj(query.detach())), -1)
        keys = F.normalize(self.keys[:n], -1)
        sim  = (keys @ q.T).squeeze(-1)
        k    = min(self.n_retrieve, n)
        top_sim, top_idx = sim.topk(k)
        vals = self.values[top_idx]
        if k < self.n_retrieve:
            vals    = F.pad(vals,    (0,0,0,self.n_retrieve-k))
            top_sim = F.pad(top_sim, (0, self.n_retrieve-k), value=-1e9)
        attn = F.softmax(top_sim, 0)
        return vals, attn

    def read_aggregated(self, query: torch.Tensor) -> torch.Tensor:
        vals, attn = self.read(query)
        return (vals * attn.unsqueeze(-1)).sum(0)


class WorkingMemory:
    """Rolling short-term context buffer."""
    def __init__(self, capacity: int = 20):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float):
        self.buffer.append({"state": state, "action": action, "reward": reward})

    def recent_rewards(self, n: int = 10) -> List[float]:
        return [e["reward"] for e in list(self.buffer)[-n:]]

    def cumulative_return(self) -> float:
        r = [e["reward"] for e in self.buffer]
        return float(np.sum(r)) if r else 0.0

    def clear(self): self.buffer.clear()
    def __len__(self): return len(self.buffer)
