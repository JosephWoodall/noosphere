"""
noosphere/memory.py
===================
Experience and Episodic Memory Buffers

Features:
- Dual-Control Tracking: SequenceReplayBuffer now stores both discrete macro-intents 
  and continuous physical trajectories (xyz) to enable full-spectrum Behavioral Cloning.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Any

class SequenceReplayBuffer:
    def __init__(self, capacity: int, seq_len: int):
        self.capacity = capacity
        self.seq_len = seq_len
        self.ptr = 0
        self.size = 0
        
        self.obs_buffer: List[Dict[str, np.ndarray]] = [{} for _ in range(capacity)]
        self.action_buffer = np.zeros(capacity, dtype=np.int64)
        self.continuous_action_buffer = np.zeros((capacity, 3), dtype=np.float32) # New: Physical tracking
        self.reward_buffer = np.zeros(capacity, dtype=np.float32)
        self.done_buffer = np.zeros(capacity, dtype=bool)

    def add_step(self, obs: Dict[str, Any], action: int, continuous_action: np.ndarray, reward: float, done: bool):
        numpy_obs = {}
        for k, v in obs.items():
            if isinstance(v, torch.Tensor): numpy_obs[k] = v.detach().cpu().numpy()
            else: numpy_obs[k] = np.asarray(v)

        self.obs_buffer[self.ptr] = numpy_obs
        self.action_buffer[self.ptr] = action
        
        # Ensure correct shape for 3D continuous intent
        if continuous_action is not None:
            self.continuous_action_buffer[self.ptr] = np.asarray(continuous_action, dtype=np.float32).flatten()[:3]
            
        self.reward_buffer[self.ptr] = reward
        self.done_buffer[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        if self.size < self.seq_len: return {}

        batch = {"actions": [], "continuous_actions": [], "rewards": [], "dones": []}
        obs_keys = self.obs_buffer[0].keys()
        for k in obs_keys: batch[k] = []

        valid_starts = []
        for i in range(self.size):
            end_idx = (i + self.seq_len) % self.capacity
            if not (i <= self.ptr < end_idx or end_idx < i <= self.ptr):
                valid_starts.append(i)

        if len(valid_starts) < batch_size: return {}

        start_indices = np.random.choice(valid_starts, batch_size, replace=False)

        for start in start_indices:
            seq_indices = [(start + i) % self.capacity for i in range(self.seq_len)]
            batch["actions"].append(self.action_buffer[seq_indices])
            batch["continuous_actions"].append(self.continuous_action_buffer[seq_indices])
            batch["rewards"].append(self.reward_buffer[seq_indices])
            batch["dones"].append(self.done_buffer[seq_indices])
            for k in obs_keys:
                modality_seq = [self.obs_buffer[idx].get(k, np.zeros_like(self.obs_buffer[start][k])) for idx in seq_indices]
                batch[k].append(np.stack(modality_seq))

        out = {
            "actions": torch.tensor(np.stack(batch["actions"]), dtype=torch.long, device=device),
            "continuous_actions": torch.tensor(np.stack(batch["continuous_actions"]), dtype=torch.float32, device=device),
            "rewards": torch.tensor(np.stack(batch["rewards"]), dtype=torch.float32, device=device),
            "dones": torch.tensor(np.stack(batch["dones"]), dtype=torch.float32, device=device)
        }
        for k in obs_keys:
            out[k] = torch.tensor(np.stack(batch[k]), dtype=torch.float32, device=device)

        return out

    def __len__(self) -> int: return self.size

class EpisodicMemory:
    def __init__(self, state_dim: int, value_dim: int, capacity: int = 5000):
        self.capacity = capacity; self.state_dim = state_dim
        self.ptr = 0; self.size = 0
        self.keys = torch.zeros(capacity, state_dim)
        self.values = torch.zeros(capacity, value_dim)

    def write(self, state: torch.Tensor, value: torch.Tensor):
        B = state.shape[0]
        for i in range(B):
            self.keys[self.ptr] = state[i].detach().cpu()
            self.values[self.ptr] = value[i].detach().cpu()
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def read(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0: raise ValueError("Episodic memory is empty.")
        dev = query.device
        valid_keys = self.keys[:self.size].to(dev)
        valid_values = self.values[:self.size].to(dev)
        
        query_norm = torch.nn.functional.normalize(query, dim=-1)
        keys_norm = torch.nn.functional.normalize(valid_keys, dim=-1)
        sim = torch.matmul(query_norm, keys_norm.transpose(-1, -2)) 
        
        topk_sim, topk_idx = torch.topk(sim, min(k, self.size), dim=-1)
        attn = torch.nn.functional.softmax(topk_sim, dim=-1)
        
        B = query.shape[0]
        gathered_values = torch.stack([valid_values[topk_idx[b]] for b in range(B)]) 
        return gathered_values, attn

class WorkingMemory:
    def __init__(self, capacity: int = 20):
        self.capacity = capacity; self.rewards = []
    def push(self, obs: np.ndarray, action: int, reward: float):
        self.rewards.append(reward)
        if len(self.rewards) > self.capacity: self.rewards.pop(0)
    def recent_rewards(self, n: int = 10) -> List[float]: return self.rewards[-n:]
    def clear(self): self.rewards.clear()