"""
noosphere/gnn.py
================
Learned-Adjacency Graph Neural Network — Stream C (Kinematics)

Joint/body states are embedded as graph nodes. Edges are fully learned
from data rather than hardcoded from a skeleton, so the model discovers
which physical couplings matter for each task.

Adjacency learning
------------------
    A_raw ∈ ℝ^(N×N)           — learnable, unconstrained
    A     = sigmoid(A_raw)     — soft ∈ [0, 1]
    Ã     = D^(-½) A D^(-½)   — symmetric normalisation (GCN-style)

Each message-passing layer has its own independent A_raw, allowing
layer 1 to learn local couplings and later layers to capture longer-range
task-relevant interactions.

Sparsity regularisation L1 = λ·‖A‖₁ encourages pruning weak edges —
call gnn.sparsity_loss() and add to the world-model training loss.

Output
------
    graph_token    (B, 1, d_model)  — global summary for single injection
    graph_sequence (B, N, d_model)  — per-node embeddings for cross-attention
    adj_matrices   list of (N, N)   — current learned adjacency per layer
    sparsity_loss  scalar tensor
    node_attn      (B, N)           — attention weights over nodes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ── Learned adjacency ─────────────────────────────────────────────────────────

class LearnedAdjacency(nn.Module):
    """
    Single learned adjacency matrix with temperature annealing.

    temperature controls the sharpness of the sigmoid — anneal toward 0
    over training to encourage hard sparse topology.
    """
    def __init__(self, n_nodes: int, temperature: float = 1.0,
                 sparse_reg: float = 0.01):
        super().__init__()
        self.n_nodes    = n_nodes
        self.sparse_reg = sparse_reg
        self.W = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)
        self.register_buffer("temp", torch.tensor(temperature))

    def forward(self) -> torch.Tensor:
        A = torch.sigmoid(self.W / self.temp)
        A = A + torch.eye(self.n_nodes, device=A.device)   # self-loops
        D = A.sum(-1).clamp(min=1e-6)
        D_inv = torch.diag(D.pow(-0.5))
        return D_inv @ A @ D_inv

    def sparsity_loss(self) -> torch.Tensor:
        return self.sparse_reg * torch.sigmoid(self.W).abs().mean()

    def anneal(self, factor: float = 0.999):
        self.temp.mul_(factor).clamp_(min=0.01)


# ── Graph attention layer ─────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Graph attention layer with multi-head attention over neighbourhood.

    h_i^(l+1) = LN(h_i + W_self·h_i + Σⱼ Ãᵢⱼ · Attn(h_i, h_j))
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.W_self  = nn.Linear(d_model, d_model, bias=False)
        self.W_msg   = nn.Linear(d_model, d_model, bias=False)
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        n = self.norm1(h)
        # Weighted message aggregation: Ã·W_msg·h
        agg = torch.einsum("nm,bmd->bnd", A, self.W_msg(n))
        # Self-message
        h   = h + self.drop(self.W_self(n) + agg)
        h   = h + self.drop(self.ffn(self.norm2(h)))
        return h


# ── Node feature encoder ──────────────────────────────────────────────────────

class NodeEncoder(nn.Module):
    """Projects raw node features to d_model with optional temporal encoding."""
    def __init__(self, feature_dim: int, d_model: int, use_temporal: bool = False,
                 temporal_len: int = 10):
        super().__init__()
        self.temporal = use_temporal
        if use_temporal:
            self.proj = nn.Linear(feature_dim * temporal_len, d_model)
        else:
            self.proj = nn.Linear(feature_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal and x.dim() == 4:
            B, N, T, F = x.shape
            x = x.reshape(B, N, T * F)
        return self.norm(self.proj(x))


# ── Full GNN ──────────────────────────────────────────────────────────────────

class KinematicGNN(nn.Module):
    """
    Multi-layer GNN with per-layer learned adjacency.

    Parameters
    ----------
    n_nodes        : number of joints / bodies
    node_feature_dim: features per node (e.g. 12 for pos+vel+angle+torque)
    d_model        : embedding dimension
    n_layers       : number of message-passing layers
    n_heads        : attention heads per GAT layer
    sparse_reg     : L1 penalty coefficient on adjacency entries
    """
    def __init__(
        self,
        n_nodes:          int,
        node_feature_dim: int,
        d_model:          int   = 256,
        n_layers:         int   = 3,
        n_heads:          int   = 4,
        dropout:          float = 0.1,
        temperature:      float = 1.0,
        sparse_reg:       float = 0.01,
        use_temporal:     bool  = False,
        temporal_len:     int   = 10,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model

        self.node_enc   = NodeEncoder(node_feature_dim, d_model, use_temporal, temporal_len)
        self.adjacencies = nn.ModuleList([
            LearnedAdjacency(n_nodes, temperature, sparse_reg) for _ in range(n_layers)
        ])
        self.gat_layers = nn.ModuleList([
            GATLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.global_attn = nn.Linear(d_model, 1)
        self.global_proj = nn.Linear(d_model, d_model)
        self.drop        = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor) -> Dict:
        h     = self.node_enc(node_features)
        adjs  = []
        total_sparse = torch.tensor(0.0, device=h.device)

        for adj_mod, gat in zip(self.adjacencies, self.gat_layers):
            A = adj_mod()
            adjs.append(A.detach())
            total_sparse = total_sparse + adj_mod.sparsity_loss()
            h = gat(h, A)

        h = self.drop(h)

        w       = F.softmax(self.global_attn(h).squeeze(-1), dim=-1)
        summary = self.global_proj((w.unsqueeze(-1) * h).sum(1))

        return {
            "graph_token":    summary.unsqueeze(1),    # (B, 1, d_model)
            "graph_sequence": h,                        # (B, N, d_model)
            "adj_matrices":   adjs,
            "sparsity_loss":  total_sparse,
            "node_attn":      w,
        }

    def anneal_adjacencies(self, factor: float = 0.999):
        for adj in self.adjacencies:
            adj.anneal(factor)

    def get_adjacency_summary(self) -> Dict[str, float]:
        results = {}
        eye = None
        for i, adj in enumerate(self.adjacencies):
            A = adj().detach()
            if eye is None:
                eye = torch.eye(self.n_nodes, device=A.device)
            off = A * (1 - eye)
            results[f"adj_{i}_sparsity"]   = (off < 0.1).float().mean().item()
            results[f"adj_{i}_mean_weight"] = off.mean().item()
            results[f"adj_{i}_asymmetry"]  = (A - A.t()).abs().mean().item()
        return results
