"""
noosphere/gnn.py
================
Learned-Adjacency Graph Neural Network — Stream C (Kinematics)

Changes in v1.3.1
-----------------
LearnedAdjacency.forward: replaced D_inv @ A @ D_inv (O(N³) matrix multiply)
with element-wise scaling using the diagonal of D_inv.

    Old:  D_inv_matrix @ A @ D_inv_matrix   — two full (N,N)×(N,N) matmuls
    New:  d[:, None] * A * d[None, :]        — two (N,N) element-wise scales

For N=20 joints: 20³=8000 ops → 20²=400 ops per normalisation call.
Called once per layer per forward pass; savings compound at 30fps training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ── Learned adjacency ─────────────────────────────────────────────────────────

class LearnedAdjacency(nn.Module):
    """
    Single learned adjacency matrix with temperature annealing.
    Sparsity regularisation drives weak edges toward zero over training.
    """
    def __init__(self, n_nodes: int, temperature: float = 1.0,
                 sparse_reg: float = 0.01):
        super().__init__()
        self.n_nodes    = n_nodes
        self.sparse_reg = sparse_reg
        self.W = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)
        self.register_buffer("temp", torch.tensor(temperature))

    def forward(self) -> torch.Tensor:
        A  = torch.sigmoid(self.W / self.temp)
        A  = A + torch.eye(self.n_nodes, device=A.device)   # self-loops
        D  = A.sum(-1).clamp(min=1e-6)        # (N,) degree vector
        d  = D.pow(-0.5)                       # (N,) D^(-½) diagonal
        # Symmetric normalisation: Ã = D^(-½) A D^(-½)
        # Element-wise: Ã[i,j] = d[i] * A[i,j] * d[j]
        # O(N²) — not O(N³) as with matrix multiply
        return d[:, None] * A * d[None, :]

    def sparsity_loss(self) -> torch.Tensor:
        return self.sparse_reg * torch.sigmoid(self.W).abs().mean()

    def anneal(self, factor: float = 0.999):
        self.temp.mul_(factor).clamp_(min=0.01)


# ── Graph attention layer ─────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Graph attention layer with multi-head attention over neighbourhood.
    h_i^(l+1) = LN(h_i + W_self·h_i + Σⱼ Ãᵢⱼ · W_msg·h_j)
    """
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.W_self = nn.Linear(d_model, d_model, bias=False)
        self.W_msg  = nn.Linear(d_model, d_model, bias=False)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.drop   = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        n   = self.norm1(h)
        agg = torch.einsum("nm,bmd->bnd", A, self.W_msg(n))
        h   = h + self.drop(self.W_self(n) + agg)
        h   = h + self.drop(self.ffn(self.norm2(h)))
        return h


# ── Node feature encoder ──────────────────────────────────────────────────────

class NodeEncoder(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, use_temporal: bool = False,
                 temporal_len: int = 10):
        super().__init__()
        self.temporal = use_temporal
        in_dim = feature_dim * temporal_len if use_temporal else feature_dim
        self.proj = nn.Linear(in_dim, d_model)
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
    node_feature_dim: features per node
    d_model        : embedding dimension
    n_layers       : message-passing depth
    sparse_reg     : L1 coefficient on adjacency entries
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

        self.node_enc    = NodeEncoder(node_feature_dim, d_model, use_temporal, temporal_len)
        self.adjacencies = nn.ModuleList([
            LearnedAdjacency(n_nodes, temperature, sparse_reg) for _ in range(n_layers)
        ])
        self.gat_layers  = nn.ModuleList([
            GATLayer(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.global_attn = nn.Linear(d_model, 1)
        self.global_proj = nn.Linear(d_model, d_model)
        self.drop        = nn.Dropout(dropout)

    def forward(self, node_features: torch.Tensor) -> Dict:
        h            = self.node_enc(node_features)
        adjs         = []
        total_sparse = torch.tensor(0.0, device=h.device)

        for adj_mod, gat in zip(self.adjacencies, self.gat_layers):
            A            = adj_mod()
            adjs.append(A.detach())
            total_sparse = total_sparse + adj_mod.sparsity_loss()
            h            = gat(h, A)

        h = self.drop(h)
        w = F.softmax(self.global_attn(h).squeeze(-1), dim=-1)
        summary = self.global_proj((w.unsqueeze(-1) * h).sum(1))

        return {
            "graph_token":    summary.unsqueeze(1),   # (B, 1, d)
            "graph_sequence": h,                       # (B, N, d)
            "adj_matrices":   adjs,
            "sparsity_loss":  total_sparse,
            "node_attn":      w,
        }

    def anneal_adjacencies(self, factor: float = 0.999):
        for adj in self.adjacencies:
            adj.anneal(factor)

    def get_adjacency_summary(self) -> Dict[str, float]:
        results = {}
        eye     = None
        for i, adj in enumerate(self.adjacencies):
            A = adj().detach()
            if eye is None:
                eye = torch.eye(self.n_nodes, device=A.device)
            off = A * (1 - eye)
            results[f"adj_{i}_sparsity"]   = (off < 0.1).float().mean().item()
            results[f"adj_{i}_mean_weight"] = off.mean().item()
            results[f"adj_{i}_asymmetry"]   = (A - A.t()).abs().mean().item()
        return results
