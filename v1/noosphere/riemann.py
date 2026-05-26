import torch
import numpy as np
from typing import List, Dict, Optional

def sinkhorn_ot_mapping(x_source: torch.Tensor, x_target: torch.Tensor, epsilon: float = 0.1, num_iters: int = 20) -> torch.Tensor:
    """
    Maps x_source to x_target using Sinkhorn Optimal Transport.
    x_source: (N, D)
    x_target: (M, D)
    Returns x_aligned: (N, D)
    """
    # Compute cost matrix C
    # C_ij = ||x_s_i - x_t_j||^2
    N, D = x_source.shape
    M, _ = x_target.shape
    
    C = torch.cdist(x_source, x_target) ** 2
    
    # Sinkhorn-Knopp
    K = torch.exp(-C / epsilon)
    u = torch.ones(N, device=x_source.device) / N
    v = torch.ones(M, device=x_source.device) / M
    
    log_K = -C / epsilon
    f = torch.zeros(N, device=x_source.device)
    g = torch.zeros(M, device=x_source.device)
    
    for _ in range(num_iters):
        # f = epsilon * (log(u) - logsumexp(log_K + g/epsilon))
        f = epsilon * (torch.log(torch.tensor(1.0/N)) - torch.logsumexp(log_K + g.unsqueeze(0) / epsilon, dim=1))
        g = epsilon * (torch.log(torch.tensor(1.0/M)) - torch.logsumexp(log_K + f.unsqueeze(1) / epsilon, dim=0))
        
    P = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) + log_K) / epsilon) # (N, M)
    
    # Barycentric mapping
    P_sum = P.sum(dim=1, keepdim=True)
    P_norm = P / (P_sum + 1e-8)
    
    x_aligned = torch.matmul(P_norm, x_target)
    return x_aligned

def compute_ea_reference(trials: torch.Tensor) -> torch.Tensor:
    """
    Computes the Euclidean Alignment reference matrix.
    trials: (N, C, T)
    Returns R_inv_sq: (C, C)
    """
    N, C, T = trials.shape
    covs = []
    for i in range(N):
        x = trials[i]
        c = torch.matmul(x, x.t()) / (T - 1)
        covs.append(c)
    R = torch.stack(covs).mean(dim=0)
    L, Q = torch.linalg.eigh(R)
    R_inv_sq = Q @ torch.diag(1.0 / torch.sqrt(L + 1e-8)) @ Q.t()
    return R_inv_sq

def apply_ea(x: torch.Tensor, R_inv_sq: torch.Tensor) -> torch.Tensor:
    """
    Applies Euclidean Alignment.
    x: (..., C, T)
    R_inv_sq: (C, C)
    """
    return torch.matmul(R_inv_sq, x)
