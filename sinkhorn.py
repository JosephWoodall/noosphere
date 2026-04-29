import torch

def sinkhorn_ot_mapping(x_source, x_target, epsilon=0.1, num_iters=20):
    """
    Maps x_source to x_target using Sinkhorn Optimal Transport.
    x_source: (N, D)
    x_target: (M, D)
    Returns x_aligned: (N, D)
    """
    # Cost matrix
    C = torch.cdist(x_source, x_target) ** 2
    
    # Sinkhorn iterations
    log_K = -C / epsilon
    N, M = C.shape
    f = torch.zeros(N, 1, device=C.device)
    g = torch.zeros(1, M, device=C.device)
    
    for _ in range(num_iters):
        f = -torch.logsumexp(log_K + g, dim=1, keepdim=True)
        g = -torch.logsumexp(log_K + f, dim=0, keepdim=True)
        
    P = torch.exp(log_K + f + g) # (N, M)
    
    # Barycentric mapping
    P_sum = P.sum(dim=1, keepdim=True)
    P_norm = P / (P_sum + 1e-8)
    
    x_aligned = torch.matmul(P_norm, x_target)
    return x_aligned
