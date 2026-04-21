import re

with open('noosphere/s4_eeg.py', 'r') as f:
    content = f.read()

# Replace RiemannianStem entirely
new_stem = '''class RiemannianStem(nn.Module):
    def __init__(self, in_ch: int, d_model: int, window: int = 64, stride: int = 16, **kwargs):
        super().__init__()
        self.in_ch = in_ch
        self.ts_dim = in_ch * (in_ch + 1) // 2
        self.window = window
        self.stride = stride
        self.proj = nn.Sequential(
            nn.Linear(self.ts_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        w = min(self.window, T)
        x_unfold = x.unfold(dimension=-1, size=w, step=self.stride)
        B, C, L, W = x_unfold.shape
        x_w = x_unfold.transpose(1, 2).contiguous().view(B*L, C, W)
        
        x_c = x_w - x_w.mean(-1, keepdim=True)
        cov = torch.bmm(x_c, x_c.transpose(1, 2)) / (W - 1 + 1e-8)
        trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        eps = (1e-3 * trace / C) + 1e-4
        cov = cov + eps * torch.eye(C, device=x.device).unsqueeze(0)
        
        try:
            cov_det = cov.detach()  # PREVENT EIGH GRADIENT EXPLOSION
            vals, vecs = torch.linalg.eigh(cov_det.to(torch.float64))
            vals = torch.log(vals.clamp(min=1e-7))
            log_cov = torch.bmm(vecs, torch.bmm(torch.diag_embed(vals), vecs.transpose(1, 2))).to(torch.float32)
        except Exception:
            log_cov = cov.detach()
            
        idx = torch.triu_indices(C, C, device=x.device)
        ts_vector = log_cov[:, idx[0], idx[1]]
        seq = self.proj(ts_vector).view(B, L, -1)
        return seq

# ── Attention Pooling ──────────────────────────────────────────────────────────'''

content = re.sub(r'class RiemannianStem\(nn\.Module\):.*?# ── Attention Pooling ──────────────────────────────────────────────────────────', new_stem, content, flags=re.DOTALL)

with open('noosphere/s4_eeg.py', 'w') as f:
    f.write(content)

print("Patch applied to RiemannianStem.")
