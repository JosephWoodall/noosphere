import re
import sys

def patch():
    with open('noosphere/s4_eeg.py', 'r') as f:
        content = f.read()

    # 1. Replace S4DLayer with MambaS6Layer
    content = re.sub(
        r'class S4DLayer\(nn\.Module\):.*?class SelectiveS4DBlock',
        '''class S4DLayer(nn.Module):
    """
    True Mamba S6 Selective Scan Layer.
    - B, C, dt are input-dependent.
    - Processed sequentially (highly efficient for short L~13 sliding windows).
    """
    def __init__(
        self, d_model: int, d_state: int = 64, dt_min: float = 0.001,
        dt_max: float = 0.1, bidirectional: bool = True, dropout: float = 0.1,
        selective: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.bidir = bidirectional
        
        A_diag = torch.abs(torch.diagonal(_hippo_a(d_state)))
        self.A_log = nn.Parameter(torch.log(A_diag).unsqueeze(0).expand(d_model, -1).clone())
        
        self.dt_proj = nn.Linear(d_model, d_model)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))
        
        log_dt = (torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.dt_bias = nn.Parameter(log_dt)
        
        if bidirectional:
            self.out_bi = nn.Linear(d_model * 2, d_model)
        self.out = nn.Linear(d_model, d_model)

    def _scan(self, u: torch.Tensor) -> torch.Tensor:
        B, L, H = u.shape
        D_S = self.d_state
        
        dt = F.softplus(self.dt_proj(u) + self.dt_bias)
        B_val = self.B_proj(u)
        C_val = self.C_proj(u)
        
        A = -torch.exp(self.A_log)
        
        state = torch.zeros(B, H, D_S, device=u.device, dtype=u.dtype)
        y = torch.zeros(B, L, H, device=u.device, dtype=u.dtype)
        
        for t in range(L):
            u_t = u[:, t, :]
            dt_t = dt[:, t, :].unsqueeze(-1)
            B_t = B_val[:, t, :].unsqueeze(1)
            C_t = C_val[:, t, :].unsqueeze(1)
            
            A_bar = torch.exp(dt_t * A.unsqueeze(0))
            B_bar = (A_bar - 1.0) / (A.unsqueeze(0) + 1e-8) * B_t
            
            state = A_bar * state + B_bar * u_t.unsqueeze(-1)
            y[:, t, :] = (state * C_t).sum(-1) + u_t * self.D
            
        return y
        
    def forward(self, u: torch.Tensor, inference: bool = False):
        y = self._scan(u)
        if self.bidir:
            yr = torch.flip(self._scan(torch.flip(u, [1])), [1])
            y  = self.out_bi(torch.cat([y, yr], dim=-1))
        else:
            y = self.out(y)
        return y, None

class SelectiveS4DBlock''',
        content,
        flags=re.DOTALL
    )

    # 2. Replace RiemannianStem with SlidingRiemannianStem
    content = re.sub(
        r'class RiemannianStem\(nn\.Module\):.*?class SpectralGraphWaveletStem',
        '''class RiemannianStem(nn.Module):
    """
    Sliding Window Tangent-space Riemannian geometry encoder.
    Returns (B, L_windows, ts_dim) sequence instead of static token.
    Uses Ledoit-Wolf shrinkage approx.
    """
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
            vals, vecs = torch.linalg.eigh(cov.to(torch.float64))
            vals = torch.log(vals.clamp(min=1e-7))
            log_cov = torch.bmm(vecs, torch.bmm(torch.diag_embed(vals), vecs.transpose(1, 2))).to(torch.float32)
        except Exception:
            log_cov = cov
            
        idx = torch.triu_indices(C, C, device=x.device)
        ts_vector = log_cov[:, idx[0], idx[1]]
        seq = self.proj(ts_vector).view(B, L, -1)
        return seq

class SpectralGraphWaveletStem''',
        content,
        flags=re.DOTALL
    )

    # 3. Update S4EEGEncoder.__init__
    content = content.replace(
        '''        # ── Temporal encoding (spectral graph wavelets) ──────────────────────
        self.spectral_wavelet = SpectralGraphWaveletStem(in_channels, d_model,
                                                          downsample=downsample)

        # ── S4D MoE blocks for high and low frequency paths ──────────────────''',
        '''        # ── S4D MoE blocks for high and low frequency paths ──────────────────'''
    )

    content = content.replace(
        '''        # FiLM: spatial features condition temporal processing
        self.film_high = nn.ModuleList([FiLM(d_model) for _ in range(n_hi)])
        self.film_low  = nn.ModuleList([FiLM(d_model) for _ in range(n_lo)])''',
        ''''''
    )

    # 4. Update S4EEGEncoder.forward_momentum (just simplify it to support the new forward)
    content = re.sub(
        r'def forward_momentum.*?def forward\(',
        '''def forward_momentum(self, eeg: torch.Tensor) -> torch.Tensor:
        self._init_momentum()
        gater    = self.artifact_gater(eeg).unsqueeze(-1)
        eeg_g    = eeg * gater
        hurst    = self.hurst(eeg).unsqueeze(-1).clamp(min=0.1)
        x      = self.momentum_spatial(eeg_g) * hurst
        x_hi = x
        x_lo = x
        for blk in self.momentum_blocks_hi: x_hi = blk(x_hi)
        for blk in self.momentum_blocks_lo: x_lo = blk(x_lo)
        return torch.cat([x_hi, x_lo], dim=1).mean(dim=1)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(''',
        content,
        flags=re.DOTALL
    )

    # 5. Update S4EEGEncoder.forward
    content = re.sub(
        r'# ── Temporal path \(spectral graph wavelets\).*?x = torch\.cat\(\[x_hi, x_lo\], dim=1\)',
        '''# ── Temporal sequence is now entirely Riemannian Manifolds ────────────
        x_hi = x_spatial * hurst
        x_lo = x_spatial * hurst

        # ── S4D MoE blocks (True Mamba S6 Sequential Scan) ────────────────────
        for blk in self.blocks_high:
            x_hi = blk(x_hi)

        for blk in self.blocks_low:
            x_lo = blk(x_lo)

        x = torch.cat([x_hi, x_lo], dim=1)''',
        content,
        flags=re.DOTALL
    )

    # 6. Update get_flops and update_momentum to remove wavelet/film references
    content = content.replace(
        '''(self.spectral_wavelet, self.momentum_temporal),''',
        ''''''
    )
    content = content.replace(
        '''for p in self.momentum_temporal.parameters():  p.requires_grad = False''',
        ''''''
    )

    with open('noosphere/s4_eeg.py', 'w') as f:
        f.write(content)
        
    print("s4_eeg.py patched successfully.")

if __name__ == '__main__':
    patch()
