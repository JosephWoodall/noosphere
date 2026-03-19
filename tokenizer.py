"""
noosphere/tokenizer.py
======================
Multimodal Patch Tokenizer — Stream A

Converts spatially-structured sensor data into token sequences for the
shared transformer backbone. Each modality is handled by a dedicated
tokenizer that produces (B, N, d_model) tokens.

Supported modalities
--------------------
    rgb        (B, 3, H, W)         — colour image
    depth      (B, 1, H, W)         — metric depth map
    rgb_right  (B, 3, H, W)         — stereo right view
    lidar      (B, N_pts, 3+F)      — point cloud with optional features
    audio      (B, 1, n_mels, T)    — log-mel spectrogram

New modalities are registered via UnifiedTokenizer.register_modality()
with any nn.Module that maps its input to (B, N, d_model). No other
code changes required.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


MODALITY_IDS = {
    "rgb":       0,
    "depth":     1,
    "rgb_right": 2,
    "lidar":     3,
    "audio":     4,
}
MAX_MODALITIES = 8


# ── Positional encodings ──────────────────────────────────────────────────────

class SinusoidalPosEnc(nn.Module):
    """Fixed sinusoidal PE — extrapolates beyond training length."""
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.shape[1]].unsqueeze(0)


class Learned2DPosEnc(nn.Module):
    """Factored row+col learned PE for image patches."""
    def __init__(self, d_model: int, max_h: int = 64, max_w: int = 64):
        super().__init__()
        self.row = nn.Embedding(max_h, d_model // 2)
        self.col = nn.Embedding(max_w, d_model - d_model // 2)

    def forward(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        r = self.row(torch.arange(h, device=device)).unsqueeze(1).expand(h, w, -1)
        c = self.col(torch.arange(w, device=device)).unsqueeze(0).expand(h, w, -1)
        return torch.cat([r, c], dim=-1).reshape(h * w, -1)


# ── Per-modality tokenizers ───────────────────────────────────────────────────

class ImagePatchTokenizer(nn.Module):
    """
    ViT-style patch embedding.

    Splits (B, C, H, W) into non-overlapping P×P patches and linearly
    projects each to d_model. Token count N = (H/P)·(W/P).
    """
    def __init__(self, in_channels: int, d_model: int, patch_size: int = 8,
                 max_h: int = 64, max_w: int = 64):
        super().__init__()
        P = patch_size
        self.patch_size = P
        self.proj    = nn.Conv2d(in_channels, d_model, kernel_size=P, stride=P)
        self.pos_enc = Learned2DPosEnc(d_model, max_h // P + 1, max_w // P + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.proj(x)                            # (B, d, H/P, W/P)
        h, w = t.shape[2], t.shape[3]
        t = t.flatten(2).transpose(1, 2)            # (B, N, d)
        return t + self.pos_enc(h, w, x.device).unsqueeze(0)


class LiDARTokenizer(nn.Module):
    """
    Point cloud tokenizer via farthest-point sampling + PointNet aggregation.
    Produces M tokens from N input points with local neighbourhood context.
    """
    def __init__(self, in_features: int = 3, d_model: int = 256,
                 n_samples: int = 64, k_neighbors: int = 16):
        super().__init__()
        self.n_samples = n_samples
        self.k          = k_neighbors
        self.mlp = nn.Sequential(
            nn.Linear(in_features + 3, 64), nn.ReLU(),
            nn.Linear(64, 128),             nn.ReLU(),
            nn.Linear(128, d_model),
        )
        self.pos_enc = SinusoidalPosEnc(d_model)

    def _fps(self, xyz: torch.Tensor, n: int) -> torch.Tensor:
        B, N, _ = xyz.shape
        sel  = torch.zeros(B, n, dtype=torch.long, device=xyz.device)
        dist = torch.full((B, N), float("inf"), device=xyz.device)
        cur  = torch.zeros(B, dtype=torch.long, device=xyz.device)
        for i in range(n):
            sel[:, i]  = cur
            c_xyz      = xyz[torch.arange(B), cur].unsqueeze(1)
            dist       = torch.min(dist, (xyz - c_xyz).pow(2).sum(-1))
            cur        = dist.argmax(dim=1)
        return sel

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        B, N, C = points.shape
        M   = min(self.n_samples, N)
        idx = self._fps(points[..., :3], M)
        sampled = torch.gather(points, 1, idx.unsqueeze(-1).expand(B, M, C))
        return self.pos_enc(self.mlp(sampled))


# ── Registry ──────────────────────────────────────────────────────────────────

class UnifiedTokenizer(nn.Module):
    """
    Converts all present modalities to a single flat token sequence.

    Token = content_embed + modality_type_embed + position_embed

    Usage
    -----
        tokenizer = UnifiedTokenizer(d_model=256)
        tokenizer.register_modality("rgb",   ImagePatchTokenizer(3, 256))
        tokenizer.register_modality("depth", ImagePatchTokenizer(1, 256))
        tokens, mask = tokenizer({"rgb": rgb_tensor})
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model       = d_model
        self.cls_token     = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.modality_emb  = nn.Embedding(MAX_MODALITIES + 1, d_model)
        self.tokenizers    = nn.ModuleDict()
        self._order: List[str] = []

    def register_modality(self, name: str, tokenizer: nn.Module) -> "UnifiedTokenizer":
        """Register a new modality tokenizer. Chainable."""
        self.tokenizers[name] = tokenizer
        if name not in self._order:
            self._order.append(name)
        return self

    def forward(
        self, inputs: Dict[str, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        tokens   (B, 1+N_total, d_model)
        pad_mask (B, 1+N_total)  — True at padding positions
        """
        B      = next(v.shape[0] for v in inputs.values() if v is not None)
        device = next(v.device   for v in inputs.values() if v is not None)

        token_list, mask_list = [], []
        for name in self._order:
            if name not in inputs or inputs[name] is None:
                continue
            if name not in self.tokenizers:
                continue
            toks = self.tokenizers[name](inputs[name])           # (B, N, d)
            mid  = MODALITY_IDS.get(name, MAX_MODALITIES)
            toks = toks + self.modality_emb(
                torch.tensor(mid, device=device)
            ).unsqueeze(0).unsqueeze(0)
            token_list.append(toks)
            mask_list.append(torch.zeros(B, toks.shape[1], dtype=torch.bool, device=device))

        if not token_list:
            raise ValueError("At least one modality must be present.")

        cls    = self.cls_token.expand(B, 1, -1)
        c_mask = torch.zeros(B, 1, dtype=torch.bool, device=device)
        tokens = torch.cat([cls]  + token_list, dim=1)
        mask   = torch.cat([c_mask] + mask_list, dim=1)
        return tokens, mask


def build_tokenizer(d_model: int = 256, patch_size: int = 8) -> UnifiedTokenizer:
    """Convenience factory — registers all standard visual modalities."""
    t = UnifiedTokenizer(d_model)
    t.register_modality("rgb",       ImagePatchTokenizer(3, d_model, patch_size))
    t.register_modality("depth",     ImagePatchTokenizer(1, d_model, patch_size))
    t.register_modality("rgb_right", ImagePatchTokenizer(3, d_model, patch_size))
    t.register_modality("audio",     ImagePatchTokenizer(1, d_model, patch_size=4))
    t.register_modality("lidar",     LiDARTokenizer(3, d_model))
    return t
