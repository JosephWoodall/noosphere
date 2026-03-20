"""
noosphere/perception.py
=======================
Hybrid Perception Model

Three streams fused into one transformer via four strategies.

    Stream A  patch tokenizer  →  transformer    RGB · depth · LiDAR · audio
    Stream B  S4 SSM                             EEG (continuous, no windowing)
    Stream C  learned-adj GNN                    kinematics (joints / bodies)

Fusion strategies
-----------------
    1  Single injection      S4 + GNN summaries prepended as tokens at layer 0
    2  Multi-scale injection summaries re-injected at layers in `inject_layers`
    3  Cross-attention       transformer tokens attend into S4 sequence / GNN nodes
    4  Gated fusion          γ = σ(W[pool_trans; pool_ext]) — dynamic modality gate

The four strategies alternate across layers so the transformer never
double-counts any stream within a single block.

Latency profiling
-----------------
    model.enable_profiling()
    ...
    model.print_profile()

Disabling strategies at inference
----------------------------------
    model.disable_fusion_strategy("cross_attn")   # drop cross-attn
    model.disable_fusion_strategy("inject")        # drop multi-scale injection
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from noosphere.tokenizer import build_tokenizer
from noosphere.s4_eeg    import S4EEGEncoder
from noosphere.gnn       import KinematicGNN


# ── Profiler ──────────────────────────────────────────────────────────────────

class FusionProfiler:
    def __init__(self):
        self._t: Dict[str, List[float]] = {}
        self.on = False

    def record(self, key: str, ms: float):
        if self.on:
            self._t.setdefault(key, []).append(ms)

    def summary(self, n: int = 100) -> Dict[str, float]:
        import numpy as np
        return {k: float(np.mean(v[-n:])) for k, v in self._t.items()}

    def reset(self):  self._t.clear()
    def enable(self): self.on = True;  return self
    def disable(self):self.on = False; return self


_PROFILER = FusionProfiler()


# ── Gated cross-attention ─────────────────────────────────────────────────────

class GatedCrossAttention(nn.Module):
    """
    Transformer tokens (Q) attend to external stream (K/V).
    Dynamic gate γ = σ(W[pool_q; pool_kv]) controls contribution.
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.xattn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),     nn.Sigmoid(),
        )
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.out     = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)

    def forward(
        self,
        q:    torch.Tensor,              # (B, N, d)
        kv:   torch.Tensor,              # (B, M, d)
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c, _ = self.xattn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                           key_padding_mask=mask, need_weights=False)
        c    = self.out(c)
        g    = self.gate_net(torch.cat([q.mean(1), kv.mean(1)], -1)).unsqueeze(1)
        return q + g * self.drop(c)


# ── Residual injection ────────────────────────────────────────────────────────

class ResidualInjection(nn.Module):
    """Gated residual addition of an external summary into a token position."""
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, tokens: torch.Tensor, ext: torch.Tensor, pos: int) -> torch.Tensor:
        e = self.proj(ext)
        g = self.gate(torch.cat([tokens[:, pos], e], -1))
        tokens = tokens.clone()
        tokens[:, pos] = tokens[:, pos] + g * e
        return tokens


# ── Hybrid transformer block ──────────────────────────────────────────────────

class HybridBlock(nn.Module):
    """
    Pre-LN transformer block with optional cross-attention and injection hooks.
    Pass None for any external stream to skip that fusion strategy.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1,
                 cross_s4: bool = False, cross_gnn: bool = False):
        super().__init__()
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.sattn   = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop        = nn.Dropout(dropout)
        self.xattn_s4    = GatedCrossAttention(d_model, n_heads, dropout) if cross_s4  else None
        self.xattn_gnn   = GatedCrossAttention(d_model, n_heads, dropout) if cross_gnn else None
        self.inject_s4   = ResidualInjection(d_model)
        self.inject_gnn  = ResidualInjection(d_model)

    def forward(
        self, x: torch.Tensor,
        pad_mask:    Optional[torch.Tensor] = None,
        s4_seq:      Optional[torch.Tensor] = None,
        gnn_nodes:   Optional[torch.Tensor] = None,
        s4_summary:  Optional[torch.Tensor] = None,
        gnn_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h, _ = self.sattn(self.norm1(x), self.norm1(x), self.norm1(x),
                          key_padding_mask=pad_mask, need_weights=False)
        x = x + self.drop(h)
        if self.xattn_s4  and s4_seq    is not None: x = self.xattn_s4(x, s4_seq)
        if self.xattn_gnn and gnn_nodes is not None: x = self.xattn_gnn(x, gnn_nodes)
        if s4_summary  is not None: x = self.inject_s4(x, s4_summary, pos=1)
        if gnn_summary is not None: x = self.inject_gnn(x, gnn_summary, pos=2)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ── Hybrid perception model ───────────────────────────────────────────────────

class HybridPerceptionModel(nn.Module):
    """
    Unified multimodal perception model.

    Parameters
    ----------
    d_model            : shared embedding dimension
    n_heads            : transformer attention heads
    n_layers           : transformer depth
    n_eeg_channels     : EEG electrode count
    s4_d_state         : S4 state dimension
    s4_n_blocks        : number of S4 blocks
    s4_downsample      : temporal downsampling factor before S4
    n_kinematic_nodes  : number of joints/bodies
    node_feature_dim   : features per kinematic node
    gnn_n_layers       : GNN depth
    cross_attn_layers  : transformer layers where cross-attention runs (default: [1,3,5])
    inject_layers      : transformer layers where multi-scale injection runs (default: [2,4])
    patch_size         : image patch size for vision tokenizer
    dropout            : dropout rate throughout

    Forward inputs (all optional except at least one must be present)
    -----------------------------------------------------------------
        rgb             (B, 3, H, W)
        depth           (B, 1, H, W)
        rgb_right       (B, 3, H, W)
        lidar           (B, N_pts, 3)
        audio           (B, 1, n_mels, T)
        eeg             (B, n_eeg_channels, T_samples)
        electrode_mask  (B, n_eeg_channels)
        kinematics      (B, n_kinematic_nodes, node_feature_dim)

    Returns
    -------
        embed      (B, d_model)  — CLS token, unified world observation
        s4_out     dict or None
        gnn_out    dict or None
        all_tokens (B, N_total, d_model)
    """

    def __init__(
        self,
        d_model:           int   = 256,
        n_heads:           int   = 8,
        n_layers:          int   = 6,
        n_eeg_channels:    int   = 64,
        s4_d_state:        int   = 64,
        s4_n_blocks:       int   = 4,
        s4_downsample:     int   = 4,
        n_kinematic_nodes: int   = 20,
        node_feature_dim:  int   = 12,
        gnn_n_layers:      int   = 3,
        cross_attn_layers: Optional[List[int]] = None,
        inject_layers:     Optional[List[int]] = None,
        patch_size:        int   = 8,
        dropout:           float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self._cross = set(cross_attn_layers or [1, 3, 5])
        self._inj   = set(inject_layers     or [2, 4])

        # Stream A
        self.tokenizer = build_tokenizer(d_model, patch_size)

        # Stream B
        self.s4 = S4EEGEncoder(
            n_channels=n_eeg_channels, d_model=d_model,
            d_state=s4_d_state, n_blocks=s4_n_blocks,
            downsample=s4_downsample, dropout=dropout,
        )

        # Stream C
        self.gnn = KinematicGNN(
            n_nodes=n_kinematic_nodes, node_feature_dim=node_feature_dim,
            d_model=d_model, n_layers=gnn_n_layers, dropout=dropout,
        )

        # Shared transformer
        self.blocks = nn.ModuleList([
            HybridBlock(
                d_model, n_heads, d_model * 4, dropout,
                cross_s4=(l in self._cross), cross_gnn=(l in self._cross),
            )
            for l in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Strategy-1 projection heads
        self.s4_proj  = nn.Linear(d_model, d_model)
        self.gnn_proj = nn.Linear(d_model, d_model)
        self.cls      = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.profiler = _PROFILER

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        inputs:    Dict[str, Optional[torch.Tensor]],
        inference: bool = False,
    ) -> Dict:
        t0 = time.perf_counter()
        B  = next(v.shape[0] for v in inputs.values() if v is not None)
        dev= next(v.device   for v in inputs.values() if v is not None)

        # Track which streams are present — absent streams are fully masked
        # so zero-padded tokens never pollute attention from present streams.
        has_eeg   = inputs.get("eeg") is not None
        has_kin   = inputs.get("kinematics") is not None
        has_vis   = any(inputs.get(k) is not None
                        for k in ("rgb","depth","rgb_right","lidar","audio"))

        # Stream B: S4 EEG
        t = time.perf_counter()
        s4_out      = None
        s4_summary  = torch.zeros(B, self.d_model, device=dev)
        s4_seq      = torch.zeros(B, 1, self.d_model, device=dev)
        if has_eeg:
            s4_out     = self.s4(inputs["eeg"], inputs.get("electrode_mask"), inference)
            s4_summary = s4_out["summary"]
            s4_seq     = s4_out["sequence"]
        self.profiler.record("s4_ms", (time.perf_counter() - t) * 1000)

        # Stream C: GNN
        t = time.perf_counter()
        gnn_out     = None
        gnn_summary = torch.zeros(B, self.d_model, device=dev)
        gnn_nodes   = torch.zeros(B, 1, self.d_model, device=dev)
        if has_kin:
            gnn_out     = self.gnn(inputs["kinematics"])
            gnn_summary = gnn_out["graph_token"].squeeze(1)
            gnn_nodes   = gnn_out["graph_sequence"]
        self.profiler.record("gnn_ms", (time.perf_counter() - t) * 1000)

        # Stream A: vision tokens
        t = time.perf_counter()
        vis_inputs = {k: v for k, v in inputs.items()
                      if k in ("rgb","depth","rgb_right","lidar","audio") and v is not None}
        vis_tokens = None   # None means absent — do NOT inject
        pad_mask   = None
        if has_vis:
            try:
                raw, pad_mask = self.tokenizer(vis_inputs)
                vis_tokens    = raw[:, 1:]          # strip tokenizer CLS
            except Exception:
                pass
        self.profiler.record("vis_ms", (time.perf_counter() - t) * 1000)

        # Assemble token sequence.
        # Only present streams contribute tokens.
        # CLS is always present; S4/GNN tokens carry zeros but ARE included
        # so the positional structure is stable — they are masked to True
        # (excluded from attention) when the stream is absent.
        token_parts = [self.cls.expand(B, 1, -1)]
        mask_parts  = [torch.zeros(B, 1, dtype=torch.bool, device=dev)]  # CLS never masked

        s4_tok = self.s4_proj(s4_summary).unsqueeze(1)   # (B,1,d) — zero if absent
        gnn_tok= self.gnn_proj(gnn_summary).unsqueeze(1) # (B,1,d) — zero if absent
        token_parts.append(s4_tok)
        token_parts.append(gnn_tok)
        # Mask absent streams so their zero tokens are excluded from attention
        mask_parts.append(torch.full((B, 1), not has_eeg, dtype=torch.bool, device=dev))
        mask_parts.append(torch.full((B, 1), not has_kin,  dtype=torch.bool, device=dev))

        if vis_tokens is not None:
            token_parts.append(vis_tokens)
            if pad_mask is not None:
                mask_parts.append(pad_mask)
            else:
                mask_parts.append(torch.zeros(B, vis_tokens.shape[1], dtype=torch.bool, device=dev))
        # else: no vision tokens appended at all — sequence is shorter

        x        = torch.cat(token_parts, dim=1)
        pad_mask = torch.cat(mask_parts, dim=1) if any(
            m.any().item() for m in mask_parts) else None

        # Transformer with fusion hooks
        t = time.perf_counter()
        for l, blk in enumerate(self.blocks):
            x = blk(
                x, pad_mask=pad_mask,
                s4_seq      = s4_seq      if l in self._cross else None,
                gnn_nodes   = gnn_nodes   if l in self._cross else None,
                s4_summary  = s4_summary  if l in self._inj   else None,
                gnn_summary = gnn_summary if l in self._inj   else None,
            )
        x = self.final_norm(x)
        self.profiler.record("transformer_ms", (time.perf_counter() - t) * 1000)
        self.profiler.record("total_ms", (time.perf_counter() - t0) * 1000)

        return {
            "embed":      x[:, 0],
            "s4_out":     s4_out,
            "gnn_out":    gnn_out,
            "all_tokens": x,
        }

    # ── utilities ─────────────────────────────────────────────────────────────

    def enable_profiling(self) -> "HybridPerceptionModel":
        self.profiler.enable(); return self

    def print_profile(self):
        s = self.profiler.summary()
        print("\n── Fusion Profiler ───────────────────────────────")
        for k, ms in sorted(s.items()):
            print(f"  {k:<22} {ms:6.2f}ms  {'█' * int(ms/2)}")
        total = s.get("total_ms", 0)
        print(f"  {'TOTAL':<22} {total:6.2f}ms  ({total/50*100:.0f}% of 50ms budget)")
        print("──────────────────────────────────────────────────\n")

    def disable_fusion_strategy(self, strategy: str):
        if strategy == "cross_attn": self._cross.clear()
        elif strategy == "inject":   self._inj.clear()
