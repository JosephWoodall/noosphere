"""
noosphere/perception.py
=======================
Multimodal Hybrid Perception Model

Features:
- Synchronized Injection: S4 only passes its collapsed, current-timestep `embed` 
  into the Fusion Transformer, preventing temporal smearing with current visual frames.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from noosphere.s4_eeg import S4EEGEncoder
from noosphere.gnn import KinematicGNN

class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int, d_model: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return self.norm(x)

class HybridPerceptionModel(nn.Module):
    def __init__(
        self, d_model: int = 256, n_heads: int = 8, n_layers: int = 6, n_eeg_channels: int = 3,
        s4_d_state: int = 64, s4_n_blocks: int = 4, s4_downsample: int = 4, n_kinematic_nodes: int = 20,
        node_feature_dim: int = 12, gnn_n_layers: int = 3, patch_size: int = 8, max_reach: float = 0.70
    ):
        super().__init__()
        self.d_model = d_model
        self.max_reach = max_reach

        self.s4 = S4EEGEncoder(
            in_channels=n_eeg_channels, d_model=d_model, d_state=s4_d_state,
            n_blocks=s4_n_blocks, downsample=s4_downsample
        )
        
        self.gnn = KinematicGNN(
            n_nodes=n_kinematic_nodes, node_feature_dim=node_feature_dim,
            d_model=d_model, n_layers=gnn_n_layers, n_heads=n_heads
        )
        
        self.rgb_enc = VisionEncoder(in_channels=3, d_model=d_model, patch_size=patch_size)
        self.depth_enc = VisionEncoder(in_channels=1, d_model=d_model, patch_size=patch_size)
        
        self.structured_proj = nn.Sequential(nn.Linear(64, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

        self.modality_embed = nn.Parameter(torch.randn(5, d_model) * 0.02)
        
        fusion_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=0.1, activation='gelu', batch_first=True)
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=n_layers)
        
        self.out_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        
        self.xyz_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), 
            nn.Linear(d_model, 3), nn.Tanh()
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self.device = next(self.parameters()).device
        B = list(inputs.values())[0].shape[0]
        
        tokens = []; s4_out = None; gnn_out = None

        if "eeg" in inputs:
            s4_out = self.s4(inputs["eeg"], inputs.get("electrode_mask"))
            if "_mock_intent" in inputs:
                mock_probs = inputs["_mock_intent"]
                if mock_probs.ndim == 1: mock_probs = mock_probs.unsqueeze(0).expand(B, -1)
                s4_out["intent_probs"] = mock_probs.to(self.device)
                s4_out["confidence"] = torch.ones_like(s4_out["confidence"])
            
            # TEMPORAL SMEARING FIX: Inject only the current 1D cognitive state `embed`, not the `sequence`
            eeg_token = s4_out["embed"].unsqueeze(1) + self.modality_embed[0]
            tokens.append(eeg_token)

        if "kinematics" in inputs:
            gnn_out = self.gnn(inputs["kinematics"])
            tokens.append(gnn_out["graph_sequence"] + self.modality_embed[1])

        if "rgb" in inputs: tokens.append(self.rgb_enc(inputs["rgb"]) + self.modality_embed[2])
        if "depth" in inputs: tokens.append(self.depth_enc(inputs["depth"]) + self.modality_embed[3])
        if "structured" in inputs: tokens.append(self.structured_proj(inputs["structured"]).unsqueeze(1) + self.modality_embed[4])

        if not tokens:
            empty = torch.zeros(B, 1, self.d_model, device=self.device)
            return {"embed": empty.squeeze(1), "s4_out": None, "gnn_out": None, "xyz_pred": torch.zeros(B, 3, device=self.device)}

        concat_tokens = torch.cat(tokens, dim=1)
        fused_seq = self.fusion(concat_tokens)
        fused_embed = self.out_proj(fused_seq.mean(dim=1))
        
        xyz_pred = self.xyz_head(fused_embed) * self.max_reach

        return {"embed": fused_embed, "s4_out": s4_out, "gnn_out": gnn_out, "xyz_pred": xyz_pred}