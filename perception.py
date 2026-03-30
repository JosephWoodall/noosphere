"""
noosphere/perception.py
=======================
Multimodal Hybrid Perception Model

Fuses high-bandwidth spatial context (Vision, Kinematics) with 
low-latency biological intent (EEG).

Features:
- S4 state-space model for raw EEG sequence processing.
- KinematicGNN for flexible, plug-and-play body topology.
- Transformer-based late fusion to create a unified world model embedding.
- Native injection point for testing mock macro-intents.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from noosphere.s4_eeg import S4EEGEncoder
from noosphere.gnn import KinematicGNN

class VisionEncoder(nn.Module):
    """Lightweight CNN to process RGB/Depth into a sequence of patch embeddings."""
    def __init__(self, in_channels: int, d_model: int, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, d_model, H/P, W/P)
        x = self.proj(x)
        # Flatten spatial dimensions: (B, d_model, N_patches)
        x = x.flatten(2)
        # Transpose to sequence: (B, N_patches, d_model)
        x = x.transpose(1, 2)
        return self.norm(x)

class HybridPerceptionModel(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        n_eeg_channels: int = 3,
        s4_d_state: int = 64,
        s4_n_blocks: int = 4,
        s4_downsample: int = 4,
        n_kinematic_nodes: int = 20,
        node_feature_dim: int = 12,
        gnn_n_layers: int = 3,
        patch_size: int = 8,
    ):
        super().__init__()
        self.device = None # Set dynamically during forward pass
        self.d_model = d_model

        # ── Modality Encoders ─────────────────────────────────────────────────
        
        self.s4 = S4EEGEncoder(
            in_channels=n_eeg_channels,
            d_model=d_model,
            d_state=s4_d_state,
            n_blocks=s4_n_blocks,
            downsample=s4_downsample
        )
        
        self.gnn = KinematicGNN(
            n_nodes=n_kinematic_nodes,
            node_feature_dim=node_feature_dim,
            d_model=d_model,
            n_layers=gnn_n_layers,
            n_heads=n_heads
        )
        
        self.rgb_enc = VisionEncoder(in_channels=3, d_model=d_model, patch_size=patch_size)
        self.depth_enc = VisionEncoder(in_channels=1, d_model=d_model, patch_size=patch_size)
        
        self.structured_proj = nn.Sequential(
            nn.Linear(64, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        # ── Fusion Transformer ────────────────────────────────────────────────
        
        # Modality type embeddings to help the transformer distinguish sources
        self.modality_embed = nn.Parameter(torch.randn(5, d_model) * 0.02)
        
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.fusion = nn.TransformerEncoder(fusion_layer, num_layers=n_layers)
        
        # Final projection to fixed size for the RSSM
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

        self._profiling = False

    def enable_profiling(self):
        self._profiling = True

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        self.device = next(self.parameters()).device
        B = list(inputs.values())[0].shape[0]
        
        tokens = []
        s4_out = None
        gnn_out = None

        # 1. EEG & Biological Intent
        if "eeg" in inputs:
            s4_out = self.s4(inputs["eeg"], inputs.get("electrode_mask"))
            
            # ── INJECTION POINT: Mock Intent Override ─────────────────────────
            if "_mock_intent" in inputs:
                mock_probs = inputs["_mock_intent"]
                if mock_probs.ndim == 1:
                    mock_probs = mock_probs.unsqueeze(0).expand(B, -1)
                
                s4_out["intent_probs"] = mock_probs.to(self.device)
                s4_out["confidence"] = torch.ones_like(s4_out["confidence"])
            # ──────────────────────────────────────────────────────────────────
            
            eeg_seq = s4_out["sequence"] + self.modality_embed[0]
            tokens.append(eeg_seq)

        # 2. Kinematics (GNN)
        if "kinematics" in inputs:
            gnn_out = self.gnn(inputs["kinematics"])
            kin_seq = gnn_out["graph_sequence"] + self.modality_embed[1]
            tokens.append(kin_seq)

        # 3. Vision (RGB & Depth)
        if "rgb" in inputs:
            rgb_seq = self.rgb_enc(inputs["rgb"]) + self.modality_embed[2]
            tokens.append(rgb_seq)
            
        if "depth" in inputs:
            depth_seq = self.depth_enc(inputs["depth"]) + self.modality_embed[3]
            tokens.append(depth_seq)

        # 4. Digital State (OS / Shell Context)
        if "structured" in inputs:
            # Add a sequence dimension for the transformer: (B, 1, d_model)
            struct_seq = self.structured_proj(inputs["structured"]).unsqueeze(1) + self.modality_embed[4]
            tokens.append(struct_seq)

        # 5. Fallback if no observations
        if not tokens:
            empty = torch.zeros(B, 1, self.d_model, device=self.device)
            return {"embed": empty.squeeze(1), "s4_out": None, "gnn_out": None}

        # 6. Late Fusion
        concat_tokens = torch.cat(tokens, dim=1)
        fused_seq = self.fusion(concat_tokens)
        
        # Global average pooling over the sequence length
        fused_embed = fused_seq.mean(dim=1)
        fused_embed = self.out_proj(fused_embed)

        return {
            "embed": fused_embed,
            "s4_out": s4_out,
            "gnn_out": gnn_out
        }