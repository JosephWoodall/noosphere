"""
CNS (central nervous system) spike train encoder for cross-modal JEPA.

Architecture mirrors StreamEncoder but takes (B, T, n_neurons) spike rate input
instead of multi-modal EEG. Uses the same ZOH-SSM blocks for architectural
consistency so that latent covariance structures are compatible.

Purpose: pretrained on NLB MC_Maze macaque M1 data in a self-supervised JEPA loop,
then frozen and used as a cross-modal teacher for the EEG student encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, List

from v2_digital_self_replication.core.stream_encoder import BiosignalSSMBlock


class CNSEncoder(nn.Module):
    """
    ZOH-SSM encoder for spike train data.

    Input:  (B, T, n_neurons) — binned firing rates at 250/256 Hz
    Output: (B, T, d_model) encoded sequence + hidden states

    Identical block stack to StreamEncoder; only the input projection differs.
    """

    def __init__(
        self,
        n_neurons: int = 182,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers

        # Project spike rates to model dimension; LayerNorm stabilises the
        # wide dynamic range of firing rates (2–200 Hz) before the SSM stack
        self.input_proj = nn.Sequential(
            nn.Linear(n_neurons, d_model, bias=False),
            nn.LayerNorm(d_model),
        )
        self.blocks = nn.ModuleList([
            BiosignalSSMBlock(d_model, d_state, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, spikes: torch.Tensor):
        """
        spikes: (B, T, n_neurons) float32
        Returns: (B, T, d_model), hidden_states list
        """
        x = self.input_proj(spikes)           # (B, T, d_model)
        hidden_states: List[torch.Tensor] = []
        for block in self.blocks:
            x, h_final = block(x)
            hidden_states.append(h_final)
        return self.norm_out(x), hidden_states

    def encode_pooled(self, spikes: torch.Tensor) -> torch.Tensor:
        """Returns mean-pooled latent (B, d_model)."""
        out, _ = self.forward(spikes)
        return out.mean(dim=1)

    def zero_hidden(
        self,
        batch_size: int = 1,
        device: str = "cpu",
        dtype=torch.float32,
    ) -> List[torch.Tensor]:
        return [
            torch.zeros(batch_size, self.d_model, self.d_state, device=device, dtype=dtype)
            for _ in range(self.n_layers)
        ]


def load_frozen_cns_encoder(checkpoint_path: str, device: str = "cpu") -> CNSEncoder:
    """
    Load a pretrained CNS encoder from checkpoint and freeze all parameters.
    Used as the teacher in cross-modal JEPA training.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Support checkpoints saved by CNSJEPATrainer (has 'cns_encoder' key)
    # or generic state_dict saved directly
    if "cns_encoder" in ckpt:
        state_dict = ckpt["cns_encoder"]
    elif "encoder" in ckpt:
        state_dict = ckpt["encoder"]
    else:
        state_dict = ckpt

    # Infer n_neurons from the input_proj weight shape
    in_proj_weight = state_dict.get("input_proj.0.weight")
    if in_proj_weight is not None:
        n_neurons = in_proj_weight.shape[1]
        d_model = in_proj_weight.shape[0]
    else:
        n_neurons, d_model = 182, 128

    encoder = CNSEncoder(n_neurons=n_neurons, d_model=d_model)
    encoder.load_state_dict(state_dict)

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    import logging
    logging.getLogger(__name__).info(
        "CNS encoder loaded from %s (n_neurons=%d, d_model=%d) — frozen",
        checkpoint_path, n_neurons, d_model,
    )
    return encoder.to(device)
