"""
Streaming biosignal encoder: MultiModalFusion → ZOH SSM stack.

Design principles:
  - ZOH discretization (bar_A = exp(dt * A)): provably stable for A < 0
  - Bilinear binding term: h = bar_A*h + bar_B*x + bar_W*tanh(h*x)
    captures nonlinear cross-temporal correlation without breaking causal recurrence
  - GLU (gated linear unit) output per block following Mamba/TSSM pattern
  - decode_step() provides O(1) single-step streaming inference via cached hidden states
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _bilinear_scan(
    x: torch.Tensor,
    bar_A: torch.Tensor,
    bar_B: torch.Tensor,
    bar_W: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Bilinear SSM scan over sequence dimension T.

    x:     (B, T, d_model)
    bar_A: (d_model, d_state)  — ZOH transition
    bar_B: (d_model, d_state)
    bar_W: (d_model, d_state)  — bilinear binding weight
    C:     (d_model, d_state)  — output projection
    D:     (d_model,)          — skip connection

    Returns: outputs (B, T, d_model), h_final (B, d_model, d_state)
    """
    B = x.shape[0]
    T = x.shape[1]
    d_model = x.shape[2]
    d_state = bar_A.shape[1]

    h = torch.zeros(B, d_model, d_state, device=x.device, dtype=x.dtype)
    out = torch.zeros(B, T, d_model, device=x.device, dtype=x.dtype)

    for t in range(T):
        x_t = x[:, t, :].unsqueeze(-1)              # (B, d_model, 1)
        h = bar_A * h + bar_B * x_t + bar_W * torch.tanh(h * x_t)
        out[:, t, :] = (C * h).sum(-1) + D * x[:, t, :]

    return out, h


_bilinear_scan = torch.compile(_bilinear_scan, dynamic=True)


class BiosignalSSMCell(nn.Module):
    """ZOH SSM cell with bilinear state binding for streaming biosignals."""

    def __init__(self, d_model: int, d_state: int = 64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A must stay negative for ZOH stability: exp(dt*A) < 1 iff A < 0
        self.A = nn.Parameter(torch.randn(d_model, d_state) - 5.0)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))
        # Small init prevents bilinear term from dominating early in training
        self.W = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.dt_proj = nn.Parameter(torch.randn(d_model) * 0.01)

    def _discretize(self):
        dt = F.softplus(self.dt_proj).unsqueeze(-1)  # (d_model, 1), always > 0
        bar_A = torch.exp(dt * self.A)               # (d_model, d_state), ZOH
        bar_B = dt * self.B                          # (d_model, d_state)
        bar_W = dt * self.W                          # (d_model, d_state)
        return bar_A, bar_B, bar_W

    def forward(self, x: torch.Tensor):
        """Full sequence. x: (B, T, d_model) → (B, T, d_model), h_final (B, d_model, d_state)."""
        bar_A, bar_B, bar_W = self._discretize()
        return _bilinear_scan(x, bar_A, bar_B, bar_W, self.C, self.D)

    def decode_step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        """Single step. x_t: (B, d_model), h_prev: (B, d_model, d_state)."""
        bar_A, bar_B, bar_W = self._discretize()
        x_exp = x_t.unsqueeze(-1)  # (B, d_model, 1)
        h = bar_A * h_prev + bar_B * x_exp + bar_W * torch.tanh(h_prev * x_exp)
        y = (self.C * h).sum(-1) + self.D * x_t
        return y, h


class BiosignalSSMBlock(nn.Module):
    """SSM cell wrapped in GLU projection + residual, matching the TSSM block pattern."""

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.ssm = BiosignalSSMCell(d_model, d_state)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_model) → (B, T, d_model), h_final (B, d_model, d_state)."""
        residual = x
        x_proj = self.in_proj(self.norm(x))           # (B, T, 2*d_model)
        x_ssm, gate = x_proj.chunk(2, dim=-1)
        y_ssm, h_final = self.ssm(x_ssm)
        y = y_ssm * F.silu(gate)                       # GLU
        out = self.dropout(self.out_proj(y))
        return residual + out, h_final

    def decode_step(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        """x_t: (B, d_model), h_prev: (B, d_model, d_state)."""
        residual = x_t
        x_proj = self.in_proj(self.norm(x_t))         # (B, 2*d_model)
        x_ssm, gate = x_proj.chunk(2, dim=-1)
        y_ssm, h_new = self.ssm.decode_step(x_ssm, h_prev)
        y = y_ssm * F.silu(gate)
        out = self.dropout(self.out_proj(y))
        return residual + out, h_new


class MultiModalFusion(nn.Module):
    """Projects each biosignal modality into a shared d_model space and sums them."""

    def __init__(self, n_eeg: int = 21, n_prop: int = 6, d_model: int = 128):
        super().__init__()
        self.eeg_proj = nn.Linear(n_eeg, d_model, bias=False)
        self.hrv_proj = nn.Linear(1, d_model, bias=False)
        self.gsr_proj = nn.Linear(1, d_model, bias=False)
        self.prop_proj = nn.Linear(n_prop, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        eeg: torch.Tensor,
        hrv: Optional[torch.Tensor] = None,
        gsr: Optional[torch.Tensor] = None,
        prop: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Accepts (B, T, C) or (B, C) inputs for each modality."""
        out = self.eeg_proj(eeg)
        if hrv is not None:
            out = out + self.hrv_proj(hrv)
        if gsr is not None:
            out = out + self.gsr_proj(gsr)
        if prop is not None:
            out = out + self.prop_proj(prop)
        return self.norm(out)


class StreamEncoder(nn.Module):
    """
    Full streaming encoder: MultiModalFusion → N × BiosignalSSMBlock.

    Training: forward() processes a full (B, T, C) window.
    Inference: decode_step() processes one sample at a time using cached hidden states.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        n_eeg: int = 21,
        n_prop: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.fusion = MultiModalFusion(n_eeg=n_eeg, n_prop=n_prop, d_model=d_model)
        self.blocks = nn.ModuleList([
            BiosignalSSMBlock(d_model, d_state, dropout) for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        eeg: torch.Tensor,
        hrv: Optional[torch.Tensor] = None,
        gsr: Optional[torch.Tensor] = None,
        prop: Optional[torch.Tensor] = None,
    ):
        """Returns (B, T, d_model) encoded sequence and list of final hidden states per layer."""
        x = self.fusion(eeg, hrv, gsr, prop)
        hidden_states = []
        for block in self.blocks:
            x, h_final = block(x)
            hidden_states.append(h_final)
        return self.norm_out(x), hidden_states

    def decode_step(
        self,
        eeg_t: torch.Tensor,
        hrv_t: Optional[torch.Tensor] = None,
        gsr_t: Optional[torch.Tensor] = None,
        prop_t: Optional[torch.Tensor] = None,
        hidden_states: Optional[list] = None,
    ):
        """
        Process one timestep. Inputs: (B, C) each.
        hidden_states: list of (B, d_model, d_state) per layer, or None to zero-init.
        Returns: (B, d_model) output, updated hidden_states list.
        """
        if hidden_states is None:
            hidden_states = [None] * self.n_layers

        x_t = self.fusion(eeg_t, hrv_t, gsr_t, prop_t)  # (B, d_model)
        new_hidden = []
        for i, block in enumerate(self.blocks):
            h_prev = hidden_states[i]
            if h_prev is None:
                h_prev = torch.zeros(
                    x_t.shape[0], self.d_model, self.d_state,
                    device=x_t.device, dtype=x_t.dtype,
                )
            x_t, h_new = block.decode_step(x_t, h_prev)
            new_hidden.append(h_new)

        return self.norm_out(x_t), new_hidden

    def zero_hidden(self, batch_size: int = 1, device: str = "cpu", dtype=torch.float32):
        """Return fresh zero hidden states for the start of a new streaming session."""
        return [
            torch.zeros(batch_size, self.d_model, self.d_state, device=device, dtype=dtype)
            for _ in range(self.n_layers)
        ]
