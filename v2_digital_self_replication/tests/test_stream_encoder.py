"""Tests for StreamEncoder: shapes, gradient stability, decode_step == forward."""

import pytest
import torch
import numpy as np

from v2_digital_self_replication.core.stream_encoder import (
    BiosignalSSMCell, BiosignalSSMBlock, MultiModalFusion, StreamEncoder
)


@pytest.fixture
def encoder():
    return StreamEncoder(d_model=32, d_state=16, n_layers=2, n_eeg=21, n_prop=6)


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_ssm_cell_forward_shape():
    cell = BiosignalSSMCell(d_model=16, d_state=8)
    x = torch.randn(2, 10, 16)
    y, h = cell(x)
    assert y.shape == (2, 10, 16)
    assert h.shape == (2, 16, 8)


def test_ssm_block_forward_shape():
    block = BiosignalSSMBlock(d_model=16, d_state=8)
    x = torch.randn(3, 20, 16)
    out, h = block(x)
    assert out.shape == x.shape
    assert h.shape == (3, 16, 8)


def test_fusion_forward_shape():
    fusion = MultiModalFusion(n_eeg=21, n_prop=6, d_model=32)
    eeg  = torch.randn(2, 10, 21)
    hrv  = torch.randn(2, 10, 1)
    gsr  = torch.randn(2, 10, 1)
    prop = torch.randn(2, 10, 6)
    out = fusion(eeg, hrv, gsr, prop)
    assert out.shape == (2, 10, 32)


def test_encoder_forward_shape(encoder):
    eeg = torch.randn(4, 64, 21)
    out, hs = encoder(eeg)
    assert out.shape == (4, 64, 32)
    assert len(hs) == 2


def test_encoder_eeg_only(encoder):
    """Encoder should work with only EEG (HRV/GSR/prop are optional)."""
    eeg = torch.randn(2, 32, 21)
    out, _ = encoder(eeg)
    assert out.shape == (2, 32, 32)


# ── decode_step consistency ───────────────────────────────────────────────────

def test_decode_step_matches_forward():
    """Single-step decode must produce same output as sequential full-forward on last step."""
    enc = StreamEncoder(d_model=16, d_state=8, n_layers=2, n_eeg=21)
    enc.eval()
    T = 5
    eeg = torch.randn(1, T, 21)

    # Full forward
    with torch.no_grad():
        out_full, _ = enc(eeg)
        last_full = out_full[:, -1, :]

    # Step-by-step via decode_step
    hidden = enc.zero_hidden(batch_size=1)
    last_step = None
    with torch.no_grad():
        for t in range(T):
            eeg_t = eeg[:, t, :]
            h_out, hidden = enc.decode_step(eeg_t, hidden_states=hidden)
            last_step = h_out

    assert torch.allclose(last_full, last_step, atol=1e-5), (
        f"decode_step diverges from forward. max diff = {(last_full - last_step).abs().max():.6f}"
    )


# ── Gradient stability ────────────────────────────────────────────────────────

def test_no_nan_gradients():
    """ZOH discretization must not produce NaN gradients on short sequences."""
    enc = StreamEncoder(d_model=16, d_state=8, n_layers=2, n_eeg=21, n_prop=6)
    eeg  = torch.randn(2, 32, 21)
    hrv  = torch.randn(2, 32, 1)
    gsr  = torch.randn(2, 32, 1)
    prop = torch.randn(2, 32, 6)
    out, _ = enc(eeg, hrv, gsr, prop)  # all modalities active → all projections used
    loss = out.mean()
    loss.backward()
    for name, p in enc.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


def test_stable_long_sequence():
    """Encoder should remain numerically stable on 1024-step sequences."""
    enc = StreamEncoder(d_model=32, d_state=16, n_layers=2, n_eeg=21)
    eeg = torch.randn(1, 1024, 21)
    with torch.no_grad():
        out, _ = enc(eeg)
    assert not torch.isnan(out).any(), "NaN in long-sequence output"
    assert not torch.isinf(out).any(), "Inf in long-sequence output"
