"""Tests for IntentDecoder: output bounds, uncertainty positivity, ERN probability range."""

import torch
import numpy as np

from v2_digital_self_replication.core.intent_decoder import IntentDecoder, IntentLoss


def test_mu_bounded():
    dec = IntentDecoder(d_model=32, n_dof=6)
    h = torch.randn(4, 32)
    intent = dec(h)
    assert (intent.mu >= -1.0).all() and (intent.mu <= 1.0).all(), "mu must be in [-1, 1]"


def test_sigma_positive():
    dec = IntentDecoder(d_model=32, n_dof=6)
    h = torch.randn(8, 32)
    intent = dec(h)
    assert (intent.sigma > 0).all(), "sigma must be strictly positive"


def test_ern_prob_in_unit_interval():
    dec = IntentDecoder(d_model=32, n_dof=6)
    h = torch.randn(4, 32)
    intent = dec(h)
    assert (intent.ern_prob >= 0).all() and (intent.ern_prob <= 1).all(), "ern_prob must be in [0, 1]"


def test_seq_input_uses_last_timestep():
    """Decoder with (B, T, d) input should give same result as (B, d) from last step."""
    dec = IntentDecoder(d_model=32, n_dof=6)
    dec.eval()
    h_seq = torch.randn(2, 10, 32)
    h_last = h_seq[:, -1, :]
    with torch.no_grad():
        intent_seq  = dec(h_seq)
        intent_last = dec(h_last)
    assert torch.allclose(intent_seq.mu, intent_last.mu, atol=1e-6)


def test_loss_backward():
    dec = IntentDecoder(d_model=32, n_dof=6)
    loss_fn = IntentLoss()
    h = torch.randn(4, 32)
    intent = dec(h)
    target = torch.randn(4, 6)
    ern_lbl = torch.zeros(4)
    loss = loss_fn(intent, target, ern_label=ern_lbl)
    loss.backward()
    for name, p in dec.named_parameters():
        assert p.grad is not None, f"No grad for {name}"
        assert not torch.isnan(p.grad).any(), f"NaN grad in {name}"


def test_loss_without_ern_label():
    dec = IntentDecoder(d_model=32, n_dof=6)
    loss_fn = IntentLoss()
    h = torch.randn(4, 32)
    intent = dec(h)
    target = torch.randn(4, 6)
    loss = loss_fn(intent, target)  # no ern_label — should not crash
    assert loss.item() > 0
