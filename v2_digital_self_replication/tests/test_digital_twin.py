"""Tests for DigitalTwin: step, observe_outcome, adapt (online learning)."""

import numpy as np
import pytest
import torch


@pytest.fixture
def twin():
    from v2_digital_self_replication.agent.digital_twin import DigitalTwin
    t = DigitalTwin()
    t.reset_state()
    return t


def test_step_returns_command(twin):
    eeg = np.random.randn(21).astype(np.float32)
    cmd = twin.step(eeg)
    assert cmd is None or (isinstance(cmd, np.ndarray) and cmd.shape == (6,))


def test_step_stores_latent(twin):
    eeg = np.random.randn(21).astype(np.float32)
    twin.step(eeg)
    assert twin._last_latent is not None
    assert twin._last_latent.shape == (1, twin.cfg.encoder.d_model)


def _fill_memory(twin, n):
    """Run n steps and record outcomes so memory is populated with latents."""
    eeg = np.random.randn(21).astype(np.float32)
    actual_pos = np.random.randn(6).astype(np.float32)
    for _ in range(n):
        twin.step(eeg)
        twin.observe_outcome(actual_pos, eeg_window=eeg.reshape(1, -1))


def test_adapt_runs_without_error(twin):
    """adapt() must not raise — key regression test for the grad_fn bug."""
    _fill_memory(twin, twin.cfg.online.adapt_batch_size + 5)
    twin.adapt()  # must not raise RuntimeError: element 0 does not require grad


def test_adapt_on_cuda_no_device_mismatch(twin):
    """EMA params and hidden state must follow model to CUDA — no device mismatch."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    twin.cuda()
    twin.reset_state()  # re-init hidden state on cuda after move
    _fill_memory(twin, twin.cfg.online.adapt_batch_size + 5)
    twin.adapt()  # must not raise RuntimeError: device mismatch
