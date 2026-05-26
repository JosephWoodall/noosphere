"""Tests for synthetic data generators: shape, amplitude, ERD/ERS patterns."""

import numpy as np
import pytest

from v2_digital_self_replication.data.synthetic_eeg import (
    EEGStreamGenerator, smooth_motor_trajectory, make_training_batch, N_CH,
    IDX_C3, IDX_C4, IDX_FCZ
)
from v2_digital_self_replication.data.synthetic_physio import generate_hrv_stream, generate_gsr_stream
from v2_digital_self_replication.data.stream_buffer import StreamBuffer, MultiModalBuffer


# ── EEG generator ─────────────────────────────────────────────────────────────

def test_eeg_step_output_shape():
    gen = EEGStreamGenerator(seed=0, subject_id=1)
    sample = gen.step(np.zeros(6, dtype=np.float32))
    assert sample.shape == (N_CH,), f"Expected ({N_CH},), got {sample.shape}"
    assert sample.dtype == np.float32


def test_eeg_stream_shape():
    gen = EEGStreamGenerator(seed=0, subject_id=1)
    T = 128
    cmds = np.zeros((T, 6), dtype=np.float32)
    eeg = gen.stream(cmds)
    assert eeg.shape == (T, N_CH)


def test_eeg_amplitude_reasonable():
    """EEG should be in ±500 µV range (typical for synthetic)."""
    gen = EEGStreamGenerator(seed=42, subject_id=1)
    T = 256
    cmds = np.zeros((T, 6), dtype=np.float32)
    eeg = gen.stream(cmds)
    max_amp = np.abs(eeg).max()
    assert max_amp < 1000.0, f"EEG amplitude too large: {max_amp:.1f} µV"
    assert max_amp > 1.0, f"EEG amplitude suspiciously small: {max_amp:.3f} µV"


def test_erd_pattern():
    """
    Motor imagery (right arm) should produce different EEG at C3 vs rest.
    Specifically, C3 variance should differ between rest and active conditions.
    """
    T = 512
    rest_cmd  = np.zeros((T, 6), dtype=np.float32)
    active_cmd = np.zeros((T, 6), dtype=np.float32)
    active_cmd[:, 1] = 0.8  # shoulder_pitch active

    gen_rest   = EEGStreamGenerator(seed=100, subject_id=2)
    gen_active = EEGStreamGenerator(seed=100, subject_id=2)

    eeg_rest   = gen_rest.stream(rest_cmd)
    eeg_active = gen_active.stream(active_cmd)

    var_rest   = float(np.var(eeg_rest[:, IDX_C3]))
    var_active = float(np.var(eeg_active[:, IDX_C3]))
    # ERD reduces alpha/beta variance at C3 — active condition SHOULD differ
    # We just verify that the two conditions produce different signals
    assert abs(var_rest - var_active) > 0, "Rest and active EEG should differ at C3"


def test_ern_trigger():
    """ERN should activate at FCz when intent reverses rapidly."""
    gen = EEGStreamGenerator(seed=0, subject_id=1)
    # Prime with positive intent
    for _ in range(20):
        gen.step(np.array([0.8, 0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    # Sharp reversal → ERN
    gen.step(np.array([-0.8, -0.8, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert gen.ern_active(), "ERN should activate after intent reversal"


def test_smooth_trajectory_shape():
    cmd = smooth_motor_trajectory(n_dof=6, T=256, n_segments=4)
    assert cmd.shape == (256, 6)
    assert np.all(cmd >= -1.0) and np.all(cmd <= 1.0)


def test_make_training_batch():
    batch = make_training_batch(n_subjects=2, n_trials=3, trial_duration_s=1.0, fs=64, seed=0)
    assert len(batch) == 2
    for sub_id, data in batch.items():
        assert "eeg" in data
        assert "commands" in data
        assert data["eeg"].shape[2] == N_CH
        assert data["commands"].shape[2] == 6


# ── HRV generator ─────────────────────────────────────────────────────────────

def test_hrv_shape():
    hrv = generate_hrv_stream(duration_s=2.0, fs=64, seed=0)
    assert hrv.shape == (128, 1)


def test_hrv_realistic_range():
    hrv = generate_hrv_stream(duration_s=4.0, fs=256, hr_mean=70.0, seed=0)
    assert hrv.min() > 30.0, "HR should not drop below 30 BPM"
    assert hrv.max() < 220.0, "HR should not exceed 220 BPM"


# ── GSR generator ─────────────────────────────────────────────────────────────

def test_gsr_shape():
    gsr = generate_gsr_stream(duration_s=2.0, fs=64, seed=0)
    assert gsr.shape == (128, 1)


def test_gsr_positive():
    gsr = generate_gsr_stream(duration_s=4.0, fs=256, seed=42)
    assert np.all(gsr > 0), "GSR must be positive (skin conductance)"


# ── Stream buffer ─────────────────────────────────────────────────────────────

def test_stream_buffer_basic():
    buf = StreamBuffer(n_channels=21, capacity=100)
    for _ in range(50):
        buf.append(np.ones(21, dtype=np.float32))
    w = buf.get_window(20)
    assert w.shape == (20, 21)
    assert np.allclose(w, 1.0)


def test_stream_buffer_zero_pads_when_underfull():
    buf = StreamBuffer(n_channels=21, capacity=100)
    buf.append(np.ones(21, dtype=np.float32))
    w = buf.get_window(10)
    assert w.shape == (10, 21)
    assert np.allclose(w[0], 0.0)   # front is zero-padded
    assert np.allclose(w[-1], 1.0)  # last sample is the one we appended


def test_stream_buffer_wraps_correctly():
    """Ring buffer must return correct order after wrap-around."""
    buf = StreamBuffer(n_channels=1, capacity=10)
    for i in range(15):
        buf.append(np.array([float(i)], dtype=np.float32))
    w = buf.get_window(5)
    expected = np.array([[10.0], [11.0], [12.0], [13.0], [14.0]])
    assert np.allclose(w, expected), f"Expected {expected.T}, got {w.T}"
