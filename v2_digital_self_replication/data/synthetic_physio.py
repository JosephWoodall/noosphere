"""
Synthetic physiological signal generators: HRV and GSR.

HRV (Heart Rate Variability):
  Uses the Integral Pulse Frequency Modulation (IPFM) model — the standard
  physiologically validated HRV synthesis approach.
  Outputs: instantaneous heart rate signal at EEG sampling rate.

GSR (Galvanic Skin Response / Electrodermal Activity):
  Tonic component: slow drift correlated with arousal.
  Phasic component: Skin Conductance Response (SCR) peaks triggered by stimuli.
  Model: Convolution of sparse event train with double-exponential SCR kernel.

Both signals are correlated with the motor intent profile to simulate
realistic co-variation during prosthetic use.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ── HRV via IPFM ─────────────────────────────────────────────────────────────

def _ipfm_rr_series(
    n_beats: int,
    hr_mean: float = 70.0,
    lf_power: float = 0.08,
    hf_power: float = 0.06,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate RR interval series using IPFM model.
    Returns (n_beats,) RR intervals in seconds.
    """
    if rng is None:
        rng = np.random.default_rng()

    mean_rr = 60.0 / hr_mean
    rr = np.full(n_beats, mean_rr)

    # LF modulation (0.04–0.15 Hz, Mayer waves, sympathetic + parasympathetic)
    lf_freq = rng.uniform(0.07, 0.12)
    lf_phase = rng.uniform(0, 2 * math.pi)
    lf_amp = math.sqrt(2 * lf_power) * mean_rr

    # HF modulation (0.15–0.4 Hz, respiratory sinus arrhythmia, parasympathetic)
    hf_freq = rng.uniform(0.18, 0.35)
    hf_phase = rng.uniform(0, 2 * math.pi)
    hf_amp = math.sqrt(2 * hf_power) * mean_rr

    t = np.cumsum(rr)
    rr += lf_amp * np.sin(2 * math.pi * lf_freq * t + lf_phase)
    rr += hf_amp * np.sin(2 * math.pi * hf_freq * t + hf_phase)
    rr += rng.standard_normal(n_beats) * 0.003  # measurement noise

    return np.clip(rr, 0.4, 1.5).astype(np.float32)


def generate_hrv_stream(
    duration_s: float,
    fs: int = 256,
    hr_mean: float = 70.0,
    arousal_profile: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate an instantaneous heart rate signal sampled at `fs`.

    arousal_profile: (T,) float in [0, 1] modulates HR upward (sympathetic activation).
    Returns: (T, 1) float — instantaneous HR in beats-per-minute.
    """
    rng = np.random.default_rng(seed)
    T = int(duration_s * fs)
    n_beats = int(duration_s * hr_mean / 60 * 2) + 10  # oversample

    rr = _ipfm_rr_series(n_beats, hr_mean=hr_mean, rng=rng)
    beat_times = np.cumsum(rr)
    # Instantaneous HR: 60 / RR_at_t
    t_axis = np.arange(T) / fs
    rr_interp = np.interp(t_axis, beat_times[:-1], rr[:-1])
    ihr = 60.0 / (rr_interp + 1e-6)  # instantaneous HR in BPM

    if arousal_profile is not None:
        stress = np.interp(np.arange(T), np.linspace(0, T - 1, len(arousal_profile)), arousal_profile)
        ihr += stress * 15.0  # up to +15 BPM under high arousal

    return ihr.reshape(-1, 1).astype(np.float32)


# ── GSR via convolution model ─────────────────────────────────────────────────

def _scr_kernel(fs: int, rise_s: float = 1.0, decay_s: float = 4.0) -> np.ndarray:
    """Double-exponential SCR (Skin Conductance Response) kernel."""
    T_kernel = int((rise_s + decay_s * 4) * fs)
    t = np.arange(T_kernel) / fs
    kernel = np.exp(-t / decay_s) - np.exp(-t / rise_s)
    kernel = np.maximum(kernel, 0)
    return (kernel / (kernel.max() + 1e-8)).astype(np.float32)


def generate_gsr_stream(
    duration_s: float,
    fs: int = 256,
    tonic_level: float = 5.0,
    event_rate: float = 0.3,
    arousal_profile: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a GSR (electrodermal activity) signal.

    tonic_level: baseline skin conductance in µS.
    event_rate: spontaneous SCR events per second.
    arousal_profile: (T,) float in [0, 1] increases event rate.
    Returns: (T, 1) float — skin conductance in µS.
    """
    rng = np.random.default_rng(seed)
    T = int(duration_s * fs)
    t_axis = np.arange(T) / fs

    # Tonic: slow drift (0.01–0.1 Hz)
    drift_freq = rng.uniform(0.01, 0.05)
    tonic = tonic_level + np.sin(2 * math.pi * drift_freq * t_axis) * 0.5

    # Phasic: sparse event train convolved with SCR kernel
    kernel = _scr_kernel(fs)
    if arousal_profile is not None:
        ap = np.interp(np.arange(T), np.linspace(0, T - 1, len(arousal_profile)), arousal_profile)
        local_rate = event_rate + ap * 0.5
    else:
        local_rate = np.full(T, event_rate)

    event_train = (rng.random(T) < local_rate / fs).astype(np.float32)
    event_train *= rng.uniform(0.5, 2.0, T)  # variable amplitude

    phasic = np.convolve(event_train, kernel, mode="full")[:T]

    gsr = tonic + phasic
    gsr += rng.standard_normal(T) * 0.05  # sensor noise
    gsr = np.maximum(gsr, 0.01)

    return gsr.reshape(-1, 1).astype(np.float32)


# ── Convenience batch generator ───────────────────────────────────────────────

def make_physio_batch(
    n_trials: int = 20,
    duration_s: float = 4.0,
    fs: int = 256,
    motor_activity: Optional[np.ndarray] = None,
    seed: int = 0,
) -> dict:
    """
    Generate paired HRV + GSR batches correlated with motor activity.

    motor_activity: (n_trials, T) float in [0, 1] — total motor activity per timestep.
    Returns: {
      "hrv": (n_trials, T, 1) float32 — instantaneous HR in BPM,
      "gsr": (n_trials, T, 1) float32 — skin conductance in µS,
    }
    """
    T = int(duration_s * fs)
    hrv_all = np.empty((n_trials, T, 1), dtype=np.float32)
    gsr_all = np.empty((n_trials, T, 1), dtype=np.float32)

    for i in range(n_trials):
        arousal = motor_activity[i] if motor_activity is not None else None
        hrv_all[i] = generate_hrv_stream(duration_s, fs, seed=seed + i, arousal_profile=arousal)
        gsr_all[i] = generate_gsr_stream(duration_s, fs, seed=seed + i * 31, arousal_profile=arousal)

    return {"hrv": hrv_all, "gsr": gsr_all}
