"""
21-channel synthetic EEG generator for prosthetic arm control.

Extends the Kuramoto oscillator model from v1/noosphere/synth.py to:
  - 21 BCI-standard channels (10-20 system)
  - 6-DOF continuous intent vectors (instead of discrete class labels)
  - ERD/ERS patterns at C3/C4 proportional to arm motor activity
  - ERN simulation at FCz triggered by rapid intent reversals
  - Hyper-realistic noise model (pink 1/f background, eye blinks, muscle artifacts,
    line noise, slow impedance drift)

Usage:
    gen = EEGStreamGenerator(seed=42, subject_id=1, fs=256)
    for frame in gen.stream(motor_commands_6dof):   # (T, 6) continuous
        eeg_frame = frame  # (21,)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── Channel layout ────────────────────────────────────────────────────────────

CHANNELS_21 = [
    "Fp1", "Fp2",
    "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "Oz", "O2",
    "FCz",
]
N_CH = len(CHANNELS_21)   # 21

# Channel indices relevant to motor control
IDX_C3  = CHANNELS_21.index("C3")   # 8
IDX_CZ  = CHANNELS_21.index("Cz")   # 9
IDX_C4  = CHANNELS_21.index("C4")   # 10
IDX_FCZ = CHANNELS_21.index("FCz")  # 20
IDX_T7  = CHANNELS_21.index("T7")   # 7
IDX_T8  = CHANNELS_21.index("T8")   # 11

# ── Neural source model ───────────────────────────────────────────────────────

class KuramotoOscillator:
    """Vectorized Kuramoto phase oscillator network."""

    def __init__(self, n: int, f_mean: float, f_std: float, coupling: float, dt: float, rng):
        self.n = n
        self.dt = dt
        self.omega = rng.normal(f_mean * 2 * math.pi, f_std * 2 * math.pi, n)
        self.theta = rng.uniform(0, 2 * math.pi, n)
        K = np.ones((n, n)) * coupling
        np.fill_diagonal(K, 0.0)
        self.K = K

    def step(self, coupling_override: Optional[np.ndarray] = None) -> np.ndarray:
        K = coupling_override if coupling_override is not None else self.K
        dtheta = np.sin(self.theta[None, :] - self.theta[:, None])
        coupling_term = (K * dtheta).sum(axis=1) / self.n
        self.theta += (self.omega + coupling_term) * self.dt
        self.theta %= 2 * math.pi
        return np.sin(self.theta)

    def reset(self, rng):
        self.theta = rng.uniform(0, 2 * math.pi, self.n)


def _pink_noise(n: int, dt: float, rng: np.random.Generator) -> np.ndarray:
    """1/f^0.75 power-law aperiodic background."""
    freqs = np.fft.rfftfreq(n, d=dt)
    freqs[0] = freqs[1]
    amp = 1.0 / (freqs + 1e-8) ** 0.75
    phase = rng.uniform(0, 2 * math.pi, len(freqs))
    noise = np.fft.irfft(amp * np.exp(1j * phase), n=n)
    return noise / (np.std(noise) + 1e-8)

# ── Subject profile ───────────────────────────────────────────────────────────

@dataclass
class SubjectProfile:
    id: int
    iaf: float         # individual alpha frequency (Hz)
    ibf: float         # individual beta frequency (Hz)
    leadfield: np.ndarray  # (21, n_sources) volume conduction matrix
    blink_rate: float  # blinks per second
    impedance_drift: float

    @classmethod
    def create(cls, subject_id: int, n_sources: int = 8, rng_seed: Optional[int] = None):
        sub_rng = np.random.default_rng(rng_seed if rng_seed is not None else subject_id * 7919)
        iaf = float(sub_rng.normal(10.5, 1.2))
        ibf = float(sub_rng.normal(21.0, 2.5))

        # Sparse leadfield: each source affects nearby channels with falloff
        lf = sub_rng.uniform(0.05, 0.6, (N_CH, n_sources))
        # Make motor sources (sources 0-2) especially strong at C3, Cz, C4
        for motor_src in range(3):
            lf[IDX_C3, motor_src] *= sub_rng.uniform(2.0, 3.5)
            lf[IDX_CZ, motor_src] *= sub_rng.uniform(1.5, 2.5)
            lf[IDX_C4, motor_src] *= sub_rng.uniform(1.0, 2.0)
        # ERN source strong at FCz
        lf[IDX_FCZ, 3] *= sub_rng.uniform(3.0, 5.0)
        # Row-normalize so total power is consistent
        lf /= lf.sum(axis=1, keepdims=True) + 1e-8
        return cls(
            id=subject_id,
            iaf=iaf,
            ibf=ibf,
            leadfield=lf.astype(np.float32),
            blink_rate=float(sub_rng.uniform(0.1, 0.4)),
            impedance_drift=float(sub_rng.uniform(0.001, 0.008)),
        )


# ── Main generator ────────────────────────────────────────────────────────────

class EEGStreamGenerator:
    """
    Continuous streaming 21-channel EEG generator.

    Intent is a 6-DOF vector in [-1, 1].  Motor activity level = sum(|intent|) / 6.
    High motor activity → ERD (alpha/beta desynchronization) at C3.
    Low motor activity after movement → ERS (beta rebound) at C3.
    Rapid intent reversal → ERN at FCz.
    """

    N_SOURCES = 8

    def __init__(self, seed: int = 42, subject_id: int = 1, fs: int = 256):
        self.fs = fs
        self.dt = 1.0 / fs
        self.rng = np.random.default_rng(seed)
        self.profile = SubjectProfile.create(subject_id, self.N_SOURCES)

        # Oscillator networks per frequency band
        self._alpha = KuramotoOscillator(
            self.N_SOURCES, self.profile.iaf, 0.5, coupling=2.0, dt=self.dt, rng=self.rng
        )
        self._beta = KuramotoOscillator(
            self.N_SOURCES, self.profile.ibf, 1.2, coupling=1.5, dt=self.dt, rng=self.rng
        )
        self._gamma = KuramotoOscillator(
            self.N_SOURCES, 45.0, 5.0, coupling=0.5, dt=self.dt, rng=self.rng
        )
        self._theta = KuramotoOscillator(
            self.N_SOURCES, 6.0, 0.8, coupling=1.0, dt=self.dt, rng=self.rng
        )

        # Coupling matrices
        self._K_sync = np.ones((self.N_SOURCES, self.N_SOURCES)) * 4.0
        np.fill_diagonal(self._K_sync, 0.0)
        self._K_desync = self._K_sync * 0.15

        # ERN and rebound state
        self._prev_intent = np.zeros(6, dtype=np.float32)
        self._rebound_timer = 0       # samples remaining in beta rebound
        self._ern_timer = 0           # samples remaining in ERN
        self._blink_countdown = self._next_blink_interval()
        self._muscle_burst_timer = 0
        self._drift_state = self.rng.standard_normal(N_CH).astype(np.float32) * 3.0
        self._t = 0.0

    def _next_blink_interval(self) -> int:
        mean_s = 1.0 / (self.profile.blink_rate + 1e-8)
        return int(self.rng.exponential(mean_s) * self.fs)

    def _erd_strength(self, intent: np.ndarray) -> float:
        """ERD is proportional to total motor activity level in [0,1]."""
        return float(np.mean(np.abs(intent)))

    def _is_error(self, intent: np.ndarray) -> bool:
        """Rapid reversal of intent direction ≈ unintended movement → ERN."""
        dot = float(np.dot(self.profile.iaf * 0 + intent, self._prev_intent))
        activity = float(np.linalg.norm(self._prev_intent)) * float(np.linalg.norm(intent))
        if activity < 0.1:
            return False
        cosine = dot / (activity + 1e-8)
        return cosine < -0.7  # near-reversal

    def step(self, intent: np.ndarray) -> np.ndarray:
        """
        Generate one EEG sample given 6-DOF intent vector.
        intent: (6,) float in [-1, 1]
        Returns: (21,) float EEG in microvolts.
        """
        erd = self._erd_strength(intent)

        # ERN trigger: intent reversal detected
        if self._is_error(intent) and self._ern_timer == 0:
            self._ern_timer = int(0.25 * self.fs)  # 250 ms ERN window
        self._prev_intent = intent.copy()

        # Beta rebound: when transitioning from active to rest
        if erd < 0.05 and np.linalg.norm(self._prev_intent) > 0.3:
            self._rebound_timer = int(1.5 * self.fs)

        # Desynchronization coupling: more activity → less coupling → more ERD
        K_alpha = self._K_desync if erd > 0.3 else self._K_sync
        K_beta  = self._K_desync if erd > 0.3 else self._K_sync
        if self._rebound_timer > 0:
            K_beta = self._K_sync * 1.5  # beta rebound — hypersynchronized
            self._rebound_timer -= 1

        # Oscillator outputs (N_SOURCES,)
        alpha_src = self._alpha.step(K_alpha) * 20.0 * (1 - erd * 0.6)
        beta_src  = self._beta.step(K_beta)  * 12.0 * (1 - erd * 0.5)
        gamma_src = self._gamma.step()       *  4.0 * (1 + erd * 0.4)
        theta_src = self._theta.step()       * 10.0

        sources = alpha_src + beta_src + gamma_src + theta_src  # (N_SOURCES,)

        # Pink noise background
        sources += self.rng.standard_normal(self.N_SOURCES) * 8.0

        # ── Artifact injection ────────────────────────────────────────────────

        # Eye blink (large slow deflection, strongest at Fp1, Fp2)
        self._blink_countdown -= 1
        if self._blink_countdown <= 0:
            blink_amp = self.rng.uniform(80, 200)
            sources[0] += blink_amp  # blink source
            self._blink_countdown = self._next_blink_interval()

        # Muscle burst (high-frequency noise at temporal sources)
        if self._muscle_burst_timer <= 0 and self.rng.random() < 0.005:
            self._muscle_burst_timer = self.rng.integers(10, 60)
        if self._muscle_burst_timer > 0:
            sources[4:6] += self.rng.standard_normal(2) * 35.0
            self._muscle_burst_timer -= 1

        # ── Leadfield projection: sources → scalp electrodes ─────────────────
        eeg = self.profile.leadfield @ sources  # (21,)

        # ── ERN: negative deflection at FCz ──────────────────────────────────
        if self._ern_timer > 0:
            decay = self._ern_timer / (0.25 * self.fs)
            ern_amp = -15.0 * decay
            eeg[IDX_FCZ] += ern_amp
            eeg[IDX_CZ]  += ern_amp * 0.4
            self._ern_timer -= 1

        # ── Sensor-level noise ───────────────────────────────────────────────
        t = self._t
        for ch in range(N_CH):
            # 50/60 Hz line noise
            line_f = 60.0 if ch % 2 == 0 else 50.0
            eeg[ch] += math.sin(2 * math.pi * line_f * t) * 6.0
            # Impedance drift
            self._drift_state[ch] += self.rng.standard_normal() * self.profile.impedance_drift
            eeg[ch] += self._drift_state[ch]
            # White sensor noise
            eeg[ch] += self.rng.standard_normal() * 1.5

        self._t += self.dt
        return eeg.astype(np.float32)

    def stream(self, motor_commands: np.ndarray) -> np.ndarray:
        """
        Generate a full EEG window correlated with continuous motor commands.
        motor_commands: (T, 6) float in [-1, 1]
        Returns: (T, 21) EEG in microvolts.
        """
        T = len(motor_commands)
        eeg = np.empty((T, N_CH), dtype=np.float32)
        for t in range(T):
            eeg[t] = self.step(motor_commands[t])
        return eeg

    def ern_active(self) -> bool:
        return self._ern_timer > 0


# ── Batch data builders ───────────────────────────────────────────────────────

def smooth_motor_trajectory(
    n_dof: int = 6,
    T: int = 1024,
    n_segments: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate smooth continuous motor command trajectory using cosine interpolation.
    Returns (T, n_dof) float in [-1, 1].
    """
    if rng is None:
        rng = np.random.default_rng()
    waypoints = rng.uniform(-1, 1, (n_segments + 1, n_dof))
    # Zero out some DOFs to simulate partial movement
    inactive = rng.choice(n_dof, size=rng.integers(0, n_dof // 2), replace=False)
    waypoints[:, inactive] = 0.0

    t_full = np.linspace(0, n_segments, T)
    cmd = np.empty((T, n_dof), dtype=np.float32)
    for d in range(n_dof):
        cmd[:, d] = np.interp(t_full, np.arange(n_segments + 1), waypoints[:, d])

    # Cosine smoothing via sinusoidal basis
    blend = 0.5 * (1 - np.cos(np.pi * (t_full % 1)))
    segments = t_full.astype(int).clip(0, n_segments - 1)
    alpha_vals = waypoints[segments, :]
    beta_vals  = waypoints[(segments + 1).clip(0, n_segments), :]
    cmd = alpha_vals + (beta_vals - alpha_vals) * blend[:, None]
    return cmd.astype(np.float32)


def make_training_batch(
    n_subjects: int = 5,
    n_trials: int = 20,
    trial_duration_s: float = 4.0,
    fs: int = 256,
    seed: int = 0,
) -> dict:
    """
    Build a dict of training data keyed by subject ID.

    Returns:
        {
          subject_id: {
            "eeg": (n_trials, T, 21) float32,
            "commands": (n_trials, T, 6) float32,
            "ern_labels": (n_trials, T) bool,
          }
        }
    """
    T = int(trial_duration_s * fs)
    dataset = {}
    for sub_id in range(n_subjects):
        gen = EEGStreamGenerator(seed=seed + sub_id * 1000, subject_id=sub_id, fs=fs)
        eeg_all = np.empty((n_trials, T, N_CH), dtype=np.float32)
        cmd_all = np.empty((n_trials, T, 6), dtype=np.float32)
        ern_all = np.zeros((n_trials, T), dtype=bool)

        for trial in range(n_trials):
            cmds = smooth_motor_trajectory(n_dof=6, T=T, rng=np.random.default_rng(seed + sub_id * 100 + trial))
            gen2 = EEGStreamGenerator(seed=seed + sub_id * 1000 + trial, subject_id=sub_id, fs=fs)
            for t in range(T):
                eeg_all[trial, t] = gen2.step(cmds[t])
                ern_all[trial, t] = gen2.ern_active()
            cmd_all[trial] = cmds

        dataset[sub_id] = {"eeg": eeg_all, "commands": cmd_all, "ern_labels": ern_all}
    return dataset
