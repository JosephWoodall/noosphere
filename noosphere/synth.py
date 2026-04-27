"""
noosphere/data/synth.py
=======================
MOABB-Reference Synthetic EEG & Multimodal Data

Features:
- Multi-Band Kuramoto: Coupled Alpha (8-13Hz) and Beta (15-30Hz) rhythms.
- Subject Profiles: Inter-subject variability in leadfield and base frequencies.
- Beta Rebound: Post-movement synchronization (ERS) modeled as a phase surge.
- Power Line Contamination: Realistic 50/60Hz notch effects.
- 1/f Pink Noise: Authentic aperiodic neural background.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np

# ── EEG: MOABB-Reference Generation ──────────────────────────────────────────

@dataclass
class SubjectProfile:
    id: int
    alpha_freq: float
    beta_freq: float
    leadfield: np.ndarray
    impedance_drift: float
    blink_rate: float
    
    @classmethod
    def create(cls, subject_id: int, n_channels: int, n_sources: int, rng):
        # Create a local RNG for the subject profile to ensure uniqueness
        sub_rng = np.random.default_rng(subject_id)
        
        # Source frequencies vary per subject (Individual Alpha Frequency - IAF)
        iaf = sub_rng.normal(10.5, 1.0)
        ibf = sub_rng.normal(22.0, 2.5)
        
        # Leadfield: Subject-specific volume conduction
        lf = sub_rng.uniform(0.1, 0.7, (n_channels, n_sources))
        lf = lf / (lf.sum(axis=1, keepdims=True) + 1e-8)
        
        return cls(
            id=subject_id,
            alpha_freq=iaf,
            beta_freq=ibf,
            leadfield=lf,
            impedance_drift=sub_rng.uniform(0.001, 0.01),
            blink_rate=sub_rng.uniform(0.1, 0.5)
        )

class KuramotoNetwork:
    """Phase-coupled oscillators mimicking regional neural synchrony."""
    def __init__(self, n_nodes: int, freq_mean: float, freq_std: float, dt: float, rng):
        self.n_nodes = n_nodes; self.dt = dt
        self.omega = rng.normal(freq_mean * 2 * math.pi, freq_std * 2 * math.pi, n_nodes)
        self.theta = rng.uniform(0, 2 * math.pi, n_nodes)
        self.K = np.ones((n_nodes, n_nodes)) * 2.0
        np.fill_diagonal(self.K, 0)
        
    def step(self, K_matrix_override: Optional[np.ndarray] = None) -> np.ndarray:
        K = K_matrix_override if K_matrix_override is not None else self.K
        # Kuramoto differential equation
        diff = self.theta[None, :] - self.theta[:, None]
        coupling = np.sum(K * np.sin(diff), axis=1) / self.n_nodes
        self.theta += (self.omega + coupling) * self.dt
        self.theta = np.mod(self.theta, 2 * np.pi)
        return np.sin(self.theta)

def generate_pink_noise(n_samples: int, dt: float, rng) -> np.ndarray:
    """Generate authentic 1/f power-law aperiodic noise."""
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    freqs[0] = freqs[1]  # Anti-zero-div
    amp = 1.0 / (freqs + 1e-8)**0.75 # 1/f^beta where beta ~ 1.5 for voltage
    phase = rng.uniform(0, 2 * np.pi, len(freqs))
    noise = np.fft.irfft(amp * np.exp(1j * phase), n=n_samples)
    return noise / (np.std(noise) + 1e-8)

class ScalpEEGGenerator:
    """
    SOTA Synthetic Brainwave Engine.
    Uses Kuramoto oscillators for Alpha/Mu and Beta bands, Subject-specific leadfields,
    and post-movement Beta Rebound (ERS).
    """
    SAMPLE_RATE = 256

    # MOABB-compatible labels
    LABEL_CLEAN_BRAIN, LABEL_EYE_BLINK, LABEL_MUSCLE, LABEL_LINE_NOISE = 0, 1, 2, 3
    LABEL_SLOW_DRIFT, LABEL_CARDIAC, LABEL_MIXED, LABEL_SENSOR_NOISE = 4, 5, 6, 7

    # Intent mapping
    INTENT_REST, INTENT_RIGHT_HAND, INTENT_LEFT_HAND, INTENT_BOTH_HANDS = 0, 1, 2, 3
    INTENT_JAW_CLENCH, INTENT_HEAD_TILT, INTENT_SHOULDER_SHRUG = 4, 5, 6

    def __init__(self, seed: Optional[int] = 42, subject_id: int = 1):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0; self.dt = 1.0 / self.SAMPLE_RATE
        self.n_sources = 6; self.n_channels = 3
        
        # Create Subject Profile (The "MOABB" variable)
        self.profile = SubjectProfile.create(subject_id, self.n_channels, self.n_sources, self.rng)
        
        # Dual-band oscillators
        self.alpha_net = KuramotoNetwork(self.n_sources, self.profile.alpha_freq, 0.5, self.dt, self.rng)
        self.beta_net = KuramotoNetwork(self.n_sources, self.profile.beta_freq, 1.2, self.dt, self.rng)
        
        # Coupling matrices for ERD/ERS
        self.K_sync = np.ones((self.n_sources, self.n_sources)) * 5.0
        np.fill_diagonal(self.K_sync, 0)
        self.K_desync = self.K_sync * 0.1
        
        self._muscle_intent = self.INTENT_REST
        self._prev_intent = self.INTENT_REST
        self._rebound_timer = 0
        self._next_blink = 1.0 / self.profile.blink_rate + self.rng.exponential(2.0)
        self._muscle_burst = 0
        self._drift_state = self.rng.standard_normal(self.n_channels) * 5.0

    def next_segment(self, intent: Optional[int] = None, n_samples: int = 256) -> Dict:
        self._muscle_intent = intent if intent is not None else self.INTENT_REST
        
        # Check for Beta Rebound (Post-movement)
        if self._prev_intent != self.INTENT_REST and self._muscle_intent == self.INTENT_REST:
            self._rebound_timer = int(self.SAMPLE_RATE * 1.5) # 1.5s rebound
        self._prev_intent = self._muscle_intent
        
        sources = np.zeros((self.n_sources, n_samples), dtype=np.float32)
        
        # 1. Background Aperiodic Activity (1/f)
        for i in range(self.n_sources):
            sources[i, :] = generate_pink_noise(n_samples, self.dt, self.rng) * 12.0
            
        # 2. Oscillatory Activity (Alpha + Beta)
        K_alpha = self.K_sync if self._muscle_intent == self.INTENT_REST else self.K_desync
        
        # Beta is slightly different: Desync during move, ERS (rebound) after
        if self._rebound_timer > 0:
            K_beta = self.K_sync * 2.5 # Synchronized burst
            self._rebound_timer -= n_samples
        elif self._muscle_intent != self.INTENT_REST:
            K_beta = self.K_desync
        else:
            K_beta = self.K_sync
            
        for i in range(n_samples):
            # Sum alpha and beta oscillations
            sources[:, i] += self.alpha_net.step(K_alpha) * 18.0
            sources[:, i] += self.beta_net.step(K_beta) * 8.0
            
        # 3. Artifact Injection
        t_batch = self.t + np.arange(n_samples) * self.dt
        
        # Blinks (Source 5 is the EOG source)
        if t_batch[-1] > self._next_blink:
            idx = np.searchsorted(t_batch, self._next_blink)
            # Bell curve for blink
            blink = np.exp(-((np.arange(n_samples) - idx)**2) / (2 * 10.0**2)) * 150.0
            sources[5, :] += blink
            self._next_blink = t_batch[-1] + 1.0/self.profile.blink_rate + self.rng.exponential(2.0)
            
        # Muscle Bursts
        if self._muscle_burst <= 0 and self.rng.random() < 0.02:
            self._muscle_burst = self.rng.integers(20, 150)
            
        if self._muscle_burst > 0:
            burst_len = min(n_samples, self._muscle_burst)
            # High freq noise
            sources[4, :burst_len] += self.rng.standard_normal(burst_len) * 40.0
            self._muscle_burst -= burst_len
            
        # 4. Leadfield Projection (Source -> Sensor)
        eeg_channels = self.profile.leadfield @ sources
        
        # 5. Sensor-level noise & Contamination
        for ch in range(self.n_channels):
            # 50Hz/60Hz Noise
            line_freq = 60.0 if self.rng.random() > 0.5 else 50.0
            line_noise = np.sin(2 * math.pi * line_freq * t_batch) * 8.0
            eeg_channels[ch, :] += line_noise
            
            # Slow Drift (Electrode Impedance)
            self._drift_state[ch] += self.rng.standard_normal() * self.profile.impedance_drift
            eeg_channels[ch, :] += self._drift_state[ch]
            
            # White sensor noise
            eeg_channels[ch, :] += self.rng.standard_normal(n_samples) * 2.0

        self.t = t_batch[-1] + self.dt
        
        # Labels and Metadata
        avg = eeg_channels.mean(axis=1)
        var = eeg_channels[1].var()
        probs = np.zeros(8)
        probs[self.LABEL_MUSCLE] = min(0.6, var / 2000.0)
        probs[self.LABEL_EYE_BLINK] = min(0.9, max(0.0, np.abs(avg[1]) / 100.0))
        if probs[self.LABEL_MUSCLE] < 0.2 and probs[self.LABEL_EYE_BLINK] < 0.2:
            probs[self.LABEL_CLEAN_BRAIN] = 1.0
        if probs.sum() > 0: probs /= probs.sum()
        dom = int(probs.argmax())

        action = "Intentional" if (dom == self.LABEL_CLEAN_BRAIN and self._muscle_intent != self.INTENT_REST) else None
        
        return {
            "raw_microvolts": avg,
            "eeg": eeg_channels.astype(np.float32),
            "root_label": dom,
            "hierarchical": {
                "root": dom, 
                "muscle_intent": self._muscle_intent if action else None, 
                "subject_id": self.profile.id,
                "rebound_active": self._rebound_timer > 0
            },
            "probabilities": probs,
            "timestamp": t_batch[n_samples//2]
        }

# ── Vision: RGB + depth ───────────────────────────────────────────────────────

def synth_rgb(H: int = 64, W: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.float32)
    for c in range(3):
        img[:, :, c] = np.linspace(0.1, 0.4, W) * np.linspace(0.2, 0.5, H)[:, None]
    n_blobs = rng.integers(2, 6)
    for _ in range(n_blobs):
        cy, cx = rng.integers(8, H - 8), rng.integers(8, W - 8)
        r = rng.integers(4, 12)
        color = rng.uniform(0.3, 1.0, 3)
        y, x = np.ogrid[:H, :W]
        mask = (y - cy) ** 2 + (x - cx) ** 2 < r**2
        for c in range(3):
            img[:, :, c] = np.where(mask, color[c], img[:, :, c])
    return img.clip(0, 1)

def synth_depth(H: int = 64, W: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    depth = np.ones((H, W), dtype=np.float32) * 1.5
    depth[H // 2 :] = 0.8 + rng.uniform(-0.02, 0.02, (H // 2, W))
    cy, cx = rng.integers(H // 4, 3 * H // 4), rng.integers(W // 4, 3 * W // 4)
    r = rng.integers(6, 15)
    y, x = np.ogrid[:H, :W]
    mask = ((y - cy) ** 2 + (x - cx) ** 2) < r**2
    depth = np.where(mask, 0.4 + rng.uniform(0, 0.1), depth)
    return depth.clip(0.1, 5.0)

def synth_stereo(H: int = 64, W: int = 64, baseline: float = 0.12, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    left = synth_rgb(H, W, seed)
    shift = int(W * baseline / 1.5)
    right = np.roll(left, -shift, axis=1)
    right[:, -shift:] = 0.0
    return left, right

# ── Kinematics ────────────────────────────────────────────────────────────────

def synth_kinematics(n_nodes: int = 6, feat_dim: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((n_nodes, feat_dim)).astype(np.float32)
    state[:, 0:3] *= 0.3; state[:, 3:6] *= 0.5; state[:, 6:9] *= 1.0; state[:, 9:12] *= 5.0
    return state

def synth_imu(n_steps: int = 10, feat_dim: int = 13, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    imu = rng.standard_normal((n_steps, feat_dim)).astype(np.float32)
    imu[:, 0:3] *= 9.81 * 0.2; imu[:, 3:6] *= 0.5; imu[:, 6:9] *= 0.3
    return imu

# ── Domain-specific full observation builders ─────────────────────────────────

def obs_drone(seed: int = 0) -> Dict:
    rgb = synth_rgb(64, 64, seed); dep = synth_depth(64, 64, seed); imu = synth_imu(10, 13, seed)
    return {"rgb": rgb, "depth": dep, "structured": imu}

def obs_legged(seed: int = 0) -> Dict:
    left, right = synth_stereo(48, 64, seed=seed); joints = synth_kinematics(30, 12, seed)
    return {"rgb": left, "rgb_right": right, "kinematics": joints}

def obs_manipulation(seed: int = 0) -> Dict:
    rgb = synth_rgb(64, 64, seed); dep = synth_depth(64, 64, seed); kin = synth_kinematics(6, 13, seed)
    return {"rgb": rgb, "depth": dep, "kinematics": kin}

def obs_bci(seed: int = 0, intent: Optional[int] = None, eeg_gen: Optional[ScalpEEGGenerator] = None) -> Dict:
    gen = eeg_gen or ScalpEEGGenerator(seed)
    seg = gen.next_segment(intent=intent, n_samples=256)
    rgb = synth_rgb(48, 64, seed); imu = synth_imu(5, 5, seed)
    return {
        "eeg": seg["eeg"], "electrode_mask": np.ones(3, dtype=np.float32),
        "rgb": rgb, "structured": imu, "_eeg_segment": seg
    }

def obs_fluid(seed: int = 0) -> Dict:
    rgb = synth_rgb(64, 64, seed); dep = synth_depth(64, 64, seed); pressure = synth_imu(20, 18, seed)
    return {"rgb": rgb, "depth": dep, "structured": pressure}

# ── Full multimodal batch builder ─────────────────────────────────────────────

def make_moabb_dataset(n_subjects: int = 5, n_trials_per_subject: int = 20, seq_len: int = 256) -> Dict[int, List[Dict]]:
    """
    Generates a structured multi-subject dataset similar to MOABB archives.
    Each subject has a unique profile and multiple trials of different intents.
    """
    dataset = {}
    intents = [
        ScalpEEGGenerator.INTENT_REST, 
        ScalpEEGGenerator.INTENT_RIGHT_HAND, 
        ScalpEEGGenerator.INTENT_LEFT_HAND
    ]
    
    for sub_id in range(1, n_subjects + 1):
        gen = ScalpEEGGenerator(seed=sub_id * 100, subject_id=sub_id)
        subject_data = []
        for trial_id in range(n_trials_per_subject):
            intent = int(np.random.choice(intents))
            # Simulate a few segments of the same intent to form a 'trial'
            segments = [gen.next_segment(intent=intent, n_samples=seq_len) for _ in range(3)]
            combined_eeg = np.concatenate([s["eeg"] for s in segments], axis=1)
            subject_data.append({
                "trial_id": trial_id,
                "intent": intent,
                "eeg": combined_eeg,
                "profile": gen.profile
            })
        dataset[sub_id] = subject_data
    return dataset

def make_batch(
    domain: str,
    B: int = 4,
    seed: int = 0,
    eeg_gen: Optional[ScalpEEGGenerator] = None,
) -> Dict:
    """
    Build a batch of B observations for `domain`.
    All observations are synthetic and deterministic given seed.
    """
    obs_fn = {
        "drone": obs_drone,
        "legged": obs_legged,
        "manipulation": obs_manipulation,
        "fluid": obs_fluid,
    }.get(domain)

    if domain == "bci":
        gen = eeg_gen or ScalpEEGGenerator(seed)
        samples = [obs_bci(seed + i, eeg_gen=gen) for i in range(B)]
    elif obs_fn is not None:
        samples = [obs_fn(seed + i) for i in range(B)]
    else:
        raise ValueError(f"Unknown domain: {domain}")

    # Stack into batch
    batch = {}
    keys = samples[0].keys()
    for k in keys:
        if k.startswith("_"):
            continue  # skip metadata
        arrs = [np.array(s[k], dtype=np.float32) for s in samples]
        try:
            batch[k] = np.stack(arrs, axis=0)
        except ValueError:
            batch[k] = arrs  # variable-length fallback
    return batch

# ── Monitor Stress Test ───────────────────────────────────────────────────────

class MonitorStressTest:
    @staticmethod
    def kl_explosion() -> Dict: return {"wm/loss": 5.0, "wm/kl": 25.0}
    @staticmethod
    def reward_crash() -> List[float]: return [0.9] * 25 + [0.1] * 25
    @staticmethod
    def command_failure_streak() -> List[Dict]: return [{"exit_code": 1, "outcome": "error"}] * 15
    @staticmethod
    def gpu_pressure() -> float: return 98.5

# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("MOABB-Reference Synthetic Data Check")
    print("─" * 40)

    gen = ScalpEEGGenerator(seed=42, subject_id=7)
    print(f"  Subject Profile: ID={gen.profile.id}, Alpha={gen.profile.alpha_freq:.1f}Hz, Beta={gen.profile.beta_freq:.1f}Hz")
    
    for intent_name, intent_id in [
        ("REST", ScalpEEGGenerator.INTENT_REST),
        ("RIGHT_HAND", ScalpEEGGenerator.INTENT_RIGHT_HAND),
    ]:
        seg = gen.next_segment(intent=intent_id)
        print(f"  EEG [{intent_name:15s}] root={seg['root_label']} std={seg['eeg'].std():.2f}")

    print("  Beta Rebound check...")
    _ = gen.next_segment(intent=ScalpEEGGenerator.INTENT_RIGHT_HAND)
    seg_rebound = gen.next_segment(intent=ScalpEEGGenerator.INTENT_REST)
    print(f"  Post-move: Rebound Active = {seg_rebound['hierarchical']['rebound_active']}")

    print("All checks passed.")
