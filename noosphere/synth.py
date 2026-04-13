"""
noosphere/data/synth.py
=======================
Synthetic Test Data — All Modalities

Single file for all synthetic sensor data generation.
Replace individual generators with real hardware drivers as you integrate
physical systems. All generators produce NumPy arrays in the format
expected by NoosphereAgent.step(obs).

EEG configuration:
    3 electrodes on the scalp (e.g., C3, Cz, C4).
    Motor cortex placement means:
        - Strong alpha / desynchronization (ERD) — this IS the intentional signal
        - Prominent eye blinks (frontend artifacts)
        - Muscle artifacts (EMG from jaw/neck) — this is NOISE
        - Good access to: motor imagery, cognitive intent

    This follows standard BCI paradigms where cognitive brainwaves form
    the intentional signal, and muscle twitches must be rejected.

Artifact hierarchy (from mechanicus, adapted for 3-ch neck):
    RootArtifactLabel:
        CleanBrain      — intent/motor imagery or resting baseline
        EyeBlink        — frontend artifacts (high amplitude noise)
        MuscleArtifact  — jaw/neck EMG (high frequency noise)
            → RightHand, LeftHand, BothHands, JawClench,
              HeadTilt, ShoulderShrug, FingerFlexion,
              WristExtension, EyebrowRaise
        LineNoise       — 60Hz contamination
        SlowDrift       — electrode drift
        Cardiac         — weak at neck
        MixedArtifact   — combinations
        SensorNoiseOnly — pure noise floor

All generators are deterministic given a seed, so tests are reproducible.
"""

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np

# ── EEG: 3-electrode neck placement ──────────────────────────────────────────


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
        diff = self.theta[None, :] - self.theta[:, None]
        coupling = np.sum(K * np.sin(diff), axis=1) / self.n_nodes
        self.theta += (self.omega + coupling) * self.dt
        self.theta = np.mod(self.theta, 2 * np.pi)
        return np.sin(self.theta)

def generate_pink_noise(n_samples: int, dt: float, rng) -> np.ndarray:
    """Generate authentic 1/f power-law aperiodic noise."""
    freqs = np.fft.rfftfreq(n_samples, d=dt)
    freqs[0] = freqs[1]  # Anti-zero-div
    amp = 1.0 / np.sqrt(freqs)
    phase = rng.uniform(0, 2 * np.pi, len(freqs))
    noise = np.fft.irfft(amp * np.exp(1j * phase), n=n_samples)
    return noise / (np.std(noise) + 1e-8)

class ScalpEEGGenerator:
    """
    SOTA Synthetic Brainwave Engine.
    Employs Kuramoto Non-Linear Phase-Coupled Oscillators (mu/alpha ERD rhythms),
    1/f Aperiodic Pink Noise, and a Spatial Volume Conduction Leadfield projection 
    from 5 internal cortical sources -> 3 active scalp sensors.
    """
    SAMPLE_RATE = 256

    LABEL_CLEAN_BRAIN, LABEL_EYE_BLINK, LABEL_MUSCLE, LABEL_LINE_NOISE = 0, 1, 2, 3
    LABEL_SLOW_DRIFT, LABEL_CARDIAC, LABEL_MIXED, LABEL_SENSOR_NOISE = 4, 5, 6, 7

    INTENT_REST, INTENT_RIGHT_HAND, INTENT_LEFT_HAND, INTENT_BOTH_HANDS = 0, 1, 2, 3
    INTENT_JAW_CLENCH, INTENT_HEAD_TILT, INTENT_SHOULDER_SHRUG = 4, 5, 6
    INTENT_FINGER_FLEX, INTENT_WRIST_EXT, INTENT_EYEBROW_RAISE = 7, 8, 9

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0; self.dt = 1.0 / self.SAMPLE_RATE
        self.n_sources = 5; self.n_channels = 3
        
        # Simulated Volume Conduction Matrix (Leadfield)
        self.leadfield = np.array([
            [0.6, 0.2, 0.1,  0.4, 0.2],  # C3 external sensor
            [0.2, 0.5, 0.2,  0.6, 0.2],  # Cz external sensor
            [0.1, 0.2, 0.6,  0.4, 0.2]   # C4 external sensor
        ])
        
        self.kuramoto = KuramotoNetwork(3, 11.0, 0.5, self.dt, self.rng)
        self.K_rest = np.ones((3, 3)) * 4.0; np.fill_diagonal(self.K_rest, 0)
        self.K_desync_right = self.K_rest.copy()
        self.K_desync_right[0, :], self.K_desync_right[:, 0] = 0.5, 0.5 # Left C3 ERD
        self.K_desync_left = self.K_rest.copy()
        self.K_desync_left[2, :], self.K_desync_left[:, 2] = 0.5, 0.5   # Right C4 ERD
        
        self._muscle_intent = self.INTENT_REST
        self._next_blink = 2.0 + self.rng.exponential(1.5)
        self._muscle_burst = 0; self._muscle_freqs = None

    def next_segment(self, intent: Optional[int] = None, n_samples: int = 256) -> Dict:
        self._muscle_intent = intent if intent is not None else self.INTENT_REST
        sources = np.zeros((self.n_sources, n_samples), dtype=np.float32)
        
        for i in range(self.n_sources):
            sources[i, :] = generate_pink_noise(n_samples, self.dt, self.rng) * 15.0
        
        K_active = self.K_rest
        if self._muscle_intent == self.INTENT_RIGHT_HAND: K_active = self.K_desync_right
        elif self._muscle_intent == self.INTENT_LEFT_HAND: K_active = self.K_desync_left
        elif self._muscle_intent != self.INTENT_REST: K_active = self.K_rest * 0.1
        
        for i in range(n_samples):
            sources[0:3, i] += self.kuramoto.step(K_active) * 20.0
            
        t_batch = self.t + np.arange(n_samples) * self.dt
        if t_batch[-1] > self._next_blink:
            idx = np.searchsorted(t_batch, self._next_blink)
            sources[3, :] += np.exp(-((np.arange(n_samples) - idx)**2) / 100.0) * 120.0
            self._next_blink = t_batch[-1] + np.clip(self.rng.exponential(3.0), 1.0, 15.0)
            
        if self._muscle_burst == 0 and self.rng.random() < 0.05:
            self._muscle_burst = self.rng.integers(10, 80)
            self._muscle_freqs = self.rng.uniform(30.0, 100.0, size=3)
            
        if self._muscle_burst > 0:
            burst_len = min(n_samples, self._muscle_burst)
            t_emg = t_batch[:burst_len]
            sources[4, :burst_len] += np.sum([np.sin(2*math.pi*f*t_emg) for f in self._muscle_freqs], axis=0) * 20.0
            self._muscle_burst -= burst_len
            
        eeg_channels = self.leadfield @ sources
        
        for ch in range(self.n_channels):
            eeg_channels[ch, :] += math.sin(t_batch[0]/50.0)*30.0 + np.sin(2*math.pi*60*t_batch)*10.0 + self.rng.standard_normal(n_samples)*3.0

        self.t = t_batch[-1] + self.dt
        avg = eeg_channels.mean(axis=1)
        
        var = eeg_channels[1].var()
        probs = np.zeros(8)
        probs[self.LABEL_MUSCLE] = min(0.6, var / 1500.0)
        probs[self.LABEL_EYE_BLINK] = min(0.8, max(0.0, avg[1] / 60.0))
        if probs[self.LABEL_MUSCLE] < 0.3 and probs[self.LABEL_EYE_BLINK] < 0.3:
            probs[self.LABEL_CLEAN_BRAIN] = 1.0
        if probs.sum() > 0: probs /= probs.sum()
        dom = int(probs.argmax())

        action = "Intentional" if (dom == self.LABEL_CLEAN_BRAIN and self._muscle_intent != self.INTENT_REST) else None
        
        return {
            "raw_microvolts": avg,
            "eeg": eeg_channels.astype(np.float32),
            "root_label": dom,
            "hierarchical": {"root": dom, "muscle_intent": self._muscle_intent if action else None, "action": action, "kinematic": None, "custom_tag": "SOTA_Kuramoto_1f_EEG"},
            "probabilities": probs,
            "timestamp": t_batch[n_samples//2]
        }



# ── Vision: RGB + depth ───────────────────────────────────────────────────────


def synth_rgb(H: int = 64, W: int = 64, seed: int = 0) -> np.ndarray:
    """
    (H, W, 3) float32 [0,1] — synthetic RGB frame.
    Adds structured content (gradient + random blobs) so vision encoder
    sees non-trivial spatial patterns.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.float32)

    # Background gradient
    for c in range(3):
        img[:, :, c] = np.linspace(0.1, 0.4, W) * np.linspace(0.2, 0.5, H)[:, None]

    # Random blobs (simulating objects)
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
    """
    (H, W) float32 — metric depth in metres [0.1, 5.0].
    Simulates a table surface with an object on it.
    """
    rng = np.random.default_rng(seed)
    depth = np.ones((H, W), dtype=np.float32) * 1.5  # background at 1.5m

    # Floor/table at 0.8m in lower half
    depth[H // 2 :] = 0.8 + rng.uniform(-0.02, 0.02, (H // 2, W))

    # Object bump
    cy, cx = rng.integers(H // 4, 3 * H // 4), rng.integers(W // 4, 3 * W // 4)
    r = rng.integers(6, 15)
    y, x = np.ogrid[:H, :W]
    mask = ((y - cy) ** 2 + (x - cx) ** 2) < r**2
    depth = np.where(mask, 0.4 + rng.uniform(0, 0.1), depth)

    return depth.clip(0.1, 5.0)


def synth_stereo(
    H: int = 64, W: int = 64, baseline: float = 0.12, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (rgb_left, rgb_right) — slightly shifted for stereo."""
    left = synth_rgb(H, W, seed)
    # Right view: small horizontal shift
    shift = int(W * baseline / 1.5)  # approx at 1.5m depth
    right = np.roll(left, -shift, axis=1)
    right[:, -shift:] = 0.0
    return left, right


# ── Kinematics ────────────────────────────────────────────────────────────────


def synth_kinematics(
    n_nodes: int = 6,
    feat_dim: int = 12,
    seed: int = 0,
) -> np.ndarray:
    """
    (n_nodes, feat_dim) float32 — joint state vector.
    Feature layout (matching mechanicus arm): [pos(3), vel(3), angle(3), torque(3)]
    """
    rng = np.random.default_rng(seed)
    state = rng.standard_normal((n_nodes, feat_dim)).astype(np.float32)

    # Physically plausible ranges (radians / m/s / Nm)
    state[:, 0:3] *= 0.3  # position: ±0.3m
    state[:, 3:6] *= 0.5  # velocity: ±0.5 m/s
    state[:, 6:9] *= 1.0  # angle: ±1 rad
    state[:, 9:12] *= 5.0  # torque: ±5 Nm

    return state


def synth_imu(n_steps: int = 10, feat_dim: int = 13, seed: int = 0) -> np.ndarray:
    """
    (n_steps, feat_dim) float32 — IMU time series.
    [ax, ay, az, gx, gy, gz, roll, pitch, yaw, alt, vx, vy, vz]
    """
    rng = np.random.default_rng(seed)
    imu = rng.standard_normal((n_steps, feat_dim)).astype(np.float32)
    imu[:, 0:3] *= 9.81 * 0.2  # accel (fraction of g)
    imu[:, 3:6] *= 0.5  # gyro (rad/s)
    imu[:, 6:9] *= 0.3  # euler angles (rad)
    return imu


def synth_lidar(n_pts: int = 512, seed: int = 0) -> np.ndarray:
    """(n_pts, 3) float32 — 3D point cloud in metres."""
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n_pts, 3)).astype(np.float32)
    pts = pts / (np.linalg.norm(pts, axis=1, keepdims=True) + 1e-6)
    dist = rng.uniform(0.3, 2.0, (n_pts, 1)).astype(np.float32)
    return pts * dist


def synth_audio(n_mels: int = 80, T: int = 64, seed: int = 0) -> np.ndarray:
    """(1, n_mels, T) float32 — log-mel spectrogram."""
    rng = np.random.default_rng(seed)
    spec = rng.uniform(0.0, 1.0, (1, n_mels, T)).astype(np.float32)
    # Add harmonic structure
    for k in range(1, 5):
        row = min(int(k * n_mels / 8), n_mels - 1)
        spec[0, row, :] += 0.5
    return spec.clip(0, 1)


# ── Domain-specific full observation builders ─────────────────────────────────


def obs_drone(seed: int = 0) -> Dict:
    """
    Drone observation.
    Sensors: RGB downward, depth (ToF), IMU (13 features, 10 steps).
    """
    rgb = synth_rgb(64, 64, seed)
    dep = synth_depth(64, 64, seed)
    imu = synth_imu(10, 13, seed)
    return {"rgb": rgb, "depth": dep, "structured": imu}


def obs_legged(seed: int = 0) -> Dict:
    """
    Legged robot observation.
    Sensors: stereo RGB, joint state (30 DOF × 12 features).
    """
    left, right = synth_stereo(48, 64, seed=seed)
    joints = synth_kinematics(30, 12, seed)
    return {"rgb": left, "rgb_right": right, "kinematics": joints}


def obs_manipulation(seed: int = 0) -> Dict:
    """
    Manipulation arm observation.
    Sensors: RGBD, end-effector + force-torque (6 joints × 13 features).
    """
    rgb = synth_rgb(64, 64, seed)
    dep = synth_depth(64, 64, seed)
    kin = synth_kinematics(6, 13, seed)
    return {"rgb": rgb, "depth": dep, "kinematics": kin}


def obs_bci(
    seed: int = 0,
    intent: Optional[int] = None,
    eeg_gen: Optional[ScalpEEGGenerator] = None,
) -> Dict:
    """
    BCI-controlled apparatus observation.
    Sensors: 3-electrode neck EEG (256 samples), visual feedback, system state.

    intent: force a specific MuscleIntent for testing supervised learning.
    eeg_gen: reuse existing generator for temporal continuity.
    """
    gen = eeg_gen or ScalpEEGGenerator(seed)
    seg = gen.next_segment(intent=intent, n_samples=256)
    rgb = synth_rgb(48, 64, seed)
    imu = synth_imu(
        5, 5, seed
    )  # [cursor_x, cursor_y, target_x, target_y, time_remaining]
    return {
        "eeg": seg["eeg"],  # (3, 256) float32
        "electrode_mask": np.ones(3, dtype=np.float32),
        "rgb": rgb,
        "structured": imu,
        "_eeg_segment": seg,  # full label for supervised training
    }


def obs_fluid(seed: int = 0) -> Dict:
    """
    Fluid / soft-body observation.
    Sensors: RGB (high-speed cam), pressure array (18 features, 20 steps).
    """
    rgb = synth_rgb(64, 64, seed)
    dep = synth_depth(64, 64, seed)
    pressure = synth_imu(20, 18, seed)
    return {"rgb": rgb, "depth": dep, "structured": pressure}


# ── Full multimodal batch builder ─────────────────────────────────────────────


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


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Synthetic data sanity check")
    print("─" * 40)

    gen = ScalpEEGGenerator(seed=42)
    for intent_name, intent_id in [
        ("REST", ScalpEEGGenerator.INTENT_REST),
        ("RIGHT_HAND", ScalpEEGGenerator.INTENT_RIGHT_HAND),
        ("SHOULDER_SHRUG", ScalpEEGGenerator.INTENT_SHOULDER_SHRUG),
    ]:
        seg = gen.next_segment(intent=intent_id)
        print(
            f"  EEG [{intent_name:15s}]  "
            f"root={seg['root_label']}  "
            f"shape={seg['eeg'].shape}  "
            f"μV_rms={seg['eeg'].std(axis=1).mean():.1f}"
        )

    for domain in ["drone", "legged", "manipulation", "bci", "fluid"]:
        batch = make_batch(domain, B=2)
        keys = [f"{k}:{v.shape}" for k, v in batch.items() if hasattr(v, "shape")]
        print(f"  {domain:<14} {', '.join(keys)}")

    print("All checks passed.")
