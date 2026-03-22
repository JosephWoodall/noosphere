"""
noosphere/data/synth.py
=======================
Synthetic Test Data — All Modalities

Single file for all synthetic sensor data generation.
Replace individual generators with real hardware drivers as you integrate
physical systems. All generators produce NumPy arrays in the format
expected by NoosphereAgent.step(obs).

EEG configuration:
    3 electrodes on the back of the neck.
    Posterior-neck placement means:
        - Strong EMG (neck muscle artifacts) — this IS the intentional signal
        - Limited alpha (some posterior occipital leakage)
        - Minimal frontal artifacts (eye blinks attenuated ~90%)
        - Good access to: neck muscle contraction, shoulder/head movement intent

    This is fundamentally different from standard 64-channel EEG.
    The signal-to-noise profile is inverted: what is noise in clinic EEG
    (neck EMG) is signal here. The intentional movement IS the muscle artifact.

Artifact hierarchy (from mechanicus, adapted for 3-ch neck):
    RootArtifactLabel:
        CleanBrain      — resting baseline, alpha
        EyeBlink        — highly attenuated at neck (small amplitude)
        MuscleArtifact  — DOMINANT intentional signal (high amplitude)
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


class NeckEEGGenerator:
    """
    Realistic synthetic EEG from 3 neck electrodes.

    Electrode layout (posterior neck):
        Ch0: Left mastoid / posterior cervical (C7-left)
        Ch1: Central posterior cervical (C7 midline)
        Ch2: Right mastoid / posterior cervical (C7-right)

    Key physiological properties of neck EEG:
        - EMG bandwidth: 20–200 Hz (dominant at this site)
        - Alpha (8–13 Hz): weak leakage from occipital, amplitude ~5–15 μV
        - Muscle artifacts: 20–150 Hz, amplitude 50–300 μV during contraction
        - Line noise: 60 Hz, amplitude 8–15 μV
        - DC drift: slow, 10–50 μV amplitude
        - Baseline noise: ~8 μV RMS (higher than scalp due to hair/skin contact)

    Mirrors mechanicus RealWorldEEG but adapted for 3-ch neck placement.
    """

    SAMPLE_RATE = 256  # Hz

    # Root artifact labels (matches mechanicus RootArtifactLabel)
    LABEL_CLEAN_BRAIN = 0
    LABEL_EYE_BLINK = 1
    LABEL_MUSCLE = 2
    LABEL_LINE_NOISE = 3
    LABEL_SLOW_DRIFT = 4
    LABEL_CARDIAC = 5
    LABEL_MIXED = 6
    LABEL_SENSOR_NOISE = 7

    # Muscle intent labels (matches mechanicus MuscleArtifactMuscleIntent)
    INTENT_REST = 0
    INTENT_RIGHT_HAND = 1
    INTENT_LEFT_HAND = 2
    INTENT_BOTH_HANDS = 3
    INTENT_JAW_CLENCH = 4
    INTENT_HEAD_TILT = 5
    INTENT_SHOULDER_SHRUG = 6
    INTENT_FINGER_FLEX = 7
    INTENT_WRIST_EXT = 8
    INTENT_EYEBROW_RAISE = 9

    def __init__(self, seed: Optional[int] = 42):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.dt = 1.0 / self.SAMPLE_RATE

        # State
        self._env_alpha = np.zeros(3)
        self._blink_phase = 128
        self._muscle_remaining = 0
        self._muscle_freq = 50.0
        self._muscle_intent = self.INTENT_REST
        self._next_blink = 2.0

        # Per-channel parameters (neck-adapted)
        self._line_amp = self.rng.uniform(8.0, 15.0, size=3)  # μV
        self._drift_amp = self.rng.uniform(10.0, 50.0, size=3)  # μV
        self._noise_std = self.rng.uniform(6.0, 10.0, size=3)  # μV (higher than scalp)
        self._muscle_amp = np.array([120.0, 200.0, 130.0])  # μV (strong at neck)

        # Blink buffer (Gaussian kernel)
        sigma = 0.08 * self.SAMPLE_RATE / 6.0
        buf = np.exp(-0.5 * ((np.arange(128) - 64) / sigma) ** 2)
        self._blink_buf = buf
        # Blink amplitude at neck is ~10% of frontal
        self._blink_amp_neck = np.array([35.0, 20.0, 35.0])  # μV, symmetric at neck

    def _next_sample(self) -> np.ndarray:
        raw = np.zeros(3)
        t = self.t
        self.t += self.dt

        for ch in range(3):
            # 1. Slow electrode drift
            drift = self._drift_amp[ch] * 0.4 * math.sin(t / 20.0)

            # 2. 60 Hz line noise
            line = self._line_amp[ch] * math.sin(2 * math.pi * 60 * t)

            # 3. Eye blink (weak at neck — ~10% of frontal)
            blink = 0.0
            if self._blink_phase < 128:
                blink = self._blink_amp_neck[ch] * self._blink_buf[self._blink_phase]
                if ch == 2:
                    self._blink_phase += 1  # advance once per sample
            if t >= self._next_blink:
                self._blink_phase = 0
                self._next_blink = t + float(
                    np.clip(self.rng.exponential(1.0 / 0.2), 0.8, 12.0)
                )

            # 4. Neck muscle artifact (DOMINANT intentional signal)
            muscle = 0.0
            if self._muscle_remaining == 0 and self.rng.random() < 0.002:
                self._muscle_remaining = self.rng.integers(30, 200)
                self._muscle_freq = self.rng.uniform(30.0, 150.0)
            if self._muscle_remaining > 0:
                self._muscle_remaining -= 1
                muscle = self._muscle_amp[ch] * math.sin(
                    2 * math.pi * self._muscle_freq * t
                )

            # 5. Alpha rhythm (weak occipital leakage)
            self._env_alpha[ch] += self.rng.uniform(-0.8, 0.8)
            self._env_alpha[ch] *= 0.997
            # Symmetric at neck (no occipital asymmetry)
            alpha_scale = 0.3
            alpha = (
                8.0
                * alpha_scale
                * self._env_alpha[ch]
                * math.sin(2 * math.pi * 10.5 * t)
            )

            # 6. Sensor noise (higher at neck due to hair/skin)
            noise = self.rng.standard_normal() * self._noise_std[ch]

            raw[ch] = drift + line + blink + muscle + alpha + noise

        return raw

    def next_segment(
        self,
        intent: Optional[int] = None,
        n_samples: int = 256,
    ) -> Dict:
        """
        Generate one labeled EEG segment (n_samples at 256 Hz = 1 second).

        Parameters
        ----------
        intent: Optional[int]
            If set, forces this muscle intent during the segment (simulation
            of intentional movement). If None, random organic dynamics.
        n_samples: int
            Samples per segment. Default 256 = 1 second.

        Returns
        -------
        dict with keys matching SegmentLabel from mechanicus:
            raw_microvolts  np.ndarray (3,)       — per-channel mean
            eeg             np.ndarray (3, T)      — raw signal for S4 encoder
            root_label      int
            hierarchical    dict
            probabilities   np.ndarray (8,)
            timestamp       float
        """
        if intent is not None:
            self._muscle_remaining = n_samples
            self._muscle_freq = self.rng.uniform(30.0, 120.0)
            self._muscle_intent = intent

        buf = np.zeros((3, n_samples))
        for i in range(n_samples):
            buf[:, i] = self._next_sample()

        t_mid = self.t - n_samples * self.dt / 2.0
        avg = buf.mean(axis=1)

        # Feature extraction for label assignment
        var = buf[1].var()  # central channel variance
        kurt = (((buf[1] - buf[1].mean()) ** 4).mean()) / max(var**2, 1.0)
        line_power = buf.mean(axis=0)  # rough proxy

        # Probabilities (rough heuristic, mirrors mechanicus)
        probs = np.zeros(8)
        probs[self.LABEL_MUSCLE] = min(1.0, var / 3000.0)
        probs[self.LABEL_EYE_BLINK] = min(0.3, max(0.0, avg[0] / 150.0))  # attenuated
        probs[self.LABEL_LINE_NOISE] = min(0.8, self._line_amp.mean() / 20.0)
        probs[self.LABEL_CLEAN_BRAIN] = min(1.0, abs(avg[1]) / 20.0)
        probs[self.LABEL_SLOW_DRIFT] = min(0.6, abs(buf[0, 0] - buf[0, -1]) / 60.0)
        s = probs.sum()
        if s > 0:
            probs /= s

        dom = int(probs.argmax())

        # Kinematic (random plausible coordinates for intentional movements)
        kinematic = None
        m_intent = None
        action = None

        if dom == self.LABEL_MUSCLE and var > 1500.0:
            m_intent = (
                self._muscle_intent
                if intent is not None
                else int(self.rng.integers(1, 10))
            )
            kinematic = {
                "x": float(self.rng.uniform(-0.4, 0.4)),
                "y": float(self.rng.uniform(-0.4, 0.4)),
                "z": float(self.rng.uniform(0.0, 0.60)),  # positive z (above shoulder)
                "velocity": float(self.rng.uniform(0.0, 2.0)),
                "force": float(self.rng.uniform(0.0, 1.0)),
            }
            action = "Intentional"

        return {
            "raw_microvolts": avg,
            "eeg": buf.astype(np.float32),  # (3, T) for S4 encoder
            "root_label": dom,
            "hierarchical": {
                "root": dom,
                "muscle_intent": m_intent,
                "action": action,
                "kinematic": kinematic,
                "custom_tag": "NeckElectrode",
            },
            "probabilities": probs,
            "timestamp": t_mid,
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
    eeg_gen: Optional[NeckEEGGenerator] = None,
) -> Dict:
    """
    BCI-controlled apparatus observation.
    Sensors: 3-electrode neck EEG (256 samples), visual feedback, system state.

    intent: force a specific MuscleIntent for testing supervised learning.
    eeg_gen: reuse existing generator for temporal continuity.
    """
    gen = eeg_gen or NeckEEGGenerator(seed)
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
    eeg_gen: Optional[NeckEEGGenerator] = None,
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
        gen = eeg_gen or NeckEEGGenerator(seed)
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

    gen = NeckEEGGenerator(seed=42)
    for intent_name, intent_id in [
        ("REST", NeckEEGGenerator.INTENT_REST),
        ("RIGHT_HAND", NeckEEGGenerator.INTENT_RIGHT_HAND),
        ("SHOULDER_SHRUG", NeckEEGGenerator.INTENT_SHOULDER_SHRUG),
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
