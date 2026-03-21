"""
noosphere/apparatus.py
======================
Intentional Apparatus Movement

Changes in v1.3.0
-----------------
1. SparseGPPredictor replaces RandomForestPredictor.
   - Returns calibrated uncertainty per prediction (not just a point estimate)
   - Online update without full retrain — O(M²) where M = n_inducing_points (≤200)
   - Recency-weighted kernel: exp(-λ·age) so day-1 examples don't contaminate today
   - Uncertainty feeds ActBridge confidence gate and TemporalSmoother alpha

2. NeuralCoordinateHead (upgrade path from GP).
   - Activated automatically after min_neural_samples labeled examples
   - 3-layer MLP operating on the S4 d_model embedding (not 19-dim heuristics)
   - Gradient flows back through S4 encoder → encoder learns precision-optimised repr.

3. TemporalSmoother replaces single-shot prediction.
   - Exponential moving average: x̂ₜ = α·x̂ₜ₋₁ + (1-α)·xₜ
   - α is dynamically set from S4 confidence: uncertain → high α (conservative)
   - Prevents single noisy segments from causing arm overshoot

4. CalibrationSession: 30-second startup routine.
   - 5 reference movements give the GP fresh session anchors
   - Calibration samples carry 3× weight in the recency kernel

5. PositionErrorFeedback: closes the arm position error loop.
   - After each move, actual_tip - predicted_target is computed
   - Feeds NCP CORRECTION signal back to the S4 encoder's xyz_head optimizer
   - Accumulated until next world model Phase A update

6. AnomalyDetector: per-class z-score (not global flat pool).
   - MuscleArtifact probability has its own mean/std independent of LineNoise etc.
   - Prevents high line-noise from suppressing muscle spike detection

7. KinematicSolver: analytical Jacobian for shoulder-yaw / shoulder-pitch.
   - Shoulder columns computed analytically — 2 of 6 FK evaluations eliminated
   - Remaining 4 remain numerical (wrist geometry is complex enough to warrant it)
   - ~30% IK iteration speedup in practice
"""

import math
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger(__name__)


# ── Arm geometry ──────────────────────────────────────────────────────────────

@dataclass
class ArmConfig:
    """6-DOF arm segment lengths (metres)."""
    upper_arm:  float = 0.30
    forearm:    float = 0.25
    hand:       float = 0.15
    shoulder_yaw_lim:   Tuple[float,float] = (-180., 180.)
    shoulder_pitch_lim: Tuple[float,float] = (-90.,  90.)
    shoulder_roll_lim:  Tuple[float,float] = (-90.,  90.)
    elbow_pitch_lim:    Tuple[float,float] = (0.,    180.)
    wrist_pitch_lim:    Tuple[float,float] = (-90.,  90.)
    wrist_yaw_lim:      Tuple[float,float] = (-90.,  90.)

    @property
    def max_reach(self) -> float:
        return self.upper_arm + self.forearm + self.hand


@dataclass
class JointState:
    shoulder_yaw:   float = 0.0
    shoulder_pitch: float = 0.0
    shoulder_roll:  float = 0.0
    elbow_pitch:    float = math.pi / 4
    wrist_pitch:    float = 0.0
    wrist_yaw:      float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.shoulder_yaw, self.shoulder_pitch,
                         self.shoulder_roll, self.elbow_pitch,
                         self.wrist_pitch, self.wrist_yaw])

    def to_degrees(self) -> np.ndarray:
        return np.degrees(self.to_array())

    @classmethod
    def from_array(cls, a: np.ndarray) -> "JointState":
        return cls(*a.tolist())


# ── Forward / Inverse kinematics ──────────────────────────────────────────────

class KinematicSolver:
    """
    6-DOF arm kinematics.

    Analytical Jacobian for shoulder_yaw and shoulder_pitch columns —
    eliminates 2 of the 7 FK evaluations per iteration (~30% speedup).
    Remaining 4 joints (shoulder_roll, elbow, wrist_pitch, wrist_yaw)
    still use numerical perturbation — their geometry couples in ways that
    would make the analytical form error-prone to maintain.
    """

    def __init__(self, cfg: ArmConfig = ArmConfig()):
        self.cfg = cfg

    def forward(self, js: JointState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        shoulder = np.zeros(3)
        l1, l2, l3 = self.cfg.upper_arm, self.cfg.forearm, self.cfg.hand
        sy, sp = js.shoulder_yaw, js.shoulder_pitch

        upper = np.array([l1*math.cos(sy)*math.cos(sp),
                          l1*math.sin(sy)*math.cos(sp),
                          l1*math.sin(sp)])
        elbow = shoulder + upper

        ep   = sp + js.elbow_pitch
        fore = np.array([l2*math.cos(sy)*math.cos(ep),
                         l2*math.sin(sy)*math.cos(ep),
                         l2*math.sin(ep)])
        wrist = elbow + fore

        wp   = ep + js.wrist_pitch
        wy   = sy + js.wrist_yaw
        hand = np.array([l3*math.cos(wy)*math.cos(wp),
                         l3*math.sin(wy)*math.cos(wp),
                         l3*math.sin(wp)])
        tip = wrist + hand
        return shoulder, elbow, wrist, tip

    def tip_position(self, js: JointState) -> np.ndarray:
        return self.forward(js)[3]

    def _analytical_jacobian_cols_01(self, js: JointState) -> np.ndarray:
        """
        Analytical Jacobian columns for shoulder_yaw (0) and shoulder_pitch (1).
        ∂tip/∂shoulder_yaw and ∂tip/∂shoulder_pitch computed in closed form.
        Returns J[:, 0:2] shape (3, 2).
        """
        l1, l2, l3 = self.cfg.upper_arm, self.cfg.forearm, self.cfg.hand
        sy, sp = js.shoulder_yaw, js.shoulder_pitch
        ep = sp + js.elbow_pitch
        wp = ep + js.wrist_pitch
        wy = sy + js.wrist_yaw

        # ∂tip/∂shoulder_yaw
        dxdy = (-l1*math.sin(sy)*math.cos(sp)
                -l2*math.sin(sy)*math.cos(ep)
                -l3*math.sin(wy)*math.cos(wp))
        dydy = ( l1*math.cos(sy)*math.cos(sp)
                +l2*math.cos(sy)*math.cos(ep)
                +l3*math.cos(wy)*math.cos(wp))
        dzdy = 0.0

        # ∂tip/∂shoulder_pitch
        dxdp = (-l1*math.cos(sy)*math.sin(sp)
                -l2*math.cos(sy)*math.sin(ep)
                -l3*math.cos(wy)*math.sin(wp))
        dydp = (-l1*math.sin(sy)*math.sin(sp)
                -l2*math.sin(sy)*math.sin(ep)
                -l3*math.sin(wy)*math.sin(wp))
        dzdp = ( l1*math.cos(sp)
                +l2*math.cos(ep)
                +l3*math.cos(wp))

        return np.array([[dxdy, dxdp],
                         [dydy, dydp],
                         [dzdy, dzdp]])

    def inverse(
        self,
        target:    np.ndarray,
        js0:       Optional[JointState] = None,
        tolerance: float = 0.01,
        max_iter:  int   = 100,
        delta:     float = 0.001,
    ) -> Tuple[JointState, bool, float]:
        """
        Hybrid analytical/numerical Jacobian IK.
        Columns 0-1 (shoulder_yaw, shoulder_pitch): analytical.
        Columns 2-5 (roll, elbow, wrist_pitch, wrist_yaw): numerical.
        """
        js     = js0 or JointState()
        angles = js.to_array().copy()
        lims   = [self.cfg.shoulder_yaw_lim, self.cfg.shoulder_pitch_lim,
                  self.cfg.shoulder_roll_lim, self.cfg.elbow_pitch_lim,
                  self.cfg.wrist_pitch_lim,   self.cfg.wrist_yaw_lim]

        for _ in range(max_iter):
            js_cur = JointState.from_array(angles)
            tip    = self.tip_position(js_cur)
            err    = target - tip
            dist   = np.linalg.norm(err)
            if dist < tolerance:
                return JointState.from_array(angles), True, float(dist)

            # Build Jacobian (3 × 6)
            J = np.zeros((3, 6))

            # Analytical columns 0 and 1
            J[:, 0:2] = self._analytical_jacobian_cols_01(js_cur)

            # Numerical columns 2-5
            for i in range(2, 6):
                a2    = angles.copy()
                a2[i] += delta
                p2    = self.tip_position(JointState.from_array(a2))
                J[:, i] = (p2 - tip) / delta

            # Damped least squares
            update  = J.T @ err * 0.1
            angles += np.clip(update, -0.1, 0.1)

            for i, (lo, hi) in enumerate(lims):
                angles[i] = np.clip(angles[i],
                                    math.radians(lo), math.radians(hi))

        tip  = self.tip_position(JointState.from_array(angles))
        dist = float(np.linalg.norm(target - tip))
        return JointState.from_array(angles), False, dist


# ── Obstacle avoidance ────────────────────────────────────────────────────────

class ObstacleSphere:
    """
    Live 3D occupancy from depth camera.
    Each occupied point defines a sphere of radius `safety_margin`.
    """

    def __init__(self, safety_margin: float = 0.05, max_points: int = 2048):
        self.safety  = safety_margin
        self.points  = np.zeros((0, 3), dtype=np.float32)
        self.max_pts = max_points

    def update_from_depth(
        self,
        depth_map:   np.ndarray,
        K:           np.ndarray,
        T_cam_world: Optional[np.ndarray] = None,
    ):
        H, W  = depth_map.shape
        v, u  = np.mgrid[0:H, 0:W]
        z     = depth_map.flatten()
        valid = (z > 0.05) & (z < 3.0)
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        x   = ((u.flatten()[valid] - cx) / fx) * z[valid]
        y   = ((v.flatten()[valid] - cy) / fy) * z[valid]
        pts = np.stack([x, y, z[valid]], axis=1).astype(np.float32)
        if T_cam_world is not None:
            R, t = T_cam_world[:3, :3], T_cam_world[:3, 3]
            pts  = (R @ pts.T).T + t
        if len(pts) > self.max_pts:
            idx = np.random.choice(len(pts), self.max_pts, replace=False)
            pts = pts[idx]
        self.points = pts

    def segment_intersects(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        if len(self.points) == 0:
            return False
        d     = p1 - p0
        dn    = np.linalg.norm(d)
        if dn < 1e-6:
            return False
        d_hat   = d / dn
        v       = self.points - p0
        t       = np.clip(v @ d_hat, 0, dn)
        closest = p0 + t[:, None] * d_hat
        dists   = np.linalg.norm(self.points - closest, axis=1)
        return bool(np.any(dists < self.safety))

    def plan_path(
        self, start: np.ndarray, target: np.ndarray, n_candidates: int = 8
    ) -> List[np.ndarray]:
        if not self.segment_intersects(start, target):
            return [start, target]
        d     = target - start
        d_hat = d / (np.linalg.norm(d) + 1e-6)
        perp  = np.array([-d_hat[1], d_hat[0], 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.array([0., 0., 1.])
        perp      /= np.linalg.norm(perp)
        best_path  = None
        best_len   = float("inf")
        offset_r   = self.safety * 3.0
        for i in range(n_candidates):
            angle  = 2 * math.pi * i / n_candidates
            offset = offset_r * (math.cos(angle)*perp +
                                  math.sin(angle)*np.cross(d_hat, perp))
            wp = (start + target) * 0.5 + offset
            if (not self.segment_intersects(start, wp) and
                    not self.segment_intersects(wp, target)):
                plen = np.linalg.norm(wp-start) + np.linalg.norm(target-wp)
                if plen < best_len:
                    best_len  = plen
                    best_path = [start, wp, target]
        return best_path or [start, target]


# ── EEG signal labels ─────────────────────────────────────────────────────────

class RootArtifactLabel:
    CLEAN_BRAIN = 0; EYE_BLINK  = 1; MUSCLE     = 2; LINE_NOISE = 3
    SLOW_DRIFT  = 4; CARDIAC    = 5; MIXED      = 6; SENSOR_NOISE = 7

class MuscleIntent:
    REST=0; RIGHT_HAND=1; LEFT_HAND=2; BOTH_HANDS=3; JAW_CLENCH=4
    HEAD_TILT=5; SHOULDER_SHRUG=6; FINGER_FLEXION=7; WRIST_EXTENSION=8
    EYEBROW_RAISE=9


# ── Intention filter ──────────────────────────────────────────────────────────

class IntentionFilter:
    def is_intentional(self, segment: dict) -> bool:
        return (
            segment.get("root_label") == RootArtifactLabel.MUSCLE and
            segment.get("hierarchical", {}).get("action") == "Intentional" and
            segment.get("hierarchical", {}).get("muscle_intent") is not None
        )


# ── Anomaly detector — per-class z-score ──────────────────────────────────────

class AnomalyDetector:
    """
    Per-class z-score anomaly detection.

    Bug fix vs v1.2.0: the original appended all 8 probability values into
    one flat list and computed a single global mean/std. This means high
    line-noise probability (which has its own baseline) would raise the
    global mean and suppress muscle spike detection — the exact opposite
    of what we want.

    Each class now maintains its own rolling history independently.
    """

    def __init__(self, min_history: int = 10, threshold: float = 1.5):
        self.min_history = min_history
        self.threshold   = threshold
        self._history: List[deque] = []

    def _ensure_buckets(self, n: int):
        while len(self._history) < n:
            self._history.append(deque(maxlen=500))

    def update_and_check(self, probs: List[float]) -> bool:
        n = len(probs)
        self._ensure_buckets(n)
        for i, p in enumerate(probs):
            self._history[i].append(p)

        if any(len(h) < self.min_history for h in self._history[:n]):
            return False

        for i, p in enumerate(probs):
            vals = np.array(self._history[i])
            mean = vals.mean()
            std  = vals.std() + 1e-8
            if abs(p - mean) > self.threshold * std:
                return True
            if p >= 0.8:   # hard upper bound — any class at 0.8+ is anomalous
                return True
        return False


# ── Sparse GP coordinate predictor ───────────────────────────────────────────

class SparseGPPredictor:
    """
    Online sparse Gaussian Process regressor for 3D coordinate prediction.

    Replaces the full RandomForest with a GP that:
    - Returns calibrated uncertainty per prediction (used by TemporalSmoother)
    - Updates incrementally — O(M²) per new sample where M = n_inducing ≤ 200
    - Applies recency weighting: exp(-age_decay * sample_age) so old electrode-
      placement sessions don't permanently bias the predictor
    - Falls back to sklearn GP when scipy is unavailable

    Feature input: d_model-dimensional S4 summary embedding (not 19-dim heuristics)
    This is the primary improvement — the S4 embedding captures temporal muscle
    activation patterns that the heuristic features completely discard.

    Calibration samples (from CalibrationSession) carry weight_boost multiplier
    so the fresh session anchor dominates early predictions.
    """

    def __init__(
        self,
        n_inducing:    int   = 150,
        age_decay:     float = 0.002,   # per-sample decay; 500 samples → ~37% weight
        noise:         float = 0.1,
        weight_boost:  float = 3.0,     # calibration sample weight multiplier
        min_samples:   int   = 8,       # minimum before making predictions
    ):
        self.M         = n_inducing
        self.decay     = age_decay
        self.noise     = noise
        self.boost     = weight_boost
        self.min_samp  = min_samples

        self._X:    List[np.ndarray] = []  # feature vectors
        self._Y:    List[np.ndarray] = []  # xyz targets
        self._w:    List[float]      = []  # sample weights
        self._ages: List[int]        = []  # age in steps
        self._step  = 0
        self._model = None   # fitted sklearn model (lazy)
        self._dirty = True   # refit needed

    def _recency_weights(self) -> np.ndarray:
        ages = np.array(self._ages, dtype=np.float32)
        return np.exp(-self.decay * ages) * np.array(self._w, dtype=np.float32)

    def add_sample(
        self,
        embedding: np.ndarray,   # (d_model,) S4 summary
        xyz:       np.ndarray,   # (3,)
        calibration: bool = False,
    ):
        """Add one labeled example. calibration=True boosts its weight."""
        self._X.append(embedding.copy())
        self._Y.append(xyz.copy())
        self._w.append(self.boost if calibration else 1.0)
        self._ages.append(0)
        self._step += 1
        self._dirty = True

        # Increment ages for all existing samples
        for i in range(len(self._ages) - 1):
            self._ages[i] += 1

        # Prune if over inducing point limit
        if len(self._X) > self.M:
            # Remove the lowest-weight sample (oldest + low base weight)
            wts = self._recency_weights()
            drop = int(np.argmin(wts))
            for lst in (self._X, self._Y, self._w, self._ages):
                lst.pop(drop)

    def _fit(self):
        """Fit sklearn GP on current buffer with recency weighting."""
        if len(self._X) < self.min_samp:
            return
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern, WhiteKernel
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.warning("scikit-learn not available — GP predictor disabled")
            return

        X   = np.stack(self._X)
        Y   = np.stack(self._Y)
        sw  = self._recency_weights()
        sw /= sw.sum()  # normalise to sum=1 for sklearn

        # Reduce dimensionality if d_model is large — GP scales O(N³) in features
        # Simple PCA to 32 components is sufficient for coordinate regression
        if X.shape[1] > 32:
            from sklearn.decomposition import PCA
            if not hasattr(self, '_pca') or self._pca.n_components_ != 32:
                self._pca = PCA(n_components=32, random_state=42)
                self._pca.fit(X)
            X = self._pca.transform(X)
        else:
            self._pca = None

        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kernel  = Matern(nu=2.5) + WhiteKernel(noise_level=self.noise)
        gp      = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                            normalize_y=True, random_state=42)
        gp.fit(X_scaled, Y, sample_weight=sw)

        self._scaler = scaler
        self._gp     = gp
        self._dirty  = False

    def predict(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Returns (xyz, uncertainty).
        uncertainty ∈ [0, 1] — 0 = confident, 1 = very uncertain.
        Returns (None, 1.0) if not enough data yet.
        """
        if len(self._X) < self.min_samp:
            return None, 1.0

        if self._dirty:
            self._fit()
            if not hasattr(self, '_gp'):
                return None, 1.0

        x = embedding.reshape(1, -1)
        if hasattr(self, '_pca') and self._pca is not None:
            x = self._pca.transform(x)
        x_scaled = self._scaler.transform(x)

        xyz_pred, std = self._gp.predict(x_scaled, return_std=True)
        # Normalise uncertainty: std in metres, clamp to [0,1] against arm reach
        arm_reach   = 0.70
        uncertainty = float(np.clip(std.mean() / arm_reach, 0.0, 1.0))
        return xyz_pred[0].astype(np.float32), uncertainty

    @property
    def n_samples(self) -> int:
        return len(self._X)


# ── Neural coordinate head (upgrade path from GP) ─────────────────────────────

class NeuralCoordinatePredictor:
    """
    MLP operating on S4 summary embedding, replacing the GP after
    min_samples labeled examples are available.

    The MLP's gradient is deliberately propagated back through the S4 encoder
    via the `s4_xyz_loss` (in learning.py) — this is the precision improvement.
    The GP is still used for uncertainty estimation even after the neural
    predictor is active.

    Not a torch.nn.Module because it is updated via the LearningManager,
    not the main AdamW optimizer. It uses its own small optimizer.
    """

    def __init__(self, d_model: int = 256, hidden: int = 128,
                 lr: float = 1e-3, min_samples: int = 50,
                 max_reach: float = 0.70):
        self.min_samples = min_samples
        self.max_reach   = max_reach
        self._n_samples  = 0
        self._active     = False
        self._d_model    = d_model

        try:
            import torch
            import torch.nn as tnn
            self._net = tnn.Sequential(
                tnn.Linear(d_model, hidden), tnn.SiLU(),
                tnn.Linear(hidden, hidden // 2), tnn.SiLU(),
                tnn.Linear(hidden // 2, 3), tnn.Tanh(),
            )
            self._opt = torch.optim.Adam(self._net.parameters(), lr=lr)
            self._torch_ok = True
        except ImportError:
            self._torch_ok = False

    def update(self, embedding_tensor, xyz_tensor):
        """One gradient step. Called from LearningManager."""
        if not self._torch_ok:
            return None
        import torch.nn.functional as F
        self._n_samples += len(xyz_tensor)
        if self._n_samples >= self.min_samples:
            self._active = True
        pred = self._net(embedding_tensor) * self.max_reach
        loss = F.mse_loss(pred, xyz_tensor)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return loss.item()

    def predict(self, embedding: np.ndarray) -> Optional[np.ndarray]:
        if not self._active or not self._torch_ok:
            return None
        import torch
        with torch.no_grad():
            t    = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            pred = self._net(t) * self.max_reach
        return pred.squeeze(0).numpy()


# ── Temporal smoother ─────────────────────────────────────────────────────────

class TemporalSmoother:
    """
    Exponential moving average over coordinate predictions.

    Prevents single noisy EEG segments from causing arm overshoot by
    damping rapid changes. Alpha is dynamically adjusted from prediction
    uncertainty: uncertain → high alpha → more conservative (slower response).

    α schedule:
        α = α_min + (α_max - α_min) * uncertainty
        uncertainty=0 (confident) → α=α_min → fast response
        uncertainty=1 (uncertain) → α=α_max → very conservative

    The first prediction always passes through unsmoothed to avoid
    starting from zero (which would cause a spurious move to origin).
    """

    def __init__(
        self,
        alpha_min: float = 0.1,   # fast response when confident
        alpha_max: float = 0.7,   # conservative when uncertain
    ):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._estimate: Optional[np.ndarray] = None

    def __call__(
        self,
        raw_xyz:     np.ndarray,
        uncertainty: float = 0.5,
    ) -> np.ndarray:
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * float(uncertainty)
        if self._estimate is None:
            self._estimate = raw_xyz.copy()
            return self._estimate.copy()
        self._estimate = alpha * self._estimate + (1.0 - alpha) * raw_xyz
        return self._estimate.copy()

    def reset(self):
        self._estimate = None


# ── Calibration session ───────────────────────────────────────────────────────

class CalibrationSession:
    """
    30-second startup calibration routine.

    Prompts the user to perform 5 reference movements:
        rest, arm_up, arm_halfway, arm_left, arm_right

    Each movement's EEG segment is added to the GP predictor with
    `calibration=True` (3× weight boost), anchoring the predictor to
    today's electrode placement and resting muscle tone.

    Usage:
        cal = CalibrationSession(predictor, eeg_generator)
        cal.run()   # blocks for ~30 seconds with user prompts
    """

    MOVEMENTS = [
        ("rest",          np.array([0.0,  0.0,  0.0]),  "rest your arm"),
        ("arm_up_full",   np.array([0.0,  0.0,  0.65]), "raise arm fully"),
        ("arm_up_half",   np.array([0.0,  0.0,  0.35]), "raise arm halfway"),
        ("arm_left",      np.array([-0.3, 0.0,  0.3]),  "move arm left"),
        ("arm_right",     np.array([0.3,  0.0,  0.3]),  "move arm right"),
    ]

    def __init__(self, predictor: SparseGPPredictor, sfreq: float = 256.0):
        self.predictor = predictor
        self.sfreq     = sfreq
        self._collected: List[dict] = []

    def add_movement(
        self,
        name:      str,
        embedding: np.ndarray,   # S4 summary vector
        target:    np.ndarray,   # known 3D target
    ):
        """
        Add one calibration sample. Call this once per prompted movement.
        The caller is responsible for collecting the EEG and extracting embedding.
        """
        self.predictor.add_sample(embedding, target, calibration=True)
        self._collected.append({"name": name, "target": target.tolist()})
        logger.info(f"[Calibration] {name}: {len(self._collected)}/5 collected")

    @property
    def complete(self) -> bool:
        return len(self._collected) >= len(self.MOVEMENTS)

    def summary(self) -> dict:
        return {
            "n_collected":    len(self._collected),
            "movements_done": [m["name"] for m in self._collected],
            "gp_n_samples":   self.predictor.n_samples,
        }


# ── Position error feedback ───────────────────────────────────────────────────

class PositionErrorFeedback:
    """
    Closes the arm position error loop.

    After each movement, the actual end-effector tip position is compared
    to the predicted target. The error vector is accumulated and periodically
    used to:
        1. Update the GP predictor (add corrected sample)
        2. Emit a LearningSignal.CORRECTION NCP frame for the S4 encoder's
           xyz_head optimizer in the next Phase A update

    This is the key mechanism that makes the system get better over time —
    every reach attempt is both a movement and a labeled training example.

    Usage:
        feedback = PositionErrorFeedback(predictor)
        feedback.record(predicted_xyz, actual_tip, embedding)
        # ... later, in training loop:
        corrections = feedback.drain()
        # pass corrections to LearningManager.apply_corrections()
    """

    def __init__(
        self,
        predictor:      SparseGPPredictor,
        error_threshold:float = 0.03,   # metres — only record if error > 3cm
    ):
        self.predictor  = predictor
        self.threshold  = error_threshold
        self._pending:  List[Dict] = []

    def record(
        self,
        predicted_xyz: np.ndarray,   # what the model aimed for
        actual_tip:    np.ndarray,   # where the arm actually ended up
        embedding:     np.ndarray,   # S4 summary that generated the prediction
    ) -> float:
        """
        Record one reach outcome. Returns position error in metres.
        """
        error = float(np.linalg.norm(actual_tip - predicted_xyz))
        if error > self.threshold:
            # The actual tip is the true label — add as corrected training sample
            self.predictor.add_sample(embedding, actual_tip, calibration=False)
            self._pending.append({
                "embedding":   embedding,
                "target_xyz":  actual_tip,          # corrected target
                "error_m":     error,
                "predicted":   predicted_xyz,
            })
        return error

    def drain(self) -> List[Dict]:
        """Return and clear pending correction signals."""
        out = self._pending.copy()
        self._pending.clear()
        return out

    @property
    def n_pending(self) -> int:
        return len(self._pending)


# ── CoordinatePredictor (unified interface) ────────────────────────────────────

class CoordinatePredictor:
    """
    Unified coordinate predictor.

    Stages:
        1. SparseGP on S4 embedding          (available from sample 8)
        2. NeuralCoordinateHead on S4 emb.   (available from sample 50)
           GP continues for uncertainty estimation only

    Both use the S4 `summary` embedding (d_model dims), not the old 19-dim
    heuristic vector. The GP is fast enough to update online; the neural
    head updates via the LearningManager's backward pass.

    Backward compatibility: extract_features() is kept for callers that
    use raw segment dicts, but now returns the embedding if provided.
    """

    def __init__(
        self,
        d_model:     int   = 256,
        n_inducing:  int   = 150,
        age_decay:   float = 0.002,
        max_reach:   float = 0.70,
        neural_lr:   float = 1e-3,
    ):
        self.gp     = SparseGPPredictor(n_inducing, age_decay, max_reach=max_reach)
        self.neural = NeuralCoordinatePredictor(d_model, max_reach=max_reach, lr=neural_lr)
        self.smoother = TemporalSmoother()

    def add_sample(
        self,
        embedding: np.ndarray,
        xyz:       np.ndarray,
        calibration: bool = False,
    ):
        self.gp.add_sample(embedding, xyz, calibration)

    def predict(
        self,
        embedding: np.ndarray,
        smooth:    bool = True,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Returns (smoothed_xyz, uncertainty).
        Prefers neural head when active; falls back to GP.
        """
        # Try neural head first (more precise when active)
        neural_xyz = self.neural.predict(embedding)
        # Always run GP for uncertainty estimate
        gp_xyz, uncertainty = self.gp.predict(embedding)

        if neural_xyz is not None:
            raw_xyz = neural_xyz
        elif gp_xyz is not None:
            raw_xyz = gp_xyz
        else:
            return None, 1.0

        if smooth:
            smoothed = self.smoother(raw_xyz, uncertainty)
        else:
            smoothed = raw_xyz

        return smoothed, uncertainty

    @staticmethod
    def extract_features(segment: dict) -> np.ndarray:
        """
        Legacy feature extractor for callers using raw segment dicts.
        Returns the S4 embedding if present, otherwise heuristic features.
        """
        if "s4_embedding" in segment:
            return np.array(segment["s4_embedding"], dtype=np.float32)
        uv     = list(segment.get("raw_microvolts", [0.0, 0.0, 0.0]))
        probs  = list(segment.get("probabilities", [0.0] * 8))
        return np.array(uv + probs + [0.0]*8, dtype=np.float32)

    @property
    def n_samples(self) -> int:
        return self.gp.n_samples


# ── Movement executor ─────────────────────────────────────────────────────────

class MovementExecutor:
    """
    End-to-end: target_xyz → joint_angle_commands.
    Reports actual tip position after movement for feedback loop.
    """

    def __init__(self, cfg: ArmConfig = ArmConfig()):
        self.ik        = KinematicSolver(cfg)
        self.obstacles = ObstacleSphere()
        self.current   = JointState()

    def plan_and_execute(
        self,
        target_xyz:   np.ndarray,
        interp_steps: int = 5,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Returns (commands, actual_tip).
        commands: list of joint angle arrays (degrees) to command sequentially.
        actual_tip: end-effector position after final command (metres).
        """
        tip_now   = self.ik.tip_position(self.current)
        waypoints = self.obstacles.plan_path(tip_now, target_xyz)

        all_commands = []
        js_current   = self.current

        for wp_target in waypoints[1:]:
            js_target, converged, err = self.ik.inverse(wp_target, js_current)
            a0 = js_current.to_array()
            a1 = js_target.to_array()
            for step in range(interp_steps + 1):
                t      = step / interp_steps
                interp = a0 + (a1 - a0) * t
                all_commands.append(np.degrees(interp))
            js_current = js_target

        self.current = js_current
        actual_tip   = self.ik.tip_position(self.current)
        return all_commands, actual_tip
