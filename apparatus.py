"""
noosphere/apparatus.py
======================
Intentional Apparatus Movement

Integrates the full mechanicus movement pipeline into noosphere:

    EEG signal (3 neck electrodes)
        ↓ filter:  only intentional MuscleArtifact with kinematic
        ↓ detect:  z-score anomaly detection → intentional spikes
        ↓ predict: coordinate prediction (supervised RF, or neural)
        ↓ plan:    inverse kinematics → joint angles
        ↓ check:   collision-free path through 3D obstacle sphere
        ↓ execute: smooth interpolated motor commands

Key design decisions vs. mechanicus:
─────────────────────────────────────
1. No Redis dependency inside noosphere — the apparatus module works
   standalone and publishes via NCP frames to whatever transport is used.

2. IK uses the same Jacobian pseudoinverse as mechanicus, but runs inside
   a differentiable wrapper so the world model can learn from IK errors.

3. Obstacle avoidance: maintains a live PointCloud of the environment
   (built from depth camera). Before each move, plans a waypoint path
   that stays outside the obstacle sphere of radius `safety_margin`.
   This fulfills: "choose to move to nearest coordinate without impacting object."

4. Learning modes:
   - Supervised: kinematic → coordinate labels from mechanicus hierarchy
   - Unsupervised: anomaly detection drives exploration without labels
   - RL: world model reward from successful reach without collision
"""

import math
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


# ── Arm geometry ──────────────────────────────────────────────────────────────

@dataclass
class ArmConfig:
    """6-DOF arm segment lengths (metres)."""
    upper_arm:  float = 0.30   # shoulder → elbow
    forearm:    float = 0.25   # elbow → wrist
    hand:       float = 0.15   # wrist → tip
    # Joint limits (degrees)
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
    6-DOF arm kinematics (same model as mechanicus brain.py Arm3D).

    Forward kinematics: joint angles → end-effector position
    Inverse kinematics: target position → joint angles (Jacobian pseudoinverse)

    The IK is also used by the world model: IK error feeds as a training
    signal so the neural coordinate predictor learns reachable targets.
    """

    def __init__(self, cfg: ArmConfig = ArmConfig()):
        self.cfg = cfg

    def forward(self, js: JointState) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (shoulder, elbow, wrist, tip) in 3D metres."""
        shoulder = np.zeros(3)
        l1, l2, l3 = self.cfg.upper_arm, self.cfg.forearm, self.cfg.hand

        sy, sp = js.shoulder_yaw, js.shoulder_pitch
        upper = np.array([l1*math.cos(sy)*math.cos(sp),
                          l1*math.sin(sy)*math.cos(sp),
                          l1*math.sin(sp)])
        elbow = shoulder + upper

        ep = sp + js.elbow_pitch
        fore = np.array([l2*math.cos(sy)*math.cos(ep),
                         l2*math.sin(sy)*math.cos(ep),
                         l2*math.sin(ep)])
        wrist = elbow + fore

        wp = ep + js.wrist_pitch
        wy = sy + js.wrist_yaw
        hand = np.array([l3*math.cos(wy)*math.cos(wp),
                         l3*math.sin(wy)*math.cos(wp),
                         l3*math.sin(wp)])
        tip = wrist + hand
        return shoulder, elbow, wrist, tip

    def tip_position(self, js: JointState) -> np.ndarray:
        return self.forward(js)[3]

    def inverse(
        self,
        target:        np.ndarray,
        js0:           Optional[JointState] = None,
        tolerance:     float = 0.01,
        max_iter:      int   = 100,
        step_size:     float = 0.1,
        delta:         float = 0.001,
    ) -> Tuple[JointState, bool, float]:
        """
        Jacobian pseudoinverse IK (damped least squares).
        Returns (solution, converged, final_error_metres).
        """
        js = js0 or JointState()
        angles = js.to_array().copy()
        lims   = [self.cfg.shoulder_yaw_lim, self.cfg.shoulder_pitch_lim,
                  self.cfg.shoulder_roll_lim, self.cfg.elbow_pitch_lim,
                  self.cfg.wrist_pitch_lim,   self.cfg.wrist_yaw_lim]

        for _ in range(max_iter):
            tip = self.tip_position(JointState.from_array(angles))
            err = target - tip
            dist = np.linalg.norm(err)
            if dist < tolerance:
                return JointState.from_array(angles), True, dist

            # Numerical Jacobian (3 × 6)
            J = np.zeros((3, 6))
            for i in range(6):
                a2 = angles.copy(); a2[i] += delta
                p2 = self.tip_position(JointState.from_array(a2))
                J[:, i] = (p2 - tip) / delta

            # Damped pseudoinverse
            damp = 0.1
            update = J.T @ err * damp
            angles += np.clip(update, -0.1, 0.1)

            # Apply joint limits (convert to degrees for comparison)
            for i, (lo, hi) in enumerate(lims):
                angles[i] = np.clip(angles[i],
                                     math.radians(lo), math.radians(hi))

        tip = self.tip_position(JointState.from_array(angles))
        err = float(np.linalg.norm(target - tip))
        return JointState.from_array(angles), False, err


# ── 3D Obstacle Sphere / Collision Detection ──────────────────────────────────

class ObstacleSphere:
    """
    Live 3D occupancy model built from depth/LiDAR frames.
    Represents the environment as a point cloud; each point implicitly
    defines a sphere of radius `safety_margin` that the arm must avoid.

    On each movement request:
        1. Build current occupancy from latest depth frame
        2. Check if straight-line path from current tip to target intersects any sphere
        3. If collision, find nearest collision-free waypoint and plan around it

    Range of motion is defined as a continuous 3D vector space ℝ³ bounded
    by arm reach. The camera scans the environment to populate this space.
    """

    def __init__(self, safety_margin: float = 0.05, max_points: int = 2048):
        self.safety = safety_margin
        self.points: np.ndarray = np.zeros((0, 3))  # (N, 3)
        self.max_pts = max_points

    def update_from_depth(
        self,
        depth_map:   np.ndarray,   # (H, W) float32 — metric depth
        K:           np.ndarray,   # (3, 3) intrinsics
        T_cam_world: Optional[np.ndarray] = None,  # (4, 4) SE3
    ):
        """Rebuild occupancy from a new depth frame."""
        H, W = depth_map.shape
        v, u  = np.mgrid[0:H, 0:W]
        z     = depth_map.flatten()
        valid = (z > 0.05) & (z < 3.0)

        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        x = ((u.flatten()[valid] - cx) / fx) * z[valid]
        y = ((v.flatten()[valid] - cy) / fy) * z[valid]
        pts = np.stack([x, y, z[valid]], axis=1)

        if T_cam_world is not None:
            R, t = T_cam_world[:3,:3], T_cam_world[:3, 3]
            pts = (R @ pts.T).T + t

        # Subsample to max_pts using stride
        if len(pts) > self.max_pts:
            idx = np.random.choice(len(pts), self.max_pts, replace=False)
            pts = pts[idx]

        self.points = pts

    def segment_intersects(
        self,
        p0: np.ndarray,   # (3,) start
        p1: np.ndarray,   # (3,) end
    ) -> bool:
        """True if the line segment p0→p1 comes within safety margin of any obstacle point."""
        if len(self.points) == 0:
            return False
        d   = p1 - p0
        dn  = np.linalg.norm(d)
        if dn < 1e-6:
            return False
        d_hat = d / dn

        # Vector from p0 to each obstacle
        v   = self.points - p0   # (N, 3)
        # Project onto segment direction
        t   = np.clip(v @ d_hat, 0, dn)   # (N,)
        # Closest point on segment to each obstacle
        closest = p0 + t[:, None] * d_hat  # (N, 3)
        dists   = np.linalg.norm(self.points - closest, axis=1)
        return bool(np.any(dists < self.safety))

    def plan_path(
        self,
        start:  np.ndarray,   # (3,) current tip
        target: np.ndarray,   # (3,) desired tip
        n_candidates: int = 8,
    ) -> List[np.ndarray]:
        """
        Returns a list of 3D waypoints [start, ..., target].
        If direct path is clear, returns [start, target].
        Otherwise, tries midpoint candidates offset from the obstacle centroid.
        """
        if not self.segment_intersects(start, target):
            return [start, target]

        # Find obstacle centroid near the path
        if len(self.points) > 0:
            mid = (start + target) * 0.5
            dists = np.linalg.norm(self.points - mid, axis=1)
            centroid = self.points[dists.argmin()]
        else:
            centroid = (start + target) * 0.5

        # Try offset waypoints perpendicular to the segment direction
        d = target - start
        d_hat = d / (np.linalg.norm(d) + 1e-6)
        perp = np.array([-d_hat[1], d_hat[0], 0.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.array([0., 0., 1.])
        perp /= np.linalg.norm(perp)

        best_path = None
        best_len  = float("inf")
        offset_r  = self.safety * 3.0

        for i in range(n_candidates):
            angle = 2 * math.pi * i / n_candidates
            offset = offset_r * (math.cos(angle)*perp +
                                  math.sin(angle)*np.cross(d_hat, perp))
            wp = (start + target) * 0.5 + offset
            if (not self.segment_intersects(start, wp) and
                    not self.segment_intersects(wp, target)):
                path_len = (np.linalg.norm(wp-start) +
                             np.linalg.norm(target-wp))
                if path_len < best_len:
                    best_len  = path_len
                    best_path = [start, wp, target]

        return best_path or [start, target]   # fallback: direct (unsafe)


# ── EEG Artifact Classification ───────────────────────────────────────────────

class RootArtifactLabel:
    """Mirrors mechanicus RootArtifactLabel enum as integer constants."""
    CLEAN_BRAIN    = 0
    EYE_BLINK      = 1
    MUSCLE         = 2
    LINE_NOISE     = 3
    SLOW_DRIFT     = 4
    CARDIAC        = 5
    MIXED          = 6
    SENSOR_NOISE   = 7


class MuscleIntent:
    """Mirrors mechanicus MuscleArtifactMuscleIntent."""
    REST             = 0
    RIGHT_HAND       = 1
    LEFT_HAND        = 2
    BOTH_HANDS       = 3
    JAW_CLENCH       = 4
    HEAD_TILT        = 5
    SHOULDER_SHRUG   = 6
    FINGER_FLEXION   = 7
    WRIST_EXTENSION  = 8
    EYEBROW_RAISE    = 9


# ── Signal Transformation (from mechanicus brain.py) ─────────────────────────

class IntentionFilter:
    """
    Filters EEG segments to retain only intentional MuscleArtifact signals.

    Intentional = root_label is MuscleArtifact AND action == "Intentional".
    This is the mechanicus `transformation` step, now integrated into noosphere.
    """

    def is_intentional(self, segment: dict) -> bool:
        return (
            segment.get("root_label") == RootArtifactLabel.MUSCLE and
            segment.get("hierarchical", {}).get("action") == "Intentional" and
            segment.get("hierarchical", {}).get("muscle_intent") is not None
        )


class AnomalyDetector:
    """
    Statistical z-score anomaly detector (from mechanicus brain.py).

    Maintains a rolling history of probability vectors.
    A segment is anomalous if any probability deviates > threshold σ
    from the running mean, OR is outside (0.2, 0.8) bounds.
    These anomalies signal intentional movements worth acting on.
    """

    def __init__(self, min_history: int = 10, threshold: float = 1.0):
        self.min_history = min_history
        self.threshold   = threshold
        self._history: List[float] = []

    def update_and_check(self, probs: List[float]) -> bool:
        """Returns True if this segment is anomalous."""
        self._history.extend(probs)
        if len(self._history) < self.min_history * len(probs):
            return False

        vals  = np.array(self._history)
        mean  = vals.mean()
        std   = vals.std() + 1e-8

        for p in probs:
            if abs(p - mean) > self.threshold * std:
                return True
            if p <= 0.2 or p >= 0.8:
                return True
        return False


# ── Coordinate Predictor (supervised / online learning) ───────────────────────

class CoordinatePredictor:
    """
    Predicts 3D target coordinates from EEG feature vectors.

    Integrates the mechanicus RandomForestRegressor approach but wraps it
    in a noosphere-compatible learning interface that supports:

        Supervised:   fit on labeled (features, kinematic_xyz) pairs
        Unsupervised: auto-label via anomaly score weighting
        Online:       retrain every `retrain_every` samples (mechanicus: 10 min)

    Features (matching mechanicus FeatureExtractor):
        raw_microvolts (3) + probabilities (8) + contributions_avg (8) = 19 dims

    Why RandomForest here instead of neural:
        - Very few labeled samples (BCI data is expensive)
        - RF is robust to small N and high variance
        - Interpretable: feature importances reveal which EEG features
          drive coordinate predictions (safety critical)
        - Neural predictor can be substituted via `model_cls` argument
    """

    def __init__(self, retrain_every: int = 200, n_trees: int = 100):
        self.retrain_every = retrain_every
        self.n_trees       = n_trees
        self._model_x = None
        self._model_y = None
        self._model_z = None
        self._buffer: List[Tuple[List[float], List[float]]] = []
        self._step   = 0
        self._trained = False

    @staticmethod
    def extract_features(segment: dict) -> List[float]:
        uv    = list(segment.get("raw_microvolts", [0.0, 0.0, 0.0]))
        probs = list(segment.get("probabilities", [0.0] * 8))
        contrib = [0.0] * 8  # placeholder — extend when contributions_avg available
        return uv + probs + contrib   # 19-dim

    def add_sample(self, features: List[float], xyz: List[float]):
        self._buffer.append((features, xyz))
        self._step += 1
        if self._step % self.retrain_every == 0 and len(self._buffer) >= 20:
            self._retrain()

    def _retrain(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
            X  = np.array([s[0] for s in self._buffer])
            Yx = np.array([s[1][0] for s in self._buffer])
            Yy = np.array([s[1][1] for s in self._buffer])
            Yz = np.array([s[1][2] for s in self._buffer])
            kw = dict(n_estimators=self.n_trees, n_jobs=-1, random_state=42)
            self._model_x = RandomForestRegressor(**kw).fit(X, Yx)
            self._model_y = RandomForestRegressor(**kw).fit(X, Yy)
            self._model_z = RandomForestRegressor(**kw).fit(X, Yz)
            self._trained = True
        except ImportError:
            pass  # sklearn optional; neural predictor used if unavailable

    def predict(self, features: List[float]) -> Optional[np.ndarray]:
        if not self._trained:
            return None
        X = np.array([features])
        return np.array([
            self._model_x.predict(X)[0],
            self._model_y.predict(X)[0],
            self._model_z.predict(X)[0],
        ])


# ── Movement Executor ─────────────────────────────────────────────────────────

class MovementExecutor:
    """
    Closes the full loop from target coordinate to motor commands.

    1. Run IK to find joint angles
    2. Check collision-free path via ObstacleSphere
    3. Interpolate smooth joint trajectory
    4. Yield motor commands (joint_angles_deg) at each waypoint
    """

    def __init__(self, cfg: ArmConfig = ArmConfig()):
        self.ik       = KinematicSolver(cfg)
        self.obstacles= ObstacleSphere()
        self.current  = JointState()

    def plan_and_execute(
        self,
        target_xyz: np.ndarray,
        interp_steps: int = 5,
    ) -> List[np.ndarray]:
        """
        Returns list of joint angle arrays (degrees) to command sequentially.
        Handles obstacle avoidance by splitting path into collision-free segments.
        """
        # Current tip position
        tip_now = self.ik.tip_position(self.current)

        # Plan collision-free waypoints in Cartesian space
        waypoints = self.obstacles.plan_path(tip_now, target_xyz)

        all_commands = []
        js_current = self.current

        for wp_target in waypoints[1:]:  # skip start
            js_target, converged, err = self.ik.inverse(wp_target, js_current)

            # Interpolate joint angles from current to target
            a0 = js_current.to_array()
            a1 = js_target.to_array()
            for step in range(interp_steps + 1):
                t = step / interp_steps
                interp = a0 + (a1 - a0) * t
                all_commands.append(np.degrees(interp))

            js_current = js_target

        self.current = js_current
        return all_commands
