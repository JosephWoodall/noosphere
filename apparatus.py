"""
noosphere/apparatus.py
======================
Intentional Apparatus Movement 

Features:
- TemporalSmoother: Dynamically adjusts movement fluidity based on S4 confidence. 
  Low confidence smoothly halts the arm; high confidence allows crisp, responsive tracking.
- ObstacleSphere: Local safety gate preventing the IK solver from driving the arm 
  into the desk or the user, analogous to the World Model's termination predictor.
- MovementExecutor: The continuous physical executor. Closes the loop by returning 
  the 'actual_tip' position so the LearningManager can compute PositionErrorFeedback.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

# ── Configuration & State ─────────────────────────────────────────────────────

@dataclass
class ArmConfig:
    """Physical parameters of the 6-DOF robotic limb."""
    link_lengths: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.25, 0.10])
    max_reach: float = 0.70
    joint_limits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-180.0, 180.0), # Base Pan
        (-90.0, 90.0),   # Shoulder Pitch
        (-180.0, 180.0), # Elbow Pitch
        (-90.0, 90.0),   # Wrist Pitch
        (-180.0, 180.0), # Wrist Roll
        (-20.0, 20.0),   # Gripper
    ])

class JointState:
    """Represents the current angular configuration of the apparatus."""
    def __init__(self, angles: Optional[np.ndarray] = None):
        self.angles = angles if angles is not None else np.zeros(6, dtype=np.float32)

    def to_array(self) -> np.ndarray:
        return self.angles.copy()

# ── Safety & Smoothing ────────────────────────────────────────────────────────

class ObstacleSphere:
    """
    Local physical safety bounds. Prevents trajectories that intersect with 
    known physical exclusions (e.g., the table surface, the user's body).
    """
    def __init__(self):
        # Format: (x, y, z, radius)
        self.exclusions = [
            np.array([0.0, 0.0, -0.1, 0.15]), # Base mount / desk
        ]

    def is_safe(self, target_xyz: np.ndarray) -> bool:
        for ex in self.exclusions:
            center, radius = ex[:3], ex[3]
            if np.linalg.norm(target_xyz - center) < radius:
                return False
        return True

    def clamp_to_safety(self, start_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
        """If the target is inside an obstacle, project it to the safe boundary."""
        if self.is_safe(target_xyz):
            return target_xyz
            
        for ex in self.exclusions:
            center, radius = ex[:3], ex[3]
            vec = target_xyz - center
            dist = np.linalg.norm(vec)
            if dist < radius:
                # Push outward to the boundary surface + 1cm margin
                correction_dir = vec / (dist + 1e-8)
                return center + correction_dir * (radius + 0.01)
        return start_xyz

class TemporalSmoother:
    """
    Filters high-frequency jitter from the S4 continuous spatial head.
    The smoothing factor (alpha) is inversely proportional to biological confidence.
    """
    def __init__(self, base_alpha: float = 0.4, min_confidence: float = 0.2):
        self.base_alpha = base_alpha
        self.min_confidence = min_confidence
        self.last_target: Optional[np.ndarray] = None

    def smooth(self, raw_target: np.ndarray, confidence: float) -> np.ndarray:
        # If confidence is too low, alpha goes to 1.0 (freeze in place)
        if confidence < self.min_confidence:
            dynamic_alpha = 1.0
        else:
            # High confidence (1.0) -> lower alpha -> fast tracking
            dynamic_alpha = self.base_alpha + (1.0 - confidence) * (1.0 - self.base_alpha)
            dynamic_alpha = min(0.98, max(0.1, dynamic_alpha))

        if self.last_target is None:
            self.last_target = raw_target
            return raw_target

        smoothed = dynamic_alpha * self.last_target + (1.0 - dynamic_alpha) * raw_target
        self.last_target = smoothed
        return smoothed

# ── Kinematics & Execution ────────────────────────────────────────────────────

class KinematicSolver:
    """Handles Forward and Inverse Kinematics for the configured ArmConfig."""
    def __init__(self, cfg: ArmConfig):
        self.cfg = cfg

    def tip_position(self, state: JointState) -> np.ndarray:
        # Simplified placeholder for actual FK (e.g., via Pinocchio or analytical Jacobian)
        # Returns current (X, Y, Z) in meters.
        reach = sum(self.cfg.link_lengths)
        return np.array([reach, 0.0, 0.0], dtype=np.float32)

    def inverse(self, target_xyz: np.ndarray, current: JointState) -> Tuple[JointState, bool, float]:
        """
        Attempts to find joint angles that reach target_xyz.
        Returns: (New JointState, Converged Bool, Position Error in meters)
        """
        # Distances bounded by max reach
        norm = np.linalg.norm(target_xyz)
        if norm > self.cfg.max_reach:
            target_xyz = target_xyz * (self.cfg.max_reach / norm)

        # Placeholder for numeric IK convergence (e.g., CCD or Jacobian Transpose)
        # Assuming successful convergence for the structural outline:
        simulated_angles = current.angles + np.random.normal(0, 0.01, 6) # Mock angle delta
        
        # Clamp to joint limits
        for i, (lower, upper) in enumerate(self.cfg.joint_limits):
            simulated_angles[i] = np.clip(simulated_angles[i], lower, upper)
            
        return JointState(simulated_angles), True, 0.005 # Returning 5mm residual error

class MovementExecutor:
    """
    The Continuous Execution Engine.
    Maps continuous biological intent (xyz_pred) into smoothed, safe physical motion.
    """
    def __init__(self, cfg: ArmConfig = ArmConfig()):
        self.ik = KinematicSolver(cfg)
        self.obstacles = ObstacleSphere()
        self.smoother = TemporalSmoother()
        self.current = JointState()

    def execute_continuous(
        self, 
        raw_xyz_pred: np.ndarray, 
        confidence: float
    ) -> Dict[str, np.ndarray]:
        """
        Called at high frequency (e.g., 60Hz).
        1. Smooths the raw prediction based on S4 confidence.
        2. Clamps the target to prevent physical collisions.
        3. Solves Inverse Kinematics for the safe target.
        4. Returns motor commands and the actual resolved tip position.
        """
        
        # 1. Intent Smoothing (Freeze if confidence is low)
        smoothed_target = self.smoother.smooth(raw_xyz_pred, confidence)
        
        # 2. Local Safety Verification (Physical boundary checks)
        safe_target = self.obstacles.clamp_to_safety(self.ik.tip_position(self.current), smoothed_target)
        
        # 3. Inverse Kinematics
        next_state, converged, error_m = self.ik.inverse(safe_target, self.current)
        
        # 4. State Update
        self.current = next_state
        
        # Determine actual tip position achieved (for PositionErrorFeedback)
        actual_tip = self.ik.tip_position(self.current)
        
        return {
            "joint_commands": self.current.to_array(), # To be sent to hardware.py ServoController
            "actual_tip": actual_tip,                  # To be queued in LearningManager
            "converged": np.array([converged]),
            "error_m": np.array([error_m]),
            "safe_target": safe_target                 # The clamped target
        }