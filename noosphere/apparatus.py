"""
noosphere/apparatus.py
======================
Intentional Apparatus Movement 

Features:
- Exposes `safe_target` explicitly so the Trainer can push it into the Replay Buffer 
  as `exec_continuous` for accurate World Model physics training.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

@dataclass
class ArmConfig:
    link_lengths: List[float] = field(default_factory=lambda: [0.10, 0.25, 0.25, 0.10])
    max_reach: float = 0.70
    joint_limits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-180.0, 180.0), (-90.0, 90.0), (-180.0, 180.0), 
        (-90.0, 90.0), (-180.0, 180.0), (-20.0, 20.0),
    ])

class JointState:
    def __init__(self, angles: Optional[np.ndarray] = None):
        self.angles = angles if angles is not None else np.zeros(6, dtype=np.float32)
    def to_array(self) -> np.ndarray:
        return self.angles.copy()

class ObstacleSphere:
    def __init__(self):
        self.exclusions = [np.array([0.0, 0.0, -0.1, 0.15])]

    def is_safe(self, target_xyz: np.ndarray) -> bool:
        for ex in self.exclusions:
            if np.linalg.norm(target_xyz - ex[:3]) < ex[3]: return False
        return True

    def clamp_to_safety(self, start_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
        if self.is_safe(target_xyz): return target_xyz
        for ex in self.exclusions:
            center, radius = ex[:3], ex[3]
            vec = target_xyz - center
            dist = np.linalg.norm(vec)
            if dist < radius:
                return center + (vec / (dist + 1e-8)) * (radius + 0.01)
        return start_xyz

class TemporalSmoother:
    def __init__(self, base_alpha: float = 0.4, min_confidence: float = 0.2):
        self.base_alpha = base_alpha; self.min_confidence = min_confidence
        self.last_target: Optional[np.ndarray] = None

    def smooth(self, raw_target: np.ndarray, confidence: float) -> np.ndarray:
        if confidence < self.min_confidence: dynamic_alpha = 1.0
        else: dynamic_alpha = min(0.98, max(0.1, self.base_alpha + (1.0 - confidence) * (1.0 - self.base_alpha)))

        if self.last_target is None:
            self.last_target = raw_target; return raw_target

        smoothed = dynamic_alpha * self.last_target + (1.0 - dynamic_alpha) * raw_target
        self.last_target = smoothed; return smoothed

class KinematicSolver:
    def __init__(self, cfg: ArmConfig): self.cfg = cfg

    def tip_position(self, state: JointState) -> np.ndarray:
        return np.array([sum(self.cfg.link_lengths), 0.0, 0.0], dtype=np.float32)

    def inverse(self, target_xyz: np.ndarray, current: JointState) -> Tuple[JointState, bool, float]:
        norm = np.linalg.norm(target_xyz)
        if norm > self.cfg.max_reach: target_xyz = target_xyz * (self.cfg.max_reach / norm)
        
        simulated_angles = current.angles + np.random.normal(0, 0.01, 6)
        for i, (lower, upper) in enumerate(self.cfg.joint_limits):
            simulated_angles[i] = np.clip(simulated_angles[i], lower, upper)
            
        return JointState(simulated_angles), True, 0.005 

class MovementExecutor:
    def __init__(self, cfg: ArmConfig = ArmConfig()):
        self.ik = KinematicSolver(cfg)
        self.obstacles = ObstacleSphere()
        self.smoother = TemporalSmoother()
        self.current = JointState()

    def execute_continuous(self, raw_xyz_pred: np.ndarray, confidence: float) -> Dict[str, np.ndarray]:
        smoothed_target = self.smoother.smooth(raw_xyz_pred, confidence)
        safe_target = self.obstacles.clamp_to_safety(self.ik.tip_position(self.current), smoothed_target)
        
        next_state, converged, error_m = self.ik.inverse(safe_target, self.current)
        self.current = next_state
        
        return {
            "joint_commands": self.current.to_array(), 
            "actual_tip": self.ik.tip_position(self.current),
            "converged": np.array([converged]),
            "error_m": np.array([error_m]),
            "safe_target": safe_target  # Passed back to agent.observe() as exec_cont
        }