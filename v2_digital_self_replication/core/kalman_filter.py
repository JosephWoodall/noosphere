"""
Adaptive Kalman filter for smooth motor command tracking.

State:  x = [position (n_dof), velocity (n_dof)]  ∈ R^{2*n_dof}
Model:  constant-velocity: pos_{t+1} = pos_t + dt*vel_t, vel stays constant
Obs:    z = position part of x  ∈ R^{n_dof}

The key adaptation: measurement noise R = diag(sigma^2) is set from the
decoder's aleatoric uncertainty each step.  When the neural decoder is
uncertain, the filter trusts prediction momentum over the noisy observation.
"""

from __future__ import annotations

import numpy as np


class AdaptiveKalmanFilter:
    def __init__(self, n_dof: int = 6, dt: float = 1.0 / 256, process_noise: float = 0.01):
        self.n_dof = n_dof
        n = n_dof * 2  # [pos, vel]
        self.n = n

        self.x = np.zeros(n, dtype=np.float64)
        self.P = np.eye(n, dtype=np.float64) * 0.1

        # Constant-velocity transition matrix
        self.F = np.eye(n, dtype=np.float64)
        self.F[:n_dof, n_dof:] = np.eye(n_dof) * dt

        # Observation matrix: observe position only
        self.H = np.zeros((n_dof, n), dtype=np.float64)
        self.H[:, :n_dof] = np.eye(n_dof)

        self.Q = np.eye(n, dtype=np.float64) * process_noise
        self._default_R = np.eye(n_dof, dtype=np.float64) * 0.1

    def predict(self) -> np.ndarray:
        """Predict step. Returns predicted position."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[: self.n_dof].copy()

    def update(
        self,
        z: np.ndarray,
        sigma: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Update step.
        z: (n_dof,) predicted command from decoder.
        sigma: (n_dof,) decoder uncertainty — sets measurement noise R adaptively.
        Returns smoothed position estimate.
        """
        R = np.diag(sigma**2) if sigma is not None else self._default_R
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.solve(S.T, np.eye(self.n_dof)).T
        innov = z - self.H @ self.x
        self.x = self.x + K @ innov
        self.P = (np.eye(self.n) - K @ self.H) @ self.P
        return self.x[: self.n_dof].copy()

    def step(self, z: np.ndarray, sigma: np.ndarray | None = None) -> np.ndarray:
        """Predict + update in one call. Returns smoothed position."""
        self.predict()
        return self.update(z, sigma)

    def reset(self):
        self.x[:] = 0.0
        self.P[:] = np.eye(self.n) * 0.1

    @property
    def position(self) -> np.ndarray:
        return self.x[: self.n_dof].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[self.n_dof :].copy()
