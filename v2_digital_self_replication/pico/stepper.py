"""
Pure-Python stepper state machine for the 28BYJ-48 motor.

Works on both CPython (for testing) and MicroPython (on the Pico).
No hardware dependencies — all math, no Pin/time imports.

Implements the same protocol as arduino_bridge.py:
  Command:  M<idx> <pwm>  (pwm 0–255)
  Feedback: P<idx> <angle> (angle 0–180)
"""


class StepperSim:
    """
    28BYJ-48 half-step state machine.

    Call set_pwm() when a command arrives, tick() each loop iteration,
    and angle() to get the feedback value to report.
    """

    HALF_STEP = [
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ]

    def __init__(self, max_steps: int = 512):
        self.max_steps   = max_steps
        self.current_pos = 0
        self.target_pos  = 0
        self.phase_idx   = 0

    def set_pwm(self, pwm: int) -> None:
        """Receive M command value (0–255) and update target position."""
        self.target_pos = self._pwm_to_target(int(pwm))

    def centre(self) -> None:
        """Watchdog / halt: drive back to zero position."""
        self.target_pos = 0

    def tick(self) -> bool:
        """
        Advance one half-step toward target.
        Returns True if the motor moved, False if already at target.
        """
        if self.current_pos < self.target_pos:
            self.phase_idx   = (self.phase_idx + 1) % 8
            self.current_pos += 1
            return True
        if self.current_pos > self.target_pos:
            self.phase_idx   = (self.phase_idx - 1) % 8
            self.current_pos -= 1
            return True
        return False

    def run_to_target(self, max_ticks: int = 2048) -> int:
        """Tick until at target or max_ticks reached. Returns steps taken."""
        steps = 0
        while steps < max_ticks and self.tick():
            steps += 1
        return steps

    def angle(self) -> int:
        """Current position as angle 0–180 for the P<idx> feedback message."""
        return self._pos_to_angle(self.current_pos)

    def phase(self) -> list:
        """Which coils are energised (for hardware verification tests)."""
        return list(self.HALF_STEP[self.phase_idx % 8])

    def at_target(self) -> bool:
        return self.current_pos == self.target_pos

    # ── Conversion helpers (mirrors arduino_bridge._dof_to_pwm / _angle_to_dof) ──

    def _pwm_to_target(self, pwm: int) -> int:
        return int((pwm / 255.0) * 2 * self.max_steps - self.max_steps)

    def _pos_to_angle(self, pos: int) -> int:
        return int((pos + self.max_steps) / (2 * self.max_steps) * 180)
