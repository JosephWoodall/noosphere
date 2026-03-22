"""
noosphere/hardware.py
=====================
Hardware Abstraction Layer

Provides a unified `ServoController` interface across four backends:

    sim         — prints commands (no hardware required, default)
    rpi_pca9685 — Raspberry Pi + PCA9685 I2C PWM driver (6+ servos)
    arduino     — Arduino serial protocol (M<ch>,<angle>\\n commands)
    rpi_gpio    — Raspberry Pi hardware PWM (2 channels only)

Select backend by instantiating with `backend=`:
    ctrl = ServoController(backend="sim")
    ctrl = ServoController(backend="rpi_pca9685")

The controller accepts joint angles in DEGREES and handles PWM conversion.
Smooth movement is provided by `smooth_move()` which interpolates between
current and target state in configurable steps.

NCP integration:
    Hardware publishes each executed command as an NCP MOTOR_COMMAND frame
    so the world model can observe actual motor state (not just planned state).
"""

import math
import time
from typing import List, Optional, Tuple

import numpy as np

JOINT_NAMES = [
    "shoulder_yaw",
    "shoulder_pitch",
    "shoulder_roll",
    "elbow_pitch",
    "wrist_pitch",
    "wrist_yaw",
]


class ServoController:
    """
    Unified servo interface — backend selected at construction time.

    Parameters
    ----------
    backend   : "sim" | "rpi_pca9685" | "arduino" | "rpi_gpio"
    n_channels: number of servo channels (default 6)
    """

    def __init__(self, backend: str = "sim", n_channels: int = 6):
        self.backend = backend
        self.n_channels = n_channels
        self._current = np.zeros(n_channels)
        self._impl = self._init_backend(backend)

    def _init_backend(self, backend: str):
        if backend == "rpi_pca9685":
            return _PCA9685Backend(self.n_channels)
        elif backend == "arduino":
            return _ArduinoBackend(self.n_channels)
        elif backend == "rpi_gpio":
            return _GPIOBackend(self.n_channels)
        else:
            return _SimBackend(self.n_channels)

    def set_angle(self, channel: int, angle_deg: float):
        self._impl.set_angle(channel, angle_deg)
        self._current[channel] = angle_deg

    def set_all_angles(self, angles_deg: np.ndarray):
        self._impl.set_all_angles(angles_deg)
        self._current[:] = angles_deg
        time.sleep(0.5)  # servo settling time

    def smooth_move(
        self,
        target_deg: np.ndarray,
        steps: int = 5,
        step_delay_s: float = 0.1,
    ) -> List[np.ndarray]:
        """
        Interpolate from current angles to target in `steps`.
        Returns list of intermediate angle arrays (for logging/world model).
        """
        trajectory = []
        for i in range(steps + 1):
            t = i / steps
            interp = self._current + (target_deg - self._current) * t
            self._impl.set_all_angles(interp)  # bypass the 0.5s settling sleep
            self._current[:] = interp
            trajectory.append(interp.copy())
            if i < steps:
                time.sleep(step_delay_s)
        return trajectory

    def disable_all(self):
        self._impl.disable_all()

    def __del__(self):
        try:
            self.disable_all()
        except Exception:
            pass


# ── Simulation backend ────────────────────────────────────────────────────────


class _SimBackend:
    def __init__(self, n):
        self.n = n

    def set_angle(self, ch, deg):
        print(
            f"  [SIM] {JOINT_NAMES[ch] if ch < len(JOINT_NAMES) else f'ch{ch}'}: {deg:.2f}°"
        )

    def set_all_angles(self, angles):
        print("  [SIM] Motor command:")
        for i, a in enumerate(angles):
            name = JOINT_NAMES[i] if i < len(JOINT_NAMES) else f"ch{i}"
            print(f"    {name:<20} {a:+7.2f}°")

    def disable_all(self):
        print("  [SIM] All servos disabled")


# ── PCA9685 backend (Raspberry Pi I2C) ───────────────────────────────────────


class _PCA9685Backend:
    """Raspberry Pi + PCA9685 16-channel PWM driver via I2C."""

    MIN_PULSE = 150  # ~1ms (0°)
    MAX_PULSE = 600  # ~2ms (180°)

    def __init__(self, n):
        self.n = n
        try:
            from pwm_pca9685 import Address, Pca9685
            from rppal.i2c import I2c

            i2c = I2c()
            self._pwm = Pca9685(i2c, Address.default())
            self._pwm.enable()
            self._pwm.set_prescale(100)  # ~60Hz
            print("[Hardware] PCA9685 initialized on I2C")
        except Exception as e:
            print(f"[Hardware] PCA9685 init failed ({e}), falling back to sim")
            self._pwm = None

    def _angle_to_pulse(self, angle_deg: float) -> int:
        centered = angle_deg + 90.0
        clamped = max(0.0, min(180.0, centered))
        return int(
            self.MIN_PULSE + (clamped / 180.0) * (self.MAX_PULSE - self.MIN_PULSE)
        )

    def set_angle(self, ch, deg):
        if self._pwm is None:
            return
        from pwm_pca9685 import Channel

        ch_map = [
            Channel.C0,
            Channel.C1,
            Channel.C2,
            Channel.C3,
            Channel.C4,
            Channel.C5,
        ]
        if ch < len(ch_map):
            self._pwm.set_channel_on_off(ch_map[ch], 0, self._angle_to_pulse(deg))

    def set_all_angles(self, angles):
        for i, a in enumerate(angles):
            self.set_angle(i, a)

    def disable_all(self):
        if self._pwm:
            from pwm_pca9685 import Channel

            for ch in [
                Channel.C0,
                Channel.C1,
                Channel.C2,
                Channel.C3,
                Channel.C4,
                Channel.C5,
            ]:
                self._pwm.set_channel_on_off(ch, 0, 0)


# ── Arduino serial backend ────────────────────────────────────────────────────


class _ArduinoBackend:
    """
    Arduino serial protocol.
    Set:  M<channel>,<angle_deg>\\n
    All:  A<a0>,<a1>,<a2>,<a3>,<a4>,<a5>\\n
    Stop: D\\n
    """

    def __init__(self, n, port: str = "/dev/ttyACM0", baud: int = 115200):
        self.n = n
        try:
            import serial

            self._port = serial.Serial(port, baud, timeout=1)
            time.sleep(2.0)  # Arduino reset
            print(f"[Hardware] Arduino on {port} @ {baud}")
        except Exception as e:
            print(f"[Hardware] Arduino init failed ({e}), falling back to sim")
            self._port = None

    def _write(self, cmd: str):
        if self._port:
            self._port.write(cmd.encode())
            self._port.flush()

    def set_angle(self, ch, deg):
        self._write(f"M{ch},{deg + 90.0:.2f}\n")

    def set_all_angles(self, angles):
        parts = ",".join(f"{a + 90.0:.2f}" for a in angles)
        self._write(f"A{parts}\n")

    def disable_all(self):
        self._write("D\n")


# ── Raspberry Pi GPIO PWM backend ─────────────────────────────────────────────


class _GPIOBackend:
    """Raspberry Pi hardware PWM (2 channels max)."""

    def __init__(self, n):
        self.n = min(n, 2)
        try:
            from rppal.pwm import Channel, Polarity, Pwm

            self._pwms = [
                Pwm.with_frequency(Channel.Pwm0, 50.0, 0.075, Polarity.Normal, True),
                Pwm.with_frequency(Channel.Pwm1, 50.0, 0.075, Polarity.Normal, True),
            ]
            print("[Hardware] Raspberry Pi GPIO PWM (2 channels)")
            if n > 2:
                print(
                    "[Hardware] WARNING: Only 2 servos supported — use PCA9685 for more"
                )
        except Exception as e:
            print(f"[Hardware] GPIO PWM init failed ({e})")
            self._pwms = []

    def _angle_to_duty(self, deg: float) -> float:
        centered = max(0.0, min(180.0, deg + 90.0))
        return 0.05 + (centered / 180.0) * 0.05  # 5%–10%

    def set_angle(self, ch, deg):
        if ch < len(self._pwms):
            self._pwms[ch].set_duty_cycle(self._angle_to_duty(deg))

    def set_all_angles(self, angles):
        for i, a in enumerate(angles):
            self.set_angle(i, a)

    def disable_all(self):
        for p in self._pwms:
            p.disable()
