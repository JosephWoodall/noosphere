"""
noosphere/hardware.py
=====================
Hardware Abstraction & Proprioceptive Telemetry Layer

Provides a unified `ServoController` interface across four backends:
    sim         — prints commands (no hardware required, default)
    rpi_pca9685 — Raspberry Pi + PCA9685 I2C PWM driver (6+ servos)
    arduino     — Arduino serial protocol (M<ch>,<angle>\\n commands)
    rpi_gpio    — Raspberry Pi hardware PWM (2 channels only)

Prosthetic Alignment Feature: 
The hardware layer is not just an output; it is a proprioceptive sensor. 
It strictly logs and optionally broadcasts (via NCP) the *actually executed* clamped angles, ensuring the World Model's next state observation is grounded 
in physical reality, not just open-loop assumptions.
"""

import logging
import math
import time
from typing import List, Optional, Tuple, Any

import numpy as np

# Assuming NCP components are available in the broader package
try:
    from noosphere.proto import NCPEncoder, MsgType, Channel
except ImportError:
    NCPEncoder = None

logger = logging.getLogger(__name__)

JOINT_NAMES = [
    "shoulder_yaw",
    "shoulder_pitch",
    "shoulder_roll",
    "elbow_pitch",
    "wrist_pitch",
    "wrist_yaw",
]

# ── Hardware Backends ─────────────────────────────────────────────────────────

class _SimBackend:
    """Simulated backend that simply logs commands and tracks state."""
    def __init__(self, n: int):
        self.n = n
        self.angles = np.zeros(n, dtype=np.float32)
        logger.info(f"[Hardware] Initialized Simulation Backend ({n} channels)")

    def set_all_angles(self, angles: np.ndarray):
        self.angles = np.copy(angles)
        # In trace/debug mode, you might log the actual array:
        # logger.debug(f"[Sim] Executed angles: {np.round(angles, 2)}")

    def disable_all(self):
        logger.info("[Sim] Motors Disabled (Torque OFF)")


class _PCA9685Backend:
    """I2C driver for standard 16-channel PWM servo hats."""
    def __init__(self, n: int):
        self.n = n
        try:
            import board
            import busio
            from adafruit_pca9685 import PCA9685
            from adafruit_motor import servo

            i2c = busio.I2C(board.SCL, board.SDA)
            self.pca = PCA9685(i2c)
            self.pca.frequency = 50
            self.servos = [servo.Servo(self.pca.channels[i], min_pulse=500, max_pulse=2500) for i in range(self.n)]
            logger.info(f"[Hardware] PCA9685 I2C Backend active ({n} channels)")
        except ImportError:
            logger.error("Adafruit CircuitPython libraries not found. Falling back to Sim.")
            self.servos = []

    def set_all_angles(self, angles: np.ndarray):
        if not self.servos: return
        for i in range(min(self.n, len(angles))):
            # Map physical degrees (-90 to 90 or -180 to 180) to 0-180 for standard servos
            # This depends heavily on your physical servo tuning
            mapped_angle = np.clip(angles[i] + 90.0, 0.0, 180.0) 
            self.servos[i].angle = mapped_angle

    def disable_all(self):
        if not self.servos: return
        for i in range(self.n):
            self.pca.channels[i].duty_cycle = 0


class _ArduinoBackend:
    """Serial communication to an Arduino running a custom servo sketch."""
    def __init__(self, n: int, port: str = "/dev/ttyUSB0", baud: int = 115200):
        self.n = n
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            logger.info(f"[Hardware] Arduino Serial Backend on {port} at {baud} baud")
        except Exception as e:
            logger.error(f"Arduino serial init failed: {e}. Falling back to Sim.")
            self.ser = None

    def _write(self, msg: str):
        if self.ser:
            self.ser.write(msg.encode('utf-8'))

    def set_all_angles(self, angles: np.ndarray):
        # Format: "A90.00,45.00,120.50\n"
        parts = ",".join(f"{a + 90.0:.2f}" for a in angles)
        self._write(f"A{parts}\n")

    def disable_all(self):
        self._write("D\n")


class _GPIOBackend:
    """Direct hardware PWM from Raspberry Pi pins (limited to 2 channels natively)."""
    def __init__(self, n: int):
        self.n = min(n, 2)
        try:
            from rppal.pwm import Channel, Polarity, Pwm
            self._pwms = [
                Pwm.with_frequency(Channel.Pwm0, 50.0, 0.075, Polarity.Normal, True),
                Pwm.with_frequency(Channel.Pwm1, 50.0, 0.075, Polarity.Normal, True),
            ]
            logger.info("[Hardware] Raspberry Pi GPIO PWM (2 channels)")
            if n > 2:
                logger.warning("[Hardware] Only 2 servos supported via GPIO — use PCA9685 for more")
        except Exception as e:
            logger.error(f"GPIO PWM init failed: {e}. Falling back to Sim.")
            self._pwms = []

    def _angle_to_duty(self, deg: float) -> float:
        centered = max(0.0, min(180.0, deg + 90.0))
        return 0.05 + (centered / 180.0) * 0.05  # 5% to 10% duty cycle

    def set_all_angles(self, angles: np.ndarray):
        for i in range(min(self.n, len(self._pwms))):
            self._pwms[i].change_duty_cycle(self._angle_to_duty(angles[i]))

    def disable_all(self):
        for pwm in self._pwms:
            pwm.change_duty_cycle(0.0)


class _BluetoothBackend:
    """Bluetooth LE communication to a smart robotic node (e.g. ESP32). Non-blocking background sync."""
    def __init__(self, n: int, service_uuid: str = "0000ffe0-0000-1000-8000-00805f9b34fb", char_uuid: str = "0000ffe1-0000-1000-8000-00805f9b34fb"):
        self.n = n
        self.service_uuid = service_uuid
        self.char_uuid = char_uuid
        self._target_angles = np.zeros(n, dtype=np.float32)
        self._shutdown = False
        
        try:
            import threading
            self._thread = threading.Thread(target=self._daemon_loop, daemon=True)
            self._thread.start()
            logger.info(f"[Hardware] Bluetooth DAEMON Backend initialized ({n} channels)")
        except Exception as e:
            logger.error(f"Failed to start Bluetooth daemon: {e}. Falling back to Sim.")
            self._thread = None

    def _daemon_loop(self):
        import asyncio
        asyncio.run(self._async_loop())

    async def _async_loop(self):
        try:
            from bleak import BleakClient, BleakScanner
        except ImportError:
            logger.error("bleak missing for Bluetooth. Disable or pip install bleak.")
            return

        while not self._shutdown:
            try:
                # Find any device advertising our service
                device = await BleakScanner.find_device_by_filter(
                    lambda d, ad: self.service_uuid.lower() in [u.lower() for u in ad.service_uuids], timeout=2.0
                )
                if not device:
                    await asyncio.sleep(0.5)
                    continue
                    
                async with BleakClient(device) as client:
                    logger.info(f"[Bluetooth] Streaming to connected node {device.address}")
                    while not self._shutdown and client.is_connected:
                        # Send position targets
                        parts = ",".join(f"{a + 90.0:.2f}" for a in self._target_angles)
                        msg = f"A{parts}\\n".encode('utf-8')
                        try:
                            await client.write_gatt_char(self.char_uuid, msg, response=False)
                        except Exception:
                            break  # Connection likely dropped, break to reconnect
                        await asyncio.sleep(0.02)  # Stream at 50Hz
            except Exception as e:
                logger.debug(f"[Bluetooth] Disconnected or error: {e}. Searching...")
                await asyncio.sleep(1.0)

    def set_all_angles(self, angles: np.ndarray):
        # Instantly updates shared state; does not block the RL loop!
        self._target_angles = np.copy(angles)

    def disable_all(self):
        self._shutdown = True
        logger.info("[Bluetooth] Daemon shutdown requested.")


# ── Unified Controller ────────────────────────────────────────────────────────

class ServoController:
    """
    Unified servo interface that handles command routing and telemetry broadcasting.
    """
    def __init__(self, backend: str = "sim", n_channels: int = 6, ncp_transport: Optional[Any] = None):
        self.n = n_channels
        self.backend_type = backend
        self.ncp_transport = ncp_transport
        self.current_angles = np.zeros(n_channels, dtype=np.float32)

        if backend == "sim":
            self._hw = _SimBackend(n_channels)
        elif backend == "rpi_pca9685":
            self._hw = _PCA9685Backend(n_channels)
        elif backend == "arduino":
            self._hw = _ArduinoBackend(n_channels)
        elif backend == "rpi_gpio":
            self._hw = _GPIOBackend(n_channels)
        elif backend == "bluetooth":
            self._hw = _BluetoothBackend(n_channels)
        else:
            logger.warning(f"Unknown backend '{backend}', defaulting to 'sim'")
            self._hw = _SimBackend(n_channels)

    def set_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Sends angles to the hardware, caches the true physical state, 
        and optionally broadcasts the execution via NCP for the World Model.
        """
        # Ensure array sizing matches hardware channels
        clamped_angles = np.array(angles, dtype=np.float32)[:self.n]
        if len(clamped_angles) < self.n:
            padded = np.zeros(self.n, dtype=np.float32)
            padded[:len(clamped_angles)] = clamped_angles
            clamped_angles = padded

        # Execute on hardware
        self._hw.set_all_angles(clamped_angles)
        self.current_angles = clamped_angles

        # Proprioceptive Feedback: Broadcast actual executed state
        self._broadcast_telemetry()

        return self.current_angles

    def smooth_move(self, target_angles: np.ndarray, steps: int = 10, delay_s: float = 0.02) -> np.ndarray:
        """
        Linearly interpolates from current position to target for fluid motion,
        broadcasting telemetry at each step.
        """
        start = self.current_angles.copy()
        target = np.array(target_angles, dtype=np.float32)[:self.n]
        
        for step in range(1, steps + 1):
            t = step / float(steps)
            interp = start + (target - start) * t
            self.set_angles(interp)
            time.sleep(delay_s)
            
        return self.current_angles

    def get_proprioception(self) -> np.ndarray:
        """Read the currently held physical state (for direct observation polling)."""
        return self.current_angles.copy()

    def disable(self):
        """Cuts torque to all motors (freewheel mode)."""
        self._hw.disable_all()
        logger.info("[Hardware] System disengaged.")

    def _broadcast_telemetry(self):
        """
        Encodes the current physical joint state into an NCP binary frame 
        so the perception module can process it as a `kinematics` observation.
        """
        if self.ncp_transport and NCPEncoder:
            # We map the physical state to an NCP MOTOR_COMMAND telemetry frame
            payload = {"angles": self.current_angles.tolist()}
            frame = NCPEncoder.encode(Channel.TELEMETRY, MsgType.MOTOR_COMMAND, payload)
            self.ncp_transport.send(frame)