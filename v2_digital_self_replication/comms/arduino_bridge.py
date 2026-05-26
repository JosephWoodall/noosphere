"""
Arduino bridge: ZMQ SUB → pyserial → Arduino motor controller.

Subscribes to the ZMQBridge PUB port (5555) and forwards motor commands
to an Arduino over serial. The Arduino sketch (arduino/motor_control.ino)
receives "M<idx> <pwm>\\n" and calls analogWrite(pin, pwm).

DOF mapping (configurable via DOF_MOTOR_MAP):
  DOF 0 (shoulder_yaw)  → M0
  DOF 5 (grip_aperture) → M1

DOF values in [-1.0, 1.0] are linearly mapped to PWM [0, 255].

Usage:
    python -m v2_digital_self_replication.comms.arduino_bridge \\
        --port /dev/ttyACM0 \\
        --zmq-port 5555
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Map DOF index → motor index sent to Arduino
DOF_MOTOR_MAP: dict[int, int] = {
    0: 0,   # shoulder_yaw   → M0
    5: 1,   # grip_aperture  → M1
}


def _dof_to_pwm(value: float) -> int:
    """Map DOF value in [-1, 1] to Arduino PWM [0, 255]."""
    pwm = int((value + 1.0) / 2.0 * 255)
    return max(0, min(255, pwm))


class ArduinoBridge:
    """
    ZMQ subscriber that forwards motor commands to an Arduino over serial.

    Runs synchronously — call run() in main thread or a dedicated thread.
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 115200,
        zmq_host: str = "localhost",
        zmq_port: int = 5555,
        dof_map: Optional[dict[int, int]] = None,
    ):
        self.port     = port
        self.baud     = baud
        self.zmq_host = zmq_host
        self.zmq_port = zmq_port
        self.dof_map  = dof_map or DOF_MOTOR_MAP
        self._running = False

    def run(self):
        """Block and forward commands until KeyboardInterrupt."""
        serial = self._open_serial()
        sock   = self._open_zmq()
        if sock is None:
            return

        self._running = True
        logger.info("Arduino bridge running (Ctrl-C to stop)")

        try:
            while self._running:
                try:
                    raw = sock.recv_string()
                    msg = json.loads(raw)
                    self._dispatch(msg, serial)
                except Exception:
                    pass  # ZMQ timeout or parse error — keep polling
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            sock.close()
            sock.context.term()
            if serial is not None:
                serial.close()
                logger.info("Serial port closed")

    def _dispatch(self, msg: dict, serial):
        halt = msg.get("halt", False)
        cmd  = msg.get("cmd", [0.0] * 6)

        for dof_idx, motor_idx in self.dof_map.items():
            value = 0.0 if halt else float(cmd[dof_idx])
            pwm   = _dof_to_pwm(value)
            line  = f"M{motor_idx} {pwm}\n"
            if serial is not None:
                try:
                    serial.write(line.encode())
                except Exception as e:
                    logger.error("Serial write error: %s", e)
            else:
                logger.debug("SIM serial → %s", line.strip())

    def _open_serial(self):
        try:
            import serial as pyserial
            s = pyserial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2.0)  # wait for Arduino reset after DTR toggle
            logger.info("Arduino connected: %s @ %d baud", self.port, self.baud)
            return s
        except ImportError:
            logger.warning("pyserial not installed — serial output disabled")
        except Exception as e:
            logger.warning("Serial open failed (%s) — running in dry-run mode", e)
        return None

    def _open_zmq(self):
        try:
            import zmq
        except ImportError:
            logger.error("pyzmq not installed — arduino_bridge cannot receive commands")
            return None

        ctx  = zmq.Context()
        sock = ctx.socket(zmq.SUB)
        sock.connect(f"tcp://{self.zmq_host}:{self.zmq_port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.setsockopt(zmq.RCVTIMEO, 100)
        sock.context = ctx  # keep reference for cleanup
        logger.info("ZMQ SUB connected to tcp://%s:%d", self.zmq_host, self.zmq_port)
        return sock


def main():
    p = argparse.ArgumentParser(description="Arduino motor bridge for the v2 digital twin.")
    p.add_argument("--port",     type=str, default="/dev/ttyACM0")
    p.add_argument("--baud",     type=int, default=115200)
    p.add_argument("--zmq-host", type=str, default="localhost")
    p.add_argument("--zmq-port", type=int, default=5555)
    p.add_argument("--log-level", type=str, default="INFO")
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    bridge = ArduinoBridge(
        port=args.port,
        baud=args.baud,
        zmq_host=args.zmq_host,
        zmq_port=args.zmq_port,
    )
    bridge.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
