"""
Arduino bridge: ZMQ SUB → pyserial → Arduino → ZMQ PUB (feedback).

Subscribes to ZMQBridge command port (5555), forwards motor commands to
the Arduino over serial, reads position feedback from serial, and publishes
it back on ZMQBridge feedback port (5556) so the twin closes the loop.

Command protocol  (bridge → Arduino):  "M<idx> <pwm>\\n"   pwm 0–255
Feedback protocol (Arduino → bridge):  "P<idx> <angle>\\n" angle 0–180

DOF mapping:
  DOF 0 (shoulder_yaw)   → M0  (servo on pin 9)
  DOF 5 (grip_aperture)  → M1  (servo on pin 10)

Servo angle 0–180° maps back to DOF value [-1, 1] for the twin's proprioception.

Usage:
    python -m v2_digital_self_replication.comms.arduino_bridge --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# DOF index → motor index
DOF_MOTOR_MAP: dict[int, int] = {
    0: 0,   # shoulder_yaw   → M0
    5: 1,   # grip_aperture  → M1
}

# Motor index → DOF index (inverse map for feedback)
MOTOR_DOF_MAP: dict[int, int] = {v: k for k, v in DOF_MOTOR_MAP.items()}

N_DOF = 6


def _dof_to_pwm(value: float) -> int:
    """DOF [-1, 1] → PWM [0, 255]."""
    return max(0, min(255, int((value + 1.0) / 2.0 * 255)))


def _angle_to_dof(angle: float) -> float:
    """Servo angle [0, 180] → DOF [-1, 1]."""
    return (angle / 180.0) * 2.0 - 1.0


class ArduinoBridge:
    """
    Bidirectional bridge: ZMQ commands → Arduino servos → ZMQ feedback.

    Two threads:
      - command thread: ZMQ SUB → serial write
      - feedback thread: serial read → ZMQ PUB on feedback port
    """

    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 115200,
        zmq_host: str = "localhost",
        zmq_cmd_port: int = 5555,
        zmq_fb_port: int = 5556,
        dof_map: Optional[dict[int, int]] = None,
    ):
        self.port         = port
        self.baud         = baud
        self.zmq_host     = zmq_host
        self.zmq_cmd_port = zmq_cmd_port
        self.zmq_fb_port  = zmq_fb_port
        self.dof_map      = dof_map or DOF_MOTOR_MAP

        self._serial   = None
        self._running  = False
        self._position = np.zeros(N_DOF, dtype=np.float32)  # DOF space, last known
        self._pos_lock = threading.Lock()

    def run(self):
        """Open serial + ZMQ and block until Ctrl-C."""
        self._serial  = self._open_serial()
        zmq_ctx, cmd_sock, fb_sock = self._open_zmq()
        if cmd_sock is None:
            return

        self._running = True

        fb_thread = threading.Thread(
            target=self._feedback_loop,
            args=(fb_sock,),
            daemon=True,
        )
        fb_thread.start()

        logger.info("Arduino bridge running — commands from :%d, feedback to :%d",
                    self.zmq_cmd_port, self.zmq_fb_port)
        logger.info("Ctrl-C to stop")

        try:
            while self._running:
                try:
                    raw = cmd_sock.recv_string()
                    msg = json.loads(raw)
                    self._dispatch(msg)
                except Exception:
                    pass
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            cmd_sock.close()
            if fb_sock:
                fb_sock.close()
            zmq_ctx.term()
            if self._serial is not None:
                self._serial.close()
                logger.info("Serial port closed")

    def _dispatch(self, msg: dict):
        """Write motor commands to serial."""
        halt = msg.get("halt", False)
        cmd  = msg.get("cmd", [0.0] * N_DOF)

        for dof_idx, motor_idx in self.dof_map.items():
            value = 0.0 if halt else float(cmd[dof_idx])
            pwm   = _dof_to_pwm(value)
            line  = f"M{motor_idx} {pwm}\n"
            if self._serial is not None:
                try:
                    self._serial.write(line.encode())
                except Exception as e:
                    logger.error("Serial write error: %s", e)
            else:
                logger.debug("SIM → %s", line.strip())

    def _feedback_loop(self, fb_sock):
        """Read 'P<idx> <angle>\\n' from Arduino, publish FeedbackMsg to ZMQ."""
        while self._running:
            if self._serial is None:
                time.sleep(0.05)
                continue
            try:
                line = self._serial.readline().decode(errors="ignore").strip()
                if not line.startswith("P"):
                    continue
                parts = line[1:].split()
                if len(parts) != 2:
                    continue
                motor_idx = int(parts[0])
                angle     = float(parts[1])
                dof_idx   = MOTOR_DOF_MAP.get(motor_idx)
                if dof_idx is None:
                    continue

                with self._pos_lock:
                    self._position[dof_idx] = _angle_to_dof(angle)
                    pos_snapshot = self._position.copy()

                if fb_sock is not None:
                    msg = {"t": time.time(), "pos": pos_snapshot.tolist(), "ern": 0.0}
                    try:
                        fb_sock.send_string(json.dumps(msg))
                    except Exception:
                        pass
            except Exception:
                pass

    def _open_serial(self):
        try:
            import serial as pyserial
            s = pyserial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2.0)  # Arduino resets on DTR toggle — wait for READY
            # Drain any startup messages
            s.read_all()
            logger.info("Arduino connected: %s @ %d baud", self.port, self.baud)
            return s
        except ImportError:
            logger.warning("pyserial not installed — running dry (no serial)")
        except Exception as e:
            logger.warning("Serial open failed (%s) — dry-run mode", e)
        return None

    def _open_zmq(self):
        try:
            import zmq
        except ImportError:
            logger.error("pyzmq not installed — cannot start bridge")
            return None, None, None

        ctx = zmq.Context()

        cmd_sock = ctx.socket(zmq.SUB)
        cmd_sock.connect(f"tcp://{self.zmq_host}:{self.zmq_cmd_port}")
        cmd_sock.setsockopt_string(zmq.SUBSCRIBE, "")
        cmd_sock.setsockopt(zmq.RCVTIMEO, 100)
        logger.info("ZMQ SUB on tcp://%s:%d (commands)", self.zmq_host, self.zmq_cmd_port)

        fb_sock = ctx.socket(zmq.PUB)
        fb_sock.connect(f"tcp://{self.zmq_host}:{self.zmq_fb_port}")
        logger.info("ZMQ PUB on tcp://%s:%d (feedback)", self.zmq_host, self.zmq_fb_port)

        return ctx, cmd_sock, fb_sock


def main():
    p = argparse.ArgumentParser(description="Arduino servo bridge for the v2 digital twin.")
    p.add_argument("--port",         type=str, default="/dev/ttyACM0")
    p.add_argument("--baud",         type=int, default=115200)
    p.add_argument("--zmq-host",     type=str, default="localhost")
    p.add_argument("--zmq-cmd-port", type=int, default=5555)
    p.add_argument("--zmq-fb-port",  type=int, default=5556)
    p.add_argument("--log-level",    type=str, default="INFO")
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
        zmq_cmd_port=args.zmq_cmd_port,
        zmq_fb_port=args.zmq_fb_port,
    )
    bridge.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
