"""
ZeroMQ bridge: DigitalTwin ↔ prosthetic hardware.

Protocol:
  PUB  port 5555  — publishes MotorCommandMsg (JSON) to hardware controller
  SUB  port 5556  — receives FeedbackMsg (JSON) from hardware (actual position + ERN)

Message schemas:
  MotorCommandMsg: {"t": float, "cmd": [6 floats], "sigma": [6 floats], "halt": bool, "reason": str}
  FeedbackMsg:     {"t": float, "pos": [6 floats], "ern": float}

Heartbeat: published every heartbeat_ms milliseconds regardless of command updates.
Watchdog: if no feedback received for watchdog_s, emits a HALT command automatically.

Usage:
    bridge = ZMQBridge(config.zmq)
    bridge.start()
    ...
    bridge.send_command(cmd, sigma, halt=False)
    feedback = bridge.latest_feedback()
    bridge.stop()
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ZMQBridge:
    """
    ZeroMQ PUB/SUB bridge for prosthetic hardware communication.

    zmq (pyzmq) is an optional dependency — the bridge degrades gracefully
    to a no-op mock when it is not installed, so the rest of the pipeline
    runs in simulation mode without hardware.
    """

    def __init__(
        self,
        pub_port: int = 5555,
        sub_port: int = 5556,
        heartbeat_ms: int = 100,
        watchdog_s: float = 2.0,
    ):
        self.pub_port = pub_port
        self.sub_port = sub_port
        self.heartbeat_ms = heartbeat_ms
        self.watchdog_s = watchdog_s

        self._context = None
        self._publisher = None
        self._subscriber = None
        self._running = False

        self._feedback_lock = threading.Lock()
        self._latest_feedback: Optional[dict] = None
        self._last_feedback_t: float = time.monotonic()

        self._pub_thread: Optional[threading.Thread] = None
        self._sub_thread: Optional[threading.Thread] = None

        self._pending_cmd: Optional[dict] = None
        self._cmd_lock = threading.Lock()

        self._zmq_available = self._check_zmq()

    def _check_zmq(self) -> bool:
        try:
            import zmq  # noqa: F401
            return True
        except ImportError:
            logger.warning("ZMQBridge: pyzmq not installed — running in simulation mode")
            return False

    def start(self):
        if not self._zmq_available:
            logger.info("ZMQBridge: simulation mode, no actual sockets opened")
            self._running = True
            return

        import zmq
        self._context = zmq.Context()

        self._publisher = self._context.socket(zmq.PUB)
        self._publisher.bind(f"tcp://*:{self.pub_port}")

        self._subscriber = self._context.socket(zmq.SUB)
        self._subscriber.connect(f"tcp://localhost:{self.sub_port}")
        self._subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self._subscriber.setsockopt(zmq.RCVTIMEO, 50)  # 50 ms poll timeout

        self._running = True

        self._pub_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._pub_thread.start()

        self._sub_thread = threading.Thread(target=self._feedback_loop, daemon=True)
        self._sub_thread.start()

        logger.info("ZMQBridge started: PUB=%d SUB=%d", self.pub_port, self.sub_port)

    def stop(self):
        self._running = False
        if self._pub_thread:
            self._pub_thread.join(timeout=1.0)
        if self._sub_thread:
            self._sub_thread.join(timeout=1.0)
        if self._context:
            self._context.term()
        logger.info("ZMQBridge stopped")

    def send_command(
        self,
        cmd: Optional[np.ndarray],
        sigma: Optional[np.ndarray] = None,
        halt: bool = False,
        reason: str = "",
    ):
        """Queue a motor command for the next heartbeat publication."""
        msg = {
            "t": time.time(),
            "cmd": cmd.tolist() if cmd is not None else [0.0] * 6,
            "sigma": sigma.tolist() if sigma is not None else [1.0] * 6,
            "halt": halt or cmd is None,
            "reason": reason,
        }
        with self._cmd_lock:
            self._pending_cmd = msg

    def latest_feedback(self) -> Optional[dict]:
        """Return the most recent hardware feedback, or None if not available."""
        with self._feedback_lock:
            return self._latest_feedback

    def watchdog_ok(self) -> bool:
        return (time.monotonic() - self._last_feedback_t) < self.watchdog_s

    # ── Background threads ─────────────────────────────────────────────────────

    def _heartbeat_loop(self):
        interval = self.heartbeat_ms / 1000.0
        while self._running:
            with self._cmd_lock:
                msg = self._pending_cmd
            if msg is not None and self._publisher is not None:
                try:
                    self._publisher.send_string(json.dumps(msg))
                except Exception as e:
                    logger.error("ZMQBridge publish error: %s", e)
            time.sleep(interval)

    def _feedback_loop(self):
        while self._running:
            if self._subscriber is None:
                time.sleep(0.05)
                continue
            try:
                raw = self._subscriber.recv_string()
                fb = json.loads(raw)
                with self._feedback_lock:
                    self._latest_feedback = fb
                self._last_feedback_t = time.monotonic()
            except Exception:
                pass  # timeout or parse error — continue polling


# ── Simulation mock ───────────────────────────────────────────────────────────

class SimulatedHardware:
    """
    Simple physics mock for testing without real hardware.
    Models the arm as a first-order lag system: pos → pos + dt * (cmd - pos).
    """

    def __init__(self, n_dof: int = 6, dt: float = 1.0 / 256, lag: float = 0.05):
        self.n_dof = n_dof
        self.dt = dt
        self.lag = lag  # time constant in seconds
        self.position = np.zeros(n_dof, dtype=np.float32)

    def step(self, command: np.ndarray) -> np.ndarray:
        alpha = self.dt / (self.lag + self.dt)
        self.position += alpha * (command - self.position)
        return self.position.copy()

    def feedback_message(self) -> dict:
        return {"t": time.time(), "pos": self.position.tolist(), "ern": 0.0}
