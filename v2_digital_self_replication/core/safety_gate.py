"""
Safety gate for prosthetic arm control.

Three independent mechanisms (any can halt independently):
  1. ERN gate       — neural error signal detected by decoder (P(ERN) > threshold)
  2. Uncertainty    — decoder sigma exceeds safe threshold (command unreliable)
  3. Watchdog       — no valid input received for > timeout_s (connection lost)
  4. Emergency stop — explicit call, latches until reset()

HALT returns None.  The gate tracks all events for post-session audit.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SafetyConfig:
    ern_threshold: float = 0.7
    sigma_threshold: float = 1.5
    ern_halt_duration: float = 0.5
    sigma_halt_duration: float = 0.1
    watchdog_timeout: float = 2.0


@dataclass
class HaltEvent:
    timestamp: float
    reason: str
    ern_prob: float
    max_sigma: float


class SafetyGate:
    def __init__(self, config: Optional[SafetyConfig] = None):
        self.cfg = config or SafetyConfig()
        self._emergency: bool = False
        self._halt_until: float = 0.0
        self._halt_reason: str = ""
        self._last_valid_t: float = time.monotonic()
        self._event_log: list[HaltEvent] = []
        self._counts: dict[str, int] = {
            "ern": 0, "uncertainty": 0, "watchdog": 0, "emergency": 0
        }

    def check(
        self,
        command: np.ndarray,
        sigma: np.ndarray,
        ern_prob: float,
    ) -> tuple[Optional[np.ndarray], str]:
        """
        Validate command before sending to hardware.
        Returns (clipped_command, "") if safe, or (None, reason) if halted.
        """
        now = time.monotonic()
        self._last_valid_t = now

        if self._emergency:
            return self._halt("emergency", ern_prob, sigma, now, latch=True)

        if now < self._halt_until:
            return None, self._halt_reason

        if now - self._last_valid_t > self.cfg.watchdog_timeout:
            return self._halt("watchdog", ern_prob, sigma, now,
                              duration=1.0)

        if ern_prob > self.cfg.ern_threshold:
            return self._halt("ern", ern_prob, sigma, now,
                              duration=self.cfg.ern_halt_duration)

        if float(np.max(sigma)) > self.cfg.sigma_threshold:
            return self._halt("uncertainty", ern_prob, sigma, now,
                              duration=self.cfg.sigma_halt_duration)

        return np.clip(command, -1.0, 1.0), ""

    def _halt(
        self,
        reason: str,
        ern_prob: float,
        sigma: np.ndarray,
        now: float,
        duration: float = 0.0,
        latch: bool = False,
    ) -> tuple[None, str]:
        self._halt_reason = reason
        if not latch:
            self._halt_until = now + duration
        self._counts[reason] = self._counts.get(reason, 0) + 1
        self._event_log.append(
            HaltEvent(now, reason, float(ern_prob), float(np.max(sigma)))
        )
        logger.warning("SafetyGate HALT: %s (ern=%.3f, max_σ=%.3f)", reason, ern_prob, np.max(sigma))
        return None, reason

    def emergency_stop(self):
        self._emergency = True
        logger.critical("SafetyGate: EMERGENCY STOP")

    def reset(self):
        """Clear emergency latch and any active halt."""
        self._emergency = False
        self._halt_until = 0.0
        self._halt_reason = ""
        self._last_valid_t = time.monotonic()
        logger.info("SafetyGate: reset")

    @property
    def stats(self) -> dict:
        return {**self._counts, "total_halts": sum(self._counts.values())}

    @property
    def event_log(self) -> list[HaltEvent]:
        return list(self._event_log)
