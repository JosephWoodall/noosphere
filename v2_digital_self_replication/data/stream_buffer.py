"""
Ring buffer for real-time multi-channel biosignal streaming.

Provides a thread-safe, fixed-capacity sliding window over incoming samples.
The buffer is pre-allocated so append() never triggers heap allocation.

Usage:
    buf = StreamBuffer(n_channels=21, capacity=512)  # 2s @ 256 Hz
    buf.append(eeg_sample_21)           # one sample (21,)
    window = buf.get_window(256)        # last 256 samples → (256, 21)
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class StreamBuffer:
    """
    Lock-protected ring buffer for streaming biosignal samples.

    append() and get_window() are thread-safe via a reentrant lock.
    get_window() returns a contiguous copy — safe to pass to PyTorch.
    """

    def __init__(self, n_channels: int, capacity: int, dtype=np.float32):
        self.n_channels = n_channels
        self.capacity = capacity
        self._buf = np.zeros((capacity, n_channels), dtype=dtype)
        self._head = 0       # next write position
        self._count = 0      # number of valid samples
        self._lock = threading.RLock()

    def append(self, sample: np.ndarray):
        """Append one sample (n_channels,) or batch (T, n_channels)."""
        with self._lock:
            if sample.ndim == 1:
                self._buf[self._head % self.capacity] = sample
                self._head += 1
                self._count = min(self._count + 1, self.capacity)
            else:
                for row in sample:
                    self._buf[self._head % self.capacity] = row
                    self._head += 1
                    self._count = min(self._count + 1, self.capacity)

    def get_window(self, n: Optional[int] = None) -> np.ndarray:
        """
        Return last `n` samples as a contiguous (n, n_channels) array.
        If n > valid samples, zero-pads from the front.
        """
        with self._lock:
            n = n if n is not None else self.capacity
            if self._count == 0:
                return np.zeros((n, self.n_channels), dtype=self._buf.dtype)

            valid = min(self._count, n)
            tail = self._head % self.capacity
            start = (tail - valid) % self.capacity

            if start + valid <= self.capacity:
                chunk = self._buf[start: start + valid]
            else:
                part1 = self._buf[start:]
                part2 = self._buf[: valid - len(part1)]
                chunk = np.concatenate([part1, part2], axis=0)

            if valid < n:
                pad = np.zeros((n - valid, self.n_channels), dtype=self._buf.dtype)
                return np.concatenate([pad, chunk], axis=0)

            return chunk.copy()

    def clear(self):
        with self._lock:
            self._buf[:] = 0.0
            self._head = 0
            self._count = 0

    @property
    def n_samples(self) -> int:
        with self._lock:
            return self._count


class MultiModalBuffer:
    """
    Synchronized buffers for EEG, HRV, GSR, and proprioception.
    All channels share the same capacity and can be fetched in one call.
    """

    def __init__(
        self,
        n_eeg: int = 21,
        n_prop: int = 6,
        capacity: int = 512,
    ):
        self.eeg  = StreamBuffer(n_eeg,  capacity)
        self.hrv  = StreamBuffer(1,       capacity)
        self.gsr  = StreamBuffer(1,       capacity)
        self.prop = StreamBuffer(n_prop,  capacity)

    def append(
        self,
        eeg: np.ndarray,
        hrv: Optional[np.ndarray] = None,
        gsr: Optional[np.ndarray] = None,
        prop: Optional[np.ndarray] = None,
    ):
        self.eeg.append(eeg)
        if hrv is not None:
            self.hrv.append(hrv)
        if gsr is not None:
            self.gsr.append(gsr)
        if prop is not None:
            self.prop.append(prop)

    def get_window(self, n: int) -> dict:
        return {
            "eeg":  self.eeg.get_window(n),
            "hrv":  self.hrv.get_window(n),
            "gsr":  self.gsr.get_window(n),
            "prop": self.prop.get_window(n),
        }

    def clear(self):
        for buf in (self.eeg, self.hrv, self.gsr, self.prop):
            buf.clear()
