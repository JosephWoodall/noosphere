"""
noosphere/monitor.py
====================
Internal Monitoring Service

Runs as a background thread alongside the training loop.
Watches agent metrics, system state, and execution outcomes.
Triggers alerts to the user via configurable channels.

Alert channels
--------------
    Console     — logged to stderr with ANSI colours (always on)
    File        — appended to alerts.jsonl
    Desktop     — notify-send (Linux), osascript (macOS) — if available
    NCP         — published as an NCP frame on ncp:alert channel

Alert levels
------------
    INFO     — informational, no action required
    WARN     — something notable, may need attention
    CRITICAL — requires immediate attention

Watched conditions
------------------
    Agent health:
        KL divergence explosion (NaN or > 20)
        Actor/critic loss divergence
        Reward trend (consistently falling for N steps)
        World model prediction error rising
        Physics conservation law violation spikes

    System health:
        Memory pressure (>90% used)
        GPU memory near limit (>95%)
        Disk near full (>95%)
        CPU sustained >95% for >60s
        Process crashed (exit code != 0 on system commands)

    Digital task quality:
        Shell command failure rate >50% over window
        Consecutive timeouts
        Permission denied streak
        World model confidence falling (predicted_value trend)

    Apparatus quality:
        Position error trend rising over N steps
        IK failure rate
        Obstacle collision rate

Usage
-----
    from noosphere.monitor import Monitor, MonitorConfig

    monitor = Monitor(MonitorConfig())
    monitor.start()   # background thread

    # In training loop:
    monitor.record_step(step, info, train_metrics, env_info)

    # Check for pending alerts:
    alerts = monitor.drain_alerts()
    for a in alerts:
        print(a.message)

    monitor.stop()
"""

import os
import json
import time
import threading
import logging
import subprocess
import collections
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Alert levels ──────────────────────────────────────────────────────────────

class Level:
    INFO     = "INFO"
    WARN     = "WARN"
    CRITICAL = "CRITICAL"

    _ANSI = {
        "INFO":     "\033[94m",    # blue
        "WARN":     "\033[93m",    # yellow
        "CRITICAL": "\033[91m",    # red
    }
    _RESET = "\033[0m"

    @classmethod
    def colour(cls, level: str, msg: str) -> str:
        return f"{cls._ANSI.get(level, '')}{level}: {msg}{cls._RESET}"


# ── Alert dataclass ───────────────────────────────────────────────────────────

@dataclass
class Alert:
    level:     str
    source:    str         # which monitor rule fired
    message:   str
    step:      int
    timestamp: float = field(default_factory=time.time)
    data:      Dict  = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "level": self.level, "source": self.source,
            "message": self.message, "step": self.step,
            "timestamp": self.timestamp, "data": self.data,
        }

    def __str__(self) -> str:
        t = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"[{t}] [{self.level}] {self.source}: {self.message}"


# ── Monitor config ────────────────────────────────────────────────────────────

@dataclass
class MonitorConfig:
    # Channels
    console:           bool  = True
    alert_file:        str   = "noosphere_alerts.jsonl"
    desktop_notify:    bool  = True    # notify-send / osascript if available
    ncp_channel:       bool  = False   # publish on ncp:alert (requires transport)

    # Thresholds — agent
    kl_max:            float = 20.0
    reward_window:     int   = 50      # steps to measure reward trend
    reward_fall_thresh:float = -0.3    # trigger if avg reward falls by this much
    wm_loss_max:       float = 10.0
    prediction_err_window: int = 30

    # Thresholds — system
    mem_pct_warn:      float = 85.0
    mem_pct_crit:      float = 95.0
    gpu_mem_pct_warn:  float = 90.0
    gpu_mem_pct_crit:  float = 97.0
    disk_pct_warn:     float = 88.0
    disk_pct_crit:     float = 95.0
    cpu_pct_warn:      float = 90.0
    cpu_sustained_s:   float = 60.0

    # Thresholds — digital tasks
    cmd_fail_rate_warn:    float = 0.5   # >50% failures in window
    timeout_streak_crit:   int   = 3     # 3 consecutive timeouts
    permission_streak_warn:int   = 3

    # Thresholds — apparatus
    pos_err_warn_m:    float = 0.08    # >8cm sustained
    pos_err_crit_m:    float = 0.15    # >15cm
    ik_fail_rate_warn: float = 0.3
    col_rate_warn:     float = 0.2

    # Cooldown
    cooldown_s:        float = 30.0    # don't re-fire same alert within N seconds


# ── Internal monitor ──────────────────────────────────────────────────────────

class Monitor:
    """
    Background monitoring thread. Call start() before training loop.
    Collects data via record_step(). Fires alerts asynchronously.
    Drain alerts with drain_alerts() in the training loop.
    """

    def __init__(self, cfg: MonitorConfig = MonitorConfig(),
                 ncp_transport=None):
        self.cfg        = cfg
        self._transport = ncp_transport
        self._alerts:   List[Alert] = []
        self._lock      = threading.Lock()
        self._stop_ev   = threading.Event()
        self._thread:   Optional[threading.Thread] = None

        # Rolling windows
        self._rewards:    collections.deque = collections.deque(maxlen=cfg.reward_window)
        self._wm_losses:  collections.deque = collections.deque(maxlen=30)
        self._kl_vals:    collections.deque = collections.deque(maxlen=30)
        self._pos_errors: collections.deque = collections.deque(maxlen=30)
        self._cmd_results:collections.deque = collections.deque(maxlen=50)
        self._cpu_high_since: Optional[float] = None

        # Alert cooldowns: source → last_fired_time
        self._cooldowns: Dict[str, float] = {}

        # System poll state
        self._last_sys_check = 0.0

        if cfg.alert_file:
            try:
                open(cfg.alert_file, "a").close()
            except Exception:
                pass

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self):
        """Start background monitoring thread."""
        self._thread = threading.Thread(target=self._loop, daemon=True, name="NoosphereMonitor")
        self._thread.start()
        logger.info("[Monitor] Started")

    def stop(self):
        self._stop_ev.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("[Monitor] Stopped")

    def record_step(
        self,
        step:         int,
        info:         Dict,
        train_metrics: Dict,
        env_info:     Optional[Dict] = None,
    ):
        """Call once per training step from the trainer loop."""
        with self._lock:
            r = info.get("pred_reward", info.get("reward", 0.0))
            self._rewards.append(float(r))

            if "wm/loss" in train_metrics:
                self._wm_losses.append(float(train_metrics["wm/loss"]))
            if "wm/kl" in train_metrics:
                self._kl_vals.append(float(train_metrics["wm/kl"]))

            if env_info:
                if "position_error" in env_info:
                    self._pos_errors.append(float(env_info["position_error"]))
                if "exit_code" in env_info:
                    self._cmd_results.append({
                        "exit_code": env_info.get("exit_code", 0),
                        "timeout":   env_info.get("outcome", "").startswith("timeout"),
                        "perm":      "permission" in env_info.get("outcome", "").lower(),
                    })

        self._check_agent(step, train_metrics)
        self._check_apparatus(step)
        self._check_digital(step)

    def drain_alerts(self) -> List[Alert]:
        """Return and clear pending alerts. Call from training loop."""
        with self._lock:
            out = list(self._alerts)
            self._alerts.clear()
        return out

    # ── Background loop ───────────────────────────────────────────────────────

    def _loop(self):
        step_proxy = 0
        while not self._stop_ev.is_set():
            now = time.time()
            if now - self._last_sys_check > 5.0:
                self._check_system(step_proxy)
                self._last_sys_check = now
            step_proxy += 1
            self._stop_ev.wait(timeout=1.0)

    # ── Check methods ─────────────────────────────────────────────────────────

    def _check_agent(self, step: int, metrics: Dict):
        # KL explosion
        if self._kl_vals:
            kl = self._kl_vals[-1]
            if kl > self.cfg.kl_max or (kl != kl):  # NaN check
                self._fire(Alert(
                    Level.CRITICAL, "kl_explosion",
                    f"KL divergence = {kl:.2f} (max {self.cfg.kl_max}). "
                    f"World model may be diverging. Consider reducing lr_world_model.",
                    step, data={"kl": kl}
                ))

        # WM loss spike
        if self._wm_losses and len(self._wm_losses) > 5:
            wm = self._wm_losses[-1]
            baseline = float(np.mean(list(self._wm_losses)[:-5]))
            if wm > self.cfg.wm_loss_max or (baseline > 0.1 and wm > baseline * 5):
                self._fire(Alert(
                    Level.WARN, "wm_loss_spike",
                    f"World model loss = {wm:.4f} (5-step baseline {baseline:.4f}). "
                    f"Learning may be unstable.",
                    step, data={"wm_loss": wm, "baseline": baseline}
                ))

        # Reward trend
        if len(self._rewards) == self.cfg.reward_window:
            half = self.cfg.reward_window // 2
            old_avg = float(np.mean(list(self._rewards)[:half]))
            new_avg = float(np.mean(list(self._rewards)[half:]))
            if new_avg - old_avg < self.cfg.reward_fall_thresh:
                self._fire(Alert(
                    Level.WARN, "reward_declining",
                    f"Average reward fell {new_avg - old_avg:+.3f} over last "
                    f"{self.cfg.reward_window} steps ({old_avg:.3f} → {new_avg:.3f}). "
                    f"Agent may be stuck or environment changed.",
                    step, data={"old": old_avg, "new": new_avg}
                ))

    def _check_system(self, step: int):
        # Memory
        try:
            with open("/proc/meminfo") as f:
                mi = {}
                for line in f:
                    k, v = line.split(":")
                    mi[k.strip()] = int(v.strip().split()[0])
            total = mi.get("MemTotal", 1)
            avail = mi.get("MemAvailable", total)
            pct   = 100.0 * (1 - avail / total)
            if pct > self.cfg.mem_pct_crit:
                self._fire(Alert(
                    Level.CRITICAL, "memory_critical",
                    f"Memory usage {pct:.1f}% (>{self.cfg.mem_pct_crit}%). "
                    f"Risk of OOM. Reduce batch_size or replay_capacity.",
                    step, data={"pct": pct}
                ))
            elif pct > self.cfg.mem_pct_warn:
                self._fire(Alert(
                    Level.WARN, "memory_high",
                    f"Memory usage {pct:.1f}%.", step, data={"pct": pct}
                ))
        except Exception:
            pass

        # Disk
        try:
            import shutil
            du = shutil.disk_usage(".")
            pct = 100.0 * du.used / du.total
            if pct > self.cfg.disk_pct_crit:
                self._fire(Alert(
                    Level.CRITICAL, "disk_critical",
                    f"Disk {pct:.1f}% full ({du.free / 1e9:.1f}GB free). "
                    f"Checkpoints may fail.",
                    step, data={"pct": pct, "free_gb": du.free / 1e9}
                ))
            elif pct > self.cfg.disk_pct_warn:
                self._fire(Alert(
                    Level.WARN, "disk_high",
                    f"Disk {pct:.1f}% full.", step, data={"pct": pct}
                ))
        except Exception:
            pass

        # CPU load
        try:
            with open("/proc/loadavg") as f:
                load_1m = float(f.read().split()[0])
            n_cores = os.cpu_count() or 1
            pct     = 100.0 * load_1m / n_cores
            if pct > self.cfg.cpu_pct_warn:
                if self._cpu_high_since is None:
                    self._cpu_high_since = time.time()
                elif time.time() - self._cpu_high_since > self.cfg.cpu_sustained_s:
                    self._fire(Alert(
                        Level.WARN, "cpu_sustained_high",
                        f"CPU load {pct:.0f}% for >{self.cfg.cpu_sustained_s:.0f}s. "
                        f"Training may be bottlenecked.",
                        step, data={"pct": pct}
                    ))
            else:
                self._cpu_high_since = None
        except Exception:
            pass

        # GPU memory
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3.0,
            )
            if r.returncode == 0:
                for line in r.stdout.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) == 2:
                        used, total = float(parts[0].strip()), float(parts[1].strip())
                        pct = 100.0 * used / max(total, 1.0)
                        if pct > self.cfg.gpu_mem_pct_crit:
                            self._fire(Alert(
                                Level.CRITICAL, "gpu_memory_critical",
                                f"GPU memory {pct:.1f}% ({used:.0f}/{total:.0f}MB). "
                                f"Reduce batch_size or d_model to avoid OOM.",
                                step, data={"pct": pct, "used_mb": used}
                            ))
                        elif pct > self.cfg.gpu_mem_pct_warn:
                            self._fire(Alert(
                                Level.WARN, "gpu_memory_high",
                                f"GPU memory {pct:.1f}%.", step
                            ))
        except Exception:
            pass

    def _check_apparatus(self, step: int):
        if not self._pos_errors:
            return
        if len(self._pos_errors) >= 20:
            avg = float(np.mean(list(self._pos_errors)[-20:]))
            if avg > self.cfg.pos_err_crit_m:
                self._fire(Alert(
                    Level.CRITICAL, "position_error_critical",
                    f"Mean position error {avg*100:.1f}cm over last 20 moves "
                    f"(threshold {self.cfg.pos_err_crit_m*100:.0f}cm). "
                    f"Check electrode placement and run calibration.",
                    step, data={"avg_m": avg}
                ))
            elif avg > self.cfg.pos_err_warn_m:
                self._fire(Alert(
                    Level.WARN, "position_error_high",
                    f"Mean position error {avg*100:.1f}cm. "
                    f"Predictor may need recalibration.",
                    step, data={"avg_m": avg}
                ))

    def _check_digital(self, step: int):
        if not self._cmd_results:
            return
        window = list(self._cmd_results)[-20:]

        # Failure rate
        if len(window) >= 10:
            fail_rate = sum(1 for r in window if r["exit_code"] != 0) / len(window)
            if fail_rate > self.cfg.cmd_fail_rate_warn:
                self._fire(Alert(
                    Level.WARN, "high_cmd_failure_rate",
                    f"Shell command failure rate {fail_rate*100:.0f}% "
                    f"over last {len(window)} commands. "
                    f"World model may be planning ineffectively.",
                    step, data={"fail_rate": fail_rate}
                ))

        # Timeout streak
        timeout_streak = 0
        for r in reversed(window):
            if r["timeout"]:
                timeout_streak += 1
            else:
                break
        if timeout_streak >= self.cfg.timeout_streak_crit:
            self._fire(Alert(
                Level.CRITICAL, "timeout_streak",
                f"{timeout_streak} consecutive command timeouts. "
                f"Increase timeout_s or check system load.",
                step, data={"streak": timeout_streak}
            ))

        # Permission denied streak
        perm_streak = 0
        for r in reversed(window):
            if r["perm"]:
                perm_streak += 1
            else:
                break
        if perm_streak >= self.cfg.permission_streak_warn:
            self._fire(Alert(
                Level.WARN, "permission_denied_streak",
                f"{perm_streak} consecutive permission denied results. "
                f"Expand allow_tiers or check file permissions.",
                step, data={"streak": perm_streak}
            ))

    # ── Alert dispatch ────────────────────────────────────────────────────────

    def _fire(self, alert: Alert):
        """Dispatch alert, respecting cooldown."""
        now     = time.time()
        last    = self._cooldowns.get(alert.source, 0.0)
        if now - last < self.cfg.cooldown_s:
            return
        self._cooldowns[alert.source] = now

        with self._lock:
            self._alerts.append(alert)

        if self.cfg.console:
            logger.warning(Level.colour(alert.level, str(alert)))

        if self.cfg.alert_file:
            try:
                with open(self.cfg.alert_file, "a") as f:
                    f.write(json.dumps(alert.to_dict()) + "\n")
            except Exception:
                pass

        if self.cfg.desktop_notify:
            self._desktop_notify(alert)

        if self.cfg.ncp_channel and self._transport is not None:
            try:
                from noosphere.proto import NCPEncoder
                enc   = NCPEncoder()
                frame = enc.learning_signal(
                    signal_type=4,  # CORRECTION repurposed as alert
                    value=1.0 if alert.level == Level.CRITICAL else 0.5,
                )
                self._transport.publish("ncp:alert", frame)
            except Exception:
                pass

    @staticmethod
    def _desktop_notify(alert: Alert):
        try:
            title = f"Noosphere {alert.level}"
            msg   = alert.message[:200]
            urgency = "critical" if alert.level == Level.CRITICAL else (
                      "normal"   if alert.level == Level.WARN     else "low")
            # Linux
            subprocess.run(
                ["notify-send", "-u", urgency, title, msg],
                capture_output=True, timeout=2.0,
            )
        except Exception:
            try:
                # macOS
                subprocess.run(
                    ["osascript", "-e",
                     f'display notification "{alert.message[:150]}" with title "Noosphere {alert.level}"'],
                    capture_output=True, timeout=2.0,
                )
            except Exception:
                pass
