#!/usr/bin/env python3
"""
Step 4 — Run the digital twin inference loop.

Streams synthetic EEG through the trained twin, drives a simulated prosthetic arm,
triggers online adaptation every N steps, and publishes motor commands over ZMQ.

Live dashboard (--dashboard) shows 6-DOF arm state, throughput, safety status, and ZMQ
state using rich.live, updating at ~10 Hz.
"""

import argparse
import json
import signal
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the v2 digital twin inference loop."
    )
    p.add_argument("--checkpoint",   type=str,   default="v2_digital_self_replication/checkpoints/supervised_best.pt")
    p.add_argument("--subject-id",   type=int,   default=1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n-steps",      type=int,   default=2560,
                   help="Steps to run (0 = run until Ctrl-C)")
    p.add_argument("--intent",       type=float, nargs=6,
                   default=[0.5, 0.3, 0.0, 0.2, 0.0, 0.4])
    p.add_argument("--log-interval", type=int,   default=256)
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--session-log",  type=str,   default="v2_digital_self_replication/logs/session_latest.json")
    p.add_argument("--log-level",    type=str,   default="INFO")
    p.add_argument("--log-file",     type=str,   default=None)
    p.add_argument("--zmq",          action="store_true",  default=False,
                   help="Enable ZMQ bridge (publishes commands on port 5555)")
    p.add_argument("--no-dashboard", action="store_true",  default=False,
                   help="Disable rich live dashboard, print plain log lines instead")
    return p.parse_args()


_DOF_NAMES = [
    "shoulder_yaw", "shoulder_pitch", "shoulder_roll",
    "elbow_flex",   "wrist_rotate",   "grip_aperture",
]


def _make_dashboard(step, hz, n_cmd, n_halt, mean_err, hw_pos, safety_stats, zmq_active):
    from rich.console import Group
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text

    BAR_W = 22

    tbl = Table.grid(padding=(0, 1))
    tbl.add_column(style="cyan",        width=18)
    tbl.add_column(width=BAR_W + 2)
    tbl.add_column(style="bright_white", width=7, justify="right")

    for name, val in zip(_DOF_NAMES, hw_pos):
        filled = max(0, min(BAR_W, int((val + 1.0) / 2.0 * BAR_W)))
        bar = f"[green]{'█' * filled}[/][dim]{'░' * (BAR_W - filled)}[/]"
        tbl.add_row(name, bar, f"{val:+.3f}")

    header = Text.from_markup(
        f"[bold]step[/] {step:>6}  "
        f"[bold]Hz[/] {hz:>5.0f}  "
        f"[bold]cmds[/] {n_cmd}  "
        f"[bold]halts[/] {n_halt}  "
        f"[bold]err[/] {mean_err:.4f}"
    )

    halts = safety_stats.get("total_halts", 0)
    safety_str = "[green]OK[/]" if halts == 0 else f"[red]halts={halts}[/]"
    zmq_str    = "[green]active[/]" if zmq_active else "[dim]sim[/]"
    footer = Text.from_markup(f"safety {safety_str}  zmq {zmq_str}")

    return Panel(
        Group(header, Rule(style="dim"), tbl, Rule(style="dim"), footer),
        title="[bold cyan]EEG Digital Twin[/]",
        border_style="cyan",
    )


def main():
    args = parse_args()

    from v2_digital_self_replication.cli.utils import configure_logging
    configure_logging(args.log_level, args.log_file)

    import logging
    import numpy as np

    from v2_digital_self_replication.agent.digital_twin import DigitalTwin
    from v2_digital_self_replication.data.synthetic_eeg import EEGStreamGenerator
    from v2_digital_self_replication.comms.zmq_bridge import ZMQBridge, SimulatedHardware

    log = logging.getLogger("run_twin")

    # ── Load twin ─────────────────────────────────────────────────────────────
    twin = DigitalTwin()
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        twin.load(str(ckpt))
        log.info("Loaded checkpoint: %s", ckpt)
    else:
        log.warning("Checkpoint not found at %s — running with untrained weights", ckpt)

    if args.device != "cpu":
        import torch
        twin = twin.to(args.device)
        log.info("Model on %s", args.device)

    twin.eval()
    twin.reset_state()

    # ── Sensor and hardware mocks ─────────────────────────────────────────────
    gen    = EEGStreamGenerator(seed=args.seed, subject_id=args.subject_id)
    hw     = SimulatedHardware()
    intent = np.array(args.intent, dtype=np.float32)

    # ── ZMQ bridge (optional) ─────────────────────────────────────────────────
    bridge = None
    if args.zmq:
        bridge = ZMQBridge()
        bridge.start()
        log.info("ZMQBridge started")

    log.info("Starting inference loop:")
    log.info("  Steps:   %s", args.n_steps if args.n_steps > 0 else "∞ (Ctrl-C to stop)")
    log.info("  Intent:  %s", np.round(intent, 2).tolist())
    log.info("  ZMQ:     %s", "enabled" if bridge else "disabled")
    log.info("  Dashboard: %s", "disabled" if args.no_dashboard else "enabled")

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    _running = [True]
    def _stop(sig, frame):
        log.info("Caught signal %d — stopping after current step.", sig)
        _running[0] = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── Inference loop ────────────────────────────────────────────────────────
    n_cmd  = 0
    n_halt = 0
    t_start = time.monotonic()
    step = 0

    DASHBOARD_INTERVAL = 50  # update display every 50 steps (~10 Hz at 490 Hz throughput)

    def _run_loop(live=None):
        nonlocal n_cmd, n_halt, step

        while _running[0]:
            if args.n_steps > 0 and step >= args.n_steps:
                break

            eeg  = gen.step(intent)
            hrv  = np.array([70.0 + np.random.randn() * 3],  dtype=np.float32)
            gsr  = np.array([5.0  + np.random.rand()  * 0.5], dtype=np.float32)

            # Use ZMQ feedback position when available, else simulated hardware
            if bridge is not None:
                fb = bridge.latest_feedback()
                prop = np.array(fb["pos"], dtype=np.float32) if fb else hw.position
            else:
                prop = hw.position

            cmd = twin.step(eeg, hrv=hrv, gsr=gsr, prop=prop)

            # Forward command to ZMQ and simulated hardware
            sigma = getattr(twin, "_last_sigma", None)
            if cmd is not None:
                if bridge is not None:
                    bridge.send_command(cmd, sigma)
                hw.step(cmd)
                twin.observe_outcome(hw.position, eeg_window=eeg.reshape(1, -1))
                n_cmd += 1
            else:
                if bridge is not None:
                    bridge.send_command(None, sigma, halt=True)
                n_halt += 1

            if twin.should_adapt():
                twin.adapt()

            step += 1

            if live is not None and step % DASHBOARD_INTERVAL == 0:
                elapsed = time.monotonic() - t_start
                hz      = step / max(elapsed, 1e-6)
                live.update(_make_dashboard(
                    step, hz, n_cmd, n_halt,
                    twin.session_mean_error,
                    hw.position,
                    twin.safety_stats,
                    bridge is not None and bridge.watchdog_ok() if bridge else False,
                ))
            elif live is None and step % args.log_interval == 0:
                elapsed = time.monotonic() - t_start
                hz      = step / max(elapsed, 1e-6)
                log.info("step %5d  cmds=%d  halts=%d  mean_err=%.4f  pos=%s  %.0f Hz",
                         step, n_cmd, n_halt,
                         twin.session_mean_error,
                         np.round(hw.position, 3).tolist(),
                         hz)

    if args.no_dashboard:
        _run_loop(live=None)
    else:
        try:
            from rich.live import Live
            with Live(_make_dashboard(0, 0, 0, 0, float("nan"), hw.position,
                                      twin.safety_stats, False),
                      refresh_per_second=10, screen=False) as live:
                _run_loop(live=live)
        except ImportError:
            log.warning("rich not installed — falling back to plain logging")
            _run_loop(live=None)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if bridge is not None:
        bridge.stop()

    # ── Session summary ───────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    summary = {
        "total_steps":    step,
        "commands_sent":  n_cmd,
        "halts":          n_halt,
        "mean_error":     twin.session_mean_error,
        "duration_s":     round(elapsed, 2),
        "steps_per_s":    round(step / max(elapsed, 1e-6), 1),
        "safety_stats":   twin.safety_stats,
        "final_position": hw.position.tolist(),
        "checkpoint":     str(ckpt),
    }

    log.info("Session complete: %d steps in %.1fs (%.0f Hz)",
             step, elapsed, step / max(elapsed, 1e-6))
    log.info("  Commands: %d  Halts: %d  Mean error: %.4f",
             n_cmd, n_halt, twin.session_mean_error)

    out = Path(args.session_log)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("Session log written to %s", out)

    twin.log_session_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
