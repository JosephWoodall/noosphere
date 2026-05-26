#!/usr/bin/env python3
"""
Step 4 — Run the digital twin inference loop.

Streams synthetic EEG through the trained twin, drives a simulated prosthetic arm,
and triggers online adaptation every N steps.

Prints a live status line and writes a JSON session summary on exit.
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
    p.add_argument("--checkpoint",     type=str,   default="v2_digital_self_replication/checkpoints/supervised_best.pt",
                   help="Twin checkpoint from step 3.")
    p.add_argument("--subject-id",     type=int,   default=1,     help="Synthetic subject profile (seed)")
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--n-steps",        type=int,   default=2560,  help="Steps to run (0 = run until Ctrl-C)")
    p.add_argument("--intent",         type=float, nargs=6,
                   default=[0.5, 0.3, 0.0, 0.2, 0.0, 0.4],
                   help="6-DOF intent vector fed to the synthetic EEG generator")
    p.add_argument("--log-interval",   type=int,   default=256,   help="Steps between status prints")
    p.add_argument("--device",         type=str,   default="cpu")
    p.add_argument("--session-log",    type=str,   default="v2_digital_self_replication/logs/session_latest.json")
    p.add_argument("--log-level",      type=str,   default="INFO")
    p.add_argument("--log-file",       type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    from v2_digital_self_replication.cli.utils import configure_logging
    configure_logging(args.log_level, args.log_file)

    import logging
    import numpy as np

    from v2_digital_self_replication.agent.digital_twin import DigitalTwin
    from v2_digital_self_replication.data.synthetic_eeg import EEGStreamGenerator
    from v2_digital_self_replication.comms.zmq_bridge import SimulatedHardware

    log = logging.getLogger("run_twin")

    # ── Load twin ─────────────────────────────────────────────────────────────
    twin = DigitalTwin()
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        twin.load(str(ckpt))
        log.info("Loaded checkpoint: %s", ckpt)
    else:
        log.warning("Checkpoint not found at %s — running with untrained weights", ckpt)

    twin.eval()
    twin.reset_state()

    # ── Sensor and hardware mocks ─────────────────────────────────────────────
    gen = EEGStreamGenerator(seed=args.seed, subject_id=args.subject_id)
    hw  = SimulatedHardware()
    intent = np.array(args.intent, dtype=np.float32)

    log.info("Starting inference loop:")
    log.info("  Steps: %s", args.n_steps if args.n_steps > 0 else "∞ (Ctrl-C to stop)")
    log.info("  Intent: %s", np.round(intent, 2).tolist())
    log.info("  Subject ID: %d  Seed: %d", args.subject_id, args.seed)

    # ── Graceful shutdown ─────────────────────────────────────────────────────
    _running = [True]
    def _stop(sig, frame):
        log.info("Caught signal %d — stopping after current step.", sig)
        _running[0] = False
    signal.signal(signal.SIGINT,  _stop)
    signal.signal(signal.SIGTERM, _stop)

    # ── Inference loop ────────────────────────────────────────────────────────
    n_cmd = 0
    n_halt = 0
    t_start = time.monotonic()
    step = 0

    while _running[0]:
        if args.n_steps > 0 and step >= args.n_steps:
            break

        eeg  = gen.step(intent)
        hrv  = np.array([70.0 + np.random.randn() * 3], dtype=np.float32)
        gsr  = np.array([5.0  + np.random.rand()  * 0.5], dtype=np.float32)
        prop = hw.position

        cmd = twin.step(eeg, hrv=hrv, gsr=gsr, prop=prop)

        if cmd is not None:
            hw.step(cmd)
            twin.observe_outcome(hw.position, eeg_window=eeg.reshape(1, -1))
            n_cmd += 1
        else:
            n_halt += 1

        if twin.should_adapt():
            twin.adapt()

        step += 1

        if step % args.log_interval == 0:
            elapsed = time.monotonic() - t_start
            hz = step / elapsed
            log.info("step %5d  cmds=%d  halts=%d  mean_err=%.4f  pos=%s  %.0f Hz",
                     step, n_cmd, n_halt,
                     twin.session_mean_error,
                     np.round(hw.position, 3).tolist(),
                     hz)

    # ── Session summary ───────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    summary = {
        "total_steps":   step,
        "commands_sent": n_cmd,
        "halts":         n_halt,
        "mean_error":    twin.session_mean_error,
        "duration_s":    round(elapsed, 2),
        "steps_per_s":   round(step / max(elapsed, 1e-6), 1),
        "safety_stats":  twin.safety_stats,
        "final_position": hw.position.tolist(),
        "checkpoint":    str(ckpt),
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
