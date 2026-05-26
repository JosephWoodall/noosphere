#!/usr/bin/env python3
"""
Step 5 — Subject calibration loop.

For each movement target, displays a countdown, captures synthetic EEG
(or real EEG in future via ZMQ/LSL), fine-tunes the twin on the captured
data, and saves a calibrated checkpoint.

Usage:
    python -m v2_digital_self_replication.cli.calibrate \\
        --checkpoint checkpoints/supervised_best.pt \\
        --output     checkpoints/calibrated.pt \\
        --n-reps     3 \\
        --capture-s  2.0
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# 6-DOF movement targets: (label, intent_vector)
CALIBRATION_TARGETS = [
    ("REACH FORWARD",    [0.8,  0.5,  0.0,  0.6,  0.0,  0.3]),
    ("ELBOW FLEX",       [0.0,  0.3,  0.0,  0.9,  0.0,  0.0]),
    ("GRIP CLOSE",       [0.0,  0.0,  0.0,  0.0,  0.0,  0.9]),
    ("GRIP OPEN",        [0.0,  0.0,  0.0,  0.0,  0.0, -0.9]),
    ("WRIST ROTATE CW",  [0.0,  0.0,  0.0,  0.0,  0.9,  0.0]),
    ("WRIST ROTATE CCW", [0.0,  0.0,  0.0,  0.0, -0.9,  0.0]),
    ("REST",             [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]),
]


def parse_args():
    p = argparse.ArgumentParser(description="Subject calibration for the v2 digital twin.")
    p.add_argument("--checkpoint", type=str,
                   default="v2_digital_self_replication/checkpoints/supervised_best.pt")
    p.add_argument("--output",     type=str,
                   default="v2_digital_self_replication/checkpoints/calibrated.pt")
    p.add_argument("--subject-id", type=int,  default=1)
    p.add_argument("--seed",       type=int,  default=0)
    p.add_argument("--n-reps",     type=int,  default=3,   help="Repetitions per target")
    p.add_argument("--capture-s",  type=float, default=2.0, help="Capture duration (seconds) per rep")
    p.add_argument("--fs",         type=int,  default=256,  help="EEG sample rate (Hz)")
    p.add_argument("--ft-epochs",  type=int,  default=5,   help="Fine-tune epochs on captured data")
    p.add_argument("--device",     type=str,  default="cpu")
    p.add_argument("--log-level",  type=str,  default="INFO")
    return p.parse_args()


def _countdown(label: str, secs: int = 3):
    print(f"\n{'─'*50}")
    print(f"  TARGET: {label}")
    print(f"{'─'*50}")
    for i in range(secs, 0, -1):
        print(f"  Get ready... {i}", end="\r", flush=True)
        time.sleep(1.0)
    print("  GO!                    ", flush=True)


def _capture_samples(gen, intent: np.ndarray, n_samples: int, fs: int):
    """Capture n_samples of synthetic EEG driven by intent."""
    eeg_buf  = np.zeros((n_samples, 21), dtype=np.float32)
    cmd_buf  = np.zeros((n_samples, 6),  dtype=np.float32)
    for i in range(n_samples):
        eeg_buf[i]  = gen.step(intent)
        cmd_buf[i]  = intent          # ground-truth label = intent
        time.sleep(1.0 / fs)
    return eeg_buf, cmd_buf


def main():
    args = parse_args()

    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("calibrate")

    import torch
    from v2_digital_self_replication.agent.digital_twin import DigitalTwin
    from v2_digital_self_replication.data.synthetic_eeg import EEGStreamGenerator
    from v2_digital_self_replication.training.online_train import SupervisedTrainer

    np.random.seed(args.seed)

    # ── Load twin ─────────────────────────────────────────────────────────────
    twin = DigitalTwin()
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        twin.load(str(ckpt))
        log.info("Loaded checkpoint: %s", ckpt)
    else:
        log.warning("No checkpoint at %s — calibrating from scratch", ckpt)

    if args.device != "cpu":
        twin = twin.to(args.device)

    gen = EEGStreamGenerator(seed=args.seed, subject_id=args.subject_id)
    n_samples = int(args.capture_s * args.fs)

    # ── Capture loop ──────────────────────────────────────────────────────────
    all_eeg  = []
    all_cmds = []

    print(f"\n{'='*50}")
    print(f"  CALIBRATION — {len(CALIBRATION_TARGETS)} movements × {args.n_reps} reps")
    print(f"  Capture: {args.capture_s}s per rep ({n_samples} samples @ {args.fs} Hz)")
    print(f"{'='*50}")
    print("\nPress ENTER to begin...")
    input()

    for label, intent_list in CALIBRATION_TARGETS:
        intent = np.array(intent_list, dtype=np.float32)
        for rep in range(args.n_reps):
            _countdown(f"{label}  (rep {rep+1}/{args.n_reps})", secs=3)
            eeg_rep, cmd_rep = _capture_samples(gen, intent, n_samples, args.fs)
            all_eeg.append(eeg_rep)
            all_cmds.append(cmd_rep)
            print(f"  Captured {n_samples} samples for [{label}]")

    eeg_data  = np.concatenate(all_eeg,  axis=0)   # (N, 21)
    cmd_data  = np.concatenate(all_cmds, axis=0)   # (N, 6)
    log.info("Total calibration samples: %d", len(eeg_data))

    # ── Fine-tune on captured data ────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Fine-tuning for {args.ft_epochs} epoch(s) on {len(eeg_data)} samples...")
    print(f"{'='*50}\n")

    # Build windowed dataset from captured raw samples
    W = twin.cfg.encoder.d_state  # use d_state as window length for speed
    dataset = {}
    n_wins = len(eeg_data) // W
    if n_wins > 0:
        eeg_wins  = eeg_data[:n_wins*W].reshape(n_wins, W, 21)
        cmd_wins  = cmd_data[W-1:n_wins*W:W]          # label = last frame of each window
        cmd_wins  = cmd_wins[:n_wins]
        ern_wins  = np.zeros(n_wins, dtype=np.float32)
        dataset[0] = {"eeg": eeg_wins, "commands": cmd_wins, "ern_labels": ern_wins}
    else:
        log.warning("Not enough samples to build windows — skipping fine-tune")
        dataset = None

    if dataset:
        trainer = SupervisedTrainer(twin, device=args.device)
        trainer.train(dataset, n_epochs=args.ft_epochs, window_len=W, batch_size=min(32, n_wins))
        log.info("Fine-tuning complete")

    # ── Save calibrated checkpoint ────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    twin.save(str(out))
    print(f"\n  Calibrated checkpoint saved → {out}")
    log.info("Calibrated checkpoint: %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
