#!/usr/bin/env python3
"""
Step 1 — Generate synthetic EEG + physiology training data.

Produces .npy archives and a metadata.json in --output-dir.
All downstream scripts load from this directory.
"""

import argparse
import sys
import time

from v2_digital_self_replication.cli.utils import configure_logging, save_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic EEG/HRV/GSR training data for v2 digital twin."
    )
    p.add_argument("--n-subjects",  type=int,   default=10,    help="Number of synthetic subjects")
    p.add_argument("--n-trials",    type=int,   default=50,    help="Trials per subject")
    p.add_argument("--duration",    type=float, default=4.0,   help="Trial duration in seconds")
    p.add_argument("--fs",          type=int,   default=256,   help="EEG sampling rate (Hz)")
    p.add_argument("--seed",        type=int,   default=42,    help="Global random seed")
    p.add_argument("--output-dir",  type=str,   default="v2_digital_self_replication/data/generated",
                   help="Where to write .npy files")
    p.add_argument("--log-level",   type=str,   default="INFO")
    p.add_argument("--log-file",    type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    configure_logging(args.log_level, args.log_file)

    import logging
    log = logging.getLogger("generate_data")

    from v2_digital_self_replication.data.synthetic_eeg import make_training_batch

    T = int(args.duration * args.fs)
    log.info("Generating data: %d subjects × %d trials × %.1fs @ %d Hz  (T=%d samples/trial)",
             args.n_subjects, args.n_trials, args.duration, args.fs, T)
    log.info("Output: %s", args.output_dir)

    t0 = time.time()
    dataset = make_training_batch(
        n_subjects=args.n_subjects,
        n_trials=args.n_trials,
        trial_duration_s=args.duration,
        fs=args.fs,
        seed=args.seed,
    )
    elapsed = time.time() - t0

    sample_sub = dataset[0]
    log.info("Generated in %.1fs", elapsed)
    log.info("  EEG shape per subject:      %s  (trials × timesteps × channels)", sample_sub["eeg"].shape)
    log.info("  Commands shape per subject: %s  (trials × timesteps × dof)", sample_sub["commands"].shape)
    log.info("  ERN labels shape:           %s  (trials × timesteps)", sample_sub["ern_labels"].shape)

    metadata = {
        "n_subjects":  args.n_subjects,
        "n_trials":    args.n_trials,
        "duration_s":  args.duration,
        "fs":          args.fs,
        "seed":        args.seed,
        "n_channels":  21,
        "n_dof":       6,
        "T":           T,
    }
    save_dataset(dataset, args.output_dir, metadata)
    log.info("Saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
