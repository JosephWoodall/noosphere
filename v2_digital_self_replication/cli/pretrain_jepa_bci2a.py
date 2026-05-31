#!/usr/bin/env python3
"""
JEPA self-supervised pretraining on BCI Competition IV Dataset 2a.

Loads all 9 subjects' EEG (self-supervised, no labels), formats the data
for JEPATrainer, and saves the encoder to --output.

Usage
-----
  python -m v2_digital_self_replication.cli.pretrain_jepa_bci2a \\
      --n-epochs 30 --device cpu \\
      --output v2_digital_self_replication/checkpoints/jepa_encoder_bci2a.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("pretrain_jepa_bci2a")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs",    type=int,   default=30)
    p.add_argument("--batch-size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--window-len",  type=int,   default=256)
    p.add_argument("--stride",      type=int,   default=64)
    p.add_argument("--subjects",    type=str,   default="1-9")
    p.add_argument("--device",      type=str,   default="cpu")
    p.add_argument("--output",      type=str,
                   default="v2_digital_self_replication/checkpoints/jepa_encoder_bci2a.pt")
    p.add_argument("--log-level",   type=str,   default="INFO")
    return p.parse_args()


def _parse_subjects(spec):
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from v2_digital_self_replication.data.bci2a_loader import load_bci2a_trials
    from v2_digital_self_replication.config import V2Config, EncoderConfig, JEPAConfig
    from v2_digital_self_replication.training.pretrain_jepa import JEPATrainer

    subjects = _parse_subjects(args.subjects)
    log.info("Loading BCI2a subjects %s for JEPA pretraining …", subjects)

    # Build eeg_dict: {subj_id: {"eeg": (n_trials, T, 21)}}
    eeg_dict = {}
    for subj in subjects:
        try:
            eeg, _ = load_bci2a_trials(
                subjects=[subj],
                classes=["left_hand", "right_hand", "feet", "tongue"],
            )
            eeg_dict[subj] = {"eeg": eeg}   # (n_trials, T, 21)
            log.info("  Subject %d: %d trials shape=%s", subj, eeg.shape[0], eeg.shape)
        except Exception as e:
            log.warning("  Subject %d failed: %s", subj, e)

    if not eeg_dict:
        log.error("No data loaded"); return 1

    total_trials = sum(v["eeg"].shape[0] for v in eeg_dict.values())
    log.info("Total: %d trials across %d subjects", total_trials, len(eeg_dict))

    cfg = V2Config()
    cfg.encoder = EncoderConfig(d_model=128, d_state=64, n_layers=4,
                                n_eeg_channels=21, dropout=0.1)
    cfg.jepa    = JEPAConfig(n_epochs=args.n_epochs, batch_size=args.batch_size,
                             lr=args.lr)
    cfg.checkpoint_dir = str(Path(args.output).parent)

    trainer = JEPATrainer(config=cfg, device=args.device)
    log.info("JEPA pretraining: %d epochs, window=%d, stride=%d, device=%s",
             args.n_epochs, args.window_len, args.stride, args.device)

    trainer.train(eeg_dict, window_len=args.window_len, stride=args.stride)

    # Save encoder only (same format as existing JEPA checkpoints)
    import torch
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": trainer.encoder.state_dict()}, str(out))
    log.info("Saved BCI2a JEPA encoder → %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
