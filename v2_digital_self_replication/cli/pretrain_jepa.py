#!/usr/bin/env python3
"""
Step 2 — JEPA self-supervised pretraining of the stream encoder.

No labels required. Predicts the latent of a future EEG window from a past context window.
Saves encoder checkpoint to --checkpoint-dir/jepa_encoder_final.pt.
"""

import argparse
import sys

from v2_digital_self_replication.cli.utils import configure_logging, load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="JEPA pretraining for the v2 stream encoder."
    )
    p.add_argument("--data-dir",       type=str,   default="v2_digital_self_replication/data/generated",
                   help="Directory produced by 01_generate_data.sh")
    p.add_argument("--checkpoint-dir", type=str,   default="v2_digital_self_replication/checkpoints",
                   help="Where to save encoder checkpoints")
    p.add_argument("--n-epochs",       type=int,   default=50,    help="Training epochs")
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--window-len",     type=int,   default=256,   help="EEG samples per JEPA window")
    p.add_argument("--stride",         type=int,   default=64,    help="Window stride (samples)")
    p.add_argument("--d-model",        type=int,   default=128)
    p.add_argument("--d-state",        type=int,   default=64)
    p.add_argument("--n-layers",       type=int,   default=4)
    p.add_argument("--device",         type=str,   default="cpu", help="cpu or cuda")
    p.add_argument("--log-level",      type=str,   default="INFO")
    p.add_argument("--log-file",       type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    configure_logging(args.log_level, args.log_file)

    import logging
    log = logging.getLogger("pretrain_jepa")

    log.info("Loading dataset from %s", args.data_dir)
    dataset, meta = load_dataset(args.data_dir)
    log.info("  %d subjects, %d trials each, T=%d, fs=%d Hz",
             meta["n_subjects"], meta["n_trials"], meta["T"], meta["fs"])

    from v2_digital_self_replication.config import V2Config, EncoderConfig, JEPAConfig
    cfg = V2Config()
    cfg.encoder        = EncoderConfig(d_model=args.d_model, d_state=args.d_state, n_layers=args.n_layers)
    cfg.jepa           = JEPAConfig(n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr)
    cfg.checkpoint_dir = args.checkpoint_dir

    from v2_digital_self_replication.training.pretrain_jepa import JEPATrainer
    trainer = JEPATrainer(config=cfg, device=args.device)

    log.info("Starting JEPA pretraining: %d epochs, window=%d, stride=%d, device=%s",
             args.n_epochs, args.window_len, args.stride, args.device)
    trainer.train(dataset, window_len=args.window_len, stride=args.stride)

    out = f"{args.checkpoint_dir}/jepa_encoder_final.pt"
    log.info("Pretraining complete. Final encoder saved to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
