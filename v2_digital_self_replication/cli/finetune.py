#!/usr/bin/env python3
"""
Step 3 — Supervised fine-tuning of the digital twin.

Loads the JEPA-pretrained encoder, optionally freezes it,
and fine-tunes on labeled (EEG window → motor command) pairs.
Saves the full twin checkpoint to --checkpoint-dir/supervised_best.pt.
"""

import argparse
import sys

from v2_digital_self_replication.cli.utils import configure_logging, load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Supervised fine-tuning for the v2 digital twin."
    )
    p.add_argument("--data-dir",        type=str,   default="v2_digital_self_replication/data/generated")
    p.add_argument("--encoder-ckpt",    type=str,   default="v2_digital_self_replication/checkpoints/jepa_encoder_final.pt",
                   help="JEPA encoder checkpoint from step 2. Omit to train from scratch.")
    p.add_argument("--checkpoint-dir",  type=str,   default="v2_digital_self_replication/checkpoints")
    p.add_argument("--n-epochs",        type=int,   default=20)
    p.add_argument("--batch-size",      type=int,   default=64)
    p.add_argument("--window-len",      type=int,   default=256)
    p.add_argument("--freeze-encoder",  action="store_true", default=True,
                   help="Freeze pretrained encoder backbone (only last block + decoder adapt)")
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    p.add_argument("--d-model",         type=int,   default=128)
    p.add_argument("--d-state",         type=int,   default=64)
    p.add_argument("--n-layers",        type=int,   default=4)
    p.add_argument("--device",          type=str,   default="cpu")
    p.add_argument("--log-level",       type=str,   default="INFO")
    p.add_argument("--log-file",        type=str,   default=None)
    return p.parse_args()


def main():
    args = parse_args()
    configure_logging(args.log_level, args.log_file)

    import logging
    from pathlib import Path
    log = logging.getLogger("finetune")

    log.info("Loading dataset from %s", args.data_dir)
    dataset, meta = load_dataset(args.data_dir)
    log.info("  %d subjects, %d trials, T=%d", meta["n_subjects"], meta["n_trials"], meta["T"])

    from v2_digital_self_replication.config import V2Config, EncoderConfig
    cfg = V2Config()
    cfg.encoder        = EncoderConfig(d_model=args.d_model, d_state=args.d_state, n_layers=args.n_layers)
    cfg.checkpoint_dir = args.checkpoint_dir

    from v2_digital_self_replication.agent.digital_twin import DigitalTwin
    from v2_digital_self_replication.training.pretrain_jepa import JEPATrainer
    from v2_digital_self_replication.training.online_train import SupervisedTrainer

    twin = DigitalTwin(config=cfg)

    # Load pretrained encoder if available
    if Path(args.encoder_ckpt).exists():
        trainer_loader = JEPATrainer(config=cfg, device=args.device)
        trainer_loader.load_encoder_into_twin(twin.encoder, args.encoder_ckpt)
        log.info("Loaded JEPA encoder from %s", args.encoder_ckpt)
    else:
        log.warning("Encoder checkpoint not found at %s — training from scratch", args.encoder_ckpt)

    sup = SupervisedTrainer(
        twin,
        config=cfg,
        freeze_encoder=args.freeze_encoder,
        device=args.device,
    )
    log.info("Fine-tuning: %d epochs, freeze_encoder=%s, device=%s",
             args.n_epochs, args.freeze_encoder, args.device)

    sup.train(dataset, n_epochs=args.n_epochs, window_len=args.window_len, batch_size=args.batch_size)

    out = f"{args.checkpoint_dir}/supervised_best.pt"
    log.info("Fine-tuning complete. Twin checkpoint saved to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
