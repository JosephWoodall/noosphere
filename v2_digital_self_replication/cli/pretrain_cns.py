"""
CLI: Phase 2 CNS pretraining + cross-modal EEG retraining.

Steps executed in order:
  1. Load NLB MC_Maze spike data (download + cache on first run)
  2. Pretrain CNS encoder via self-supervised JEPA on spike trains
  3. Freeze CNS encoder
  4. Retrain EEG encoder via cross-modal Sinkhorn OT JEPA
  5. Save EEG checkpoint to checkpoints/jepa_encoder_cns_pretrained.pt

Usage:
    .venv/bin/python -m v2_digital_self_replication.cli.pretrain_cns [options]

Then re-run LOSO:
    .venv/bin/python -m v2_digital_self_replication.cli.eval_loso \\
        --subjects 1-20 --classes left_hand right_hand feet \\
        --folds 5 --ft-epochs 10 --device cpu \\
        --checkpoint v2_digital_self_replication/checkpoints/supervised_best_cns.pt \\
        --output v2_digital_self_replication/logs/loso_results_cns.json
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pretrain_cns")


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: CNS pretraining + cross-modal JEPA")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--cns-epochs", type=int, default=50, help="CNS JEPA pretraining epochs")
    p.add_argument("--eeg-epochs", type=int, default=50, help="Cross-modal EEG epochs")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ot-epsilon", type=float, default=1.0, help="Sinkhorn regularisation (1.0 for unit-sphere latents)")
    p.add_argument("--ot-weight", type=float, default=1.0, help="OT loss weight")
    p.add_argument("--ema-weight", type=float, default=0.5, help="Unimodal EMA loss weight")
    p.add_argument("--window-len", type=int, default=256)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--skip-cns-pretrain", action="store_true",
                   help="Skip CNS pretraining if checkpoint already exists")
    p.add_argument("--cns-checkpoint", default=None,
                   help="Path to existing CNS checkpoint (skips CNS pretraining)")
    p.add_argument("--eeg-checkpoint", default=None,
                   help="Path to existing EEG checkpoint to warm-start cross-modal training")
    return p.parse_args()


def main():
    args = parse_args()

    from v2_digital_self_replication.config import V2Config
    from v2_digital_self_replication.data.nlb_loader import NLBLoader
    from v2_digital_self_replication.core.cns_encoder import CNSEncoder, load_frozen_cns_encoder
    from v2_digital_self_replication.training.cross_modal_jepa import (
        CNSJEPATrainer, CrossModalJEPATrainer,
    )
    from v2_digital_self_replication.core.stream_encoder import StreamEncoder

    cfg = V2Config()
    cfg.jepa.n_epochs = args.cns_epochs
    cfg.jepa.batch_size = args.batch_size
    cfg.jepa.lr = args.lr

    # ── Step 1: Load NLB MC_Maze ──────────────────────────────────────────────
    logger.info("Loading NLB MC_Maze spike data ...")
    nlb = NLBLoader()
    cns_train = nlb.load(smooth=True, split="train")
    logger.info("CNS data shape: %s", cns_train.shape)

    # ── Step 2: Pretrain CNS encoder ─────────────────────────────────────────
    cns_ckpt_path = f"{cfg.checkpoint_dir}/jepa_encoder_cns_best.pt"

    if args.cns_checkpoint:
        cns_ckpt_path = args.cns_checkpoint
        logger.info("Using provided CNS checkpoint: %s", cns_ckpt_path)
    elif args.skip_cns_pretrain:
        logger.info("Skipping CNS pretraining (--skip-cns-pretrain)")
    else:
        n_neurons = cns_train.shape[2]
        cns_trainer = CNSJEPATrainer(config=cfg, n_neurons=n_neurons, device=args.device)
        logger.info("CNS JEPA pretraining (%d epochs) ...", args.cns_epochs)
        cns_trainer.train(cns_train, window_len=args.window_len, stride=args.stride)
        logger.info("CNS pretraining complete.")

    # ── Step 3: Load + freeze CNS encoder ────────────────────────────────────
    cns_encoder = load_frozen_cns_encoder(cns_ckpt_path, device=args.device)
    logger.info("CNS encoder loaded and frozen.")

    # ── Step 4: Load EEG data for cross-modal training ───────────────────────
    from v2_digital_self_replication.data.real_eeg import load_trials
    logger.info("Loading PhysionetMI EEG for cross-modal training ...")
    eeg_all, _ = load_trials(
        subjects=list(range(1, 21)),
        classes=["left_hand", "right_hand", "feet"],
    )
    # Flatten to (total_T, 21) for the cross-modal trainer
    eeg_flat = eeg_all.reshape(-1, eeg_all.shape[2]).astype(np.float32)
    logger.info("EEG flat shape: %s", eeg_flat.shape)

    # ── Step 5: Cross-modal JEPA training ────────────────────────────────────
    enc_cfg = cfg.encoder
    eeg_encoder = StreamEncoder(
        d_model=enc_cfg.d_model,
        d_state=enc_cfg.d_state,
        n_layers=enc_cfg.n_layers,
        n_eeg=enc_cfg.n_eeg_channels,
        n_prop=enc_cfg.n_prop_channels,
        dropout=enc_cfg.dropout,
    )

    if args.eeg_checkpoint:
        ckpt = torch.load(args.eeg_checkpoint, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("encoder", ckpt)
        eeg_encoder.load_state_dict(state_dict, strict=False)
        logger.info("Warm-started EEG encoder from %s", args.eeg_checkpoint)

    cfg.jepa.n_epochs = args.eeg_epochs
    cross_trainer = CrossModalJEPATrainer(
        cns_encoder=cns_encoder,
        eeg_encoder=eeg_encoder,
        config=cfg,
        device=args.device,
        ot_epsilon=args.ot_epsilon,
        ot_weight=args.ot_weight,
        ema_weight=args.ema_weight,
    )
    logger.info("Cross-modal JEPA training (%d epochs) ...", args.eeg_epochs)
    cross_trainer.train(eeg_flat, cns_train, window_len=args.window_len, stride=args.stride)

    out_path = f"{cfg.checkpoint_dir}/jepa_encoder_cns_pretrained.pt"
    logger.info("Cross-modal training complete. Encoder at: %s", out_path)
    logger.info("")
    logger.info("Next: run supervised fine-tune on this encoder, then re-run LOSO:")
    logger.info(
        "  .venv/bin/python -m v2_digital_self_replication.cli.finetune \\\n"
        "      --checkpoint %s \\\n"
        "      --output %s/supervised_best_cns.pt",
        out_path, cfg.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
