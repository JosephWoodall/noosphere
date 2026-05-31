"""
Cross-modal JEPA training: frozen CNS teacher + EEG student.

The CNS encoder (pretrained on NLB MC_Maze M1 spike trains) acts as a teacher
that provides motor-manifold-aligned target latents. The EEG encoder (student)
is trained to produce latents that match the CNS latent distribution via
Sinkhorn-Knopp optimal transport alignment.

This approach works without paired EEG+neural recordings: the OT coupling
provides a soft batch-level assignment between EEG windows and CNS windows,
incentivising the EEG encoder to embed a latent geometry similar to M1
population dynamics during reaching.

Architecture (Safaie et al. 2023 biological grounding):
  M1 population dynamics during reaching are cross-species preserved.
  Aligning EEG latents to M1 latent geometry therefore embeds motor-class
  discriminative structure that the unimodal EEG JEPA objective misses.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from v2_digital_self_replication.config import V2Config, JEPAConfig
from v2_digital_self_replication.core.stream_encoder import StreamEncoder
from v2_digital_self_replication.core.cns_encoder import CNSEncoder
from v2_digital_self_replication.data.nlb_loader import NLBLoader, NLBWindowDataset
from v2_digital_self_replication.training.pretrain_jepa import EEGWindowDataset, JEPAPredictor

logger = logging.getLogger(__name__)


# ── Sinkhorn OT (adapted from v1/noosphere/riemann.py) ────────────────────────

def sinkhorn_ot_loss(
    z_eeg: torch.Tensor,
    z_cns: torch.Tensor,
    epsilon: float = 1.0,
    n_iters: int = 20,
) -> torch.Tensor:
    """
    Sinkhorn OT loss between EEG and CNS latent batches.

    Latents are L2-normalised before computing the transport plan so that
    pairwise squared distances lie in [0, 4] regardless of embedding scale.
    epsilon=1.0 gives a moderately tight matching on the unit sphere
    (pairwise cost ≈ 2 for random 128d unit vectors).

    Args:
        z_eeg: (N, d) EEG student latents
        z_cns: (M, d) CNS teacher latents (no gradient)
        epsilon: entropic regularisation; 1.0 is appropriate for unit-sphere costs
        n_iters: Sinkhorn iterations

    Returns:
        Scalar OT loss
    """
    # L2 normalise → pairwise cost in [0, 4]
    z_s = torch.nn.functional.normalize(z_eeg, dim=-1)
    z_t = torch.nn.functional.normalize(z_cns.detach(), dim=-1)

    N = z_s.shape[0]
    M = z_t.shape[0]

    # Squared L2 cost matrix (N x M)
    cost = torch.cdist(z_s, z_t) ** 2  # values in [0, 4]

    # Log-domain Sinkhorn for numerical stability
    log_K = -cost / epsilon
    f = torch.zeros(N, device=z_eeg.device)
    g = torch.zeros(M, device=z_eeg.device)

    for _ in range(n_iters):
        f = epsilon * (
            math.log(1.0 / N)
            - torch.logsumexp(log_K + g.unsqueeze(0) / epsilon, dim=1)
        )
        g = epsilon * (
            math.log(1.0 / M)
            - torch.logsumexp(log_K + f.unsqueeze(1) / epsilon, dim=0)
        )

    # Transport plan P* (N, M)
    log_P = (f.unsqueeze(1) + g.unsqueeze(0) + log_K) / epsilon
    P = torch.exp(log_P)

    # OT loss = <P, C>
    return (P * cost).sum()


# ── CNS self-supervised JEPA pretrainer ───────────────────────────────────────

class CNSJEPATrainer:
    """
    Self-supervised JEPA pretraining for the CNS encoder on NLB MC_Maze.

    Same EMA-JEPA objective as JEPATrainer but operates on spike rate windows.
    Output: checkpoint with 'cns_encoder' key for use as cross-modal teacher.
    """

    def __init__(
        self,
        config: Optional[V2Config] = None,
        n_neurons: int = 182,
        device: str = "cpu",
    ):
        self.cfg = config or V2Config()
        self.device = torch.device(device)
        self.n_neurons = n_neurons
        enc_cfg = self.cfg.encoder

        self.encoder = CNSEncoder(
            n_neurons=n_neurons,
            d_model=enc_cfg.d_model,
            d_state=enc_cfg.d_state,
            n_layers=enc_cfg.n_layers,
            dropout=enc_cfg.dropout,
        ).to(self.device)

        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.predictor = JEPAPredictor(
            d_model=enc_cfg.d_model,
            d_hidden=enc_cfg.d_model * 2,
        ).to(self.device)

        jepa = self.cfg.jepa
        self.ema_decay = jepa.ema_decay
        self.context_frac = jepa.context_fraction
        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=jepa.lr,
            weight_decay=jepa.weight_decay,
        )

        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _update_teacher(self):
        with torch.no_grad():
            for s, t in zip(self.encoder.parameters(), self.teacher.parameters()):
                t.data.mul_(self.ema_decay).add_(s.data * (1 - self.ema_decay))

    def train(self, spikes: np.ndarray, window_len: int = 256, stride: int = 64):
        """
        spikes: (n_trials, T, n_neurons) float32 from NLBLoader.load()
        """
        dataset = NLBWindowDataset(spikes, window_len=window_len, stride=stride)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.jepa.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        n_epochs = self.cfg.jepa.n_epochs
        logger.info("CNS JEPA pretraining: %d windows, %d epochs", len(dataset), n_epochs)
        best_loss = float("inf")

        for epoch in range(n_epochs):
            t0 = time.time()
            loss_val = self._train_epoch(loader, epoch, n_epochs)
            elapsed = time.time() - t0
            logger.info("Epoch %3d/%d  loss=%.6f  %.1fs", epoch + 1, n_epochs, loss_val, elapsed)

            if loss_val < best_loss:
                best_loss = loss_val
                self._save("cns_best")

            if (epoch + 1) % 10 == 0:
                self._save(f"cns_ep{epoch+1}")

        self._save("cns_final")
        logger.info("CNS JEPA done. Best loss: %.6f", best_loss)

    def _train_epoch(self, loader: DataLoader, epoch: int, total: int) -> float:
        self.encoder.train()
        self.predictor.train()
        total_loss = 0.0
        n = 0
        warmup = self.cfg.jepa.warmup_epochs
        if epoch < warmup:
            scale = (epoch + 1) / max(warmup, 1)
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.jepa.lr * scale

        for batch in loader:
            x = batch.to(self.device)  # (B, T, n_neurons)
            T = x.shape[1]
            T_ctx = int(T * self.context_frac)

            ctx_out, _ = self.encoder(x[:, :T_ctx, :])
            z_ctx = ctx_out.mean(dim=1)
            z_pred = self.predictor(z_ctx)

            with torch.no_grad():
                tgt_out, _ = self.teacher(x[:, T_ctx:, :])
                z_tgt = tgt_out.mean(dim=1)

            loss = nn.functional.mse_loss(z_pred, z_tgt)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.predictor.parameters()), 1.0
            )
            self.optimizer.step()
            self._update_teacher()
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def _save(self, tag: str):
        path = f"{self.cfg.checkpoint_dir}/jepa_encoder_{tag}.pt"
        torch.save({
            "cns_encoder": self.encoder.state_dict(),
            "cns_teacher": self.teacher.state_dict(),
            "predictor": self.predictor.state_dict(),
        }, path)


# ── Cross-modal JEPA (EEG student + frozen CNS teacher) ───────────────────────

class CrossModalJEPATrainer:
    """
    Cross-modal JEPA: frozen CNS encoder as teacher, EEG StreamEncoder as student.

    Training strategy:
      1. Sample a batch of EEG windows and a batch of CNS windows (independent).
      2. Compute EEG latents (student) and CNS latents (teacher, no grad).
      3. Use Sinkhorn OT loss to align EEG latent geometry to CNS latent geometry.
      4. Optionally combine with a self-supervised EMA loss on EEG alone.

    The OT coupling is soft: it does not require paired EEG+neural data.
    """

    def __init__(
        self,
        cns_encoder: CNSEncoder,
        eeg_encoder: Optional[StreamEncoder] = None,
        config: Optional[V2Config] = None,
        device: str = "cpu",
        ot_epsilon: float = 1.0,
        ot_iters: int = 20,
        ot_weight: float = 1.0,
        ema_weight: float = 0.5,
    ):
        self.cfg = config or V2Config()
        self.device = torch.device(device)
        self.ot_epsilon = ot_epsilon
        self.ot_iters = ot_iters
        self.ot_weight = ot_weight
        self.ema_weight = ema_weight

        enc_cfg = self.cfg.encoder

        # CNS teacher is always frozen
        self.cns_encoder = cns_encoder.to(self.device)
        for p in self.cns_encoder.parameters():
            p.requires_grad_(False)
        self.cns_encoder.eval()

        # EEG student: use provided encoder or create fresh
        if eeg_encoder is not None:
            self.eeg_encoder = eeg_encoder.to(self.device)
        else:
            self.eeg_encoder = StreamEncoder(
                d_model=enc_cfg.d_model,
                d_state=enc_cfg.d_state,
                n_layers=enc_cfg.n_layers,
                n_eeg=enc_cfg.n_eeg_channels,
                n_prop=enc_cfg.n_prop_channels,
                dropout=enc_cfg.dropout,
            ).to(self.device)

        # EMA copy of EEG encoder for unimodal self-supervised loss term
        if ema_weight > 0:
            self.eeg_teacher = copy.deepcopy(self.eeg_encoder)
            for p in self.eeg_teacher.parameters():
                p.requires_grad_(False)
        else:
            self.eeg_teacher = None

        self.predictor = JEPAPredictor(
            d_model=enc_cfg.d_model,
            d_hidden=enc_cfg.d_model * 2,
        ).to(self.device)

        jepa = self.cfg.jepa
        self.ema_decay = jepa.ema_decay
        self.context_frac = jepa.context_fraction
        self.optimizer = optim.AdamW(
            list(self.eeg_encoder.parameters()) + list(self.predictor.parameters()),
            lr=jepa.lr,
            weight_decay=jepa.weight_decay,
        )

        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _update_eeg_teacher(self):
        if self.eeg_teacher is None:
            return
        with torch.no_grad():
            for s, t in zip(self.eeg_encoder.parameters(), self.eeg_teacher.parameters()):
                t.data.mul_(self.ema_decay).add_(s.data * (1 - self.ema_decay))

    def train(
        self,
        eeg_data: np.ndarray,
        cns_data: np.ndarray,
        window_len: int = 256,
        stride: int = 64,
    ):
        """
        eeg_data: (total_T, 21) float32 EEG (concatenated across all subjects/trials)
        cns_data: (n_trials, T, 182) float32 spike rates from NLBLoader.load()
        """
        eeg_dataset = EEGWindowDataset(eeg_data, window_len=window_len, stride=stride)
        cns_dataset = NLBWindowDataset(cns_data, window_len=window_len, stride=stride)

        bs = self.cfg.jepa.batch_size
        eeg_loader = DataLoader(eeg_dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
        cns_loader = DataLoader(cns_dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)

        n_epochs = self.cfg.jepa.n_epochs
        logger.info(
            "Cross-modal JEPA: %d EEG windows, %d CNS windows, %d epochs",
            len(eeg_dataset), len(cns_dataset), n_epochs,
        )
        best_loss = float("inf")

        for epoch in range(n_epochs):
            t0 = time.time()
            loss_val = self._train_epoch(eeg_loader, cns_loader, epoch, n_epochs)
            elapsed = time.time() - t0
            logger.info("Epoch %3d/%d  loss=%.6f  %.1fs", epoch + 1, n_epochs, loss_val, elapsed)

            if loss_val < best_loss:
                best_loss = loss_val
                self._save("cns_pretrained")

            if (epoch + 1) % 10 == 0:
                self._save(f"cns_pretrained_ep{epoch+1}")

        self._save("cns_pretrained_final")
        logger.info("Cross-modal JEPA done. Best loss: %.6f", best_loss)

    def _train_epoch(
        self,
        eeg_loader: DataLoader,
        cns_loader: DataLoader,
        epoch: int,
        total: int,
    ) -> float:
        self.eeg_encoder.train()
        self.predictor.train()
        total_loss = 0.0
        n = 0

        warmup = self.cfg.jepa.warmup_epochs
        if epoch < warmup:
            scale = (epoch + 1) / max(warmup, 1)
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.jepa.lr * scale

        cns_iter = iter(cns_loader)

        for eeg_batch in eeg_loader:
            eeg_batch = eeg_batch.to(self.device)       # (B, T, 21)
            try:
                cns_batch = next(cns_iter).to(self.device)
            except StopIteration:
                cns_iter = iter(cns_loader)
                cns_batch = next(cns_iter).to(self.device)

            B, T, _ = eeg_batch.shape
            T_ctx = int(T * self.context_frac)

            # EEG student: encode full window, pool → latent
            eeg_out, _ = self.eeg_encoder(eeg_batch)    # (B, T, d)
            z_eeg = eeg_out.mean(dim=1)                 # (B, d)

            # CNS teacher: encode full CNS window → target latent (no grad)
            with torch.no_grad():
                cns_out, _ = self.cns_encoder(cns_batch)
                z_cns = cns_out.mean(dim=1)             # (B, d)

            # Cross-modal OT loss
            loss = self.ot_weight * sinkhorn_ot_loss(
                z_eeg, z_cns,
                epsilon=self.ot_epsilon,
                n_iters=self.ot_iters,
            )

            # Optional unimodal EMA JEPA loss for regularisation
            if self.ema_weight > 0 and self.eeg_teacher is not None:
                ctx_out, _ = self.eeg_encoder(eeg_batch[:, :T_ctx, :])
                z_ctx = ctx_out.mean(dim=1)
                z_pred = self.predictor(z_ctx)
                with torch.no_grad():
                    tgt_out, _ = self.eeg_teacher(eeg_batch[:, T_ctx:, :])
                    z_tgt = tgt_out.mean(dim=1)
                ema_loss = nn.functional.mse_loss(z_pred, z_tgt)
                loss = loss + self.ema_weight * ema_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.eeg_encoder.parameters()) + list(self.predictor.parameters()), 1.0
            )
            self.optimizer.step()
            self._update_eeg_teacher()

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    def _save(self, tag: str):
        path = f"{self.cfg.checkpoint_dir}/jepa_encoder_{tag}.pt"
        torch.save({
            "encoder": self.eeg_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
        }, path)
        logger.info("Saved cross-modal checkpoint: %s", path)

    def get_trained_encoder(self) -> StreamEncoder:
        """Return the trained EEG encoder for downstream use."""
        return self.eeg_encoder
