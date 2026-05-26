"""
JEPA (Joint Embedding Predictive Architecture) pretraining for the stream encoder.

Objective: predict the latent representation of a future EEG window from a past window.
This teaches the encoder causal dynamics — exactly what a proactive prosthetic controller needs.

Architecture:
  Student encoder f_θ   — online, receives gradient updates
  Predictor g_φ         — small MLP, predicts future latent from past latent
  Teacher encoder f_ψ   — EMA copy of f_θ, produces target latents (no gradient)

Loss: MSE(g_φ(f_θ(context)), stop_gradient(f_ψ(target)))
Teacher update: ψ ← τ·ψ + (1-τ)·θ  (τ = ema_decay, default 0.99)

Physics grounding: the JEPA objective can be augmented with physics Q&A from
efficient_llm_training/jepa_data.py to teach the encoder arm dynamics constraints.
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
from torch.utils.data import Dataset, DataLoader

from v2_digital_self_replication.config import V2Config
from v2_digital_self_replication.core.stream_encoder import StreamEncoder

logger = logging.getLogger(__name__)


# ── Dataset ───────────────────────────────────────────────────────────────────

class EEGWindowDataset(Dataset):
    """
    Sliding-window dataset over (T, 21) EEG sequences.
    Each item is one (window_len, 21) EEG window.
    """

    def __init__(self, eeg_data: np.ndarray, window_len: int = 256, stride: int = 64):
        """
        eeg_data: (total_T, 21) float32 EEG.
        """
        self.data = torch.from_numpy(eeg_data.astype(np.float32))
        self.window_len = window_len
        self.stride = stride
        n = eeg_data.shape[0]
        self.starts = list(range(0, n - window_len + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        return self.data[s: s + self.window_len]


class MultiSubjectEEGDataset(Dataset):
    """Concatenates EEG from multiple subjects / trials into one dataset."""

    def __init__(self, eeg_dict: dict, window_len: int = 256, stride: int = 64):
        """
        eeg_dict: {subject_id: {"eeg": (n_trials, T, 21)}} as produced by make_training_batch.
        """
        segments = []
        for sub_data in eeg_dict.values():
            eeg = sub_data["eeg"]  # (n_trials, T, 21)
            for trial in eeg:
                segments.append(trial)  # (T, 21)
        all_eeg = np.concatenate(segments, axis=0)  # (total_T, 21)
        self._ds = EEGWindowDataset(all_eeg, window_len=window_len, stride=stride)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        return self._ds[idx]


# ── Predictor (context → target latent) ──────────────────────────────────────

class JEPAPredictor(nn.Module):
    """Two-layer MLP that maps context latent to predicted target latent."""

    def __init__(self, d_model: int = 128, d_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_model),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# ── Training loop ─────────────────────────────────────────────────────────────

class JEPATrainer:
    """
    Trains the StreamEncoder via JEPA self-supervised objective.
    Checkpoints are saved to config.checkpoint_dir every epoch.
    """

    def __init__(self, config: Optional[V2Config] = None, device: str = "cpu"):
        self.cfg = config or V2Config()
        self.device = torch.device(device)
        jepa = self.cfg.jepa
        enc_cfg = self.cfg.encoder

        # Student encoder (receives gradient)
        self.encoder = StreamEncoder(
            d_model=enc_cfg.d_model,
            d_state=enc_cfg.d_state,
            n_layers=enc_cfg.n_layers,
            n_eeg=enc_cfg.n_eeg_channels,
            n_prop=enc_cfg.n_prop_channels,
            dropout=enc_cfg.dropout,
        ).to(self.device)

        # Teacher encoder (EMA, no gradient)
        self.teacher = copy.deepcopy(self.encoder)
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.predictor = JEPAPredictor(
            d_model=enc_cfg.d_model,
            d_hidden=enc_cfg.d_model * 2,
        ).to(self.device)

        self._ema_decay = jepa.ema_decay
        self._context_frac = jepa.context_fraction

        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.predictor.parameters()),
            lr=jepa.lr,
            weight_decay=jepa.weight_decay,
        )

        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _update_teacher(self):
        """EMA update: ψ ← τ·ψ + (1-τ)·θ"""
        with torch.no_grad():
            for s_param, t_param in zip(self.encoder.parameters(), self.teacher.parameters()):
                t_param.data.mul_(self._ema_decay).add_(s_param.data * (1 - self._ema_decay))

    def _cosine_schedule(self, step: int, total: int) -> float:
        return 0.5 * (1 + math.cos(math.pi * step / max(total, 1)))

    def _mean_pool(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool sequence (B, T, d) → (B, d)."""
        return x.mean(dim=1)

    def train_epoch(self, loader: DataLoader, epoch: int, total_epochs: int) -> float:
        self.encoder.train()
        self.predictor.train()
        total_loss = 0.0
        n_batches = 0

        # Cosine LR warmup
        warmup = self.cfg.jepa.warmup_epochs
        if epoch < warmup:
            scale = (epoch + 1) / max(warmup, 1)
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.jepa.lr * scale

        for batch in loader:
            x = batch.to(self.device)  # (B, T, 21)
            B, T, C = x.shape
            T_ctx = int(T * self._context_frac)

            context = x[:, :T_ctx, :]   # (B, T_ctx, 21)
            target  = x[:, T_ctx:, :]   # (B, T_tgt, 21)

            # Student: encode context → pool → predict target latent
            ctx_out, _ = self.encoder(context)           # (B, T_ctx, d)
            z_ctx = self._mean_pool(ctx_out)              # (B, d)
            z_pred = self.predictor(z_ctx)                # (B, d)

            # Teacher: encode target → pool (no gradient)
            with torch.no_grad():
                tgt_out, _ = self.teacher(target)
                z_target = self._mean_pool(tgt_out)       # (B, d)

            loss = nn.functional.mse_loss(z_pred, z_target)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.predictor.parameters()), 1.0
            )
            self.optimizer.step()
            self._update_teacher()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        eeg_dict: dict,
        window_len: int = 256,
        stride: int = 64,
    ):
        """
        Full JEPA pretraining loop.
        eeg_dict: from data.synthetic_eeg.make_training_batch()
        """
        dataset = MultiSubjectEEGDataset(eeg_dict, window_len=window_len, stride=stride)
        loader  = DataLoader(
            dataset,
            batch_size=self.cfg.jepa.batch_size,
            shuffle=True,
            num_workers=0,  # keep deterministic; bump to 4 for real data
            drop_last=True,
        )

        logger.info("JEPA pretraining: %d windows, %d epochs", len(dataset), self.cfg.jepa.n_epochs)
        best_loss = float("inf")

        for epoch in range(self.cfg.jepa.n_epochs):
            t0 = time.time()
            loss = self.train_epoch(loader, epoch, self.cfg.jepa.n_epochs)
            elapsed = time.time() - t0

            logger.info("Epoch %3d/%d  loss=%.6f  %.1fs", epoch + 1, self.cfg.jepa.n_epochs, loss, elapsed)

            if loss < best_loss:
                best_loss = loss
                self._save_checkpoint(epoch, loss, tag="best")

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, loss, tag=f"ep{epoch+1}")

        self._save_checkpoint(self.cfg.jepa.n_epochs - 1, best_loss, tag="final")
        logger.info("JEPA pretraining complete. Best loss: %.6f", best_loss)

    def _save_checkpoint(self, epoch: int, loss: float, tag: str = ""):
        path = f"{self.cfg.checkpoint_dir}/jepa_encoder_{tag}.pt"
        torch.save({
            "encoder": self.encoder.state_dict(),
            "teacher": self.teacher.state_dict(),
            "predictor": self.predictor.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }, path)

    def load_encoder_into_twin(self, twin_encoder: StreamEncoder, path: str):
        """Transfer pretrained encoder weights into a DigitalTwin's encoder."""
        ckpt = torch.load(path, map_location="cpu")
        twin_encoder.load_state_dict(ckpt["encoder"])
        logger.info("Loaded JEPA encoder weights from %s", path)
