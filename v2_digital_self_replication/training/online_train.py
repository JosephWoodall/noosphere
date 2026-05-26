"""
Online/supervised fine-tuning for the DigitalTwin.

Two training modes:
  1. Supervised — EEG windows with ground-truth motor commands.
     Used when motion-capture data is available (lab environment).
     Loss: IntentLoss (Gaussian NLL + ERN BCE + sigma regularizer).

  2. Self-supervised online — proprioceptive feedback only.
     Adapts the model to a new user/session without ground-truth labels.
     Loss: MSE(predicted_command, actual_position_delta).

The trainer maintains the JEPA-pretrained encoder as a frozen backbone
(optional) and fine-tunes only the decoder and the last encoder block.
This mirrors the transfer-learning best practice for BCI: freeze the
general dynamics representation, adapt the intent mapping.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from v2_digital_self_replication.config import V2Config
from v2_digital_self_replication.agent.digital_twin import DigitalTwin
from v2_digital_self_replication.core.intent_decoder import IntentLoss

logger = logging.getLogger(__name__)


class SupervisedEEGDataset(Dataset):
    """
    Dataset for supervised training: (EEG window, motor command, ERN label).

    eeg:      (N, T, 21)  — EEG windows
    commands: (N, 6)      — target motor command (last step of window)
    ern:      (N,)        — binary ERN label (optional, zeros if absent)
    """

    def __init__(
        self,
        eeg: np.ndarray,
        commands: np.ndarray,
        ern_labels: Optional[np.ndarray] = None,
    ):
        self.eeg = torch.from_numpy(eeg.astype(np.float32))
        self.commands = torch.from_numpy(commands.astype(np.float32))
        self.ern = torch.from_numpy(
            ern_labels.astype(np.float32) if ern_labels is not None
            else np.zeros(len(eeg), dtype=np.float32)
        )

    def __len__(self):
        return len(self.eeg)

    def __getitem__(self, idx):
        return self.eeg[idx], self.commands[idx], self.ern[idx]


def _build_supervised_dataset(eeg_dict: dict, window_len: int = 256) -> SupervisedEEGDataset:
    """
    Flatten multi-subject training data into a supervised dataset.
    Target command = mean command over the last window_len timesteps of each trial.
    """
    eeg_list, cmd_list, ern_list = [], [], []
    for sub_data in eeg_dict.values():
        eeg_trials = sub_data["eeg"]     # (n_trials, T, 21)
        cmd_trials = sub_data["commands"]  # (n_trials, T, 6)
        ern_trials = sub_data.get("ern_labels")  # (n_trials, T)

        for i in range(len(eeg_trials)):
            T = eeg_trials.shape[1]
            n_windows = T // window_len
            for w in range(n_windows):
                s = w * window_len
                eeg_list.append(eeg_trials[i, s: s + window_len])
                cmd_list.append(cmd_trials[i, s + window_len - 1])
                ern_row = ern_trials[i, s + window_len - 1] if ern_trials is not None else 0.0
                ern_list.append(float(ern_row))

    return SupervisedEEGDataset(
        np.stack(eeg_list), np.stack(cmd_list),
        np.array(ern_list, dtype=np.float32),
    )


class SupervisedTrainer:
    """
    Supervised fine-tuning of the DigitalTwin on labelled EEG data.
    The JEPA-pretrained encoder is optionally frozen to preserve the
    general dynamics representation; only the decoder adapts.
    """

    def __init__(
        self,
        twin: DigitalTwin,
        config: Optional[V2Config] = None,
        freeze_encoder: bool = True,
        device: str = "cpu",
    ):
        self.twin = twin.to(device)
        self.cfg = config or twin.cfg
        self.device = torch.device(device)
        self._loss_fn = IntentLoss(ern_weight=1.0, sigma_reg=0.01)

        if freeze_encoder:
            for p in twin.encoder.parameters():
                p.requires_grad_(False)
            # Unfreeze only the last encoder block so volume-conduction
            # artifacts specific to this user can be compensated
            for p in twin.encoder.blocks[-1].parameters():
                p.requires_grad_(True)

        trainable = [p for p in twin.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def train(
        self,
        eeg_dict: dict,
        n_epochs: int = 20,
        window_len: int = 256,
        batch_size: int = 64,
    ):
        dataset = _build_supervised_dataset(eeg_dict, window_len=window_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        logger.info("Supervised training: %d windows, %d epochs", len(dataset), n_epochs)

        best_loss = float("inf")
        for epoch in range(n_epochs):
            t0 = time.time()
            total_loss = 0.0
            self.twin.train()

            for eeg_batch, cmd_batch, ern_batch in loader:
                eeg_batch = eeg_batch.to(self.device)   # (B, T, 21)
                cmd_batch  = cmd_batch.to(self.device)   # (B, 6)
                ern_batch  = ern_batch.to(self.device)   # (B,)

                enc_out, _ = self.twin.encoder(eeg_batch)  # (B, T, d)
                intent = self.twin.decoder(enc_out)

                loss = self._loss_fn(intent, cmd_batch, ern_label=ern_batch)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.twin.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                total_loss += loss.item()

            self.scheduler.step()
            avg = total_loss / max(len(loader), 1)
            logger.info("Epoch %3d/%d  loss=%.6f  %.1fs", epoch + 1, n_epochs, avg, time.time() - t0)

            if avg < best_loss:
                best_loss = avg
                self.twin.save(f"{self.cfg.checkpoint_dir}/supervised_best.pt")

        self.twin.eval()
        logger.info("Supervised training complete. Best loss: %.6f", best_loss)
