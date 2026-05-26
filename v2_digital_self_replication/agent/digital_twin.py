"""
DigitalTwin: the main agent that continuously learns to interpret EEG intent.

The twin wraps the full inference pipeline:
  MultiModalFusion → StreamEncoder → IntentDecoder → KalmanFilter → SafetyGate

And the online learning loop:
  Experience buffer → gradient update → EMA parameter smoothing

Usage (inference loop):
    twin = DigitalTwin(config)
    twin.load("checkpoint.pt")
    hidden = twin.reset_state()
    while True:
        eeg, hrv, gsr, prop = get_sensor_data()
        command = twin.step(eeg, hrv, gsr, prop, hidden)
        if command is not None:
            send_to_hardware(command)
        twin.observe_outcome(feedback_position)
        if twin.should_adapt():
            twin.adapt()
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

from v2_digital_self_replication.config import V2Config
from v2_digital_self_replication.core.stream_encoder import StreamEncoder
from v2_digital_self_replication.core.intent_decoder import IntentDecoder, IntentLoss
from v2_digital_self_replication.core.kalman_filter import AdaptiveKalmanFilter
from v2_digital_self_replication.core.safety_gate import SafetyGate, SafetyConfig
from v2_digital_self_replication.agent.memory_store import MemoryStore, Experience

logger = logging.getLogger(__name__)


class DigitalTwin(nn.Module):
    """
    Continuously learning digital twin for prosthetic arm control.

    Parameters are updated online using recent experience (proprioceptive feedback).
    EMA smoothing prevents catastrophic forgetting between adaptation steps.
    """

    def __init__(self, config: Optional[V2Config] = None):
        super().__init__()
        self.cfg = config or V2Config()
        enc_cfg = self.cfg.encoder
        dec_cfg = self.cfg.decoder

        self.encoder = StreamEncoder(
            d_model=enc_cfg.d_model,
            d_state=enc_cfg.d_state,
            n_layers=enc_cfg.n_layers,
            n_eeg=enc_cfg.n_eeg_channels,
            n_prop=enc_cfg.n_prop_channels,
            dropout=enc_cfg.dropout,
        )
        self.decoder = IntentDecoder(
            d_model=enc_cfg.d_model,
            n_dof=dec_cfg.n_dof,
            d_hidden=dec_cfg.d_hidden,
        )
        self._loss_fn = IntentLoss(ern_weight=1.0, sigma_reg=0.01)

        self._kalman = AdaptiveKalmanFilter(
            n_dof=dec_cfg.n_dof,
            dt=self.cfg.kalman.dt,
            process_noise=self.cfg.kalman.process_noise,
        )
        self._safety = SafetyGate(SafetyConfig(
            ern_threshold=self.cfg.safety.ern_threshold,
            sigma_threshold=self.cfg.safety.sigma_threshold,
            ern_halt_duration=self.cfg.safety.ern_halt_duration,
            sigma_halt_duration=self.cfg.safety.sigma_halt_duration,
            watchdog_timeout=self.cfg.safety.watchdog_timeout,
        ))
        self._memory = MemoryStore(
            capacity=self.cfg.online.experience_capacity,
            embedding_dim=enc_cfg.d_model,
            log_dir=self.cfg.log_dir,
        )

        online_cfg = self.cfg.online
        self._optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=online_cfg.adapt_lr,
            weight_decay=1e-5,
        )
        # EMA shadow parameters — blended in after each adaptation step
        self._ema_decay = online_cfg.ema_decay
        self._ema_params = {
            name: param.data.clone()
            for name, param in self.named_parameters()
        }

        self._hidden_state: Optional[list] = None
        self._step_count: int = 0
        self._last_pred: Optional[np.ndarray] = None
        self._last_latent: Optional[np.ndarray] = None  # encoder output, stored for adapt()
        self._session_errors: list[float] = []

        Path(self.cfg.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── Inference ─────────────────────────────────────────────────────────────

    def reset_state(self):
        """Reset SSM hidden state and Kalman filter for a new session."""
        device = next(self.parameters()).device
        self._hidden_state = self.encoder.zero_hidden(batch_size=1, device=str(device))
        self._kalman.reset()
        self._safety.reset()
        return self._hidden_state

    @torch.no_grad()
    def step(
        self,
        eeg: np.ndarray,          # (21,) microvolts
        hrv: Optional[np.ndarray] = None,   # (1,) or None
        gsr: Optional[np.ndarray] = None,   # (1,) or None
        prop: Optional[np.ndarray] = None,  # (6,) — current arm position
    ) -> Optional[np.ndarray]:
        """
        Process one EEG sample and return a safe motor command or None (HALT).
        """
        if self._hidden_state is None:
            self.reset_state()

        device = next(self.parameters()).device

        def _t(arr, sz):
            if arr is None:
                return None
            return torch.tensor(arr[:sz], dtype=torch.float32, device=device).unsqueeze(0)

        eeg_t  = _t(eeg, 21)
        hrv_t  = _t(hrv, 1)
        gsr_t  = _t(gsr, 1)
        prop_t = _t(prop, 6)

        h_out, self._hidden_state = self.encoder.decode_step(
            eeg_t, hrv_t, gsr_t, prop_t, self._hidden_state
        )
        intent = self.decoder(h_out)

        mu_np    = intent.mu.squeeze(0).cpu().numpy()
        sigma_np = intent.sigma.squeeze(0).cpu().numpy()
        ern_p    = float(intent.ern_prob.squeeze().item())

        smooth_cmd = self._kalman.step(mu_np, sigma_np)
        safe_cmd, reason = self._safety.check(smooth_cmd, sigma_np, ern_p)

        self._last_pred   = mu_np
        self._last_latent = h_out.cpu().numpy()  # (1, d_model), for adapt()
        self._step_count += 1

        if reason:
            logger.debug("step %d: HALT [%s]", self._step_count, reason)
        return safe_cmd

    def observe_outcome(self, actual_position: np.ndarray, eeg_window: Optional[np.ndarray] = None):
        """
        Record the outcome of the last step for online learning.
        actual_position: (6,) — actual arm DOF state (from proprioceptive feedback).
        eeg_window: optional (T, 21) — the EEG that produced the last prediction.
        """
        if self._last_pred is None:
            return
        error = float(np.mean(np.abs(self._last_pred - actual_position)))
        self._session_errors.append(error)

        self._memory.store_experience(Experience(
            eeg=eeg_window if eeg_window is not None else np.zeros((1, 21), np.float32),
            command_pred=self._last_pred.copy(),
            command_actual=actual_position.copy(),
            ern_prob=0.0,
            latent=self._last_latent.copy() if self._last_latent is not None else None,
        ))

    # ── Online adaptation ─────────────────────────────────────────────────────

    def should_adapt(self) -> bool:
        return (self._step_count % self.cfg.online.adapt_every_n_steps == 0
                and len(self._memory.short_term) >= self.cfg.online.adapt_batch_size)

    def adapt(self):
        """Run adaptation micro-batch on recent experience."""
        experiences = self._memory.sample_for_training(self.cfg.online.adapt_batch_size)
        # Only adapt if we have experiences with stored latents
        valid = [e for e in experiences if e.latent is not None]
        if not valid:
            return

        device = next(self.parameters()).device
        self.train()

        loss = torch.tensor(0.0)
        for _ in range(self.cfg.online.adapt_n_gradient_steps):
            batch = valid[: self.cfg.online.adapt_batch_size]
            # Re-run decoder on stored latents — gradient flows through decoder params
            h_batch = torch.tensor(
                np.stack([e.latent for e in batch]), dtype=torch.float32, device=device
            )  # (B, 1, d_model) or (B, d_model)
            if h_batch.dim() == 3:
                h_batch = h_batch.squeeze(1)  # (B, d_model)
            actual_batch = torch.tensor(
                np.stack([e.command_actual for e in batch]), dtype=torch.float32, device=device
            )

            intent = self.decoder(h_batch)
            loss = nn.functional.mse_loss(intent.mu, actual_batch)
            self._optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self._optimizer.step()

        # EMA smoothing — move shadow params to model device lazily
        with torch.no_grad():
            for name, param in self.named_parameters():
                ema = self._ema_params[name].to(device)
                ema.mul_(self._ema_decay).add_(param.data * (1 - self._ema_decay))
                param.data.copy_(ema)
                self._ema_params[name] = ema

        self.eval()
        logger.info("adapt(): %d steps, mean_err=%.4f", self.cfg.online.adapt_n_gradient_steps,
                    float(loss.item()))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None):
        path = path or f"{self.cfg.checkpoint_dir}/twin_latest.pt"
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "ema_params": {k: v.cpu() for k, v in self._ema_params.items()},
            "step_count": self._step_count,
            "safety_stats": self._safety.stats,
        }, path)
        logger.info("Saved checkpoint: %s", path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self._ema_params = ckpt.get("ema_params", self._ema_params)
        self._step_count = ckpt.get("step_count", 0)
        logger.info("Loaded checkpoint: %s (step %d)", path, self._step_count)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    @property
    def session_mean_error(self) -> float:
        if not self._session_errors:
            return float("nan")
        return float(np.mean(self._session_errors[-100:]))

    @property
    def safety_stats(self) -> dict:
        return self._safety.stats

    def log_session_summary(self):
        logger.info(
            "Session summary — steps: %d, mean_error: %.4f, halts: %d",
            self._step_count, self.session_mean_error, self._safety.stats["total_halts"],
        )
        self._memory.log_episode(
            duration_s=self._step_count * self.cfg.kalman.dt,
            mean_error=self.session_mean_error,
            n_ern=self._safety.stats.get("ern", 0),
            n_halt=self._safety.stats.get("total_halts", 0),
        )
