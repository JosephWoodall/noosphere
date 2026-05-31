"""
Load PhysionetMI motor-imagery EEG via MOABB.

Returns data in the same (n_trials, T, 21) format as the synthetic pipeline,
matched to our 21-channel 10-20 layout.  All 21 channels are present verbatim
in the PhysionetMI 64-channel montage — no interpolation needed.

Preprocessing:
  1. Select the 21 channels
  2. Bandpass 8–30 Hz  (mu + beta bands carry motor imagery information)
  3. Resample to fs_target Hz  (default 256, matching synthetic training data)
  4. Epoch from tmin to tmax relative to the motor-imagery cue
  5. Baseline-correct to [tmin, 0]

Usage:
    from v2_digital_self_replication.data.real_eeg import load_trials
    eeg, labels = load_trials(subjects=[1, 2], tmin=0.5, tmax=4.5)
    # eeg: (n_trials, 1024, 21)  labels: list[str]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# The 21 channels in our model's fixed order (must match config.EEG_CHANNELS_21)
TARGET_CHANNELS = [
    "Fp1", "Fp2",
    "F7",  "F3",  "Fz",  "F4",  "F8",
    "T7",  "C3",  "Cz",  "C4",  "T8",
    "P7",  "P3",  "Pz",  "P4",  "P8",
    "O1",  "Oz",  "O2",
    "FCz",
]

# PhysionetMI class labels we care about
VALID_CLASSES = {"left_hand", "right_hand", "feet", "hands", "rest"}


def load_trials(
    subjects: list[int] | None = None,
    tmin: float = 0.5,
    tmax: float = 4.5,
    fs_target: int = 256,
    bandpass: tuple[float, float] = (8.0, 30.0),
    classes: list[str] | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """
    Load epoched PhysionetMI trials as (n_trials, T, 21) float32.

    Parameters
    ----------
    subjects   : list of subject IDs (1-109); default [1]
    tmin/tmax  : epoch window in seconds relative to cue onset
    fs_target  : resample target Hz (should match training data, default 256)
    bandpass   : (lo, hi) Hz for butter bandpass filter
    classes    : which event types to include; default all VALID_CLASSES
    verbose    : pass through to MNE

    Returns
    -------
    eeg    : np.ndarray  (n_trials, T, 21)  float32, μV scale
    labels : list[str]  class name per trial
    """
    from moabb.datasets import PhysionetMI
    import mne
    mne.set_log_level("WARNING" if not verbose else "INFO")

    subjects  = subjects or [1]
    classes   = classes  or sorted(VALID_CLASSES)
    lo, hi    = bandpass

    dataset = PhysionetMI()
    all_eeg: list[np.ndarray] = []
    all_labels: list[str]     = []

    for subject in subjects:
        logger.info("Loading subject %d …", subject)
        try:
            raw_dict = dataset.get_data(subjects=[subject])
        except Exception as e:
            logger.warning("Subject %d failed to load: %s", subject, e)
            continue

        for sess_data in raw_dict[subject].values():
            for raw in sess_data.values():
                # ── Select channels ─────────────────────────────────────────
                missing = [c for c in TARGET_CHANNELS if c not in raw.ch_names]
                if missing:
                    logger.warning("Subject %d missing channels: %s", subject, missing)
                    continue

                raw = raw.copy().pick_channels(TARGET_CHANNELS, ordered=True)

                # ── Resample ────────────────────────────────────────────────
                if raw.info["sfreq"] != fs_target:
                    raw = raw.resample(fs_target, npad="auto")

                # ── Bandpass ────────────────────────────────────────────────
                raw = raw.filter(lo, hi, method="iir", verbose=False)

                # ── Extract events ──────────────────────────────────────────
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                keep = {k: v for k, v in event_id.items()
                        if k in classes and k in VALID_CLASSES}
                if not keep:
                    continue

                # ── Epoch ───────────────────────────────────────────────────
                try:
                    epochs = mne.Epochs(
                        raw, events, event_id=keep,
                        tmin=tmin, tmax=tmax,
                        baseline=None,   # bandpass removes DC; no pre-cue window available
                        preload=True,
                        verbose=False,
                    )
                except Exception as e:
                    logger.warning("Epoching failed for subject %d: %s", subject, e)
                    continue

                data   = epochs.get_data(units="uV")   # (n_epochs, 21, T)
                data   = data.transpose(0, 2, 1)       # (n_epochs, T, 21)
                labels = [epochs.events[:, 2]]
                # Map numeric event codes back to names
                inv    = {v: k for k, v in keep.items()}
                names  = [inv[code] for code in epochs.events[:, 2]]

                all_eeg.append(data.astype(np.float32))
                all_labels.extend(names)
                logger.info("  Subject %d: %d trials", subject, len(names))

    if not all_eeg:
        raise RuntimeError(
            "No data loaded — check subject IDs and internet connection."
        )

    eeg_array = np.concatenate(all_eeg, axis=0)
    logger.info("Total: %d trials  shape=%s", len(all_labels), eeg_array.shape)
    return eeg_array, all_labels


def encode_trials(
    twin,
    eeg: np.ndarray,
    tail_samples: int = 64,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch-encode trials through the DigitalTwin encoder and decoder.

    Uses encoder.forward(B, T, C) directly — dramatically faster than
    step-by-step inference, and correct for offline evaluation.

    Parameters
    ----------
    twin         : DigitalTwin (loaded, eval mode)
    eeg          : (n_trials, T, 21) float32
    tail_samples : how many final timesteps to average for the trial embedding
    batch_size   : trials per forward pass

    Returns
    -------
    latents : (n_trials, d_model)  — trial-level encoder embeddings
    dof_pred: (n_trials, 6)        — mean DOF command over the trial tail
    """
    import torch

    n_trials, T, _ = eeg.shape
    d_model  = twin.cfg.encoder.d_model
    latents  = np.zeros((n_trials, d_model), dtype=np.float32)
    dof_pred = np.zeros((n_trials, 6),       dtype=np.float32)

    encoder = twin.encoder
    decoder = twin.decoder

    device = next(encoder.parameters()).device

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for start in range(0, n_trials, batch_size):
            end   = min(start + batch_size, n_trials)
            batch = torch.from_numpy(eeg[start:end]).to(device)   # (B, T, 21)

            enc_out, _ = encoder(batch)          # (B, T, d_model)
            tail        = enc_out[:, -tail_samples:, :]    # (B, tail, d_model)
            lat         = tail.mean(dim=1)                 # (B, d_model)

            intent = decoder(lat)                # MotorIntent(mu, sigma, ern_prob)
            cmd    = intent.mu                   # (B, 6)

            latents[start:end]  = lat.cpu().numpy()
            dof_pred[start:end] = cmd.cpu().numpy()

            logger.info("  Encoded %d/%d trials", end, n_trials)

    return latents, dof_pred
