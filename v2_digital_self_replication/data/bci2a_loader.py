"""
BCI Competition IV Dataset 2a loader via MOABB.

Returns (eeg, labels) in the same format as real_eeg.load_trials():
  eeg:    (n_trials, T, n_channels)  float32, uV-scale
  labels: list[str]  — 'left_hand', 'right_hand', 'feet', 'tongue'

Dataset: BNCI2014_001 (9 subjects, 4-class MI, 22 EEG channels, 250 Hz → 256 Hz)
PhysioNet-compatible channel subset: first 21 EEG channels used (Fz..P2),
EOG channels discarded.

Usage
-----
  from v2_digital_self_replication.data.bci2a_loader import load_bci2a_trials
  eeg, labels = load_bci2a_trials(subjects=[1], classes=['left_hand','right_hand','feet'])
"""
from __future__ import annotations

import logging
import numpy as np

log = logging.getLogger(__name__)

# BNCI2014_001 channel names (22 EEG, order as in dataset)
_BCI2A_EEG_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1",  "Pz",  "P2",  "POz",
]
N_CHANNELS_USE = 21  # drop POz (last) to match PhysioNetMI 21-ch layout


def load_bci2a_trials(
    subjects: list[int],
    classes:  list[str] | None = None,
    tmin:     float = 0.5,
    tmax:     float = 4.5,
    target_sfreq: float = 256.0,
) -> tuple[np.ndarray, list[str]]:
    """
    Load BCI2a trials for the given subjects.

    Parameters
    ----------
    subjects : list of ints in [1..9]
    classes  : subset of ['left_hand','right_hand','feet','tongue']; None = all 4
    tmin     : epoch start relative to cue onset (s)
    tmax     : epoch end relative to cue onset (s)

    Returns
    -------
    eeg    : (n_trials, T, 21) float32
    labels : list[str]
    """
    from moabb.datasets import BNCI2014_001
    import mne

    if classes is None:
        classes = ["left_hand", "right_hand", "feet", "tongue"]
    classes_set = set(classes)

    dataset = BNCI2014_001()

    all_eeg, all_labels = [], []

    for subj in subjects:
        log.info("Loading BCI2a subject %d …", subj)
        try:
            data = dataset.get_data(subjects=[subj])
        except Exception as e:
            log.warning("  Subject %d: failed to download/load: %s", subj, e)
            continue

        # data[subj] = {session: {run: raw}}
        subj_data = data[subj]
        for session_key, session_runs in subj_data.items():
            for run_key, raw in session_runs.items():
                if not hasattr(raw, "annotations"):
                    continue

                # Resample to target_sfreq if needed
                sfreq = raw.info["sfreq"]
                if abs(sfreq - target_sfreq) > 1.0:
                    raw = raw.copy().resample(target_sfreq, npad="auto")

                # Pick EEG channels only (drop EOG)
                try:
                    raw.pick_types(eeg=True, eog=False, stim=False, exclude="bads")
                except Exception:
                    raw.pick([ch for ch in raw.ch_names
                              if not ch.upper().startswith("EOG")])

                # Map MOABB event ids to class strings
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                # BNCI2014_001 uses: {'left_hand':1,'right_hand':2,'feet':3,'tongue':4}
                # but MOABB may encode them differently; map by value
                id_to_class = {}
                for class_name, ev_id in event_id.items():
                    # moabb encodes as 'left_hand', 'right_hand', etc. directly
                    normalized = class_name.lower().replace(" ", "_")
                    if normalized in classes_set:
                        id_to_class[ev_id] = normalized

                if not id_to_class:
                    continue

                # Epoch
                try:
                    epochs = mne.Epochs(
                        raw, events,
                        event_id={v: k for k, v in id_to_class.items()},
                        tmin=tmin, tmax=tmax,
                        baseline=None,
                        preload=True,
                        verbose=False,
                    )
                except Exception as e:
                    log.warning("  Epoching failed (subj %d run %s): %s",
                                subj, run_key, e)
                    continue

                data_arr = epochs.get_data()  # (n, n_ch, T)
                n_ch = min(data_arr.shape[1], N_CHANNELS_USE)
                data_arr = data_arr[:, :n_ch, :]  # (n, 21, T)

                # Transpose to (n, T, C) and rescale to µV
                data_arr = data_arr.transpose(0, 2, 1).astype(np.float32)
                if np.abs(data_arr).max() < 0.01:  # likely in V, convert
                    data_arr *= 1e6

                ep_labels = [id_to_class[ev[2]] for ev in epochs.events]

                all_eeg.append(data_arr)
                all_labels.extend(ep_labels)
                log.info("  Subject %d session=%s run=%s: %d trials shape=%s",
                         subj, session_key, run_key, data_arr.shape[0],
                         data_arr.shape)

    if not all_eeg:
        raise RuntimeError(f"No BCI2a trials loaded for subjects={subjects}")

    eeg_concat = np.concatenate(all_eeg, axis=0)  # (N, T, 21)
    log.info("Total: %d trials  shape=%s", eeg_concat.shape[0], eeg_concat.shape)
    return eeg_concat, all_labels
