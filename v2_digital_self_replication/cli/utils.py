"""Shared utilities for v2 CLI entry points."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np


def configure_logging(level: str = "INFO", log_file: str | None = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def save_dataset(dataset: dict, output_dir: str, metadata: dict):
    """Save make_training_batch() output to per-subject .npy files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sub_id, sub_data in dataset.items():
        np.save(out / f"sub{sub_id:02d}_eeg.npy",      sub_data["eeg"])
        np.save(out / f"sub{sub_id:02d}_commands.npy", sub_data["commands"])
        np.save(out / f"sub{sub_id:02d}_ern.npy",      sub_data["ern_labels"])
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_dataset(data_dir: str) -> dict:
    """Load the .npy dataset saved by generate_data.py into the training dict format."""
    p = Path(data_dir)
    meta_path = p / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}. Run 01_generate_data.sh first.")
    meta = json.loads(meta_path.read_text())
    n_subjects = meta["n_subjects"]
    dataset = {}
    for sub_id in range(n_subjects):
        eeg_path = p / f"sub{sub_id:02d}_eeg.npy"
        cmd_path = p / f"sub{sub_id:02d}_commands.npy"
        ern_path = p / f"sub{sub_id:02d}_ern.npy"
        if not eeg_path.exists():
            raise FileNotFoundError(f"Missing subject data for sub{sub_id:02d}. Re-run data generation.")
        dataset[sub_id] = {
            "eeg":       np.load(eeg_path),
            "commands":  np.load(cmd_path),
            "ern_labels": np.load(ern_path),
        }
    return dataset, meta
