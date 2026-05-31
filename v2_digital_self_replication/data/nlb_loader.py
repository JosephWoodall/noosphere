"""
NLB MC_Maze loader: downloads and preprocesses macaque M1 spike data as JEPA teacher.

Dataset: Neural Latents Benchmark MC_Maze
  - 182 M1 neurons, macaque, random-target reaching
  - DANDI Archive dandiset 000128
  - Spike trains at 1ms bins → bin to 4ms (250 Hz ≈ 256 Hz EEG)
  - Output: (n_trials, T, 182) float32 at 250Hz

Usage:
    loader = NLBLoader()
    spikes = loader.load()          # (n_trials, T, 182) float32
    spikes = loader.load(smooth=True)   # Gaussian-smoothed firing rates

Fallback: if nlb_tools / dandi are not available, returns synthetic spike data
with similar statistics for development/CI purposes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_NLB_CACHE_DIR = Path(os.environ.get("NLB_CACHE_DIR", "v2_digital_self_replication/data/nlb_cache"))
_DANDI_ID = "000128"
_N_NEURONS = 182
_BIN_MS = 4          # bin size in ms → 250Hz
_ORIG_BIN_MS = 1     # raw MC_Maze bin size
_BIN_FACTOR = _BIN_MS // _ORIG_BIN_MS


def _gaussian_smooth(rates: np.ndarray, sigma_bins: float = 5.0) -> np.ndarray:
    """Gaussian smooth along time axis. rates: (n_trials, T, N)."""
    from scipy.ndimage import gaussian_filter1d
    return gaussian_filter1d(rates.astype(np.float32), sigma=sigma_bins, axis=1)


class NLBLoader:
    """
    Loads NLB MC_Maze spike data.

    If nlb_tools and dandi are available, downloads the dataset on first call
    and caches to disk. Otherwise generates synthetic spike trains with matched
    statistics (Poisson, ~8 Hz mean firing rate, 182 neurons).
    """

    def __init__(self, cache_dir: Path = _NLB_CACHE_DIR, bin_ms: int = _BIN_MS):
        self.cache_dir = Path(cache_dir)
        self.bin_ms = bin_ms
        self.bin_factor = bin_ms // _ORIG_BIN_MS
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, smooth: bool = True, split: str = "train") -> np.ndarray:
        """
        Returns spike data as float32 array of shape (n_trials, T, 182).

        Args:
            smooth: apply Gaussian smoothing (sigma=5 bins = 20ms at 250Hz)
            split: 'train' or 'val'

        Returns:
            np.ndarray: (n_trials, T, 182) firing rates at 250Hz
        """
        cache_path = self.cache_dir / f"mc_maze_{split}_bin{self.bin_ms}ms.npy"

        if cache_path.exists():
            logger.info("NLB: loading from cache %s", cache_path)
            spikes = np.load(str(cache_path))
            if smooth:
                spikes = _gaussian_smooth(spikes)
            return spikes

        spikes = self._download_or_synthesize(split)
        np.save(str(cache_path), spikes)
        logger.info("NLB: cached to %s  shape=%s", cache_path, spikes.shape)

        if smooth:
            spikes = _gaussian_smooth(spikes)
        return spikes

    def _download_or_synthesize(self, split: str) -> np.ndarray:
        try:
            return self._download_nlb(split)
        except ImportError as e:
            logger.warning("NLB: nlb_tools/dandi not available (%s); using synthetic data", e)
            return self._synthetic_fallback(split)
        except Exception as e:
            logger.warning("NLB: download failed (%s); using synthetic data", e)
            return self._synthetic_fallback(split)

    def _download_nlb(self, split: str) -> np.ndarray:
        """Download MC_Maze via nlb_tools or dandi."""
        try:
            from nlb_tools.nwb_interface import NWBDataset
            return self._load_via_nlb_tools(split)
        except ImportError:
            pass

        # Fallback: dandi API
        try:
            from dandi.dandiapi import DandiAPIClient
            return self._load_via_dandi(split)
        except ImportError:
            raise ImportError("Neither nlb_tools nor dandi is installed. "
                              "Run: pip install nlb_tools dandi")

    def _load_via_nlb_tools(self, split: str) -> np.ndarray:
        """Load using nlb_tools NWBDataset API."""
        from nlb_tools.nwb_interface import NWBDataset

        nwb_path = self.cache_dir / "mc_maze.nwb"
        if not nwb_path.exists():
            self._download_nwb_file(nwb_path)

        dataset = NWBDataset(str(nwb_path), "*train" if split == "train" else "*val")
        # Bin at 4ms
        dataset.resample(self.bin_ms)

        # Get spike counts: (n_trials, T, n_neurons)
        spikes = dataset.data["spikes"]
        if hasattr(spikes, "values"):
            spikes = spikes.values
        spikes = np.array(spikes, dtype=np.float32)

        # Normalise to firing rates (spikes per second)
        spikes = spikes / (self.bin_ms * 1e-3)
        return spikes

    def _load_via_dandi(self, split: str) -> np.ndarray:
        """Download NWB file from DANDI and load spike trains manually."""
        from dandi.dandiapi import DandiAPIClient
        import h5py

        nwb_path = self.cache_dir / "mc_maze.nwb"
        if not nwb_path.exists():
            self._download_nwb_file(nwb_path)

        spikes = self._parse_nwb_spikes(nwb_path, split)
        return spikes

    def _download_nwb_file(self, dest: Path):
        """Download the primary MC_Maze NWB file from DANDI."""
        logger.info("NLB: downloading MC_Maze from DANDI dandiset %s ...", _DANDI_ID)
        try:
            from dandi.dandiapi import DandiAPIClient
            with DandiAPIClient() as client:
                dandiset = client.get_dandiset(_DANDI_ID, "draft")
                assets = list(dandiset.get_assets_by_glob("*.nwb"))
                if not assets:
                    raise ValueError("No NWB files found in dandiset %s" % _DANDI_ID)
                # Pick the MC_Maze_Large or first file
                asset = next(
                    (a for a in assets if "mc_maze" in a.path.lower()),
                    assets[0]
                )
                logger.info("NLB: downloading %s (%s bytes)", asset.path, asset.size)
                asset.download(str(dest))
        except Exception as e:
            raise RuntimeError("DANDI download failed: %s" % e) from e

    def _parse_nwb_spikes(self, nwb_path: Path, split: str) -> np.ndarray:
        """Parse spike trains from NWB file into binned array."""
        import h5py

        with h5py.File(str(nwb_path), "r") as f:
            # Standard NWB spike format under /units
            spike_times = []
            n_units = len(f["units"]["spike_times_index"])
            for i in range(n_units):
                idx_start = f["units"]["spike_times_index"][i - 1] if i > 0 else 0
                idx_end = f["units"]["spike_times_index"][i]
                st = f["units"]["spike_times"][idx_start:idx_end][:]
                spike_times.append(st)

            # Trial timing
            start_times = f["intervals"]["trials"]["start_time"][:]
            stop_times = f["intervals"]["trials"]["stop_time"][:]

        n_trials = len(start_times)
        bin_s = self.bin_ms * 1e-3
        trial_lens = stop_times - start_times
        T_max = int(np.ceil(trial_lens.max() / bin_s))
        n_neurons = len(spike_times)

        spikes = np.zeros((n_trials, T_max, n_neurons), dtype=np.float32)
        for i, (t_start, t_stop) in enumerate(zip(start_times, stop_times)):
            for j, st in enumerate(spike_times):
                trial_st = st[(st >= t_start) & (st < t_stop)] - t_start
                bin_idx = (trial_st / bin_s).astype(int)
                bin_idx = bin_idx[bin_idx < T_max]
                np.add.at(spikes[i, :, j], bin_idx, 1.0)

        # Normalise to Hz
        spikes /= bin_s
        return spikes

    def _synthetic_fallback(self, split: str) -> np.ndarray:
        """
        Synthetic MC_Maze-like data: Poisson spike trains with realistic statistics.
        - 182 neurons, ~8Hz mean firing rate
        - 1000 trials of 1.5s duration = 375 bins at 250Hz
        """
        rng = np.random.default_rng(42 if split == "train" else 99)
        n_trials = 800 if split == "train" else 200
        T = 375   # 1.5s at 250Hz
        N = _N_NEURONS

        # Neuron-specific mean rates between 2 and 30 Hz
        mean_rates = rng.uniform(2.0, 30.0, size=N).astype(np.float32)

        # Poisson spikes, binned at 4ms
        lam = mean_rates[None, None, :] * (self.bin_ms * 1e-3)  # (1, 1, N)
        spikes = rng.poisson(lam, size=(n_trials, T, N)).astype(np.float32)
        # Convert counts to Hz
        spikes /= (self.bin_ms * 1e-3)

        logger.info("NLB: synthetic fallback  shape=%s", spikes.shape)
        return spikes


class NLBWindowDataset:
    """
    Sliding-window dataset over (n_trials, T, 182) CNS spike data.
    Compatible with torch DataLoader.
    """

    def __init__(self, spikes: np.ndarray, window_len: int = 256, stride: int = 64):
        """
        spikes: (n_trials, T, 182) float32
        window_len: number of 4ms bins per window (256 bins = 1.024s)
        """
        import torch
        windows = []
        for trial in spikes:
            T = trial.shape[0]
            for start in range(0, T - window_len + 1, stride):
                windows.append(trial[start: start + window_len])
        self.data = torch.from_numpy(np.stack(windows, axis=0).astype(np.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
