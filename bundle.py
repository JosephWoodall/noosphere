"""
noosphere/bundle.py
===================
WorldModelBundle — Shareable World Dynamics

A bundle packages the person-independent components of a trained Noosphere
world model so that community members can share learned dynamics without
sharing anything personal.

What is transferable
--------------------
The world model's understanding of *how the world behaves* is largely
domain-specific but person-independent:
    - How git commands change repository state
    - How processes behave under load
    - How physical objects move under gravity and contact
    - What exit codes mean
    - How Docker containers respond to commands

This knowledge lives in:
    rssm.*          — learned latent dynamics (GRU, prior, posterior MLPs)
    physics.*       — physics state estimator, transition prior, residual corrector
    consequence.*   — reward/value/termination heads + digital prediction heads
    obs_decoder.*   — observation reconstruction

What is NOT transferable
------------------------
    perception.s4.*         — calibrated to one person's neck EMG patterns
    perception.tokenizer.*  — fine structurally but paired with personal data
    apparatus.predictor.*   — trained on one person's kinematic labels and
                              electrode-placement-specific GP calibration
    calibration_session.*   — personal anchor data, meaningless to others

Design
------
A bundle is a single .pt file containing:
    1. metadata dict    — version, domain tags, quality metrics, description
    2. state dicts      — only the transferable modules listed above

Loading a bundle *merges* weights into an existing agent — it does not
replace the whole model. Your S4 encoder, your calibration, your GNN
topology all remain untouched. Only the world dynamics update.

Compatibility checking
-----------------------
Bundles carry the architecture dimensions they were trained with.
load_bundle() validates that state_dim, det_dim, stoch_cats, stoch_classes
match before loading. Mismatches raise a clear error rather than silently
corrupting weights.

Usage
-----
    from noosphere.bundle import export_bundle, load_bundle, BundleMetadata

    # Export after training
    meta = BundleMetadata(
        domain_tags=["shell", "linux"],
        description="500k steps on NixOS — git, Python toolchain, Docker",
        n_training_steps=500_000,
    )
    export_bundle(agent, "my_shell_dynamics.pt", meta, train_metrics)

    # Load on another machine
    info = load_bundle(agent, "my_shell_dynamics.pt")
    print(info)
    # {"loaded": ["rssm", "physics", "consequence", "obs_decoder"],
    #  "skipped": [],
    #  "metadata": {...}}

    # Inspect bundle without loading
    from noosphere.bundle import inspect_bundle
    print(inspect_bundle("my_shell_dynamics.pt"))
"""

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

BUNDLE_FORMAT_VERSION = "1.0"


# ── Metadata ──────────────────────────────────────────────────────────────────

@dataclass
class BundleMetadata:
    """
    Human-readable description of what a bundle contains and how it was trained.
    All fields are optional except domain_tags — fill in as much as you know.
    """
    domain_tags:          List[str] = field(default_factory=list)
    description:          str       = ""
    n_training_steps:     int       = 0
    author:               str       = ""
    created_at:           str       = ""   # ISO timestamp, filled automatically
    noosphere_version:    str       = ""   # filled from __version__ automatically

    # Architecture dimensions — filled automatically on export
    state_dim:            int       = 0
    det_dim:              int       = 0
    stoch_cats:           int       = 0
    stoch_classes:        int       = 0
    consequence_type:     str       = "standard"   # "standard" | "enhanced"
    digital_state_dim:    int       = 0

    # Quality signals at export time — guidance for the recipient
    wm_loss:              float     = 0.0
    kl_loss:              float     = 0.0
    reward_avg:           float     = 0.0
    physics_loss:         float     = 0.0

    # Format version for future compatibility
    bundle_format:        str       = BUNDLE_FORMAT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BundleMetadata":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def summary(self) -> str:
        lines = [
            f"  Bundle v{self.bundle_format} | Noosphere {self.noosphere_version}",
            f"  Created:  {self.created_at}",
            f"  Author:   {self.author or '(anonymous)'}",
            f"  Domains:  {', '.join(self.domain_tags) or '(unspecified)'}",
            f"  Steps:    {self.n_training_steps:,}",
            f"  Arch:     state={self.state_dim} det={self.det_dim} "
            f"cats={self.stoch_cats}×{self.stoch_classes}",
            f"  Quality:  wm_loss={self.wm_loss:.4f}  kl={self.kl_loss:.4f}  "
            f"reward={self.reward_avg:.3f}",
        ]
        if self.description:
            lines.append(f"  Notes:    {self.description}")
        return "\n".join(lines)


# ── Module keys — which parts of the agent are included in a bundle ───────────

_BUNDLE_KEYS = [
    "rssm",         # PhysicsAugmentedRSSM (includes sub-modules below)
]

# For finer-grained inspection and selective loading, we also track sub-keys
_RSSM_SUBKEYS = [
    "rssm.rssm",            # inner RSSM (GRU, prior/posterior MLPs)
    "rssm.state_est",       # physics state estimator
    "rssm.prior",           # RK4 physics transition prior
    "rssm.corrector",       # residual corrector
    "rssm.conservation",    # conservation law module
    "rssm.phys_proj",       # physics projection layer
]

_CONSEQUENCE_KEYS = [
    "consequence",          # ConsequenceModel or EnhancedConsequenceModel
    "obs_decoder",          # ObservationDecoder
]

ALL_BUNDLE_KEYS = _BUNDLE_KEYS + _CONSEQUENCE_KEYS


# ── Export ────────────────────────────────────────────────────────────────────

def export_bundle(
    agent,
    path: str,
    metadata: Optional[BundleMetadata] = None,
    train_metrics: Optional[Dict[str, float]] = None,
) -> BundleMetadata:
    """
    Export transferable world dynamics from a trained agent.

    Parameters
    ----------
    agent         : NoosphereAgent
    path          : output file path (conventionally .pt)
    metadata      : BundleMetadata — fill in domain_tags and description
    train_metrics : latest training metrics from agent.update() — used to
                    populate quality signals in the metadata

    Returns
    -------
    BundleMetadata populated with architecture dimensions and quality signals
    """
    from noosphere import __version__

    meta = metadata or BundleMetadata()
    meta.created_at         = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta.noosphere_version  = __version__
    meta.bundle_format      = BUNDLE_FORMAT_VERSION

    # Architecture dimensions — read directly from model
    rssm_inner              = agent.rssm.rssm
    meta.state_dim          = agent.rssm.state_dim
    meta.det_dim            = rssm_inner.det_dim
    meta.stoch_cats         = rssm_inner.stoch_cats
    meta.stoch_classes      = rssm_inner.stoch_classes

    # Detect whether consequence model has digital heads
    cons = agent.consequence
    meta.consequence_type   = "enhanced" if hasattr(cons, "digital") else "standard"
    if meta.consequence_type == "enhanced":
        meta.digital_state_dim = cons.digital.digital_state_dim

    # Quality signals from training metrics
    if train_metrics:
        meta.wm_loss     = float(train_metrics.get("wm/loss",    0.0))
        meta.kl_loss     = float(train_metrics.get("wm/kl",      0.0))
        meta.reward_avg  = float(train_metrics.get("ac/return",  0.0))
        meta.physics_loss= float(train_metrics.get("wm/physics", 0.0))

    # Collect state dicts for transferable modules
    bundle_state = {}
    for key in ALL_BUNDLE_KEYS:
        module = _get_nested(agent, key)
        if module is not None:
            bundle_state[key] = module.state_dict()
            logger.debug(f"  bundled: {key} "
                         f"({sum(p.numel() for p in module.parameters()):,} params)")
        else:
            logger.warning(f"  skipped (not found): {key}")

    bundle = {
        "metadata":     meta.to_dict(),
        "state_dicts":  bundle_state,
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)

    n_params = sum(
        sum(p.numel() for p in _get_nested(agent, k).parameters())
        for k in ALL_BUNDLE_KEYS if _get_nested(agent, k) is not None
    )
    logger.info(
        f"Bundle exported → {path}\n"
        f"  Modules: {list(bundle_state.keys())}\n"
        f"  Parameters: {n_params:,}\n"
        f"{meta.summary()}"
    )
    return meta


# ── Load ──────────────────────────────────────────────────────────────────────

def load_bundle(
    agent,
    path: str,
    strict_arch: bool = True,
    modules:     Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load a bundle into an existing agent, merging only world dynamics weights.

    The agent's S4 encoder, GNN, tokenizer, and apparatus predictor are
    untouched. Only the modules listed in ALL_BUNDLE_KEYS are updated.

    Parameters
    ----------
    agent        : NoosphereAgent — architecture must match bundle dims
    path         : bundle file path
    strict_arch  : if True (default), raise on architecture mismatch
                   if False, skip mismatched modules with a warning
    modules      : if provided, only load these specific module keys
                   (subset of ALL_BUNDLE_KEYS)

    Returns
    -------
    dict with keys:
        "loaded"   : list of module keys successfully loaded
        "skipped"  : list of module keys skipped (mismatch or not in file)
        "metadata" : BundleMetadata
    """
    raw    = torch.load(path, map_location=agent.device)
    meta   = BundleMetadata.from_dict(raw["metadata"])
    states = raw["state_dicts"]

    # Architecture compatibility check
    rssm_inner = agent.rssm.rssm
    mismatches = []
    if meta.state_dim and meta.state_dim != agent.rssm.state_dim:
        mismatches.append(
            f"state_dim: bundle={meta.state_dim} agent={agent.rssm.state_dim}"
        )
    if meta.det_dim and meta.det_dim != rssm_inner.det_dim:
        mismatches.append(
            f"det_dim: bundle={meta.det_dim} agent={rssm_inner.det_dim}"
        )
    if meta.stoch_cats and meta.stoch_cats != rssm_inner.stoch_cats:
        mismatches.append(
            f"stoch_cats: bundle={meta.stoch_cats} agent={rssm_inner.stoch_cats}"
        )
    if meta.stoch_classes and meta.stoch_classes != rssm_inner.stoch_classes:
        mismatches.append(
            f"stoch_classes: bundle={meta.stoch_classes} agent={rssm_inner.stoch_classes}"
        )

    if mismatches:
        msg = "Architecture mismatch:\n" + "\n".join(f"  {m}" for m in mismatches)
        if strict_arch:
            raise ValueError(
                f"{msg}\n\n"
                f"The bundle was trained with different model dimensions.\n"
                f"Either use strict_arch=False to skip mismatched modules, "
                f"or rebuild your agent with matching AgentConfig dimensions."
            )
        logger.warning(msg)

    keys_to_load = modules if modules is not None else ALL_BUNDLE_KEYS
    loaded  = []
    skipped = []

    for key in keys_to_load:
        if key not in states:
            skipped.append(key)
            logger.debug(f"  skip (not in bundle): {key}")
            continue

        module = _get_nested(agent, key)
        if module is None:
            skipped.append(key)
            logger.warning(f"  skip (not in agent): {key}")
            continue

        try:
            missing, unexpected = module.load_state_dict(
                states[key], strict=False
            )
            if missing:
                logger.debug(f"  {key}: missing keys {missing[:3]}")
            if unexpected:
                logger.debug(f"  {key}: unexpected keys {unexpected[:3]}")
            loaded.append(key)
            n_params = sum(p.numel() for p in module.parameters())
            logger.info(f"  loaded: {key}  ({n_params:,} params)")
        except Exception as e:
            skipped.append(key)
            if strict_arch:
                raise RuntimeError(f"Failed to load {key}: {e}") from e
            logger.warning(f"  skip (load error) {key}: {e}")

    logger.info(
        f"\nBundle loaded ← {path}\n"
        f"  Loaded:  {loaded}\n"
        f"  Skipped: {skipped}\n"
        f"{meta.summary()}"
    )

    return {
        "loaded":   loaded,
        "skipped":  skipped,
        "metadata": meta,
    }


# ── Inspect ───────────────────────────────────────────────────────────────────

def inspect_bundle(path: str) -> str:
    """
    Read and describe a bundle without loading it into any model.
    Safe to call before deciding whether to load.
    """
    raw    = torch.load(path, map_location="cpu")
    meta   = BundleMetadata.from_dict(raw["metadata"])
    states = raw["state_dicts"]

    lines = [f"\nBundle: {path}", meta.summary(), "\nContents:"]
    total_params = 0
    for key, sd in states.items():
        n = sum(t.numel() for t in sd.values())
        total_params += n
        lines.append(f"  {key:<30} {n:>12,} params")
    lines.append(f"\n  Total transferable parameters: {total_params:,}")

    # Size on disk
    size_mb = Path(path).stat().st_size / 1e6
    lines.append(f"  File size: {size_mb:.1f} MB")

    return "\n".join(lines)


# ── Compatibility matrix ──────────────────────────────────────────────────────

def check_compatibility(agent, path: str) -> Dict[str, Any]:
    """
    Check whether a bundle is compatible with an agent without loading it.
    Returns a dict describing what would be loaded, skipped, and any issues.
    """
    raw    = torch.load(path, map_location="cpu")
    meta   = BundleMetadata.from_dict(raw["metadata"])
    rssm_  = agent.rssm.rssm

    issues   = []
    warnings = []

    if meta.det_dim and meta.det_dim != rssm_.det_dim:
        issues.append(f"det_dim mismatch: bundle={meta.det_dim} agent={rssm_.det_dim}")
    if meta.stoch_cats and meta.stoch_cats != rssm_.stoch_cats:
        issues.append(f"stoch_cats mismatch: bundle={meta.stoch_cats} agent={rssm_.stoch_cats}")
    if meta.stoch_classes and meta.stoch_classes != rssm_.stoch_classes:
        issues.append(f"stoch_classes mismatch: bundle={meta.stoch_classes} agent={rssm_.stoch_classes}")

    # Consequence type check
    agent_type = "enhanced" if hasattr(agent.consequence, "digital") else "standard"
    if meta.consequence_type != agent_type:
        warnings.append(
            f"consequence type: bundle={meta.consequence_type} agent={agent_type}. "
            f"Digital heads will be {'skipped' if meta.consequence_type == 'enhanced' else 'uninitialised'}."
        )

    return {
        "compatible":  len(issues) == 0,
        "issues":      issues,
        "warnings":    warnings,
        "metadata":    meta,
        "would_load":  [k for k in raw["state_dicts"] if not issues],
    }


# ── Utility ───────────────────────────────────────────────────────────────────

def _get_nested(obj: Any, dotted_key: str) -> Optional[Any]:
    """Resolve 'rssm.state_est' → obj.rssm.state_est safely."""
    parts = dotted_key.split(".")
    cur   = obj
    for part in parts:
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur
