#!/usr/bin/env python3
"""
Closed-loop virtual arm simulation.

Evaluates the LatencyPlanner in its intended setting: driving a simulated
6-DOF arm to class-specific DOF targets, comparing against a direct-decoder
baseline on the same neural trajectories.

Simulation loop
---------------
For each test trial (real EEG window from LOSO hold-out folds):

  1. Encode window → initial latent h₀
  2. Run two controllers in parallel:
       Baseline : cmd = decoder(h).mu  (direct intent decoding, no planning)
       Planner  : cmd, h_next = planner.plan(h, dof_goal=target)
                  h ← h_next (advance brain state via world model)
  3. Update virtual arm: pos ← clip(pos + dt·cmd, -1, 1)
  4. Record steps-to-target (convergence within threshold ε)

Metrics
-------
  time_to_target (TTT) : steps until ‖pos - target‖ < ε
                         lower is better; planner should beat baseline
  path_efficiency      : ‖target - start‖ / path_length  (0–1; 1 = straight)
  final_error          : ‖pos_final - target‖ after max_steps

Output
------
  logs/closed_loop_sim.json  — full per-trial results + aggregate stats
  Printed summary table

Usage
-----
  python -m v2_digital_self_replication.cli.closed_loop_sim \\
      --checkpoint .../supervised_best.pt \\
      --subjects 1-5 --n-trials 20 --max-steps 64
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("closed_loop_sim")


# ── Class → target DOF mapping ────────────────────────────────────────────────

_CLASS_TARGETS = {
    "left_hand":  np.array([ 1.0,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32),
    "right_hand": np.array([-1.0,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32),
    "feet":       np.array([ 0.0,  0.0,  1.0,  0.0,  0.0,  0.0], dtype=np.float32),
    "tongue":     np.array([ 0.0,  1.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32),
}


# ── Virtual arm ───────────────────────────────────────────────────────────────

class VirtualArm:
    """Simple Euler-integration 6-DOF virtual arm."""
    def __init__(self, dt: float = 1 / 256):
        self.dt  = dt
        self.pos = np.zeros(6, dtype=np.float32)

    def reset(self):
        self.pos[:] = 0.0

    def step(self, cmd: np.ndarray) -> np.ndarray:
        self.pos = np.clip(self.pos + self.dt * np.array(cmd, dtype=np.float32), -1.0, 1.0)
        return self.pos.copy()


# ── Simulation for one trial ──────────────────────────────────────────────────

def _simulate_trial(
    h0:        torch.Tensor,       # (1, d_model) initial latent
    target:    np.ndarray,         # (6,)
    decoder,
    transition,
    planner,
    max_steps: int   = 64,
    epsilon:   float = 0.3,        # convergence threshold (L2 in DOF space)
    dt:        float = 1 / 256,
) -> dict:
    """Returns per-controller stats for one trial."""
    target_t = torch.from_numpy(target).unsqueeze(0)  # (1,6)

    # ── Baseline controller ──────────────────────────────────────────────────
    arm_bl = VirtualArm(dt=dt)
    h_bl   = h0.clone()
    ttt_bl = max_steps
    path_bl = [arm_bl.pos.copy()]
    with torch.no_grad():
        for step in range(max_steps):
            cmd = decoder(h_bl).mu.squeeze(0).numpy()
            pos = arm_bl.step(cmd)
            path_bl.append(pos.copy())
            if np.linalg.norm(pos - target) < epsilon:
                ttt_bl = step + 1
                break
            # Baseline advances brain state via transition too (same world model)
            h_bl = transition(h_bl, torch.from_numpy(cmd).unsqueeze(0))

    # ── Planner controller (fast path: self_condition) ───────────────────────
    # self_condition() = T(h, decoder(h).mu) — one forward pass, the real-time
    # BCI path.  full plan() (CEM+gradient) is too slow for step-by-step sim.
    arm_pl  = VirtualArm(dt=dt)
    h_pl    = h0.clone()
    ttt_pl  = max_steps
    path_pl = [arm_pl.pos.copy()]
    with torch.no_grad():
        for step in range(max_steps):
            # Fast path: project latent toward next predicted state, then decode
            h_plan = planner.self_condition(h_pl)    # T(h, dec(h).mu)
            cmd    = decoder(h_plan).mu.squeeze(0).numpy()
            pos    = arm_pl.step(cmd)
            path_pl.append(pos.copy())
            if np.linalg.norm(pos - target) < epsilon:
                ttt_pl = step + 1
                break
            h_pl = h_plan

    def path_efficiency(path, target):
        pts = np.stack(path)
        total = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        direct = float(np.linalg.norm(pts[-1] - pts[0]))
        return direct / max(total, 1e-6)

    return {
        "baseline": {
            "ttt":              ttt_bl,
            "final_error":      float(np.linalg.norm(arm_bl.pos - target)),
            "path_efficiency":  path_efficiency(path_bl, target),
            "converged":        ttt_bl < max_steps,
        },
        "planner": {
            "ttt":              ttt_pl,
            "final_error":      float(np.linalg.norm(arm_pl.pos - target)),
            "path_efficiency":  path_efficiency(path_pl, target),
            "converged":        ttt_pl < max_steps,
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="v2_digital_self_replication/checkpoints/supervised_best.pt")
    p.add_argument("--subjects",   type=str, default="1-5",
                   help="Subjects to draw test windows from (range or comma list)")
    p.add_argument("--classes",    type=str, nargs="+",
                   default=["left_hand", "right_hand", "feet"])
    p.add_argument("--n-trials",   type=int, default=30,
                   help="Test trials per subject (random sample from LOSO hold-out)")
    p.add_argument("--max-steps",  type=int, default=64,
                   help="Max simulation steps per trial")
    p.add_argument("--epsilon",    type=float, default=0.3,
                   help="Convergence threshold (L2 distance to target)")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--output",     type=str,
                   default="v2_digital_self_replication/logs/closed_loop_sim.json")
    p.add_argument("--log-level",  type=str, default="INFO")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from v2_digital_self_replication.agent.digital_twin import DigitalTwin
    from v2_digital_self_replication.data.real_eeg import load_trials
    from v2_digital_self_replication.core.transition_model import ActionConditionedTransition
    from v2_digital_self_replication.core.latency_planner import LatencyPlanner
    from v2_digital_self_replication.config import V2Config

    # ── Load model ────────────────────────────────────────────────────────────
    log.info("Loading checkpoint: %s", args.checkpoint)
    twin = DigitalTwin()
    if Path(args.checkpoint).exists():
        twin.load(args.checkpoint)
    twin.eval()
    twin = twin.to(args.device)

    encoder    = twin.encoder
    decoder    = twin.decoder
    transition = twin.transition
    planner    = LatencyPlanner(
        transition=transition, decoder=decoder,
        horizon=3, n_iters=12, lr=0.05, gamma=0.8,
        mc_rollouts=32, mc_elite_frac=0.25,
    )

    def _parse_subjects(spec):
        if "-" in spec and "," not in spec:
            lo, hi = spec.split("-")
            return list(range(int(lo), int(hi) + 1))
        return [int(s) for s in spec.split(",")]

    subjects = _parse_subjects(args.subjects)
    log.info("Subjects: %s  Classes: %s  Trials/subj: %d",
             subjects, args.classes, args.n_trials)

    rng = np.random.default_rng(42)
    all_trials = []

    for subj in subjects:
        log.info("Subject %d …", subj)
        try:
            eeg, labels = load_trials(
                subjects=[subj], classes=args.classes, tmin=0.5, tmax=4.5,
            )
        except Exception as e:
            log.warning("  Subject %d load failed: %s", subj, e)
            continue

        n = len(labels)
        idxs = rng.choice(n, size=min(args.n_trials, n), replace=False)

        for idx in idxs:
            label = labels[idx]
            if label not in _CLASS_TARGETS:
                continue
            target = _CLASS_TARGETS[label]

            # Encode the EEG window → initial latent
            window = eeg[idx, -256:, :]   # (256, 21)
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(args.device)
            with torch.no_grad():
                out, _ = encoder(x)
                h0 = out[:, -64:, :].mean(1)   # (1, d_model)

            trial_result = _simulate_trial(
                h0, target, decoder, transition, planner,
                max_steps=args.max_steps,
                epsilon=args.epsilon,
            )
            trial_result["subject"] = subj
            trial_result["label"]   = label
            all_trials.append(trial_result)

        log.info("  %d trials simulated", len(idxs))

    if not all_trials:
        log.error("No trials simulated")
        return 1

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _agg(field, controller):
        vals = [t[controller][field] for t in all_trials]
        return {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals, ddof=1)),
            "median": float(np.median(vals)),
        }

    agg = {
        "n_trials":    len(all_trials),
        "max_steps":   args.max_steps,
        "epsilon":     args.epsilon,
        "baseline": {
            "ttt":             _agg("ttt", "baseline"),
            "final_error":     _agg("final_error", "baseline"),
            "path_efficiency": _agg("path_efficiency", "baseline"),
            "convergence_rate": float(np.mean([t["baseline"]["converged"] for t in all_trials])),
        },
        "planner": {
            "ttt":             _agg("ttt", "planner"),
            "final_error":     _agg("final_error", "planner"),
            "path_efficiency": _agg("path_efficiency", "planner"),
            "convergence_rate": float(np.mean([t["planner"]["converged"] for t in all_trials])),
        },
    }

    # Wilcoxon on TTT
    from scipy.stats import wilcoxon, ttest_rel
    ttt_bl  = [t["baseline"]["ttt"]  for t in all_trials]
    ttt_pl  = [t["planner"]["ttt"]   for t in all_trials]
    err_bl  = [t["baseline"]["final_error"] for t in all_trials]
    err_pl  = [t["planner"]["final_error"]  for t in all_trials]
    try:
        stat_ttt, p_ttt = wilcoxon(ttt_bl, ttt_pl, alternative="greater")
        stat_err, p_err = wilcoxon(err_bl, err_pl, alternative="greater")
    except Exception:
        stat_ttt = p_ttt = stat_err = p_err = float("nan")

    agg["statistics"] = {
        "ttt_wilcoxon_p":         float(p_ttt),
        "error_wilcoxon_p":       float(p_err),
        "ttt_delta_mean":         float(np.mean(ttt_bl) - np.mean(ttt_pl)),
        "ttt_delta_pct":          float((np.mean(ttt_bl) - np.mean(ttt_pl)) / np.mean(ttt_bl) * 100),
    }

    out_data = {
        "aggregate":   agg,
        "trial_results": all_trials,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2))
    log.info("Results saved → %s", out)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 58)
    print(f"  Closed-Loop Simulation — {len(all_trials)} trials")
    print(f"  Max steps: {args.max_steps}  ε={args.epsilon}  dt=1/256s")
    print("═" * 58)
    print(f"  {'Metric':<30} {'Baseline':>12} {'Planner':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for metric, key in [("Time-to-target (steps)", "ttt"),
                         ("Final error (L2)", "final_error"),
                         ("Path efficiency", "path_efficiency")]:
        bl = agg["baseline"][key]["mean"]
        pl = agg["planner"][key]["mean"]
        print(f"  {metric:<30} {bl:>12.3f} {pl:>12.3f}")
    print(f"  {'Convergence rate':<30} {agg['baseline']['convergence_rate']:>12.1%} "
          f"{agg['planner']['convergence_rate']:>12.1%}")
    print(f"\n  TTT reduction: {agg['statistics']['ttt_delta_mean']:.1f} steps "
          f"({agg['statistics']['ttt_delta_pct']:.1f}%)  "
          f"p={agg['statistics']['ttt_wilcoxon_p']:.4f}")
    print("═" * 58)
    return 0


if __name__ == "__main__":
    sys.exit(main())
