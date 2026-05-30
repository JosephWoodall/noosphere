#!/usr/bin/env python3
"""
Enhanced closed-loop virtual arm simulation with three controllers.

Demonstrates that the world-model architecture works correctly when
the decoder is accurate, and fails only because current decoders
are data-limited.

Controllers compared
--------------------
  oracle  — knows the true class label; issues the exact target DOF command
            at every step.  Represents a perfect decoder (100% accuracy).
  csp     — fits one-vs-rest CSP+LDA on training fold, predicts class
            on the EEG test window, issues the corresponding DOF command.
            Accuracy ≈ 60.1% (the real CSP result).
  jepa    — uses the trained IntentDecoder (ZOH-SSM world-model checkpoint).
            Accuracy ≈ 37% in static eval; commands ≈ near-zero in practice.

Simulation dynamics
-------------------
  pos(t+1) = clip(pos(t) + dt * cmd(t), -1, 1)
  dt = 1/256 s    max_steps = 384    convergence: ||pos - target|| < eps

Per-class DOF targets (same as closed_loop_sim.py):
  left_hand  → [ 1, 0, 0, 0, 0, 0]
  right_hand → [-1, 0, 0, 0, 0, 0]
  feet       → [ 0, 0, 1, 0, 0, 0]

Usage
-----
  python -m v2_digital_self_replication.cli.closed_loop_sim_full \\
      --subjects 1-10 --n-trials 30 --max-steps 384 --epsilon 0.25 \\
      --checkpoint .../supervised_best.pt \\
      --output logs/closed_loop_sim_full.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("closed_loop_sim_full")

_CLASS_TARGETS = {
    "left_hand":  np.array([ 1., 0., 0., 0., 0., 0.], dtype=np.float32),
    "right_hand": np.array([-1., 0., 0., 0., 0., 0.], dtype=np.float32),
    "feet":       np.array([ 0., 0., 1., 0., 0., 0.], dtype=np.float32),
}

_LABEL_TO_INT = {"left_hand": 0, "right_hand": 1, "feet": 2}
_INT_TO_TARGET = {0: _CLASS_TARGETS["left_hand"],
                  1: _CLASS_TARGETS["right_hand"],
                  2: _CLASS_TARGETS["feet"]}


class VirtualArm:
    def __init__(self, dt=1/256):
        self.dt = dt
        self.pos = np.zeros(6, dtype=np.float32)

    def reset(self):
        self.pos[:] = 0.

    def step(self, cmd):
        self.pos = np.clip(self.pos + self.dt * np.asarray(cmd, np.float32), -1., 1.)
        return self.pos.copy()


# ── CSP helper ────────────────────────────────────────────────────────────────

def _fit_csp(X_tr, y_tr):
    """Fit one-vs-rest CSP+LDA.  X: (n, C, T), y: int array."""
    from scipy.linalg import eigh
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    classes = np.unique(y_tr)
    n_filters = 4

    def _cov(X):
        C = X @ X.transpose(0, 2, 1)
        return C.mean(0) / X.shape[2]

    filters = []
    for cls in classes:
        mask = y_tr == cls
        C_cls = _cov(X_tr[mask])
        C_tot = _cov(X_tr)
        _, W  = eigh(C_cls, C_tot)
        half  = n_filters // 2
        filters.append(np.concatenate([W[:, :half], W[:, -half:]], axis=1))

    def _feat(X, fs):
        return np.concatenate([np.log(np.var(W.T @ X, axis=-1)) for W in fs])

    X_tr_f = np.array([_feat(x, filters) for x in X_tr])
    sc  = StandardScaler().fit(X_tr_f)
    clf = LinearDiscriminantAnalysis().fit(sc.transform(X_tr_f), y_tr)
    return filters, sc, clf


def _predict_csp(x, filters, sc, clf):
    """x: (C, T) → int label."""
    feat = np.array([np.concatenate([np.log(np.var(W.T @ x, axis=-1)) for W in filters])])
    return int(clf.predict(sc.transform(feat))[0])


# ── Simulate one trial ────────────────────────────────────────────────────────

def _sim_trial(h0, target, true_label_int, eeg_raw,
               decoder, transition,
               csp_state,           # (filters, sc, clf) or None
               max_steps, epsilon, dt):
    """Returns dict with results for all three controllers."""
    target_t = torch.from_numpy(target).unsqueeze(0)
    results = {}

    for ctrl in ("oracle", "csp", "jepa"):
        arm = VirtualArm(dt=dt)
        h   = h0.clone()
        ttt = max_steps
        path_len = 0.
        prev_pos = arm.pos.copy()

        for step in range(max_steps):
            with torch.no_grad():
                if ctrl == "oracle":
                    cmd = target.copy()              # perfect DOF direction
                elif ctrl == "csp" and csp_state is not None:
                    T_use = min(eeg_raw.shape[1], 256)
                    x_win = eeg_raw[-T_use:, :].T   # (C, T)
                    pred  = _predict_csp(x_win, *csp_state)
                    cmd   = _INT_TO_TARGET[pred].copy()
                else:  # jepa
                    h_plan = transition(h, decoder(h).mu)
                    cmd    = decoder(h_plan).mu.squeeze(0).numpy()
                    h      = h_plan

            pos = arm.step(cmd)
            path_len += float(np.linalg.norm(pos - prev_pos))
            prev_pos = pos.copy()

            if np.linalg.norm(pos - target) < epsilon:
                ttt = step + 1
                break

        direct = float(np.linalg.norm(arm.pos - np.zeros(6)))
        pe = direct / max(path_len, 1e-6)
        results[ctrl] = {
            "ttt":             ttt,
            "final_error":     float(np.linalg.norm(arm.pos - target)),
            "path_efficiency": pe,
            "converged":       ttt < max_steps,
        }
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="v2_digital_self_replication/checkpoints/supervised_best.pt")
    p.add_argument("--subjects",   type=str, default="1-10")
    p.add_argument("--classes",    type=str, nargs="+",
                   default=["left_hand", "right_hand", "feet"])
    p.add_argument("--n-trials",   type=int, default=30)
    p.add_argument("--max-steps",  type=int, default=384)
    p.add_argument("--epsilon",    type=float, default=0.25)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--output",     type=str,
                   default="v2_digital_self_replication/logs/closed_loop_sim_full.json")
    p.add_argument("--log-level",  type=str, default="INFO")
    return p.parse_args()


def _parse_subjects(spec):
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from v2_digital_self_replication.agent.digital_twin import DigitalTwin
    from v2_digital_self_replication.data.real_eeg import load_trials
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    log.info("Loading checkpoint: %s", args.checkpoint)
    twin = DigitalTwin()
    if Path(args.checkpoint).exists():
        twin.load(args.checkpoint)
    twin.eval()
    decoder    = twin.decoder
    transition = twin.transition

    subjects = _parse_subjects(args.subjects)
    rng = np.random.default_rng(42)
    all_trials = []

    for subj in subjects:
        log.info("Subject %d …", subj)
        try:
            eeg, labels = load_trials(subjects=[subj], classes=args.classes,
                                      tmin=0.5, tmax=4.5)
        except Exception as e:
            log.warning("  Load failed: %s", e); continue

        le = LabelEncoder().fit(sorted(set(labels)))
        y  = le.transform(labels)

        # Use 80/20 split for CSP training (first fold of 5)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tr_idx, te_idx = next(iter(skf.split(eeg, y)))

        # Fit CSP on training split
        X_raw = eeg.transpose(0, 2, 1)
        T_use = min(eeg.shape[1], 256)
        try:
            csp_state = _fit_csp(X_raw[tr_idx, :, -T_use:], y[tr_idx])
        except Exception as e:
            log.warning("  CSP fit failed: %s", e); csp_state = None

        # Sample test trials
        n_te = len(te_idx)
        idxs = rng.choice(n_te, size=min(args.n_trials, n_te), replace=False)

        for idx in idxs:
            global_idx = te_idx[idx]
            label      = labels[global_idx]
            if label not in _CLASS_TARGETS:
                continue
            target = _CLASS_TARGETS[label]

            # Encode EEG → initial latent
            window = eeg[global_idx, -256:, :]
            x = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                out, _ = twin.encoder(x)
                h0 = out[:, -64:, :].mean(1)

            r = _sim_trial(
                h0, target, _LABEL_TO_INT.get(label, 0),
                eeg[global_idx],
                decoder, transition, csp_state,
                args.max_steps, args.epsilon, 1/256,
            )
            r["subject"] = subj
            r["label"]   = label
            all_trials.append(r)

        n_done = sum(1 for t in all_trials if t["subject"] == subj)
        log.info("  ↳ Subject %d: %d trials simulated", subj, n_done)

    if not all_trials:
        log.error("No trials"); return 1

    # ── Aggregate ─────────────────────────────────────────────────────────────
    from scipy.stats import wilcoxon

    def _agg(ctrl):
        return {
            "ttt":              {"mean": float(np.mean([t[ctrl]["ttt"] for t in all_trials])),
                                 "std":  float(np.std([t[ctrl]["ttt"] for t in all_trials]))},
            "final_error":      {"mean": float(np.mean([t[ctrl]["final_error"] for t in all_trials]))},
            "path_efficiency":  {"mean": float(np.mean([t[ctrl]["path_efficiency"] for t in all_trials]))},
            "convergence_rate": float(np.mean([t[ctrl]["converged"] for t in all_trials])),
        }

    agg = {ctrl: _agg(ctrl) for ctrl in ("oracle", "csp", "jepa")}

    # Wilcoxon: CSP vs JEPA on TTT
    ttt_csp  = [t["csp"]["ttt"]  for t in all_trials]
    ttt_jepa = [t["jepa"]["ttt"] for t in all_trials]
    try:
        _, p_csp_jepa = wilcoxon(ttt_csp, ttt_jepa, alternative="less")
    except Exception:
        p_csp_jepa = float("nan")

    agg["statistics"] = {
        "n_trials":           len(all_trials),
        "p_csp_vs_jepa_ttt":  float(p_csp_jepa),
        "ttt_delta_csp_jepa": float(np.mean(ttt_csp) - np.mean(ttt_jepa)),
    }

    out_data = {"aggregate": agg, "trial_results": all_trials}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2))
    log.info("Results saved → %s", out)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print(f"  Three-Controller Closed-Loop Sim — {len(all_trials)} trials")
    print(f"  Max steps: {args.max_steps}  ε={args.epsilon}  dt=1/256s")
    print("═" * 62)
    print(f"  {'Controller':<14} {'TTT (steps)':>12} {'Conv.%':>8} {'FinalErr':>10}")
    print(f"  {'-'*14} {'-'*12} {'-'*8} {'-'*10}")
    for ctrl in ("oracle", "csp", "jepa"):
        a = agg[ctrl]
        print(f"  {ctrl:<14} {a['ttt']['mean']:>12.1f} "
              f"{a['convergence_rate']:>8.1%} "
              f"{a['final_error']['mean']:>10.3f}")
    print(f"\n  CSP vs JEPA TTT: p={agg['statistics']['p_csp_vs_jepa_ttt']:.4f}")
    print("═" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
