#!/usr/bin/env python3
"""
Four-controller closed-loop virtual arm simulation.

Adds the Action-Conditioned SSM (AC-SSM) as a fourth controller alongside
oracle, CSP, and the original JEPA baseline.

AC-SSM controller design
------------------------
The MLP transition model (JEPA controller) fails because it rolls out in a
latent space with no motor semantics.  The AC-SSM replaces it with iterative
self-conditioning: re-encode the static EEG window at each simulation step
with the previous decoded command as a_prev.  The SiLU-gated action injection
in the first SSM block conditions the latent on recent motor context, so the
decoded command converges toward the correct class intent.

The AC-SSM is fine-tuned per subject on the training fold (same split used
for CSP fitting) using MSE on intent vectors + EEG reconstruction loss.
This makes the comparison fair: both CSP and AC-SSM are subject-specific,
while the JEPA baseline remains the base checkpoint (no fine-tuning).

Usage
-----
  python -m v2_digital_self_replication.cli.closed_loop_sim_v2 \\
      --subjects 1-10 --n-trials 30 --max-steps 384 --epsilon 0.25 \\
      --checkpoint v2_digital_self_replication/checkpoints/jepa_encoder_best.pt \\
      --output logs/closed_loop_sim_v2.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("closed_loop_sim_v2")

_CLASS_TARGETS = {
    "left_hand":  np.array([ 1., 0., 0., 0., 0., 0.], dtype=np.float32),
    "right_hand": np.array([-1., 0., 0., 0., 0., 0.], dtype=np.float32),
    "feet":       np.array([ 0., 0., 1., 0., 0., 0.], dtype=np.float32),
}
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


# ── CSP helpers ───────────────────────────────────────────────────────────────

def _fit_csp(X_tr, y_tr):
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
    feat = np.array([np.concatenate([np.log(np.var(W.T @ x, axis=-1)) for W in filters])])
    return int(clf.predict(sc.transform(feat))[0])


# ── AC-SSM per-subject fine-tuning ────────────────────────────────────────────

def _finetune_ac_ssm(checkpoint, eeg_tr, labels_tr, device, n_epochs=10):
    """
    Fine-tune an AC-SSM encoder + linear classification head on the training fold.

    Returns (ac_enc, cls_head, sorted_classes) all in eval mode.

    At simulation time the controller:
      1. Re-encodes the EEG window with a_prev (previous issued command).
      2. Classifies via the linear head.
      3. Looks up the class intent vector (deterministic mapping, same as CSP/oracle).
      4. Issues that intent vector as the motor command.
      5. Feeds it back as a_prev for the next step.

    This removes dependence on the continuous IntentDecoder (which produces
    near-zero magnitude commands due to the data-limited decoder bottleneck).
    The convergence rate therefore reflects classifier accuracy, not decoder
    magnitude — directly comparable to CSP's convergence rate.
    """
    from v2_digital_self_replication.core.stream_encoder import StreamEncoder
    from v2_digital_self_replication.data.intent_mapping import CLASS_TO_INTENT
    from v2_digital_self_replication.config import V2Config
    from sklearn.preprocessing import LabelEncoder as _LE

    enc_cfg = V2Config().encoder
    dec_cfg = V2Config().decoder
    n_dof   = dec_cfg.n_dof
    n_eeg   = enc_cfg.n_eeg_channels

    ac_enc = StreamEncoder(
        d_model=enc_cfg.d_model, d_state=enc_cfg.d_state,
        n_layers=enc_cfg.n_layers, n_eeg=n_eeg,
        n_prop=enc_cfg.n_prop_channels, dropout=enc_cfg.dropout,
        d_dof=n_dof, n_eeg_recon=n_eeg,
    ).to(device)

    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("encoder", ckpt)
        ac_enc.load_state_dict(sd, strict=False)

    le = _LE().fit(sorted(set(labels_tr)))
    sorted_classes = le.classes_.tolist()
    n_classes = len(sorted_classes)
    y_tr = torch.tensor(le.transform(labels_tr), dtype=torch.long, device=device)

    # Intent matrix aligned to LabelEncoder order (for teacher forcing)
    intent_rows = [
        CLASS_TO_INTENT.get(c, np.zeros(n_dof, np.float32)).astype(np.float32)
        for c in sorted_classes
    ]
    intent_matrix = torch.from_numpy(np.stack(intent_rows)).to(device)  # (n_cls, n_dof)

    cls_head = nn.Linear(enc_cfg.d_model, n_classes).to(device)

    T_use = min(eeg_tr.shape[1], 256)
    X_tr  = torch.from_numpy(eeg_tr[:, -T_use:, :].astype(np.float32)).to(device)

    opt = torch.optim.Adam(
        list(ac_enc.parameters()) + list(cls_head.parameters()),
        lr=1e-3, weight_decay=0.1,
    )
    ac_enc.train(); cls_head.train()

    for _ in range(n_epochs):
        perm = torch.randperm(len(X_tr), device=device)
        for i in range(0, len(X_tr), 16):
            idx    = perm[i : i + 16]
            Xb     = X_tr[idx]
            yb     = y_tr[idx]
            a_prev = intent_matrix[yb]   # teacher forcing: ground-truth intent

            out, _ = ac_enc(Xb, a_prev=a_prev)
            h      = out[:, -64:, :].mean(1)

            loss_ce    = F.cross_entropy(cls_head(h), yb)
            eeg_hat    = ac_enc.reconstruct_eeg(out)
            loss_recon = F.mse_loss(eeg_hat, Xb) * 0.05

            loss = loss_ce + loss_recon
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(ac_enc.parameters()) + list(cls_head.parameters()), 1.0
            )
            opt.step()

    ac_enc.eval(); cls_head.eval()
    return ac_enc, cls_head, sorted_classes


# ── Simulate one trial ────────────────────────────────────────────────────────

def _sim_trial(
    h0,            # (1, d_model) — initial JEPA latent for the oracle/jepa path
    eeg_window,    # (T, C) numpy — raw EEG window for AC-SSM re-encoding
    target,        # (6,) numpy — DOF goal
    true_label_int,
    decoder,       # base IntentDecoder (for JEPA controller)
    transition,    # MLP transition (for JEPA controller)
    csp_state,     # (filters, sc, clf) or None
    ac_ssm_state,  # (ac_enc, cls_head, sorted_classes) or None
    max_steps, epsilon, dt,
    device="cpu",
):
    target_t = torch.from_numpy(target).unsqueeze(0)
    results = {}

    controllers = ["oracle", "csp", "jepa"]
    if ac_ssm_state is not None:
        controllers.append("ac_ssm")

    for ctrl in controllers:
        arm = VirtualArm(dt=dt)
        h   = h0.clone()
        ttt = max_steps
        path_len = 0.
        prev_pos = arm.pos.copy()

        # AC-SSM state: re-encode EEG window with feedback a_prev
        T_use       = min(eeg_window.shape[0], 256)
        eeg_tensor  = torch.from_numpy(
            eeg_window[-T_use:, :].astype(np.float32)
        ).unsqueeze(0).to(device)
        a_prev_ac   = torch.zeros(1, 6, device=device)

        for step in range(max_steps):
            with torch.no_grad():
                if ctrl == "oracle":
                    cmd = target.copy()

                elif ctrl == "csp" and csp_state is not None:
                    x_win = eeg_window[-T_use:, :].T   # (C, T)
                    pred  = _predict_csp(x_win, *csp_state)
                    cmd   = _INT_TO_TARGET[pred].copy()

                elif ctrl == "jepa":
                    # Original MLP transition rollout (data-limited baseline)
                    h_plan = transition(h, decoder(h).mu)
                    cmd    = decoder(h_plan).mu.squeeze(0).cpu().numpy()
                    h      = h_plan

                else:  # ac_ssm
                    # Classifier-decoder: re-encode EEG with previous command as
                    # context, classify, then look up the sim's DOF target for
                    # that class (same mapping used by oracle and CSP).
                    # Convergence reflects classifier accuracy, directly comparable
                    # to CSP which uses the same class→target lookup.
                    ac_enc, cls_head, sorted_classes = ac_ssm_state
                    out, _ = ac_enc(eeg_tensor, a_prev=a_prev_ac)
                    h_ac   = out[:, -64:, :].mean(1)
                    pred_cls = cls_head(h_ac).argmax(1).item()
                    cls_name = sorted_classes[pred_cls]
                    cmd = _CLASS_TARGETS.get(cls_name, np.zeros(6, np.float32)).copy()
                    a_prev_ac = torch.from_numpy(cmd).unsqueeze(0).to(device)

            pos = arm.step(cmd)
            path_len += float(np.linalg.norm(pos - prev_pos))
            prev_pos  = pos.copy()

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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="v2_digital_self_replication/checkpoints/supervised_best.pt",
                   help="Supervised twin checkpoint (encoder+decoder+transition)")
    p.add_argument("--jepa-checkpoint", type=str,
                   default="v2_digital_self_replication/checkpoints/jepa_encoder_best.pt",
                   help="JEPA encoder checkpoint for AC-SSM initialisation")
    p.add_argument("--subjects",   type=str, default="1-10")
    p.add_argument("--classes",    type=str, nargs="+",
                   default=["left_hand", "right_hand", "feet"])
    p.add_argument("--n-trials",   type=int, default=30)
    p.add_argument("--max-steps",  type=int, default=384)
    p.add_argument("--epsilon",    type=float, default=0.25)
    p.add_argument("--ft-epochs",  type=int, default=10,
                   help="Fine-tuning epochs for AC-SSM per subject")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--output",     type=str,
                   default="v2_digital_self_replication/logs/closed_loop_sim_v2.json")
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

    log.info("Checkpoint: %s", args.checkpoint)
    twin = DigitalTwin()
    if Path(args.checkpoint).exists():
        twin.load(args.checkpoint)
    twin.eval()
    decoder    = twin.decoder
    transition = twin.transition

    subjects  = _parse_subjects(args.subjects)
    rng       = np.random.default_rng(42)
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

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        tr_idx, te_idx = next(iter(skf.split(eeg, y)))

        X_raw = eeg.transpose(0, 2, 1)
        T_use = min(eeg.shape[1], 256)

        # CSP: fit on training fold
        try:
            csp_state = _fit_csp(X_raw[tr_idx, :, -T_use:], y[tr_idx])
        except Exception as e:
            log.warning("  CSP fit failed: %s", e); csp_state = None

        # AC-SSM: fine-tune on training fold
        eeg_tr     = eeg[tr_idx]
        labels_tr  = [labels[i] for i in tr_idx]
        try:
            log.info("  Fine-tuning AC-SSM classifier (%d trials, %d epochs) …",
                     len(labels_tr), args.ft_epochs)
            ac_enc, cls_head, sorted_classes = _finetune_ac_ssm(
                args.jepa_checkpoint, eeg_tr, labels_tr,
                device=args.device, n_epochs=args.ft_epochs,
            )
            ac_ssm_state = (ac_enc, cls_head, sorted_classes)
        except Exception as e:
            log.warning("  AC-SSM fine-tune failed: %s", e); ac_ssm_state = None

        # Sample test trials
        n_te = len(te_idx)
        idxs = rng.choice(n_te, size=min(args.n_trials, n_te), replace=False)

        for idx in idxs:
            global_idx = te_idx[idx]
            label      = labels[global_idx]
            if label not in _CLASS_TARGETS:
                continue
            target = _CLASS_TARGETS[label]

            # Initial JEPA latent (for oracle/jepa controllers)
            window = eeg[global_idx, -256:, :]
            x      = torch.from_numpy(window.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                out, _ = twin.encoder(x)
                h0 = out[:, -64:, :].mean(1)

            r = _sim_trial(
                h0, eeg[global_idx], target,
                int(y[global_idx]),
                decoder, transition,
                csp_state, ac_ssm_state,
                args.max_steps, args.epsilon, 1/256,
                device=args.device,
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
        vals = [t[ctrl] for t in all_trials if ctrl in t]
        if not vals:
            return {}
        return {
            "ttt":              {"mean": float(np.mean([v["ttt"] for v in vals])),
                                 "std":  float(np.std([v["ttt"] for v in vals]))},
            "final_error":      {"mean": float(np.mean([v["final_error"] for v in vals]))},
            "path_efficiency":  {"mean": float(np.mean([v["path_efficiency"] for v in vals]))},
            "convergence_rate": float(np.mean([v["converged"] for v in vals])),
        }

    ctrls = ["oracle", "csp", "jepa"]
    if any("ac_ssm" in t for t in all_trials):
        ctrls.append("ac_ssm")

    agg = {ctrl: _agg(ctrl) for ctrl in ctrls}

    # Wilcoxon tests
    def _ttt(ctrl):
        return [t[ctrl]["ttt"] for t in all_trials if ctrl in t]

    stats: dict = {"n_trials": len(all_trials)}
    ttt_csp  = _ttt("csp")
    ttt_jepa = _ttt("jepa")
    try:
        _, p = wilcoxon(ttt_csp, ttt_jepa, alternative="less")
        stats["p_csp_vs_jepa_ttt"] = float(p)
    except Exception:
        stats["p_csp_vs_jepa_ttt"] = float("nan")

    if "ac_ssm" in ctrls:
        ttt_ac = _ttt("ac_ssm")
        try:
            _, p = wilcoxon(ttt_ac, ttt_jepa, alternative="less")
            stats["p_ac_vs_jepa_ttt"] = float(p)
        except Exception:
            stats["p_ac_vs_jepa_ttt"] = float("nan")
        try:
            n = min(len(ttt_ac), len(ttt_csp))
            _, p = wilcoxon(ttt_ac[:n], ttt_csp[:n], alternative="two-sided")
            stats["p_ac_vs_csp_ttt"] = float(p)
        except Exception:
            stats["p_ac_vs_csp_ttt"] = float("nan")

    agg["statistics"] = stats

    out_data = {"aggregate": agg, "trial_results": all_trials}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2))
    log.info("Results saved → %s", out)

    # ── Print ─────────────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print(f"  Four-Controller Closed-Loop Sim — {len(all_trials)} trials")
    print(f"  Max steps: {args.max_steps}  ε={args.epsilon}  dt=1/256 s")
    print("═" * 72)
    print(f"  {'Controller':<14} {'TTT mean':>10} {'TTT std':>8} {'Conv.%':>8} {'FinalErr':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*8} {'-'*8} {'-'*10}")
    for ctrl in ctrls:
        a = agg.get(ctrl, {})
        if not a:
            continue
        print(f"  {ctrl:<14} {a['ttt']['mean']:>10.1f} "
              f"{a['ttt']['std']:>8.1f} "
              f"{a['convergence_rate']:>8.1%} "
              f"{a['final_error']['mean']:>10.3f}")
    print()
    print(f"  CSP vs JEPA (TTT, CSP<JEPA): p={stats.get('p_csp_vs_jepa_ttt', float('nan')):.4f}")
    if "ac_ssm" in ctrls:
        print(f"  AC-SSM vs JEPA (TTT):        p={stats.get('p_ac_vs_jepa_ttt', float('nan')):.4f}")
        print(f"  AC-SSM vs CSP  (TTT, 2-sided):p={stats.get('p_ac_vs_csp_ttt', float('nan')):.4f}")
    print("═" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
