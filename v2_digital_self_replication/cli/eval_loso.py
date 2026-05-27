#!/usr/bin/env python3
"""
Leave-one-subject-out evaluation for JBHI submission.

For each of N subjects (default 20):
  - Load PhysionetMI trials (left_hand, right_hand, feet, 3 classes)
  - Run 5-fold stratified cross-validation
  - Fine-tune from checkpoint on each training fold
  - Evaluate on held-out test fold

Conditions evaluated in every fold:
  A. Our system  — JEPA-pretrained twin, fine-tuned on training fold
  B. Ablation    — randomly-initialised twin, fine-tuned on training fold
  C. Zero-shot   — JEPA encoder latents + logistic regression (no DOF decoder)
  D. CSP + LDA   — classic BCI baseline (pyriemann)
  E. MDM         — Riemannian minimum-distance-to-mean (pyriemann)

Outputs
-------
  logs/loso_results.json   — full per-subject, per-fold results
  (summary also printed to stdout)

Usage
-----
  python -m v2_digital_self_replication.cli.eval_loso \\
      --subjects 1-20 --folds 5 --ft-epochs 10 \\
      --checkpoint v2_digital_self_replication/checkpoints/supervised_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


log = logging.getLogger("eval_loso")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="v2_digital_self_replication/checkpoints/supervised_best.pt")
    p.add_argument("--subjects",   type=str, default="1-20",
                   help="Range '1-20' or comma list '1,3,5'")
    p.add_argument("--classes",    type=str, nargs="+",
                   default=["left_hand", "right_hand", "feet"])
    p.add_argument("--tmin",       type=float, default=0.5)
    p.add_argument("--tmax",       type=float, default=4.5)
    p.add_argument("--folds",      type=int,   default=5)
    p.add_argument("--ft-epochs",  type=int,   default=10)
    p.add_argument("--device",     type=str,   default="cpu")
    p.add_argument("--output",     type=str,
                   default="v2_digital_self_replication/logs/loso_results.json")
    p.add_argument("--log-level",  type=str,   default="INFO")
    return p.parse_args()


def _parse_subjects(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


# ── Baselines ─────────────────────────────────────────────────────────────────

def _fit_eval_csp_lda(X_tr, y_tr, X_te, y_te):
    """
    One-vs-rest CSP (4 filters per class) + LDA.
    X: (n, 21, T), y: (n,) int.

    Uses scipy's generalized eigenvalue solver directly to sidestep
    pyriemann's AJD which has numpy-2.x incompatibilities.
    """
    from scipy.linalg import eigh
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    n_channels = X_tr.shape[1]
    classes     = np.unique(y_tr)
    n_filters   = 4

    def _cov(X):
        C = X @ X.transpose(0, 2, 1)
        return C.mean(0) / X.shape[2]

    # One-vs-rest CSP filters for each class
    filters = []
    for cls in classes:
        mask = y_tr == cls
        C_cls  = _cov(X_tr[mask])
        C_rest = _cov(X_tr[~mask])
        C_tot  = _cov(X_tr)
        # Generalised eigendecomposition: C_cls W = λ C_tot W
        _, W = eigh(C_cls, C_tot)
        # Keep first and last n_filters/2 eigenvectors
        half = n_filters // 2
        W_sel = np.concatenate([W[:, :half], W[:, -half:]], axis=1)
        filters.append(W_sel)

    def _apply_csp(X, filters_list):
        feats = []
        for W in filters_list:
            proj   = (W.T @ X)                  # (n_filters, T)
            logvar = np.log(np.var(proj, axis=-1))  # (n_filters,)
            feats.append(logvar)
        return np.concatenate(feats, axis=0)    # (n_classes*n_filters,)

    X_tr_feats = np.array([_apply_csp(x, filters) for x in X_tr])
    X_te_feats = np.array([_apply_csp(x, filters) for x in X_te])

    scaler = StandardScaler()
    X_tr_feats = scaler.fit_transform(X_tr_feats)
    X_te_feats = scaler.transform(X_te_feats)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_tr_feats, y_tr)
    return float((clf.predict(X_te_feats) == y_te).mean())


def _fit_eval_mdm(X_tr, y_tr, X_te, y_te):
    """Riemannian MDM.  X: (n, 21, T), y: (n,) int."""
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances

    cov  = Covariances(estimator="oas")
    mdm  = MDM(metric="riemann")

    cov_tr = cov.fit_transform(X_tr)
    cov_te = cov.transform(X_te)
    mdm.fit(cov_tr, y_tr)
    return float((mdm.predict(cov_te) == y_te).mean())


# ── Our system ────────────────────────────────────────────────────────────────

def _build_twin(checkpoint: str | None, device: str, random_init: bool = False):
    from v2_digital_self_replication.agent.digital_twin import DigitalTwin
    twin = DigitalTwin()
    if not random_init and checkpoint and Path(checkpoint).exists():
        twin.load(checkpoint)
    if device != "cpu":
        twin = twin.to(device)
    return twin


def _finetune_twin(twin, eeg_tr, labels_tr, ft_epochs: int, device: str):
    """Fine-tune twin on (n_tr, T, 21) EEG + class labels.

    Uses a per-call temp directory so the trainer's checkpoint saves never
    touch the golden supervised_best.pt used by subsequent _build_twin calls.
    """
    import tempfile
    from v2_digital_self_replication.data.intent_mapping import labels_to_intents
    from v2_digital_self_replication.training.online_train import SupervisedTrainer

    cmd_tr = labels_to_intents(labels_tr)          # (n_tr, 6)
    N, T, _ = eeg_tr.shape
    W = 256                                        # 1 s window at 256 Hz

    # Per-trial dict so _build_supervised_dataset windows each trial
    # independently (no label leakage across trial boundaries).
    cmd_bc = np.broadcast_to(cmd_tr[:, None, :], (N, T, 6)).copy()
    dataset = {
        i: {
            "eeg":        eeg_tr[i : i + 1],    # (1, T, 21)
            "commands":   cmd_bc[i : i + 1],    # (1, T, 6)
            "ern_labels": np.zeros((1, T), dtype=np.float32),
        }
        for i in range(N)
    }
    n_wins = (N * T) // W
    if n_wins < 1:
        return twin

    # Redirect trainer saves to a throwaway temp dir — never overwrites the
    # golden checkpoint that other folds need to load.
    with tempfile.TemporaryDirectory() as tmp_dir:
        twin.cfg.checkpoint_dir = tmp_dir
        trainer = SupervisedTrainer(twin, device=device)
        trainer.train(dataset, n_epochs=ft_epochs, window_len=W,
                      batch_size=min(32, max(1, n_wins)))
    return twin


def _encode_and_predict(twin, eeg, labels, batch_size: int = 32):
    """
    Batch-encode eeg (n, T, 21) → latents, dof_pred.
    Returns (balanced_acc, latents, dof_pred, sigma_mean).
    """
    import torch
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import LabelEncoder

    from v2_digital_self_replication.data.intent_mapping import predict_classes

    encoder = twin.encoder
    decoder = twin.decoder
    device  = next(encoder.parameters()).device

    encoder.eval()
    decoder.eval()

    n = eeg.shape[0]
    all_lats, all_dof, all_sigma = [], [], []

    # Truncate to 256 samples (= 1 s at 256 Hz).  This matches the training
    # window length, so it reuses the already-compiled JIT graph and avoids
    # a second ~100 s recompilation for T=1025 (full trial length).
    eeg = eeg[:, -256:, :]

    with torch.no_grad():
        for i in range(0, n, batch_size):
            chunk   = torch.from_numpy(eeg[i : i + batch_size]).to(device)
            out, _  = encoder(chunk)                     # (b, 256, d_model)
            lat     = out[:, -64:, :].mean(1)            # (b, d_model)
            intent  = decoder(lat)
            all_lats.append(lat.cpu())
            all_dof.append(intent.mu.cpu())
            all_sigma.append(intent.sigma.cpu().mean(1)) # (b,)

    latents  = torch.cat(all_lats).numpy()   # (n, d_model)
    dof_pred = torch.cat(all_dof).numpy()    # (n, 6)
    sigma    = torch.cat(all_sigma).numpy()  # (n,)

    preds = predict_classes(dof_pred)
    all_classes = sorted(set(labels) | set(preds))
    le    = LabelEncoder().fit(all_classes)
    bal_acc = balanced_accuracy_score(
        le.transform(labels), le.transform(preds)
    )
    return float(bal_acc), latents, dof_pred, sigma


def _zero_shot_probe(latents_tr, y_tr, latents_te, y_te):
    """Linear probe on encoder latents (no DOF decoder)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(latents_tr)
    X_te_s = scaler.transform(latents_te)
    clf    = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    clf.fit(X_tr_s, y_tr)
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_te, clf.predict(X_te_s)))


def _calibration_corr(sigma: np.ndarray, dof_pred: np.ndarray,
                      dof_true: np.ndarray) -> float:
    """
    Pearson r between mean sigma and mean L2 error.
    Positive r means higher uncertainty → larger error (well-calibrated direction).
    """
    from scipy.stats import pearsonr
    errors = np.linalg.norm(dof_pred - dof_true, axis=1)
    r, _   = pearsonr(sigma, errors)
    return float(r)


# ── CV loop for one subject ───────────────────────────────────────────────────

def _run_subject_cv(
    eeg: np.ndarray,          # (n_trials, T, 21)
    labels: list[str],
    checkpoint: str,
    folds: int,
    ft_epochs: int,
    device: str,
) -> dict:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    from v2_digital_self_replication.data.intent_mapping import labels_to_intents

    le      = LabelEncoder().fit(sorted(set(labels)))
    y       = le.transform(labels)
    skf     = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    X_raw   = eeg.transpose(0, 2, 1)    # (n, 21, T) for pyriemann

    fold_results = {k: [] for k in
                    ["jepa_ft", "ablation_ft", "zero_shot_probe",
                     "csp_lda", "mdm", "calib_corr"]}

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(eeg, y)):
        log.info("    fold %d/%d", fold_idx + 1, folds)

        eeg_tr, eeg_te     = eeg[tr_idx], eeg[te_idx]
        labels_tr, labels_te = [labels[i] for i in tr_idx], [labels[i] for i in te_idx]
        y_tr, y_te         = y[tr_idx], y[te_idx]
        X_tr, X_te         = X_raw[tr_idx], X_raw[te_idx]

        # ── A. Our system: JEPA-pretrained + fine-tuned ──────────────────────
        twin_ft = _build_twin(checkpoint, device, random_init=False)
        twin_ft = _finetune_twin(twin_ft, eeg_tr, labels_tr, ft_epochs, device)
        acc_ft, lats_te, dof_te, sigma_te = _encode_and_predict(twin_ft, eeg_te, labels_te)
        fold_results["jepa_ft"].append(acc_ft)

        # ── B. Ablation: random init + fine-tuned ────────────────────────────
        twin_rnd = _build_twin(None, device, random_init=True)
        twin_rnd = _finetune_twin(twin_rnd, eeg_tr, labels_tr, ft_epochs, device)
        acc_rnd, _, _, _ = _encode_and_predict(twin_rnd, eeg_te, labels_te)
        fold_results["ablation_ft"].append(acc_rnd)

        # ── C. Zero-shot: JEPA encoder → linear probe ────────────────────────
        twin_zs = _build_twin(checkpoint, device, random_init=False)
        _, lats_tr, _, _ = _encode_and_predict(twin_zs, eeg_tr, labels_tr)
        acc_zs = _zero_shot_probe(
            lats_tr, le.transform(labels_tr),
            lats_te, le.transform(labels_te),
        )
        fold_results["zero_shot_probe"].append(acc_zs)

        # ── D. CSP + LDA ─────────────────────────────────────────────────────
        try:
            acc_csp = _fit_eval_csp_lda(X_tr, y_tr, X_te, y_te)
        except Exception as e:
            log.warning("CSP+LDA failed: %s", e)
            acc_csp = float("nan")
        fold_results["csp_lda"].append(acc_csp)

        # ── E. MDM ───────────────────────────────────────────────────────────
        try:
            acc_mdm = _fit_eval_mdm(X_tr, y_tr, X_te, y_te)
        except Exception as e:
            log.warning("MDM failed: %s", e)
            acc_mdm = float("nan")
        fold_results["mdm"].append(acc_mdm)

        # ── Calibration ──────────────────────────────────────────────────────
        from v2_digital_self_replication.data.intent_mapping import labels_to_intents
        dof_true_te = labels_to_intents(labels_te)
        corr = _calibration_corr(sigma_te, dof_te, dof_true_te)
        fold_results["calib_corr"].append(corr)

    # Aggregate across folds
    summary = {}
    for key, vals in fold_results.items():
        arr = np.array([v for v in vals if not np.isnan(v)])
        summary[key] = {
            "mean": float(np.mean(arr)) if len(arr) else float("nan"),
            "std":  float(np.std(arr))  if len(arr) else float("nan"),
            "folds": [float(v) for v in vals],
        }
    return summary


# ── Statistics ────────────────────────────────────────────────────────────────

def _wilcoxon_vs_baselines(subject_results: list[dict]) -> dict:
    from scipy.stats import wilcoxon

    baselines = ["ablation_ft", "zero_shot_probe", "csp_lda", "mdm"]
    stats_out = {}
    ref = np.array([s["jepa_ft"]["mean"] for s in subject_results])

    for bl in baselines:
        comp = np.array([s[bl]["mean"] for s in subject_results
                         if not np.isnan(s[bl]["mean"])])
        ref2 = ref[:len(comp)]
        try:
            stat, p = wilcoxon(ref2, comp, alternative="greater")
        except Exception:
            stat, p = float("nan"), float("nan")
        stats_out[f"jepa_ft_vs_{bl}"] = {
            "statistic": float(stat),
            "p_value":   float(p),
            "n":         int(len(comp)),
        }
    return stats_out


def _bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95) -> dict:
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    lo = float(np.percentile(boots, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boots, (1 + ci) / 2 * 100))
    return {"mean": float(arr.mean()), "ci_lo": lo, "ci_hi": hi,
            "ci_level": ci}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from v2_digital_self_replication.data.real_eeg import load_trials

    subjects = _parse_subjects(args.subjects)
    log.info("Subjects: %s  Folds: %d  FT epochs: %d",
             subjects, args.folds, args.ft_epochs)
    log.info("Classes: %s", args.classes)

    all_subject_results = []
    subject_ids_done    = []

    for subj in subjects:
        log.info("══ Subject %d ══", subj)
        try:
            eeg, labels = load_trials(
                subjects=[subj],
                classes=args.classes,
                tmin=args.tmin, tmax=args.tmax,
            )
        except Exception as e:
            log.warning("Subject %d failed to load: %s", subj, e)
            continue

        if eeg.shape[0] < args.folds * 3:
            log.warning("Subject %d: too few trials (%d), skipping",
                        subj, eeg.shape[0])
            continue

        log.info("  Trials: %d  shape: %s", len(labels), eeg.shape)
        result = _run_subject_cv(
            eeg, labels,
            checkpoint=args.checkpoint,
            folds=args.folds,
            ft_epochs=args.ft_epochs,
            device=args.device,
        )
        all_subject_results.append(result)
        subject_ids_done.append(subj)

        # Live progress summary
        jepa_means = [s["jepa_ft"]["mean"] for s in all_subject_results]
        log.info("  ↳ Subject %d jepa_ft=%.1f%%  (running mean: %.1f±%.1f%%)",
                 subj,
                 result["jepa_ft"]["mean"] * 100,
                 np.mean(jepa_means) * 100,
                 np.std(jepa_means) * 100)

    if not all_subject_results:
        log.error("No subjects succeeded")
        return 1

    # ── Aggregate across subjects ─────────────────────────────────────────────
    conditions = ["jepa_ft", "ablation_ft", "zero_shot_probe", "csp_lda", "mdm"]
    agg = {}
    for cond in conditions:
        means = [s[cond]["mean"] for s in all_subject_results
                 if not np.isnan(s[cond]["mean"])]
        agg[cond] = _bootstrap_ci(means)

    calib_corrs = [s["calib_corr"]["mean"] for s in all_subject_results]
    agg["calib_corr"] = _bootstrap_ci(calib_corrs)

    stats = _wilcoxon_vs_baselines(all_subject_results)
    chance = 1.0 / len(args.classes)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_data = {
        "subjects":         subject_ids_done,
        "n_subjects":       len(subject_ids_done),
        "classes":          args.classes,
        "folds":            args.folds,
        "ft_epochs":        args.ft_epochs,
        "chance":           chance,
        "subject_results":  all_subject_results,
        "aggregate":        agg,
        "statistics":       stats,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2))
    log.info("Results saved → %s", out)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(f"  LOSO Evaluation — {len(subject_ids_done)} subjects, {args.folds}-fold CV")
    print(f"  Classes: {args.classes}  |  Chance: {chance*100:.1f}%")
    print("═" * 60)
    labels_map = {
        "jepa_ft":          "Our system (JEPA + fine-tune)",
        "ablation_ft":      "Ablation  (random init + fine-tune)",
        "zero_shot_probe":  "Zero-shot  (JEPA encoder + log. reg.)",
        "csp_lda":          "CSP + LDA",
        "mdm":              "MDM (Riemannian)",
    }
    for cond, label in labels_map.items():
        a = agg[cond]
        p_str = ""
        key = f"jepa_ft_vs_{cond}"
        if key in stats:
            p = stats[key]["p_value"]
            p_str = f"  (p={p:.3f} vs. JEPA)" if not np.isnan(p) else ""
        print(f"  {label:<40} {a['mean']*100:5.1f}%  "
              f"[{a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}]{p_str}")

    print(f"  {'Calibration (σ–error correlation)':<40} "
          f"r={agg['calib_corr']['mean']:+.3f}  "
          f"[{agg['calib_corr']['ci_lo']:+.3f}–{agg['calib_corr']['ci_hi']:+.3f}]")
    print("═" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
