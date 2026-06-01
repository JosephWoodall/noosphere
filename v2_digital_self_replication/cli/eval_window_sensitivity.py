#!/usr/bin/env python3
"""
Window-alignment sensitivity analysis.

Runs the primary LOSO conditions (CSP+LDA, JEPA encoder+cls, AC-SSM,
random ablation) across three epoch-window settings to confirm that the
~22 pp CSP–neural gap and the AC-SSM convergence result are not artefacts
of a single tmin/tmax choice.

Window settings tested:
  A: tmin=0.5, tmax=2.5  (2 s, early motor preparation)
  B: tmin=0.5, tmax=4.5  (4 s, full trial — primary analysis)
  C: tmin=1.0, tmax=3.5  (2.5 s, post-preparation, offset)

Usage
-----
  python -m v2_digital_self_replication.cli.eval_window_sensitivity \\
      --subjects 1-20 --folds 3 \\
      --checkpoint v2_digital_self_replication/checkpoints/jepa_encoder_best.pt \\
      --output v2_digital_self_replication/logs/window_sensitivity.json

Notes
-----
Window B (primary) results are loaded from loso_results_v5.json to avoid
re-running the full 5-fold eval. The script runs windows A and C with 3
folds each (sufficient to establish consistency without duplicating the
primary eval).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("window_sensitivity")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="v2_digital_self_replication/checkpoints/jepa_encoder_best.pt")
    p.add_argument("--subjects",   type=str, default="1-20")
    p.add_argument("--folds",      type=int, default=3,
                   help="CV folds for non-primary windows (default 3)")
    p.add_argument("--ft-epochs-cls", type=int, default=10)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--primary-results", type=str,
                   default="v2_digital_self_replication/logs/loso_results_v5.json",
                   help="Path to primary (tmin=0.5, tmax=4.5) LOSO results JSON")
    p.add_argument("--output",     type=str,
                   default="v2_digital_self_replication/logs/window_sensitivity.json")
    p.add_argument("--log-level",  type=str, default="INFO")
    return p.parse_args()


def _parse_subjects(spec: str) -> list[int]:
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def _bootstrap_ci(values: list[float], n_boot: int = 1000) -> dict:
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boots = [rng.choice(arr, len(arr), replace=True).mean() for _ in range(n_boot)]
    return {
        "mean":   float(arr.mean()),
        "ci_lo":  float(np.percentile(boots, 2.5)),
        "ci_hi":  float(np.percentile(boots, 97.5)),
    }


def _run_window(tmin: float, tmax: float, subjects: list[int],
                folds: int, ft_epochs: int, checkpoint: str,
                device: str) -> dict[str, dict]:
    """
    Run minimal LOSO for one window setting.
    Returns aggregate {condition: {mean, ci_lo, ci_hi}} dict.
    """
    import torch
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    from v2_digital_self_replication.data.real_eeg import load_trials
    from v2_digital_self_replication.cli.eval_loso import (
        _load_encoder_from_ckpt,
        _fit_eval_encoder_cls,
        _fit_eval_ac_ssm,
        _fit_eval_csp_lda,
    )

    conditions = ["csp_lda", "encoder_ft_cls", "ac_ssm", "ablation_encoder_cls"]
    subject_means: dict[str, list[float]] = {c: [] for c in conditions}

    for subj in subjects:
        try:
            eeg, labels = load_trials(subjects=[subj],
                                      classes=["left_hand", "right_hand", "feet"],
                                      tmin=tmin, tmax=tmax)
        except Exception as e:
            log.warning("Subject %d load failed: %s", subj, e)
            continue

        if eeg.shape[0] < folds * 3:
            log.warning("Subject %d: too few trials (%d)", subj, eeg.shape[0])
            continue

        le = LabelEncoder().fit(sorted(set(labels)))
        y  = le.transform(labels)
        X_raw = eeg.transpose(0, 2, 1)

        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        fold_acc: dict[str, list[float]] = {c: [] for c in conditions}

        for tr_idx, te_idx in skf.split(eeg, y):
            eeg_tr, eeg_te = eeg[tr_idx], eeg[te_idx]
            y_tr,   y_te   = y[tr_idx],   y[te_idx]
            X_tr,   X_te   = X_raw[tr_idx], X_raw[te_idx]

            # CSP
            try:
                acc = _fit_eval_csp_lda(X_tr, y_tr, X_te, y_te)
            except Exception:
                acc = float("nan")
            fold_acc["csp_lda"].append(acc)

            # JEPA encoder + cls
            enc = _load_encoder_from_ckpt(checkpoint, random_init=False)
            fold_acc["encoder_ft_cls"].append(
                _fit_eval_encoder_cls(enc, eeg_tr, y_tr, eeg_te, y_te,
                                      ft_epochs, device)
            )

            # AC-SSM
            sorted_cls = le.classes_.tolist()
            fold_acc["ac_ssm"].append(
                _fit_eval_ac_ssm(checkpoint, eeg_tr, y_tr, eeg_te, y_te,
                                 ft_epochs, device, class_names=sorted_cls)
            )

            # Ablation
            enc_rnd = _load_encoder_from_ckpt(None, random_init=True)
            fold_acc["ablation_encoder_cls"].append(
                _fit_eval_encoder_cls(enc_rnd, eeg_tr, y_tr, eeg_te, y_te,
                                      ft_epochs, device)
            )

        for cond in conditions:
            vals = [v for v in fold_acc[cond] if not np.isnan(v)]
            if vals:
                subject_means[cond].append(float(np.mean(vals)))

        log.info("  Subject %d done (tmin=%.1f, tmax=%.1f)", subj, tmin, tmax)

    agg = {}
    for cond in conditions:
        if subject_means[cond]:
            agg[cond] = _bootstrap_ci(subject_means[cond])
    return agg


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    subjects = _parse_subjects(args.subjects)

    # ── Window B (primary): load from existing results if available ───────────
    windows = {
        "A_0.5-2.5": {"tmin": 0.5, "tmax": 2.5, "label": "2s (0.5–2.5)"},
        "B_0.5-4.5": {"tmin": 0.5, "tmax": 4.5, "label": "4s (0.5–4.5) [primary]"},
        "C_1.0-3.5": {"tmin": 1.0, "tmax": 3.5, "label": "2.5s (1.0–3.5)"},
    }

    results = {}

    # Load primary from v5 results if available
    primary_path = Path(args.primary_results)
    if primary_path.exists():
        log.info("Loading primary window results from %s", primary_path)
        v5 = json.loads(primary_path.read_text())
        agg5 = v5.get("aggregate", {})
        results["B_0.5-4.5"] = {
            cond: {
                "mean":  agg5[cond]["mean"],
                "ci_lo": agg5[cond]["ci_lo"],
                "ci_hi": agg5[cond]["ci_hi"],
            }
            for cond in ["csp_lda", "encoder_ft_cls", "ac_ssm", "ablation_encoder_cls"]
            if cond in agg5
        }
        windows.pop("B_0.5-4.5")  # skip running it again

    # ── Run non-primary windows ───────────────────────────────────────────────
    for key, cfg in windows.items():
        log.info("Running window %s (%s)", key, cfg["label"])
        results[key] = _run_window(
            tmin=cfg["tmin"], tmax=cfg["tmax"],
            subjects=subjects, folds=args.folds,
            ft_epochs=args.ft_epochs_cls,
            checkpoint=args.checkpoint,
            device=args.device,
        )

    # ── Save & print ──────────────────────────────────────────────────────────
    out = {"window_configs": {
               k: {"tmin": v["tmin"], "tmax": v["tmax"], "label": v["label"]}
               for k, v in {"A_0.5-2.5": {"tmin":0.5,"tmax":2.5,"label":"2s"},
                             "B_0.5-4.5": {"tmin":0.5,"tmax":4.5,"label":"4s [primary]"},
                             "C_1.0-3.5": {"tmin":1.0,"tmax":3.5,"label":"2.5s offset"}}.items()
           },
           "results": results}

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))
    log.info("Saved → %s", args.output)

    print("\n" + "=" * 72)
    print("  Window-Alignment Sensitivity (3-class MI, PhysioNetMI)")
    print("=" * 72)
    cond_labels = {
        "csp_lda":              "CSP+LDA",
        "encoder_ft_cls":       "JEPA enc+cls",
        "ac_ssm":               "AC-SSM",
        "ablation_encoder_cls": "Random ablation",
    }
    window_order = ["A_0.5-2.5", "B_0.5-4.5", "C_1.0-3.5"]
    print(f"  {'Condition':<22}", end="")
    for wk in window_order:
        lbl = {"A_0.5-2.5":"2s(0.5-2.5)", "B_0.5-4.5":"4s(0.5-4.5)*",
               "C_1.0-3.5":"2.5s(1.0-3.5)"}[wk]
        print(f"  {lbl:>14}", end="")
    print()
    print("  " + "-" * 68)
    for cond, label in cond_labels.items():
        print(f"  {label:<22}", end="")
        for wk in window_order:
            a = results.get(wk, {}).get(cond, {})
            if a:
                print(f"  {a['mean']*100:>6.1f} [{a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}]", end="")
            else:
                print(f"  {'—':>14}", end="")
        print()
    print("  * primary analysis window")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
