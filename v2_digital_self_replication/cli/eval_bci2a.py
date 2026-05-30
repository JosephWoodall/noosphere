#!/usr/bin/env python3
"""
LOSO evaluation on BCI Competition IV Dataset 2a (BNCI2014_001).

9 subjects, 4-class MI (left_hand, right_hand, feet, tongue).
Reuses all evaluation functions from eval_loso.py.

Usage
-----
  python -m v2_digital_self_replication.cli.eval_bci2a \\
      --subjects 1-9 \\
      --classes left_hand right_hand feet tongue \\
      --folds 5 --ft-epochs-cls 5 --eegnet-epochs 50 \\
      --checkpoint .../supervised_best.pt \\
      --output .../logs/loso_bci2a.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("eval_bci2a")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=str,
                   default="v2_digital_self_replication/checkpoints/supervised_best.pt")
    p.add_argument("--subjects",     type=str, default="1-9")
    p.add_argument("--classes",      type=str, nargs="+",
                   default=["left_hand", "right_hand", "feet", "tongue"])
    p.add_argument("--folds",        type=int,   default=5)
    p.add_argument("--ft-epochs",    type=int,   default=10)
    p.add_argument("--ft-epochs-cls",type=int,   default=5)
    p.add_argument("--eegnet-epochs",type=int,   default=50)
    p.add_argument("--device",       type=str,   default="cpu")
    p.add_argument("--output",       type=str,
                   default="v2_digital_self_replication/logs/loso_bci2a.json")
    p.add_argument("--log-level",    type=str,   default="INFO")
    return p.parse_args()


def _parse_subjects(spec: str) -> list[int]:
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

    # Import eval harness functions from eval_loso
    from v2_digital_self_replication.cli.eval_loso import (
        _run_subject_cv, _bootstrap_ci,
    )
    from v2_digital_self_replication.data.bci2a_loader import load_bci2a_trials

    subjects = _parse_subjects(args.subjects)
    log.info("BCI2a subjects: %s  Classes: %s", subjects, args.classes)

    all_subject_results = []
    subject_ids_done    = []

    for subj in subjects:
        log.info("══ Subject %d ══", subj)
        try:
            eeg, labels = load_bci2a_trials(
                subjects=[subj],
                classes=args.classes,
            )
        except Exception as e:
            log.warning("Subject %d failed: %s", subj, e)
            continue

        if eeg.shape[0] < args.folds * len(args.classes):
            log.warning("Subject %d: too few trials (%d), skipping",
                        subj, eeg.shape[0])
            continue

        log.info("  Trials: %d  shape: %s", len(labels), eeg.shape)

        # Use 21 channels (BCI2a has 22 EEG; loader already trims to 21)
        result = _run_subject_cv(
            eeg, labels,
            checkpoint=args.checkpoint,
            folds=args.folds,
            ft_epochs=args.ft_epochs,
            device=args.device,
            cns_checkpoint=None,
            fast=True,          # only: encoder_ft_cls, ablation, csp_lda, mdm, planned
            ft_epochs_cls=args.ft_epochs_cls,
            eegnet=True,
            eegnet_epochs=args.eegnet_epochs,
        )
        all_subject_results.append(result)
        subject_ids_done.append(subj)

        enc_mean = result["encoder_ft_cls"]["mean"]
        eeg_mean = result.get("eegnet", {}).get("mean", float("nan"))
        csp_mean = result["csp_lda"]["mean"]
        log.info("  ↳ Subject %d  enc=%.1f%%  eegnet=%.1f%%  csp=%.1f%%",
                 subj, enc_mean*100, eeg_mean*100, csp_mean*100)

    if not all_subject_results:
        log.error("No subjects succeeded")
        return 1

    conditions = [
        "encoder_ft_cls", "encoder_ft_cls_planned",
        "ablation_encoder_cls", "eegnet", "csp_lda", "mdm",
    ]
    agg = {}
    for cond in conditions:
        means = [s[cond]["mean"] for s in all_subject_results
                 if cond in s and not np.isnan(s[cond]["mean"])]
        if means:
            agg[cond] = _bootstrap_ci(means)

    chance = 1.0 / len(args.classes)
    out_data = {
        "dataset":          "BCI2a (BNCI2014_001)",
        "subjects":         subject_ids_done,
        "n_subjects":       len(subject_ids_done),
        "classes":          args.classes,
        "folds":            args.folds,
        "chance":           chance,
        "subject_results":  all_subject_results,
        "aggregate":        agg,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2))
    log.info("Results saved → %s", out)

    labels_map = {
        "encoder_ft_cls":         "JEPA encoder + cls (e2e FT)",
        "encoder_ft_cls_planned": "JEPA + world-model cls",
        "ablation_encoder_cls":   "Random encoder + cls [ablation]",
        "eegnet":                 "EEGNet [modern neural baseline]",
        "csp_lda":                "CSP + LDA",
        "mdm":                    "MDM (Riemannian)",
    }
    print("\n" + "═" * 60)
    print(f"  BCI2a LOSO — {len(subject_ids_done)} subjects, {args.folds}-fold CV")
    print(f"  Classes: {args.classes}  |  Chance: {chance*100:.1f}%")
    print("═" * 60)
    for cond, label in labels_map.items():
        if cond not in agg:
            continue
        a = agg[cond]
        print(f"  {label:<45} {a['mean']*100:5.1f}%  "
              f"[{a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}%]")
    print("═" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
