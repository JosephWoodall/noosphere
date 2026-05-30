#!/usr/bin/env python3
"""
Fast BCI2a baseline evaluation — CSP+LDA, MDM, EEGNet only.
No JEPA encoder; no torch.compile overhead.

Rationale: The JEPA encoder was pretrained on PhysioNetMI; running it on
BCI2a measures cross-dataset transfer rather than within-dataset performance.
The three conditions here are sufficient to (a) confirm CSP/MDM are consistently
strong classical baselines and (b) establish EEGNet as the competitive modern
neural reference on a standard benchmark dataset.

Usage
-----
  python -m v2_digital_self_replication.cli.eval_bci2a_fast \\
      --subjects 1-9 --folds 5 --eegnet-epochs 50 \\
      --output logs/loso_bci2a_fast.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger("eval_bci2a_fast")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subjects",           type=str, default="1-9")
    p.add_argument("--classes",            type=str, nargs="+",
                   default=["left_hand", "right_hand", "feet", "tongue"])
    p.add_argument("--folds",              type=int, default=5)
    p.add_argument("--eegnet-epochs",      type=int, default=50)
    p.add_argument("--encoder-checkpoint", type=str, default=None,
                   help="BCI2a-pretrained JEPA encoder checkpoint. "
                        "When provided, adds encoder_ft_cls condition.")
    p.add_argument("--ft-epochs-cls",      type=int, default=5)
    p.add_argument("--device",             type=str, default="cpu")
    p.add_argument("--output",             type=str,
                   default="v2_digital_self_replication/logs/loso_bci2a_fast.json")
    p.add_argument("--log-level",          type=str, default="INFO")
    return p.parse_args()


def _parse_subjects(spec):
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in spec.split(",")]


def _bootstrap_ci(values, n_boot=1000, ci=0.95):
    rng = np.random.default_rng(42)
    arr = np.array(values)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return {
        "mean":   float(arr.mean()),
        "ci_lo":  float(np.percentile(boots, (1 - ci) / 2 * 100)),
        "ci_hi":  float(np.percentile(boots, (1 + ci) / 2 * 100)),
    }


def _fit_eval_csp_lda(X_tr, y_tr, X_te, y_te):
    from scipy.linalg import eigh
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler

    classes   = np.unique(y_tr)
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

    def _apply_csp(X, fs):
        return np.concatenate([
            np.log(np.var((W.T @ X), axis=-1)) for W in fs
        ])

    X_tr_f = np.array([_apply_csp(x, filters) for x in X_tr])
    X_te_f = np.array([_apply_csp(x, filters) for x in X_te])

    sc = StandardScaler()
    clf = LinearDiscriminantAnalysis()
    clf.fit(sc.fit_transform(X_tr_f), y_tr)
    return float((clf.predict(sc.transform(X_te_f)) == y_te).mean())


def _fit_eval_mdm(X_tr, y_tr, X_te, y_te):
    from pyriemann.classification import MDM
    from pyriemann.estimation import Covariances
    cov = Covariances(estimator="oas")
    mdm = MDM(metric="riemann")
    mdm.fit(cov.fit_transform(X_tr), y_tr)
    return float((mdm.predict(cov.transform(X_te)) == y_te).mean())


def _fit_eval_eegnet(eeg_tr, y_tr, eeg_te, y_te, n_epochs, device):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import balanced_accuracy_score
    from v2_digital_self_replication.core.eegnet import EEGNet

    T_use = min(eeg_tr.shape[1], 256)
    n_ch  = eeg_tr.shape[2]
    n_cls = len(np.unique(y_tr))

    X_tr  = torch.from_numpy(eeg_tr[:, -T_use:, :].transpose(0, 2, 1).astype(np.float32))
    X_te  = torch.from_numpy(eeg_te[:, -T_use:, :].transpose(0, 2, 1).astype(np.float32))
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    net    = EEGNet(n_classes=n_cls, n_channels=n_ch, T=T_use).to(device)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr_t),
        batch_size=min(32, len(X_tr)),
        shuffle=True, drop_last=False,
    )
    opt  = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    net.train()
    for _ in range(n_epochs):
        for Xb, yb in loader:
            loss = nn.functional.cross_entropy(net(Xb.to(device)), yb.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()

    net.eval()
    with torch.no_grad():
        preds = net(X_te.to(device)).argmax(1).cpu().numpy()
    return float(balanced_accuracy_score(y_te, preds))


def _run_subject(eeg, labels, folds, eegnet_epochs, device,
                 encoder_checkpoint=None, ft_epochs_cls=5):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    le   = LabelEncoder().fit(sorted(set(labels)))
    y    = le.transform(labels)
    skf  = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    X_raw = eeg.transpose(0, 2, 1)

    conds = ["csp_lda", "mdm", "eegnet"]
    if encoder_checkpoint:
        conds.append("encoder_ft_cls")
    results = {k: [] for k in conds}

    for fi, (tr_idx, te_idx) in enumerate(skf.split(eeg, y)):
        log.info("    fold %d/%d", fi + 1, folds)
        eeg_tr, eeg_te = eeg[tr_idx], eeg[te_idx]
        y_tr,   y_te   = y[tr_idx],   y[te_idx]
        X_tr,   X_te   = X_raw[tr_idx], X_raw[te_idx]

        try:
            results["csp_lda"].append(_fit_eval_csp_lda(X_tr, y_tr, X_te, y_te))
        except Exception as e:
            log.warning("CSP failed: %s", e); results["csp_lda"].append(float("nan"))

        try:
            results["mdm"].append(_fit_eval_mdm(X_tr, y_tr, X_te, y_te))
        except Exception as e:
            log.warning("MDM failed: %s", e); results["mdm"].append(float("nan"))

        results["eegnet"].append(
            _fit_eval_eegnet(eeg_tr, y_tr, eeg_te, y_te, eegnet_epochs, device)
        )

        if encoder_checkpoint:
            from v2_digital_self_replication.cli.eval_loso import (
                _load_encoder_from_ckpt, _fit_eval_encoder_cls,
            )
            enc = _load_encoder_from_ckpt(encoder_checkpoint, random_init=False)
            results["encoder_ft_cls"].append(
                _fit_eval_encoder_cls(enc, eeg_tr, y_tr, eeg_te, y_te,
                                      ft_epochs_cls, device)
            )

    summary = {}
    for k, vals in results.items():
        arr = np.array([v for v in vals if not np.isnan(v)])
        summary[k] = {"mean": float(arr.mean()), "std": float(arr.std()),
                      "folds": [float(v) for v in vals]}
    return summary


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    from v2_digital_self_replication.data.bci2a_loader import load_bci2a_trials

    subjects = _parse_subjects(args.subjects)
    log.info("BCI2a fast eval — subjects %s, %d folds, EEGNet %d epochs",
             subjects, args.folds, args.eegnet_epochs)

    all_results, done_subjects = [], []

    for subj in subjects:
        log.info("══ Subject %d ══", subj)
        try:
            eeg, labels = load_bci2a_trials(subjects=[subj], classes=args.classes)
        except Exception as e:
            log.warning("  Load failed: %s", e); continue

        if eeg.shape[0] < args.folds * len(args.classes):
            log.warning("  Too few trials (%d), skipping", eeg.shape[0]); continue

        log.info("  Trials: %d  shape: %s", len(labels), eeg.shape)
        result = _run_subject(eeg, labels, args.folds, args.eegnet_epochs, args.device,
                              encoder_checkpoint=args.encoder_checkpoint,
                              ft_epochs_cls=args.ft_epochs_cls)
        all_results.append(result)
        done_subjects.append(subj)

        log.info("  ↳ S%02d  csp=%.1f%%  mdm=%.1f%%  eegnet=%.1f%%",
                 subj,
                 result["csp_lda"]["mean"]*100,
                 result["mdm"]["mean"]*100,
                 result["eegnet"]["mean"]*100)

    if not all_results:
        log.error("No subjects succeeded"); return 1

    chance = 1.0 / len(args.classes)
    conds = ["csp_lda", "mdm", "eegnet"]
    if args.encoder_checkpoint:
        conds.append("encoder_ft_cls")
    agg = {
        k: _bootstrap_ci([s[k]["mean"] for s in all_results if k in s])
        for k in conds
    }

    out_data = {
        "dataset": "BCI2a (BNCI2014_001)",
        "subjects": done_subjects, "n_subjects": len(done_subjects),
        "classes": args.classes, "folds": args.folds, "chance": chance,
        "subject_results": all_results, "aggregate": agg,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_data, indent=2))
    log.info("Results saved → %s", out)

    print("\n" + "═" * 56)
    print(f"  BCI2a Fast — {len(done_subjects)} subjects, {args.folds}-fold CV")
    print(f"  Classes: {args.classes}  |  Chance: {chance*100:.1f}%")
    print("═" * 56)
    label_map = [("csp_lda","CSP+LDA"), ("mdm","MDM (Riemannian)"),
                 ("eegnet","EEGNet (Lawhern 2018)"),
                 ("encoder_ft_cls","JEPA encoder (BCI2a-pretrained)")]
    for cond, label in label_map:
        a = agg[cond]
        print(f"  {label:<28} {a['mean']*100:5.1f}%  "
              f"[{a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}%]")
    print("═" * 56)
    return 0


if __name__ == "__main__":
    sys.exit(main())
