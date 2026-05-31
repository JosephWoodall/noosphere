#!/usr/bin/env python3
"""
Evaluate the digital twin's intent decoding on real PhysionetMI EEG.

Three metrics are reported:

  1. Zero-shot accuracy
     Run real EEG through the untouched twin (trained only on synthetic data).
     Classify each trial by nearest intent vector in DOF space.
     Baseline: chance = 1/n_classes.

  2. Linear-probe accuracy
     Train a logistic regression on the encoder latents (labels only, no DOF
     decoder).  Tests whether the encoder captures class-relevant features
     independent of decoder alignment.

  3. Fine-tuned accuracy  (optional, --finetune)
     Briefly fine-tune the twin on a split of the real data, then re-evaluate.

Usage:
    python -m v2_digital_self_replication.cli.eval_real_eeg \\
        --checkpoint v2_digital_self_replication/checkpoints/supervised_best.pt \\
        --subjects 1 2 3 \\
        --output   v2_digital_self_replication/logs/real_eeg_eval.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate intent decoding on real PhysionetMI EEG."
    )
    p.add_argument("--checkpoint", type=str,
                   default="v2_digital_self_replication/checkpoints/supervised_best.pt")
    p.add_argument("--subjects",   type=int, nargs="+", default=[1],
                   help="PhysionetMI subject IDs (default: 1)")
    p.add_argument("--classes",    type=str, nargs="+",
                   default=["left_hand", "right_hand", "feet"],
                   help="Which motor imagery classes to evaluate (default excludes rest)")
    p.add_argument("--tmin",       type=float, default=0.5,
                   help="Epoch start (s) relative to cue")
    p.add_argument("--tmax",       type=float, default=4.5,
                   help="Epoch end (s) relative to cue")
    p.add_argument("--finetune",   action="store_true",
                   help="Fine-tune on 80%% of data and evaluate on 20%%")
    p.add_argument("--ft-epochs",  type=int,   default=3)
    p.add_argument("--output",     type=str,
                   default="v2_digital_self_replication/logs/real_eeg_eval.json")
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--log-level",  type=str, default="INFO")
    return p.parse_args()


def _linear_probe_accuracy(
    latents: np.ndarray,
    labels: list[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        latents, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=500, C=1.0, random_state=seed)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)

    return {
        "accuracy":  float(acc),
        "n_train":   int(len(y_tr)),
        "n_test":    int(len(y_te)),
        "classes":   list(le.classes_),
    }


def _print_confusion(mat: np.ndarray, classes: list[str], log):
    header = "  " + "  ".join(f"{c[:6]:>6}" for c in classes)
    log(header)
    for i, row in enumerate(mat):
        log("  " + f"{classes[i][:6]:>6}  " + "  ".join(f"{v:>6}" for v in row))


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("eval_real_eeg")

    from v2_digital_self_replication.agent.digital_twin import DigitalTwin
    from v2_digital_self_replication.data.real_eeg import encode_trials, load_trials
    from v2_digital_self_replication.data.intent_mapping import (
        CLASS_TO_INTENT,
        CLASSES,
        accuracy,
        confusion_matrix,
        labels_to_intents,
        predict_classes,
    )

    # ── Load twin ─────────────────────────────────────────────────────────────
    twin = DigitalTwin()
    ckpt = Path(args.checkpoint)
    if ckpt.exists():
        twin.load(str(ckpt))
        log.info("Loaded checkpoint: %s", ckpt)
    else:
        log.warning("No checkpoint found — evaluating randomly initialised twin")

    if args.device != "cpu":
        twin = twin.to(args.device)

    # ── Load real EEG ─────────────────────────────────────────────────────────
    log.info("Loading PhysionetMI  subjects=%s  classes=%s", args.subjects, args.classes)
    eeg, labels = load_trials(
        subjects=args.classes and args.subjects,
        tmin=args.tmin,
        tmax=args.tmax,
        classes=args.classes,
        verbose=False,
    )
    log.info("Loaded %d trials  shape=%s", len(labels), eeg.shape)

    n_classes = len(set(labels))
    chance    = 1.0 / n_classes
    results   = {
        "subjects":       args.subjects,
        "classes":        args.classes,
        "n_trials":       len(labels),
        "n_classes":      n_classes,
        "chance_accuracy": round(chance, 4),
    }

    # ── 1. Zero-shot evaluation ───────────────────────────────────────────────
    log.info("─── Zero-shot evaluation ───")
    latents_all, dof_all = encode_trials(twin, eeg)

    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import LabelEncoder

    zs_preds    = predict_classes(dof_all)
    zs_acc      = accuracy(labels, zs_preds)
    # Fit encoder on union of true and predicted labels so unseen predictions
    # (e.g. "rest" predicted when evaluating active classes only) are handled
    all_label_classes = sorted(set(labels) | set(zs_preds))
    le_tmp      = LabelEncoder().fit(all_label_classes)
    zs_bal_acc  = balanced_accuracy_score(
        le_tmp.transform(labels), le_tmp.transform(zs_preds)
    )

    active_classes = sorted(set(labels))
    cm = confusion_matrix(labels, zs_preds, classes=active_classes)

    log.info("Zero-shot accuracy: %.1f%%  balanced: %.1f%%  (chance %.1f%%)",
             zs_acc * 100, zs_bal_acc * 100, chance * 100)
    _print_confusion(cm, active_classes, log.info)

    results["zero_shot"] = {
        "accuracy":          round(zs_acc, 4),
        "balanced_accuracy": round(zs_bal_acc, 4),
        "confusion_matrix":  cm.tolist(),
        "classes":           active_classes,
    }

    # ── 2. Linear-probe evaluation ────────────────────────────────────────────
    log.info("─── Linear probe (encoder latents) ───")
    try:
        probe = _linear_probe_accuracy(latents_all, labels)
        log.info("Linear-probe accuracy: %.1f%%  (chance %.1f%%)",
                 probe["accuracy"] * 100, chance * 100)
        results["linear_probe"] = probe
    except Exception as e:
        log.warning("Linear probe failed: %s", e)
        results["linear_probe"] = {"error": str(e)}

    # ── 3. Fine-tuned evaluation (optional) ───────────────────────────────────
    if args.finetune:
        log.info("─── Fine-tuning on 80%% of data ───")
        from v2_digital_self_replication.training.online_train import SupervisedTrainer

        n         = len(labels)
        split     = int(n * 0.8)
        idx       = np.random.permutation(n)
        tr_idx    = idx[:split]
        te_idx    = idx[split:]

        eeg_tr   = eeg[tr_idx]
        cmd_tr   = labels_to_intents([labels[i] for i in tr_idx])
        eeg_te   = eeg[te_idx]
        labels_te = [labels[i] for i in te_idx]

        T = eeg_tr.shape[1]
        W = twin.cfg.encoder.d_state
        dataset = {
            0: {
                "eeg":        eeg_tr.reshape(1, -1, 21),
                "commands":   cmd_tr.reshape(1, -1, 6),
                "ern_labels": np.zeros((1, len(tr_idx) * T), dtype=np.float32),
            }
        }
        trainer = SupervisedTrainer(twin, device=args.device)
        trainer.train(dataset, n_epochs=args.ft_epochs, window_len=W,
                      batch_size=min(32, T // W))
        log.info("Fine-tuning complete")

        _, dof_te = encode_trials(twin, eeg_te)
        ft_preds  = predict_classes(dof_te)
        ft_acc    = accuracy(labels_te, ft_preds)
        log.info("Fine-tuned accuracy: %.1f%%  (was %.1f%% zero-shot)",
                 ft_acc * 100, zs_acc * 100)
        results["fine_tuned"] = {"accuracy": round(ft_acc, 4)}

    # ── Save results ──────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("Results saved → %s", out)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  Real-EEG Intent Decoding Evaluation")
    print("=" * 52)
    print(f"  Subjects:      {args.subjects}")
    print(f"  Trials:        {len(labels)}  ({n_classes} classes)")
    print(f"  Chance:        {chance*100:.1f}%")
    print(f"  Zero-shot:     {zs_acc*100:.1f}%")
    if "linear_probe" in results and "accuracy" in results["linear_probe"]:
        print(f"  Linear probe:  {results['linear_probe']['accuracy']*100:.1f}%")
    if "fine_tuned" in results:
        print(f"  Fine-tuned:    {results['fine_tuned']['accuracy']*100:.1f}%")
    print("=" * 52)

    return 0


if __name__ == "__main__":
    sys.exit(main())
