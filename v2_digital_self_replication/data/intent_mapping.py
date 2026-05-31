"""
Maps PhysionetMI motor-imagery class labels to 6-DOF prosthetic intent vectors.

The mapping is physiologically motivated:
  - left_hand  → shoulder rotates left, elbow flexes (reaching left)
  - right_hand → shoulder rotates right, elbow flexes (reaching right)
  - feet       → forward reach/elevation pattern (arm-swing during gait)
  - hands      → bilateral grip close
  - rest       → all zeros (no movement)

DOF ordering (from config.py):
  0  shoulder_yaw    [-1=left,  +1=right]
  1  shoulder_pitch  [-1=down,  +1=up / forward]
  2  shoulder_roll   [-1=ext,   +1=int]
  3  elbow_flex      [ 0=ext,   +1=flex]
  4  wrist_rotate    [-1=pron,  +1=sup]
  5  grip_aperture   [-1=close, +1=open]
"""

from __future__ import annotations

import numpy as np

# ── Intent vectors ─────────────────────────────────────────────────────────────

CLASS_TO_INTENT: dict[str, np.ndarray] = {
    "left_hand":  np.array([-0.8,  0.0,  0.0,  0.4,  0.0,  0.0], dtype=np.float32),
    "right_hand": np.array([ 0.8,  0.0,  0.0,  0.4,  0.0,  0.0], dtype=np.float32),
    "feet":       np.array([ 0.0,  0.6,  0.0,  0.8,  0.0,  0.0], dtype=np.float32),
    "hands":      np.array([ 0.0,  0.3,  0.0,  0.5,  0.0, -0.8], dtype=np.float32),
    "rest":       np.array([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float32),
}

# Ordered list of classes used in this project
CLASSES = list(CLASS_TO_INTENT.keys())
INTENT_MATRIX = np.stack(list(CLASS_TO_INTENT.values()))   # (n_classes, 6)


def labels_to_intents(labels: list[str]) -> np.ndarray:
    """Convert a list of class-name strings to an (N, 6) intent array."""
    out = np.zeros((len(labels), 6), dtype=np.float32)
    for i, lbl in enumerate(labels):
        if lbl in CLASS_TO_INTENT:
            out[i] = CLASS_TO_INTENT[lbl]
    return out


def predict_class(dof_pred: np.ndarray) -> str:
    """
    Nearest-neighbour classification in DOF space.
    Returns the class whose intent vector is closest to dof_pred.
    """
    dists = np.linalg.norm(INTENT_MATRIX - dof_pred, axis=1)
    return CLASSES[int(np.argmin(dists))]


def predict_classes(dof_preds: np.ndarray) -> list[str]:
    """Vectorised version of predict_class for (N, 6) predictions."""
    dists = np.linalg.norm(
        INTENT_MATRIX[None, :, :] - dof_preds[:, None, :], axis=2
    )   # (N, n_classes)
    return [CLASSES[i] for i in np.argmin(dists, axis=1)]


def accuracy(true_labels: list[str], pred_labels: list[str]) -> float:
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    return correct / len(true_labels) if true_labels else 0.0


def confusion_matrix(
    true_labels: list[str],
    pred_labels: list[str],
    classes: list[str] | None = None,
) -> np.ndarray:
    """Returns an (n_classes, n_classes) confusion matrix."""
    cls = classes or sorted(set(true_labels) | set(pred_labels))
    idx = {c: i for i, c in enumerate(cls)}
    mat = np.zeros((len(cls), len(cls)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1
    return mat
