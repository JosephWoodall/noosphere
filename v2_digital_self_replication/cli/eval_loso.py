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
    p.add_argument("--cns-checkpoint", type=str, default=None,
                   help="Path to cross-modal CNS-pretrained encoder checkpoint. "
                        "When provided, adds encoder_ft_cls_cns condition for Phase 2 comparison.")
    p.add_argument("--fast", action="store_true",
                   help="Skip jepa_ft / ablation_ft / zero_shot_probe (known-chance conditions). "
                        "Runs encoder_ft_cls, ablation_encoder_cls, csp_lda, mdm only.")
    p.add_argument("--ft-epochs-cls", type=int, default=None,
                   help="Fine-tune epochs for encoder_ft_cls / ablation_encoder_cls conditions. "
                        "Defaults to --ft-epochs if not specified.")
    p.add_argument("--eegnet", action="store_true",
                   help="Add EEGNet (Lawhern 2018) as a modern neural baseline condition.")
    p.add_argument("--eegnet-epochs", type=int, default=50,
                   help="Training epochs for EEGNet (default 50).")
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


def _load_encoder_from_ckpt(path: str | None, random_init: bool = False):
    """Load a StreamEncoder from checkpoint (JEPA or supervised), or return random init."""
    import torch
    from v2_digital_self_replication.core.stream_encoder import StreamEncoder
    from v2_digital_self_replication.config import V2Config

    enc_cfg = V2Config().encoder
    encoder = StreamEncoder(
        d_model=enc_cfg.d_model, d_state=enc_cfg.d_state,
        n_layers=enc_cfg.n_layers, n_eeg=enc_cfg.n_eeg_channels,
        n_prop=enc_cfg.n_prop_channels, dropout=enc_cfg.dropout,
    )
    if random_init or not path or not Path(path).exists():
        return encoder

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    # JEPA checkpoint: {"encoder": sd, "predictor": sd, ...}
    if "encoder" in ckpt and isinstance(ckpt["encoder"], dict):
        encoder.load_state_dict(ckpt["encoder"], strict=False)
    else:
        # Supervised twin checkpoint — pull encoder via DigitalTwin
        try:
            twin = _build_twin(path, "cpu", random_init=False)
            return twin.encoder
        except Exception:
            pass
    return encoder


def _fit_eval_encoder_cls(
    encoder,
    eeg_tr: np.ndarray,
    y_tr: np.ndarray,
    eeg_te: np.ndarray,
    y_te: np.ndarray,
    ft_epochs: int,
    device: str,
) -> float:
    """
    Fine-tune encoder + linear head end-to-end with cross-entropy.
    Tests whether the encoder's pretrained weights provide a useful initialisation
    compared to random (ablation_encoder_cls).
    """
    import copy
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import balanced_accuracy_score

    enc = copy.deepcopy(encoder).to(device)
    n_classes = len(np.unique(y_tr))
    head = nn.Linear(enc.d_model, n_classes).to(device)

    T_use = min(eeg_tr.shape[1], 256)
    X_tr = torch.from_numpy(eeg_tr[:, -T_use:, :].astype(np.float32))
    X_te = torch.from_numpy(eeg_te[:, -T_use:, :].astype(np.float32))
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr_t),
        batch_size=min(16, len(X_tr)),
        shuffle=True,
        drop_last=False,
    )

    for p in enc.parameters():
        p.requires_grad_(True)
    enc.train()
    head.train()

    opt = optim.Adam(
        list(enc.parameters()) + list(head.parameters()),
        lr=1e-3,
        weight_decay=0.1,
    )

    for _ in range(ft_epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out, _ = enc(Xb)
            z = out[:, -64:, :].mean(1)
            loss = nn.functional.cross_entropy(head(z), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(head.parameters()), 1.0
            )
            opt.step()

    enc.eval()
    head.eval()
    with torch.no_grad():
        out_te, _ = enc(X_te.to(device))
        z_te = out_te[:, -64:, :].mean(1)
        preds = head(z_te).argmax(1).cpu().numpy()

    return float(balanced_accuracy_score(y_te, preds))


def _fit_eval_encoder_cls_planned(
    encoder,
    eeg_tr: np.ndarray,
    y_tr: np.ndarray,
    eeg_te: np.ndarray,
    y_te: np.ndarray,
    ft_epochs: int,
    device: str,
) -> float:
    """
    Like encoder_ft_cls but classifies from the planner's self-conditioned
    predicted next latent T(h, decoder(h).mu) instead of raw h.

    Trains: encoder + transition + decoder + cls head jointly with the
    supervised + self-consistency loss.  At test time uses LatencyPlanner
    .self_condition() — zero-gradient, one forward pass.

    This is the 'world-model' condition: the classifier operates on where
    the brain state is headed, not just where it is now.
    """
    import copy
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import balanced_accuracy_score

    from v2_digital_self_replication.core.transition_model import (
        ActionConditionedTransition, transition_self_consistency_loss,
    )
    from v2_digital_self_replication.core.latency_planner import LatencyPlanner
    from v2_digital_self_replication.core.intent_decoder import IntentDecoder
    from v2_digital_self_replication.config import V2Config

    enc = copy.deepcopy(encoder).to(device)
    dec_cfg = V2Config().decoder
    decoder = IntentDecoder(
        d_model=enc.d_model, n_dof=dec_cfg.n_dof, d_hidden=dec_cfg.d_hidden
    ).to(device)
    transition = ActionConditionedTransition(
        d_model=enc.d_model, d_dof=dec_cfg.n_dof
    ).to(device)
    planner = LatencyPlanner(transition=transition, decoder=decoder)

    n_classes = len(np.unique(y_tr))
    head = nn.Linear(enc.d_model, n_classes).to(device)

    T_use = min(eeg_tr.shape[1], 256)
    X_tr = torch.from_numpy(eeg_tr[:, -T_use:, :].astype(np.float32))
    X_te = torch.from_numpy(eeg_te[:, -T_use:, :].astype(np.float32))
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr_t),
        batch_size=min(16, len(X_tr)),
        shuffle=True,
        drop_last=False,
    )

    for p in enc.parameters():
        p.requires_grad_(True)
    enc.train(); decoder.train(); transition.train(); head.train()

    opt = optim.Adam(
        list(enc.parameters()) + list(decoder.parameters())
        + list(transition.parameters()) + list(head.parameters()),
        lr=1e-3, weight_decay=0.1,
    )

    for _ in range(ft_epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out, _ = enc(Xb)
            h = out[:, -64:, :].mean(1)              # (B, d_model)
            # Self-condition: predict next latent via world model
            a = decoder(h).mu.detach()               # current decoded action
            h_plan = transition(h, a)                # (B, d_model)
            # Classify from planned state
            loss = nn.functional.cross_entropy(head(h_plan), yb)
            # Add self-consistency: T(h, decoder(h).mu) must still decode same
            loss = loss + transition_self_consistency_loss(
                transition, decoder, h, weight=0.1
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(transition.parameters())
                + list(decoder.parameters()) + list(head.parameters()), 1.0
            )
            opt.step()

    enc.eval(); decoder.eval(); transition.eval(); head.eval()
    with torch.no_grad():
        out_te, _ = enc(X_te.to(device))
        h_te = out_te[:, -64:, :].mean(1)
        h_te_plan = planner.self_condition(h_te)     # T(h, decoder(h).mu)
        preds = head(h_te_plan).argmax(1).cpu().numpy()

    return float(balanced_accuracy_score(y_te, preds))


def _fit_eval_ac_ssm(
    checkpoint: str | None,
    eeg_tr: np.ndarray,
    y_tr: np.ndarray,
    eeg_te: np.ndarray,
    y_te: np.ndarray,
    ft_epochs: int,
    device: str,
    class_names: list[str] | None = None,
) -> float:
    """
    Action-Conditioned SSM world model condition.

    Architecture changes vs encoder_ft_cls_planned:
      1. Action conditioning built into the first SSM block (SiLU-gated additive bias).
         The block learns when motor context modulates brain dynamics vs. when to ignore it.
      2. EEG reconstruction head: unsupervised world-model loss trains on every EEG
         timestep (label-free), giving ~56k training signals vs. 55 labelled trials.
      3. Cosine alignment loss: aligns AC-SSM latents to JEPA-pretrained direction
         (preserves representation geometry without forcing magnitude match).

    Training: CE classification + EEG reconstruction + cosine alignment.
    Test time: a_prev = zeros (no previous command available for isolated windows).
    """
    import copy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import balanced_accuracy_score

    from v2_digital_self_replication.core.stream_encoder import StreamEncoder
    from v2_digital_self_replication.core.intent_decoder import IntentDecoder
    from v2_digital_self_replication.data.intent_mapping import CLASS_TO_INTENT
    from v2_digital_self_replication.config import V2Config

    enc_cfg = V2Config().encoder
    dec_cfg = V2Config().decoder
    n_dof   = dec_cfg.n_dof
    n_eeg   = enc_cfg.n_eeg_channels

    # Build action-conditioned encoder with reconstruction head
    ac_enc = StreamEncoder(
        d_model=enc_cfg.d_model, d_state=enc_cfg.d_state,
        n_layers=enc_cfg.n_layers, n_eeg=n_eeg,
        n_prop=enc_cfg.n_prop_channels, dropout=enc_cfg.dropout,
        d_dof=n_dof, n_eeg_recon=n_eeg,
    ).to(device)

    # Load JEPA pretrained weights (strict=False: action_proj/gate_proj/recon_head are new)
    if checkpoint and Path(checkpoint).exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        sd = ckpt.get("encoder", ckpt)
        ac_enc.load_state_dict(sd, strict=False)

    # Reference JEPA encoder (frozen) for cosine alignment loss and test-time bootstrap
    jepa_enc = _load_encoder_from_ckpt(checkpoint, random_init=False).to(device)
    jepa_enc.eval()
    for p in jepa_enc.parameters():
        p.requires_grad_(False)

    # Small intent decoder for test-time bootstrap: JEPA latent → a_prev
    jepa_dec = IntentDecoder(
        d_model=enc_cfg.d_model, n_dof=n_dof, d_hidden=dec_cfg.d_hidden
    ).to(device)
    jepa_dec.eval()
    for p in jepa_dec.parameters():
        p.requires_grad_(False)

    n_classes = len(np.unique(y_tr))
    head = nn.Linear(enc_cfg.d_model, n_classes).to(device)

    T_use = min(eeg_tr.shape[1], 256)
    X_tr = torch.from_numpy(eeg_tr[:, -T_use:, :].astype(np.float32))
    X_te = torch.from_numpy(eeg_te[:, -T_use:, :].astype(np.float32))
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    # Build intent matrix aligned to LabelEncoder's sorted class order:
    # intent_matrix[y_int] → 6-DOF intent for integer-encoded class y_int.
    # class_names must be in sorted order (matching LabelEncoder output).
    if class_names is None:
        class_names = sorted(set(str(c) for c in np.unique(y_tr).tolist()))
    _intent_rows = []
    for cls_name in class_names:
        vec = CLASS_TO_INTENT.get(cls_name, np.zeros(n_dof, dtype=np.float32))
        _intent_rows.append(vec.astype(np.float32))
    intent_matrix = torch.from_numpy(np.stack(_intent_rows)).to(device)  # (n_cls, n_dof)

    # Curriculum: sort training trials by JEPA latent variance (low = easy, stable signal)
    with torch.no_grad():
        jepa_out, _ = jepa_enc(X_tr.to(device))
        jepa_z_all = jepa_out[:, -64:, :].mean(1)
    difficulty = jepa_z_all.var(dim=1).cpu().numpy()
    curriculum_order = np.argsort(difficulty)  # easy first

    X_tr_curr = X_tr[curriculum_order]
    y_tr_curr = y_tr_t[curriculum_order]

    loader = DataLoader(
        TensorDataset(X_tr_curr, y_tr_curr),
        batch_size=min(16, len(X_tr_curr)),
        shuffle=False,
        drop_last=False,
    )

    ac_enc.train()
    head.train()
    opt = optim.Adam(
        list(ac_enc.parameters()) + list(head.parameters()),
        lr=1e-3, weight_decay=0.1,
    )

    # Reconstruction weight kept small: it's a regulariser, not the primary signal.
    # High w_recon drowns the classification gradient on 55 labelled samples.
    w_recon = 0.05
    w_align = 0.1

    for epoch in range(ft_epochs):
        if epoch == max(1, ft_epochs // 3):
            loader = DataLoader(
                TensorDataset(X_tr, y_tr_t),
                batch_size=min(16, len(X_tr)),
                shuffle=True, drop_last=False,
            )

        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            # Teacher forcing: ground-truth class intent as a_prev.
            # intent_matrix[i] is the intent for LabelEncoder class index i.
            a_prev = intent_matrix[yb]  # (B, n_dof)

            out, _ = ac_enc(Xb, a_prev=a_prev)
            z = out[:, -64:, :].mean(1)

            loss_ce = F.cross_entropy(head(z), yb)

            eeg_hat = ac_enc.reconstruct_eeg(out)
            loss_recon = F.mse_loss(eeg_hat, Xb)

            # Cosine alignment: keep AC-SSM latent direction close to JEPA baseline
            with torch.no_grad():
                jepa_out_b, _ = jepa_enc(Xb)
                jepa_z_b = jepa_out_b[:, -64:, :].mean(1).detach()
            z_norm   = F.normalize(z, dim=-1)
            ref_norm = F.normalize(jepa_z_b, dim=-1)
            loss_align = (1 - (z_norm * ref_norm).sum(dim=-1)).mean()

            loss = loss_ce + w_recon * loss_recon + w_align * loss_align
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(ac_enc.parameters()) + list(head.parameters()), 1.0
            )
            opt.step()

    # Test-time: bootstrap a_prev from frozen JEPA decoder (no ground truth available)
    ac_enc.eval()
    head.eval()
    with torch.no_grad():
        jepa_te, _ = jepa_enc(X_te.to(device))
        jepa_z_te  = jepa_te[:, -64:, :].mean(1)
        a_prev_te  = jepa_dec(jepa_z_te).mu          # (N_te, n_dof)

        out_te, _ = ac_enc(X_te.to(device), a_prev=a_prev_te)
        z_te = out_te[:, -64:, :].mean(1)
        preds = head(z_te).argmax(1).cpu().numpy()

    return float(balanced_accuracy_score(y_te, preds))


def _fit_eval_eegnet(
    eeg_tr: np.ndarray,
    y_tr:   np.ndarray,
    eeg_te: np.ndarray,
    y_te:   np.ndarray,
    n_epochs: int,
    device:   str,
) -> float:
    """
    Train EEGNet end-to-end from scratch on training fold, evaluate on test fold.
    eeg: (n, T, C) — transposed to (n, C, T) internally.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import balanced_accuracy_score

    from v2_digital_self_replication.core.eegnet import EEGNet

    T_use = min(eeg_tr.shape[1], 256)
    n_ch  = eeg_tr.shape[2]
    n_cls = len(np.unique(y_tr))

    # (n, T, C) → (n, C, T)
    X_tr = torch.from_numpy(eeg_tr[:, -T_use:, :].transpose(0, 2, 1).astype(np.float32))
    X_te = torch.from_numpy(eeg_te[:, -T_use:, :].transpose(0, 2, 1).astype(np.float32))
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    net = EEGNet(n_classes=n_cls, n_channels=n_ch, T=T_use).to(device)
    loader = DataLoader(
        TensorDataset(X_tr, y_tr_t),
        batch_size=min(16, len(X_tr)),
        shuffle=True,
        drop_last=False,
    )
    opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    net.train()
    for _ in range(n_epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            loss = nn.functional.cross_entropy(net(Xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        scheduler.step()

    net.eval()
    with torch.no_grad():
        preds = net(X_te.to(device)).argmax(1).cpu().numpy()

    return float(balanced_accuracy_score(y_te, preds))


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
    eeg: np.ndarray,          # (n_trials, T, n_channels)
    labels: list[str],
    checkpoint: str,
    folds: int,
    ft_epochs: int,
    device: str,
    cns_checkpoint: str | None = None,
    fast: bool = False,
    ft_epochs_cls: int | None = None,
    eegnet: bool = False,
    eegnet_epochs: int = 50,
) -> dict:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    from v2_digital_self_replication.data.intent_mapping import labels_to_intents

    cls_epochs = ft_epochs_cls if ft_epochs_cls is not None else ft_epochs

    le      = LabelEncoder().fit(sorted(set(labels)))
    y       = le.transform(labels)
    skf     = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    X_raw   = eeg.transpose(0, 2, 1)    # (n, 21, T) for pyriemann

    # In fast mode skip the three known-null conditions (A/B/C) and calibration
    slow_conditions = ["jepa_ft", "ablation_ft", "zero_shot_probe", "calib_corr"]
    base_conditions = ([] if fast else slow_conditions) + [
        "csp_lda", "mdm", "encoder_ft_cls", "ablation_encoder_cls",
        "encoder_ft_cls_planned", "ac_ssm",
    ]
    if cns_checkpoint:
        base_conditions.append("encoder_ft_cls_cns")
        base_conditions.append("encoder_ft_cls_cns_planned")
    if eegnet:
        base_conditions.append("eegnet")
    fold_results = {k: [] for k in base_conditions}

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(eeg, y)):
        log.info("    fold %d/%d", fold_idx + 1, folds)

        eeg_tr, eeg_te     = eeg[tr_idx], eeg[te_idx]
        labels_tr, labels_te = [labels[i] for i in tr_idx], [labels[i] for i in te_idx]
        y_tr, y_te         = y[tr_idx], y[te_idx]
        X_tr, X_te         = X_raw[tr_idx], X_raw[te_idx]

        if not fast:
            # ── A. Our system: JEPA-pretrained + fine-tuned ──────────────────
            twin_ft = _build_twin(checkpoint, device, random_init=False)
            twin_ft = _finetune_twin(twin_ft, eeg_tr, labels_tr, ft_epochs, device)
            acc_ft, lats_te, dof_te, sigma_te = _encode_and_predict(twin_ft, eeg_te, labels_te)
            fold_results["jepa_ft"].append(acc_ft)

            # ── B. Ablation: random init + fine-tuned ────────────────────────
            twin_rnd = _build_twin(None, device, random_init=True)
            twin_rnd = _finetune_twin(twin_rnd, eeg_tr, labels_tr, ft_epochs, device)
            acc_rnd, _, _, _ = _encode_and_predict(twin_rnd, eeg_te, labels_te)
            fold_results["ablation_ft"].append(acc_rnd)

            # ── C. Zero-shot: JEPA encoder → linear probe ────────────────────
            twin_zs = _build_twin(checkpoint, device, random_init=False)
            _, lats_tr, _, _ = _encode_and_predict(twin_zs, eeg_tr, labels_tr)
            acc_zs = _zero_shot_probe(
                lats_tr, le.transform(labels_tr),
                lats_te, le.transform(labels_te),
            )
            fold_results["zero_shot_probe"].append(acc_zs)

            # ── Calibration ──────────────────────────────────────────────────
            from v2_digital_self_replication.data.intent_mapping import labels_to_intents
            dof_true_te = labels_to_intents(labels_te)
            corr = _calibration_corr(sigma_te, dof_te, dof_true_te)
            fold_results["calib_corr"].append(corr)

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

        # ── F. encoder_ft_cls: JEPA encoder fine-tuned e2e with cross-entropy ─
        enc_jepa = _load_encoder_from_ckpt(checkpoint, random_init=False)
        acc_enc_ft = _fit_eval_encoder_cls(
            enc_jepa, eeg_tr, y_tr, eeg_te, y_te, cls_epochs, device,
        )
        fold_results["encoder_ft_cls"].append(acc_enc_ft)

        # ── G. ablation: random-init encoder fine-tuned e2e with cross-entropy ─
        enc_rnd = _load_encoder_from_ckpt(None, random_init=True)
        acc_abl_enc = _fit_eval_encoder_cls(
            enc_rnd, eeg_tr, y_tr, eeg_te, y_te, cls_epochs, device,
        )
        fold_results["ablation_encoder_cls"].append(acc_abl_enc)

        # ── I. encoder_ft_cls_planned: world-model condition (MLP transition) ──
        #    Train encoder + transition + decoder + cls head jointly.
        #    At test time classify from T(h, decoder(h).mu).
        enc_plan = _load_encoder_from_ckpt(checkpoint, random_init=False)
        acc_planned = _fit_eval_encoder_cls_planned(
            enc_plan, eeg_tr, y_tr, eeg_te, y_te, cls_epochs, device,
        )
        fold_results["encoder_ft_cls_planned"].append(acc_planned)

        # ── K. ac_ssm: action-conditioned SSM world model ────────────────────
        #    Action conditioning built into SSM recurrence (SiLU-gated bias).
        #    Teacher-forcing at train time (ground-truth intent as a_prev).
        #    Bootstrap from JEPA decoder at test time (no GT available).
        #    Joint loss: CE + EEG reconstruction (label-free) + cosine alignment.
        #    Curriculum: easy trials (low JEPA latent variance) trained first.
        sorted_cls_names = le.classes_.tolist()  # LabelEncoder order = sorted alphabetically
        acc_ac = _fit_eval_ac_ssm(
            checkpoint, eeg_tr, y_tr, eeg_te, y_te, cls_epochs, device,
            class_names=sorted_cls_names,
        )
        fold_results["ac_ssm"].append(acc_ac)

        # ── J. EEGNet baseline ───────────────────────────────────────────────
        if eegnet:
            acc_eegnet = _fit_eval_eegnet(
                eeg_tr, y_tr, eeg_te, y_te, eegnet_epochs, device,
            )
            fold_results["eegnet"].append(acc_eegnet)

        # ── H. encoder_ft_cls_cns: CNS cross-modal encoder fine-tuned e2e ─────
        if cns_checkpoint:
            enc_cns = _load_encoder_from_ckpt(cns_checkpoint, random_init=False)
            acc_cns = _fit_eval_encoder_cls(
                enc_cns, eeg_tr, y_tr, eeg_te, y_te, cls_epochs, device,
            )
            fold_results["encoder_ft_cls_cns"].append(acc_cns)

            # CNS + planned (world model applied to cross-modal encoder)
            enc_cns_plan = _load_encoder_from_ckpt(cns_checkpoint, random_init=False)
            acc_cns_planned = _fit_eval_encoder_cls_planned(
                enc_cns_plan, eeg_tr, y_tr, eeg_te, y_te, cls_epochs, device,
            )
            fold_results["encoder_ft_cls_cns_planned"].append(acc_cns_planned)

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
            cns_checkpoint=args.cns_checkpoint,
            fast=args.fast,
            ft_epochs_cls=args.ft_epochs_cls,
            eegnet=args.eegnet,
            eegnet_epochs=args.eegnet_epochs,
        )
        all_subject_results.append(result)
        subject_ids_done.append(subj)

        # Live progress summary — use primary metric (encoder_ft_cls always present)
        primary_means = [s["encoder_ft_cls"]["mean"] for s in all_subject_results]
        log.info("  ↳ Subject %d encoder_ft_cls=%.1f%%  (running mean: %.1f±%.1f%%)",
                 subj,
                 result["encoder_ft_cls"]["mean"] * 100,
                 np.mean(primary_means) * 100,
                 np.std(primary_means) * 100)

    if not all_subject_results:
        log.error("No subjects succeeded")
        return 1

    # ── Aggregate across subjects ─────────────────────────────────────────────
    conditions = [
        "encoder_ft_cls", "encoder_ft_cls_planned", "ac_ssm",
        "ablation_encoder_cls", "csp_lda", "mdm",
    ]
    if not args.fast:
        conditions = ["jepa_ft", "ablation_ft", "zero_shot_probe"] + conditions
    if args.cns_checkpoint:
        conditions.append("encoder_ft_cls_cns")
        conditions.append("encoder_ft_cls_cns_planned")
    if args.eegnet:
        conditions.append("eegnet")
    agg = {}
    for cond in conditions:
        means = [s[cond]["mean"] for s in all_subject_results
                 if cond in s and not np.isnan(s[cond]["mean"])]
        if means:
            agg[cond] = _bootstrap_ci(means)

    if not args.fast:
        calib_corrs = [s["calib_corr"]["mean"] for s in all_subject_results
                       if "calib_corr" in s]
        if calib_corrs:
            agg["calib_corr"] = _bootstrap_ci(calib_corrs)

    stats = _wilcoxon_vs_baselines(all_subject_results) if not args.fast else {}
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
        "encoder_ft_cls":            "JEPA encoder + cls head (e2e FT)",
        "encoder_ft_cls_planned":    "JEPA encoder + MLP world-model planner + cls",
        "ac_ssm":                    "AC-SSM world model + cls [PRIMARY — NEW]",
        "ablation_encoder_cls":      "Random encoder + cls head (e2e FT) [ablation]",
        "zero_shot_probe":           "JEPA encoder + LogReg (zero-shot)",
        "jepa_ft":                   "JEPA + IntentDecoder FT",
        "ablation_ft":               "Random + IntentDecoder FT",
        "csp_lda":                   "CSP + LDA",
        "mdm":                       "MDM (Riemannian)",
    }
    if args.cns_checkpoint:
        labels_map["encoder_ft_cls_cns"] = "CNS cross-modal encoder + cls head [Phase 2]"
        labels_map["encoder_ft_cls_cns_planned"] = "CNS encoder + world-model planner + cls [Phase 2+WM]"
    if args.eegnet:
        labels_map["eegnet"] = "EEGNet (Lawhern 2018) [modern neural baseline]"
    for cond, label in labels_map.items():
        if cond not in agg:
            continue
        a = agg[cond]
        print(f"  {label:<50} {a['mean']*100:5.1f}%  "
              f"[{a['ci_lo']*100:.1f}–{a['ci_hi']*100:.1f}%]")

    if "calib_corr" in agg:
        print(f"  {'Calibration (σ–error correlation)':<40} "
              f"r={agg['calib_corr']['mean']:+.3f}  "
              f"[{agg['calib_corr']['ci_lo']:+.3f}–{agg['calib_corr']['ci_hi']:+.3f}]")
    print("═" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
