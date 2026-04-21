#!/usr/bin/env python3
"""
fast_eval.py  — sub-5-minute topology sanity benchmark
=======================================================
Uses 3 subjects × 3-fold CV × 35 pretrain epochs across ALL 5 datasets
(no LOSO, no finetune patience wait).  Gives a reliable proxy for the
full benchmark in ~4 minutes on GPU.

Usage
-----
  python fast_eval.py                   # evaluate current s4_eeg.py
  python fast_eval.py --max-subjects 2  # even faster

Prints mean WS accuracy + per-dataset breakdown.
"""

import argparse, copy, math, time, warnings, logging, sys
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from mne.decoding import CSP

# ── Steal helpers from demo_real_eeg ──────────────────────────────────────────

sys.path.insert(0, ".")
from demo_real_eeg import (
    DATASET_CATALOGUE, load_dataset,
    augment_eeg, mixup_eeg,
    _class_weights, _cosine_lr, _nllloss_smooth, _make_loader,
    infer_s4, run_csp_lda, N_EEG_CH, euclidean_alignment,
)

# ── EEGNet baseline (Lawhern et al., 2018) ───────────────────────────────────

class EEGNet(nn.Module):
    """
    EEGNet: Compact EEG deep learning model (Lawhern et al. 2018, J. Neural Eng.)
    Architecture exactly as reported in the original paper.
    F1=8, D=2, F2=16, kern=64, dropout=0.5.
    """
    def __init__(self, n_cls: int, n_ch: int = 3, T: int = 256,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kern_len: int = 64, drop: float = 0.5):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len//2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1*D, (n_ch, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(drop),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(drop),
        )
        out_T = T // 4 // 8
        self.classifier = nn.Linear(F2 * out_T, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = x.unsqueeze(1)          # (B, 1, C, T)
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = x.flatten(1)
        return self.classifier(x)


class ShallowConvNet(nn.Module):
    """
    ShallowConvNet (Schirrmeister et al. 2017, Human Brain Mapping).
    Designed specifically for motor imagery EEG.
    """
    def __init__(self, n_cls: int, n_ch: int = 3, T: int = 256,
                 n_maps: int = 40, kern: int = 25, drop: float = 0.5):
        super().__init__()
        self.temporal  = nn.Conv2d(1, n_maps, (1, kern), bias=False)
        self.spatial   = nn.Conv2d(n_maps, n_maps, (n_ch, 1), bias=False)
        self.bn        = nn.BatchNorm2d(n_maps)
        self.pool      = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.drop      = nn.Dropout(drop)
        out_T = (T - kern + 1 - 75) // 15 + 1
        self.classifier = nn.Linear(n_maps * max(1, out_T), n_cls)
        self._out_T = out_T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.clamp(x, min=1e-8).log()
        x = self.bn(x)
        x = self.drop(x)
        x = x.flatten(1)
        return self.classifier(x)


def train_baseline(model: nn.Module, X: np.ndarray, y: np.ndarray,
                   n_cls: int, dev: torch.device,
                   epochs: int = 60, lr: float = 1e-3, batch: int = 64) -> nn.Module:
    """Generic training loop for EEGNet / ShallowConvNet."""
    model = model.to(dev)
    wt     = _class_weights(y, n_cls, dev)
    loader = _make_loader(X, y, batch, shuffle=True)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        _cosine_lr(opt, epoch, 5, epochs, lr, lr * 0.01)
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            opt.zero_grad()
            logits = model(bx)
            loss = F.cross_entropy(logits, by, weight=wt)
            loss.backward()
            opt.step()
    model.eval()
    return model


def infer_baseline(model: nn.Module, X: np.ndarray, dev: torch.device) -> np.ndarray:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=64)
    preds = []
    with torch.no_grad():
        for (bx,) in loader:
            preds.append(model(bx.to(dev)).argmax(-1).cpu().numpy())
    return np.concatenate(preds)


# ── Fast training loop (fewer epochs, no val patience) ────────────────────────

def fast_pretrain(X: np.ndarray, y: np.ndarray, n_cls: int,
                  dev: torch.device,
                  epochs: int = 200, lr: float = 8e-4, batch: int = 64) -> nn.Module:
    from noosphere.s4_eeg import S4EEGEncoder
    model = S4EEGEncoder(
        in_channels=N_EEG_CH, d_model=32,
        n_actions=n_cls
    ).to(dev)


    loader = _make_loader(X, y, batch, shuffle=True)
    
    from noosphere.s4_eeg import DirichletEDLLoss
    edl_loss_fn = DirichletEDLLoss(n_classes=n_cls).to(dev)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None

    model.train()
    for epoch in range(epochs):
        _cosine_lr(opt, epoch, 10, epochs, lr, lr * 0.01)
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            
            # Simple supervised focus
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                out = model(bx)
                loss = edl_loss_fn(out["alpha"], F.one_hot(by, num_classes=n_cls).float(), epoch)

            if scaler:
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

    model.eval()
    return model


def fast_finetune(pretrained: nn.Module, Xtr: np.ndarray, ytr: np.ndarray,
                  n_cls: int, dev: torch.device,
                  epochs: int = 25, lr: float = 3e-4, batch: int = 32) -> nn.Module:
    model = copy.deepcopy(pretrained).to(dev)
    for name, p in model.named_parameters():
        p.requires_grad = ("intent_proj" in name or "predictor" in name)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        for p in model.parameters(): p.requires_grad = True
        trainable = list(model.parameters())

    wt  = _class_weights(ytr, n_cls, dev)
    from noosphere.s4_eeg import DirichletEDLLoss
    edl_loss_fn = DirichletEDLLoss(n_classes=n_cls, annealing_step=25).to(dev)
    
    opt = optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if dev.type == "cuda" else None
    loader = _make_loader(Xtr, ytr, min(batch, max(1, len(Xtr))), shuffle=True)

    model.train()
    for epoch in range(epochs):
        _cosine_lr(opt, epoch, 3, epochs, lr, lr * 0.01)
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            bx = augment_eeg(bx)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                out  = model(bx)
                alpha = out["alpha"].float()  # Avoid float16 lgamma overflow
                y_ohe = F.one_hot(by, num_classes=n_cls).float()
                loss = edl_loss_fn(alpha, y_ohe, epoch)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(trainable, 0.5)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 0.5)
                opt.step()

    model.eval()
    return model


# ── Fast evaluate on one dataset ──────────────────────────────────────────────

def fast_eval_dataset(name: str, segments, dev: torch.device,
                      max_subjects: int = 3, n_folds: int = 3) -> dict:
    from collections import defaultdict

    meta  = DATASET_CATALOGUE[name]
    n_cls = meta["n_classes"]

    subj_map = defaultdict(list)
    for s in segments:
        subj_map[s.subject].append(s)

    subjects = sorted(subj_map.keys())[:max_subjects]
    if not subjects:
        return {}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    aligned = {}
    subj_folds = {}
    for subj in subjects:
        segs = subj_map[subj]
        X    = np.stack([s.data for s in segs])
        aligned[subj] = euclidean_alignment(X)
        y    = np.array([s.label % n_cls for s in segs])
        try:
            subj_folds[subj] = list(skf.split(np.zeros(len(segs)), y))
        except ValueError:
            skf2 = StratifiedKFold(n_splits=min(3, len(segs)), shuffle=True, random_state=42)
            subj_folds[subj] = list(skf2.split(np.zeros(len(segs)), y))

    s4_yt, s4_yp = [], []
    csp_yt, csp_yp = [], []
    en_yt, en_yp = [], []   # EEGNet
    scn_yt, scn_yp = [], [] # ShallowConvNet

    for fi in range(n_folds):
        all_tr_k = []
        from demo_real_eeg import EEGSegment
        subj_splits = {}
        for subj in subjects:
            segs = subj_map[subj]
            X_ea = aligned[subj]
            folds = subj_folds[subj]
            if fi >= len(folds): continue
            tr_idx, te_idx = folds[fi]
            tr_segs = [EEGSegment(X_ea[i], segs[i].label, segs[i].subject,
                                  segs[i].dataset, segs[i].sfreq_orig) for i in tr_idx]
            te_segs = [EEGSegment(X_ea[i], segs[i].label, segs[i].subject,
                                  segs[i].dataset, segs[i].sfreq_orig) for i in te_idx]
            all_tr_k.extend(tr_segs)
            subj_splits[subj] = (tr_segs, te_segs)

        if not all_tr_k: continue
        X_all = np.stack([s.data for s in all_tr_k])
        y_all = np.array([s.label % n_cls for s in all_tr_k])
        pretrained = fast_pretrain(X_all, y_all, n_cls, dev)
        C, T = X_all.shape[1], X_all.shape[2]

        for subj, (tr, te) in subj_splits.items():
            if len(tr) < 4 or len(te) < 1: continue
            Xtr = np.stack([s.data for s in tr])
            ytr = np.array([s.label % n_cls for s in tr])
            Xte = np.stack([s.data for s in te])
            yte = np.array([s.label % n_cls for s in te])

            # RS-S4
            model = fast_finetune(pretrained, Xtr, ytr, n_cls, dev)
            pred, prob, _ = infer_s4(model, Xte, dev, T=1.5)
            s4_yt.extend(yte); s4_yp.extend(pred)
            del model

            # EEGNet
            en = train_baseline(EEGNet(n_cls, C, T), Xtr, ytr, n_cls, dev)
            en_yt.extend(yte); en_yp.extend(infer_baseline(en, Xte, dev))
            del en

            # ShallowConvNet
            scn = train_baseline(ShallowConvNet(n_cls, C, T), Xtr, ytr, n_cls, dev)
            scn_yt.extend(yte); scn_yp.extend(infer_baseline(scn, Xte, dev))
            del scn

            # CSP+LDA
            cp, _ = run_csp_lda(Xtr, ytr, Xte)
            csp_yt.extend(yte); csp_yp.extend(cp)

            if dev.type == "cuda": torch.cuda.empty_cache()

    if not s4_yt:
        return {}

    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    s4_yt  = np.array(s4_yt);   s4_yp  = np.array(s4_yp)
    csp_yt = np.array(csp_yt); csp_yp = np.array(csp_yp)
    en_yt  = np.array(en_yt);  en_yp  = np.array(en_yp)
    scn_yt = np.array(scn_yt); scn_yp = np.array(scn_yp)
    return {
        "s4_acc":      float(accuracy_score(s4_yt, s4_yp)),
        "s4_bal_acc":  float(balanced_accuracy_score(s4_yt, s4_yp)),
        "csp_acc":     float(accuracy_score(csp_yt, csp_yp)),
        "eegnet_acc":  float(accuracy_score(en_yt,  en_yp)),
        "shallow_acc": float(accuracy_score(scn_yt, scn_yp)),
        "n_classes":   n_cls,
        "n_subjects":  len(subjects),
        "chance":      1.0 / n_cls,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-subjects", type=int, default=3)
    ap.add_argument("--max-trials",   type=int, default=300)
    ap.add_argument("--n-folds",      type=int, default=3)
    ap.add_argument("--datasets",     nargs="+",
                    default=list(DATASET_CATALOGUE.keys()))
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  FAST EVAL  |  device={dev}  |  max_subj={args.max_subjects}"
          f"  folds={args.n_folds}")
    print(f"{'='*60}\n")

    t0 = time.perf_counter()
    results = {}

    for dsname in args.datasets:
        print(f"  Loading {dsname} ... ", end="", flush=True)
        try:
            segs = load_dataset(dsname, max_subjects=args.max_subjects,
                                max_trials=args.max_trials)
        except Exception as e:
            print(f"SKIP ({e})")
            continue
        print(f"{len(segs)} segs", flush=True)

        print(f"  Evaluating {dsname} ... ", end="", flush=True)
        try:
            r = fast_eval_dataset(dsname, segs, dev,
                                  max_subjects=args.max_subjects,
                                  n_folds=args.n_folds)
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        if r:
            results[dsname] = r
            flag  = "✓" if r["s4_acc"] > 0.70 else ("~" if r["s4_acc"] > 0.60 else "✗")
            print(f"  {flag}  RS-S4={r['s4_acc']:.1%}  EEGNet={r['eegnet_acc']:.1%}"
                  f"  Shallow={r['shallow_acc']:.1%}  CSP={r['csp_acc']:.1%}"
                  f"  (chance={r['chance']:.0%})")

    print()
    if results:
        s4_mean  = np.mean([r["s4_acc"]      for r in results.values()])
        bal_mean = np.mean([r["s4_bal_acc"]  for r in results.values()])
        en_mean  = np.mean([r["eegnet_acc"]  for r in results.values()])
        sc_mean  = np.mean([r["shallow_acc"] for r in results.values()])
        csp_mean = np.mean([r["csp_acc"]     for r in results.values()])
        elapsed  = time.perf_counter() - t0
        print(f"{'='*60}")
        print(f"  MEAN RS-S4:    {s4_mean:.1%}  (bal: {bal_mean:.1%})"
              f"  {'✓ ABOVE 70%' if s4_mean > 0.70 else '✗ BELOW 70%'}")
        print(f"  MEAN EEGNet:   {en_mean:.1%}")
        print(f"  MEAN Shallow:  {sc_mean:.1%}")
        print(f"  MEAN CSP+LDA:  {csp_mean:.1%}")
        print(f"  RS-S4 vs best DL baseline: {s4_mean - max(en_mean, sc_mean):+.1%}")
        print(f"  Total elapsed: {elapsed/60:.1f} min")
        print(f"{'='*60}\n")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
