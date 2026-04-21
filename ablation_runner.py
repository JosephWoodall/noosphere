#!/usr/bin/env python3
import torch, copy, time
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from noosphere.s4_eeg import S4EEGEncoder, S4DLayer, RiemannianStem
from fast_eval import fast_pretrain, fast_finetune, load_dataset, DATASET_CATALOGUE, N_EEG_CH, euclidean_alignment, _class_weights, _make_loader, _nllloss_smooth, augment_eeg, _cosine_lr, mixup_eeg, infer_s4
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore")

def do_ablation(variant_name, dataset_name="Schirrmeister2017"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segs = load_dataset(dataset_name, max_subjects=5, max_trials=300)
    n_cls = DATASET_CATALOGUE[dataset_name]["n_classes"]
    
    # Stratified CV
    from collections import defaultdict
    subj_map = defaultdict(list)
    for s in segs: subj_map[s.subject].append(s)
    subjects = sorted(subj_map.keys())
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    aligned = {}
    subj_folds = {}
    for subj in subjects:
        segs_s = subj_map[subj]
        X = np.stack([s.data for s in segs_s])
        aligned[subj] = euclidean_alignment(X)
        y = np.array([s.label % n_cls for s in segs_s])
        subj_folds[subj] = list(skf.split(np.zeros(len(segs_s)), y))

    yt, yp = [], []
    for fi in range(3):
        all_tr_k = []
        subj_splits = {}
        for subj in subjects:
            segs_s = subj_map[subj]
            X_ea = aligned[subj]
            tr_idx, te_idx = subj_folds[subj][fi]
            # Just keep raw data
            tr_d = [(X_ea[i], segs_s[i].label % n_cls) for i in tr_idx]
            te_d = [(X_ea[i], segs_s[i].label % n_cls) for i in te_idx]
            all_tr_k.extend(tr_d)
            subj_splits[subj] = (tr_d, te_d)
            
        X_all = np.stack([x[0] for x in all_tr_k])
        y_all = np.array([x[1] for x in all_tr_k])
        
        # Build Variant Model
        model = S4EEGEncoder(in_channels=N_EEG_CH, d_model=192, n_blocks=3, d_state=64, n_actions=n_cls).to(dev)
        
        if variant_name == "no_s4":
            class GRUReplacement(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.gru = nn.GRU(192, 192//2, 1, batch_first=True, bidirectional=True)
                def forward(self, x):
                    x, _ = self.gru(x)
                    return x
            for i in range(len(model.blocks_high)):
                model.blocks_high[i] = GRUReplacement().to(dev)
            for i in range(len(model.blocks_low)):
                model.blocks_low[i] = GRUReplacement().to(dev)
                
        elif variant_name == "no_riemann":
            class SimpleSpatial(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.proj = nn.Linear(N_EEG_CH, 192)
                def forward(self, x):
                    # x is (B, C, L), mean over L to get (B, C) then project
                    x = x.mean(dim=-1)
                    return self.proj(x).unsqueeze(1) # (B, 1, 192)
            model.spatial_stem = SimpleSpatial().to(dev)
            
        elif variant_name == "no_edl":
            class SoftmaxLinear(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lin = nn.Linear(192, n_cls)
                def forward(self, x):
                    logits = self.lin(x)
                    return {"alpha": logits, "intent_probs": logits}
            model.predictor = SoftmaxLinear().to(dev)
            
        # Fast Pretrain inside
        epochs = 40
        lr = 8e-4
        wt = _class_weights(y_all, n_cls, dev)
        loader = _make_loader(X_all, y_all, 64, shuffle=True)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        model.train()
        for ep in range(epochs):
            for bx, by in loader:
                bx, by = bx.to(dev), by.to(dev)
                opt.zero_grad()
                out = model(bx)
                if variant_name == "no_edl":
                    loss = F.cross_entropy(out["alpha"], by, weight=wt)
                else:
                    lp = F.log_softmax(out["alpha"], dim=-1)
                    loss = _nllloss_smooth(lp, by, wt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()
                
        # Finetune
        for subj, (tr, te) in subj_splits.items():
            if len(tr) < 4 or len(te) < 1: continue
            Xtr = np.stack([x[0] for x in tr])
            ytr = np.array([x[1] for x in tr])
            Xte = np.stack([x[0] for x in te])
            yte = np.array([x[1] for x in te])
            
            ft_model = copy.deepcopy(model)
            ft_opt = optim.AdamW(ft_model.parameters(), lr=3e-4)
            ft_loader = _make_loader(Xtr, ytr, 32, shuffle=True)
            ft_model.train()
            for ep in range(20):
                for bx, by in ft_loader:
                    bx, by = bx.to(dev), by.to(dev)
                    ft_opt.zero_grad()
                    out = ft_model(bx)
                    if variant_name == "no_edl":
                        loss = F.cross_entropy(out["alpha"], by, weight=wt)
                    else:
                        lp = F.log_softmax(out["alpha"], dim=-1)
                        loss = _nllloss_smooth(lp, by, wt)
                    loss.backward()
                    nn.utils.clip_grad_norm_(ft_model.parameters(), 0.5)
                    ft_opt.step()
                    
            ft_model.eval()
            with torch.no_grad():
                out = ft_model(torch.tensor(Xte, dtype=torch.float32).to(dev))
                pred = out["alpha"].argmax(-1).cpu().numpy()
            yt.extend(yte)
            yp.extend(pred)
            del ft_model
            
    print(f"Variant: {variant_name} -> Accuracy: {accuracy_score(yt, yp):.3f}")

if __name__ == "__main__":
    print("Running Fast Ablation (5 subj, 3 folds)...")
    do_ablation("no_edl")
