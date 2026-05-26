import torch
import numpy as np
import logging
from moabb.datasets import BNCI2014_004
from moabb.paradigms import MotorImagery
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from noosphere.riemann import sinkhorn_ot_mapping, compute_ea_reference, apply_ea

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def validate_sinkhorn_on_real_data():
    log.info("Validating Sinkhorn OT on Real Human EEG (BNCI2014_004)...")
    
    # 1. Load Data
    dataset = BNCI2014_004()
    paradigm = MotorImagery(n_classes=2, resample=256, fmin=7, fmax=35)
    
    # Load 3 subjects (1, 2 for population, 3 for target)
    subjects = [1, 2, 3]
    X, labels, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    
    # 2. Extract Population (Expert) Manifold from Subject 1 & 2
    mask_pop = meta['subject'].isin([1, 2])
    X_pop = torch.tensor(X[mask_pop], dtype=torch.float32)
    y_pop = labels[mask_pop]
    
    # Pre-align population using EA for a clean manifold
    R_pop = compute_ea_reference(X_pop)
    X_pop_ea = apply_ea(X_pop, R_pop)
    
    # Simple PSD features for the manifold
    def extract_features(x):
        # x: (N, C, T)
        # Log-variance of filtered signal
        return torch.log(torch.var(x, dim=-1) + 1e-6)

    X_pop_feat = extract_features(X_pop_ea)
    
    # Expert representatives (class means)
    expert_manifold = torch.stack([
        X_pop_feat[y_pop == 'right_hand'].mean(dim=0),
        X_pop_feat[y_pop == 'left_hand'].mean(dim=0)
    ])
    
    # 3. Target Subject (Subject 3)
    mask_target = meta['subject'] == 3
    X_target = torch.tensor(X[mask_target], dtype=torch.float32)
    y_target = labels[mask_target]
    
    # 4. Compare EA vs Sinkhorn OT
    
    # Paradigm A: Standard Euclidean Alignment
    R_target = compute_ea_reference(X_target)
    X_target_ea = apply_ea(X_target, R_target)
    X_target_ea_feat = extract_features(X_target_ea)
    
    # Paradigm B: Sinkhorn OT to Expert Manifold
    # Note: We align the raw target features to the expert manifold
    X_target_raw_feat = extract_features(X_target) # Raw features before EA
    X_target_ot_feat = sinkhorn_ot_mapping(X_target_raw_feat, expert_manifold, epsilon=0.01)
    
    # 5. Evaluation (using LDA trained on population)
    clf = Pipeline([('scaler', StandardScaler()), ('lda', LinearDiscriminantAnalysis())])
    
    # Map labels to int
    y_pop_int = np.array([1 if l == 'right_hand' else 0 for l in y_pop])
    y_target_int = np.array([1 if l == 'right_hand' else 0 for l in y_target])
    
    # Train on population
    clf.fit(X_pop_feat.numpy(), y_pop_int)
    
    acc_ea = clf.score(X_target_ea_feat.numpy(), y_target_int)
    acc_ot = clf.score(X_target_ot_feat.numpy(), y_target_int)
    
    log.info(f"Results for Subject 3 (Zero-Shot Cross-Subject):")
    log.info(f"  Accuracy (Euclidean Alignment): {acc_ea*100:.2f}%")
    log.info(f"  Accuracy (Sinkhorn OT Mapping): {acc_ot*100:.2f}%")
    
    improvement = (acc_ot - acc_ea) * 100
    if improvement > 0:
        log.info(f"  Sinkhorn OT provided a {improvement:.2f}% lift! ✓")
    else:
        log.info(f"  No lift observed on this small sample. ✗")

if __name__ == "__main__":
    validate_sinkhorn_on_real_data()
