import torch
import numpy as np
from typing import List, Dict
import logging
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sinkhorn import sinkhorn_ot_mapping

from noosphere.synth import make_moabb_dataset, ScalpEEGGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# --- Sinkhorn-Transport-Rigorous Challenge ---

def run_subject_11_challenge():
    log.info(f"Starting Subject Challenge (Sinkhorn-OT Rigorous)...")
    
    n_subjects = 16 
    n_trials = 400
    n_actions = 3
    fs = 256
    
    log.info(f"Generating Dataset...")
    dataset = make_moabb_dataset(n_subjects=n_subjects, n_trials_per_subject=n_trials)

    def extract_power_features(subject_trials: List[Dict]) -> torch.Tensor:
        all_feats = []
        for t in subject_trials:
            x = t["eeg"] # [3, T]
            f, psd = welch(x, fs=fs, nperseg=256)
            bands = [(7, 9), (9, 11), (11, 13), (14, 16), (19, 21), (24, 26)]
            p_feat = []
            for low, high in bands:
                idx = np.logical_and(f >= low, f <= high)
                p_feat.append(np.mean(psd[:, idx], axis=1))
            feat = np.concatenate(p_feat)
            all_feats.append(np.log(feat + 1e-6))
        return torch.tensor(np.array(all_feats), dtype=torch.float32)

    # 1. Population Data (Expert Manifold)
    log.info("Building Expert Manifold (Subjects 1-15)...")
    X_train_list, Y_train_list = [], []
    for sub_id in range(1, 16):
        X_sub = extract_power_features(dataset[sub_id])
        X_train_list.append(X_sub)
        Y_train_list.extend([{ScalpEEGGenerator.INTENT_REST: 0, ScalpEEGGenerator.INTENT_RIGHT_HAND: 1, ScalpEEGGenerator.INTENT_LEFT_HAND: 2}[trial["intent"]] for trial in dataset[sub_id]])
    
    X_train = torch.cat(X_train_list)
    Y_train = np.array(Y_train_list)
    
    # Representative 'Expert' samples for alignment (mean of each class)
    expert_samples = []
    for c in range(n_actions):
        expert_samples.append(X_train[Y_train == c].mean(dim=0))
    expert_manifold = torch.stack(expert_samples)

    # 2. Target Data (Subject 16)
    test_subject_id = 16
    log.info(f"Processing Target Subject {test_subject_id}...")
    X_test_raw = extract_power_features(dataset[test_subject_id])
    Y_test_all = np.array([{ScalpEEGGenerator.INTENT_REST: 0, ScalpEEGGenerator.INTENT_RIGHT_HAND: 1, ScalpEEGGenerator.INTENT_LEFT_HAND: 2}[trial["intent"]] for trial in dataset[test_subject_id]])
    
    # --- Sinkhorn Optimal Transport Alignment ---
    log.info("Mapping Target Subject to Expert Manifold via Sinkhorn OT...")
    # Map all test samples directly to the closest expert representative
    X_test_aligned = sinkhorn_ot_mapping(X_test_raw, expert_manifold, epsilon=0.01)
    X_test_aligned = X_test_aligned.numpy()
    
    # Rigorous Split
    n_cal = n_trials // 2
    X_cal, Y_cal = X_test_aligned[:n_cal], Y_test_all[:n_cal]
    X_eval, Y_eval = X_test_aligned[n_cal:], Y_test_all[n_cal:]
    
    # 3. Classifier
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis())
    ])
    
    log.info(f"Phase 2: Calibrating on Aligned Features...")
    model.fit(X_cal, Y_cal)

    log.info(f"Phase 3: Final Independent Evaluation...")
    accuracy = model.score(X_eval, Y_eval)
    
    log.info(f"Independent Subject Accuracy: {accuracy*100:.2f}%")
    if accuracy >= 0.80:
        log.info("RESULT: SUCCESS ✓")
    else:
        log.info("RESULT: FAILED ✗")
        
    return accuracy

if __name__ == "__main__":
    run_subject_11_challenge()
