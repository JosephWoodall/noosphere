import torch
import numpy as np
from typing import List, Dict
import logging
from scipy.signal import welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from noosphere.synth import make_moabb_dataset, ScalpEEGGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# --- Spectral-Rigorous Challenge ---

def run_subject_11_challenge():
    log.info(f"Starting Subject Challenge (Spectral-Rigorous)...")
    
    n_subjects = 16 
    n_trials = 400
    n_actions = 3
    fs = 256
    
    log.info(f"Generating Dataset...")
    dataset = make_moabb_dataset(n_subjects=n_subjects, n_trials_per_subject=n_trials)

    def extract_power_features(subject_trials: List[Dict]) -> np.ndarray:
        all_feats = []
        for t in subject_trials:
            x = t["eeg"] # [3, T]
            # Compute PSD using Welch
            f, psd = welch(x, fs=fs, nperseg=256)
            
            # Extract power in narrow bands around the intent frequencies
            # 8, 10, 12, 15, 20, 25 Hz
            bands = [(7, 9), (9, 11), (11, 13), (14, 16), (19, 21), (24, 26)]
            p_feat = []
            for low, high in bands:
                idx = np.logical_and(f >= low, f <= high)
                p_feat.append(np.mean(psd[:, idx], axis=1))
            
            feat = np.concatenate(p_feat)
            all_feats.append(np.log(feat + 1e-6))
            
        return np.array(all_feats)

    # 1. Target Data
    test_subject_id = 16
    log.info(f"Processing Target Subject {test_subject_id}...")
    X_all = extract_power_features(dataset[test_subject_id])
    Y_all = np.array([{ScalpEEGGenerator.INTENT_REST: 0, ScalpEEGGenerator.INTENT_RIGHT_HAND: 1, ScalpEEGGenerator.INTENT_LEFT_HAND: 2}[trial["intent"]] for trial in dataset[test_subject_id]])
    
    # Rigorous Split: first 50% for calibration, last 50% for evaluation
    n_cal = n_trials // 2
    X_cal, Y_cal = X_all[:n_cal], Y_all[:n_cal]
    X_eval, Y_eval = X_all[n_cal:], Y_all[n_cal:]
    
    # 2. Classifier
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LinearDiscriminantAnalysis())
    ])
    
    log.info(f"Phase 2: Calibrating on Subject {test_subject_id} ({n_cal} trials)...")
    model.fit(X_cal, Y_cal)

    log.info(f"Phase 3: Final Independent Evaluation (UNSEEN trials)...")
    accuracy = model.score(X_eval, Y_eval)
    
    log.info(f"Independent Subject Accuracy: {accuracy*100:.2f}%")
    if accuracy >= 0.80:
        log.info("RESULT: SUCCESS ✓")
    else:
        log.info("RESULT: FAILED ✗")
        
    return accuracy

if __name__ == "__main__":
    run_subject_11_challenge()