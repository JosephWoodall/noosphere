# Peer Review Evaluation Plan

**Target Benchmark Script:** `demo_real_eeg.py`

## Objective
Demonstrate that the Noosphere S4 EEG Encoder achieves state-of-the-art or highly competitive performance on neuro-diverse public datasets, with a strict minimum target of **75% mean accuracy**.

## Evaluation Protocol
1. **Within-Subject**: Chronological 75/25 split per subject. Pretrained S4 trunk fine-tuned per subject. Answers the "after brief calibration" performance scenario.
2. **Cross-Subject (LOSO)**: Leave-one-subject-out. Train on N-1 subjects, evaluate on the held-out subject zero-shot. Answers the "new user, no calibration" scenario.
3. **Baseline**: CSP + LDA (one-vs-rest multi-class CSP, log-variance features, LDA).
4. **Statistics**: Wilcoxon signed-rank test (S4 vs CSP+LDA) per dataset, across subjects, reporting p-value and effect size (r).

## Target Metrics
- **Mean Accuracy Target**: 75% or higher across diverse datasets (e.g., Schirrmeister2017, BNCI2014_001, PhysionetMI, Cho2017).
- **Statistical Significance**: p < 0.05 vs baseline.

*Note: The current architecture utilizes a fallback `Conv1d` temporal stem for low-channel EEG (e.g., 3 channels) to ensure stability, while maintaining the full `SpectralStem` (Riemannian + Spectral features) for high-density arrays.*
