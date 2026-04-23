# Riemannian-S4: A Novel Multi-Scale State Space Architecture

Official implementation of the **Riemannian-S4** topology for Brain-Computer Interfacing (BCI). This repository contains the core architecture and the benchmarking suite used to achieve state-of-the-art results on the MOABB (Mother of All BCI Benchmarks).

## 1. Overview
Riemannian-S4 is a hybrid neural architecture that combines:
- **Riemannian Stem:** Uses Euclidean Alignment (EA) to linearize EEG geometry.
- **Multi-Scale S4 Backbone:** Independent S4D encoders for Frontal, Motor, and Parietal cortical regions.
- **Evidential Head:** Dirichlet-based uncertainty quantification for safety-critical control.

## 2. Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## 3. Usage
To reproduce the benchmarks (Within-Subject and LOSO) for all 5 datasets:
```bash
python demo_real_eeg.py --benchmark
```

## 4. Citation
If you use this work, please cite our manuscript:
> Joseph Woodall (2026). Riemannian-S4: A Novel Multi-Scale State Space Architecture for Minimal-Calibration Brain-Computer Interfacing.
