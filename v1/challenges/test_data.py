import torch
import numpy as np
from noosphere.synth import make_moabb_dataset
dataset = make_moabb_dataset(n_subjects=1, n_trials_per_subject=1)
print(dataset[1][0]['eeg'].shape)
