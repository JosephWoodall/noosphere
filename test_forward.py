import torch
from noosphere.s4_eeg import S4EEGEncoder
model = S4EEGEncoder(in_channels=21, d_model=192, n_blocks=3, d_state=64, n_actions=4)
x = torch.randn(2, 21, 256)
x_art = model.artifact_gater(x)
x_hurst = model.hurst(x_art)
x_spatial = model.spatial_stem(x_hurst)
x_spectral = model.spectral_wavelet(x_hurst)
print("x_spatial:", x_spatial.shape)
print("x_spectral:", x_spectral.shape)
