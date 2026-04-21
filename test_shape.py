import torch
from noosphere.s4_eeg import S4EEGEncoder
model = S4EEGEncoder(in_channels=21, d_model=192, n_blocks=3, d_state=64, n_actions=4)
x = torch.randn(2, 21, 256)
out_spatial = model.spatial_stem(x)
print("Spatial out:", out_spatial.shape)
