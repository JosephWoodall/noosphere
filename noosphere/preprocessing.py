import torch
import numpy as np
from typing import Dict, Any, Optional

from noosphere.riemann import sinkhorn_ot_mapping, compute_ea_reference, apply_ea

class SinkhornOTAlignment:
    """Aligns incoming EEG to an expert manifold using Sinkhorn OT."""
    def __init__(self, expert_manifold: torch.Tensor, epsilon: float = 0.05):
        self.expert_manifold = expert_manifold
        self.epsilon = epsilon

    def __call__(self, x_features: torch.Tensor) -> torch.Tensor:
        """x_features: (B, D)"""
        return sinkhorn_ot_mapping(x_features, self.expert_manifold, epsilon=self.epsilon)

class EAAlignment:
    """Aligns incoming EEG using Euclidean Alignment."""
    def __init__(self, R_inv_sq: torch.Tensor):
        self.R_inv_sq = R_inv_sq

    def __call__(self, x_raw: torch.Tensor) -> torch.Tensor:
        """x_raw: (..., C, T)"""
        return apply_ea(x_raw, self.R_inv_sq)

class ObservationPreprocessor:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        out = {}
        modalities = ["rgb", "depth", "rgb_right", "eeg", "structured", "kinematics", "gaze", "os_window"]
        
        for k in modalities:
            val = obs.get(k)
            if val is not None:
                if torch.is_tensor(val):
                    # Already a tensor, ensure correct device
                    x_torch = val.to(self.device, dtype=torch.float32)
                    
                    # Image-like modalities: normalize and permute (H, W, C) -> (C, H, W)
                    # Note: We assume tensors might already be in correct shape (B, C, H, W) 
                    # but let's handle common cases.
                    if k in ("rgb", "depth", "rgb_right", "os_window"):
                        if x_torch.ndim == 3: # (H, W, C)
                            x_torch = x_torch.permute(2, 0, 1).unsqueeze(0)
                        elif x_torch.ndim == 2: # (H, W)
                            x_torch = x_torch.unsqueeze(0).unsqueeze(0)
                    elif k == "eeg" and x_torch.ndim == 2:
                        x_torch = x_torch.unsqueeze(0)
                    
                    out[k] = x_torch
                else:
                    x = np.asarray(val, dtype=np.float32)
                    
                    # Image-like modalities: normalize and permute (H, W, C) -> (C, H, W)
                    if k in ("rgb", "depth", "rgb_right", "os_window"):
                        if x.max() > 1.0 and k != "depth": 
                            x /= 255.0
                        if x.ndim == 2: 
                            x = np.expand_dims(x, axis=-1)
                        if x.ndim == 3: 
                            x = x.transpose(2, 0, 1)
                        x = x[None] # Add batch dim
                    
                    # Time-series / Sequence modalities
                    elif k == "eeg" and x.ndim == 2: 
                        x = x[None]
                    elif k in ("structured", "kinematics", "gaze"):
                        if x.ndim == 1: 
                            x = x[None, None]
                        elif x.ndim == 2: 
                            x = x[None]
                    
                    out[k] = torch.tensor(x, device=self.device)
        
        mask_val = obs.get("electrode_mask")
        if mask_val is not None:
            if torch.is_tensor(mask_val):
                out["electrode_mask"] = mask_val.to(self.device)
            else:
                mask = np.array(mask_val, dtype=np.float32)
                if mask.ndim == 1: 
                    mask = mask[None]
                out["electrode_mask"] = torch.tensor(mask, device=self.device)
            
        return out
