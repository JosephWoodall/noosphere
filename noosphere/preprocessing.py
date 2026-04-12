import torch
import numpy as np
from typing import Dict, Any, Optional

class ObservationPreprocessor:
    def __init__(self, device: torch.device):
        self.device = device

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        out = {}
        modalities = ["rgb", "depth", "rgb_right", "eeg", "structured", "kinematics", "gaze", "os_window"]
        
        for k in modalities:
            if obs.get(k) is not None:
                x = np.asarray(obs[k], dtype=np.float32)
                
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
        
        if "electrode_mask" in obs and obs["electrode_mask"] is not None:
            mask = np.array(obs["electrode_mask"], dtype=np.float32)
            if mask.ndim == 1: 
                mask = mask[None]
            out["electrode_mask"] = torch.tensor(mask, device=self.device)
            
        return out
