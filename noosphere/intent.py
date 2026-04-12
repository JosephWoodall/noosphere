import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

class IntentProcessor:
    """Handles BCI Intent Momentum and Probabilistic Blending."""
    def __init__(self, n_actions: int, min_confidence: float = 0.3, fast_path_threshold: float = 0.85, momentum_decay: float = 0.05):
        self.n_actions = n_actions
        self.min_confidence = min_confidence
        self.fast_path_threshold = fast_path_threshold
        self.momentum_decay = momentum_decay
        
        self.momentum = 0.0
        self.last_discrete = None
        self.last_continuous = None

    def reset(self):
        self.momentum = 0.0
        self.last_discrete = None
        self.last_continuous = None

    def update(self, s4_out: Optional[Dict], xyz_pred: Optional[torch.Tensor]) -> Dict[str, Any]:
        raw_confidence = s4_out["confidence"][0].item() if s4_out is not None else 0.0
        
        # Intent Momentum: Decay gracefully
        self.momentum = max(0.0, self.momentum - self.momentum_decay)
        
        if raw_confidence >= self.min_confidence:
            self.momentum = raw_confidence
            self.last_discrete = s4_out["intent_probs"][0]
            self.last_continuous = xyz_pred[0] if xyz_pred is not None else None

        active = self.momentum > 0.0
        effective_confidence = max(raw_confidence, self.momentum)
        
        return {
            "active": active,
            "confidence": effective_confidence,
            "discrete": self.last_discrete if active else None,
            "continuous": self.last_continuous if active else None,
            "fast_path": effective_confidence >= self.fast_path_threshold
        }

    def blend(self, bci_out: Dict, actor_probs: torch.Tensor, actor_cont: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Blends BCI intent with AI policy based on confidence."""
        if bci_out["fast_path"]:
            return bci_out["discrete"], bci_out["continuous"]
        
        if not bci_out["active"]:
            return actor_probs, actor_cont
            
        conf = bci_out["confidence"]
        p_final = conf * bci_out["discrete"] + (1.0 - conf) * actor_probs
        c_final = conf * bci_out["continuous"] + (1.0 - conf) * actor_cont
        
        return p_final, c_final
