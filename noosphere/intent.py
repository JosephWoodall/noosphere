import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

class IdentitySpace:
    """Manages 'Neural Anchors' for identity prototyping (Brain-Phone contacts)."""
    def __init__(self, embed_dim: int = 64):
        self.embed_dim = embed_dim
        self.anchors: Dict[str, torch.Tensor] = {} # contact_id -> prototype_vector

    def add_anchor(self, contact_id: str, prototype: torch.Tensor):
        self.anchors[contact_id] = F.normalize(prototype, dim=-1)

    def lookup(self, query_embed: torch.Tensor) -> Tuple[Optional[str], float]:
        if not self.anchors:
            return None, 0.0
            
        q = F.normalize(query_embed, dim=-1)
        best_id = None
        best_sim = -1.0
        
        for cid, proto in self.anchors.items():
            sim = torch.dot(q.view(-1), proto.view(-1)).item()
            if sim > best_sim:
                best_sim = sim
                best_id = cid
                
        return best_id, best_sim

class IntentArbiter:
    """Handles BCI Intent Momentum, Probabilistic Blending, Identity Mapping, and Safety Gating."""
    def __init__(self, n_actions: int, min_confidence: float = 0.3, fast_path_threshold: float = 0.85, momentum_decay: float = 0.05):
        self.n_actions = n_actions
        self.min_confidence = min_confidence
        self.fast_path_threshold = fast_path_threshold
        self.momentum_decay = momentum_decay
        
        self.momentum = 0.0
        self.last_discrete = None
        self.last_continuous = None
        
        self.identity_space = IdentitySpace(embed_dim=64)

    def reset(self):
        self.momentum = 0.0
        self.last_discrete = None
        self.last_continuous = None

    def update(self, s4_out: Optional[Dict], xyz_pred: Optional[torch.Tensor]) -> Dict[str, Any]:
        raw_confidence = s4_out["confidence"][0].item() if s4_out is not None else 0.0
        
        # Identity Lookup
        contact_id, identity_conf = None, 0.0
        if s4_out is not None and "identity_embed" in s4_out:
            contact_id, identity_conf = self.identity_space.lookup(s4_out["identity_embed"][0])

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
            "fast_path": effective_confidence >= self.fast_path_threshold,
            "contact_id": contact_id,
            "identity_confidence": identity_conf
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

    def predict_critical_failure(self, intent_command: str) -> bool:
        """
        Digital Consequence Safety Gate (v1.7.0).
        Simulates the outcome of an intent. If the command matches critical 
        destructive patterns or is predicted to cause a system-wide crash,
        intercept and block it before it reaches the OS or hardware.
        """
        # 1. Heuristic Pattern Matching (Fast Path)
        critical_patterns = ["rm -rf", "mkfs", "dd if=", "format", ":(){ :|:& };:"]
        for pattern in critical_patterns:
            if pattern in intent_command:
                return True # Failure Predicted (Blocked)
        
        # 2. Semantic Analysis (Mock - in full deployment this would use a tiny transformer)
        destructive_keywords = ["overwrite", "wipe", "destroy", "reboot"]
        if any(kw in intent_command.lower() for kw in destructive_keywords):
            # Potential threat, requires secondary confirmation or RSSM simulation
            return True 

        return False # Safe
