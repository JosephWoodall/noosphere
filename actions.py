"""
noosphere/actions.py
====================
Action Spaces, Executors, and the Plan→Act Bridge

Features:
- RL Value Veto Removed: The bridge respects the human's confidence absolutely.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

# ── [Action Space definitions remain unchanged] ──

class ActBridge:
    def __init__(self, action_space: 'ActionSpace', executor: 'Executor', min_confidence: float = 0.3, dry_run: bool = False):
        self.space = action_space
        self.executor = executor
        self.min_conf = min_confidence
        self.dry_run = dry_run
        self._history: List[Dict] = []

    def act(self, action_idx: int, s4_confidence: Optional[float] = None, info: Optional[Dict] = None) -> Dict[str, Any]:
        
        # BIOLOGICAL AUTHORITY: Never dilute human confidence with environmental value
        effective = float(s4_confidence) if s4_confidence is not None else 1.0

        if action_idx >= len(self.space): 
            return {"executed": False, "reason": "invalid index", "reward": -0.1}
        action = self.space[action_idx]

        if effective < self.min_conf:
            out = {"executed": False, "reason": f"conf {effective:.2f} < {self.min_conf}", "reward": 0.0, "action": action}
            self._history.append(out); return out

        # ABSOLUTE SAFETY GATE
        if info is not None and info.get("sim_termination", 0.0) > 0.90:
            out = {"executed": False, "reason": f"safety_gate: termination > 0.90", "reward": 0.0, "action": action}
            self._history.append(out); return out

        if self.dry_run:
            out = {"executed": False, "reason": "dry_run", "reward": 0.0, "action": action}
            self._history.append(out); return out

        exec_result = self.executor.execute(action)
        out = {
            "executed": True,
            "action": action,
            "result": exec_result,
            "reward": exec_result.get("reward", 0.0),
            "outcome": exec_result.get("outcome", ""),
            "confidence": effective,
        }
        if "structured" in exec_result: out["structured"] = exec_result["structured"]
        self._history.append(out)
        return out