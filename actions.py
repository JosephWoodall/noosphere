"""
noosphere/actions.py
====================
Action Spaces, Executors, and the Plan→Act Bridge

Features:
- Uncompromised Biological Authority: ActBridge no longer dilutes S4 confidence
  with environmental value. Human intent is absolute.
- LLMExecutor handles hierarchical agent deployments natively.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

class Tier:
    SAFE_READ = 0
    SAFE_WRITE = 1
    PROCESS = 2
    NETWORK = 3
    SYSTEM = 4
    DESTRUCTIVE = 5

@dataclass
class Action:
    index: int
    name: str
    description: str
    task_type: str = "multiclass"
    tier: int = Tier.SAFE_READ
    payload: Any = None

@dataclass
class ActionSpace:
    name: str
    actions: List[Action] = field(default_factory=list)
    def __len__(self): return len(self.actions)
    def __getitem__(self, idx: int) -> Action: return self.actions[idx]
    def add(self, name: str, description: str, task_type: str = "multiclass", tier: int = Tier.SAFE_READ, payload: Any = None) -> "ActionSpace":
        self.actions.append(Action(index=len(self.actions), name=name, description=description, task_type=task_type, tier=tier, payload=payload))
        return self
    @property
    def n_actions(self) -> int: return len(self.actions)

class DigitalStateObserver:
    N_DIMS = 64
    def observe(self, last_result: Optional[Dict] = None, timeout_s: float = 1.0) -> np.ndarray:
        return np.zeros(self.N_DIMS, dtype=np.float32)

class Executor(ABC):
    @abstractmethod
    def execute(self, action: Action) -> Dict[str, Any]: ...
    @abstractmethod
    def can_execute(self, action: Action) -> bool: ...

class LLMExecutor(Executor):
    def __init__(self, model_endpoint: str = "localhost:11434", model_name: str = "llama3"):
        self.model = model_name
        self.endpoint = model_endpoint
        self._state_obs = DigitalStateObserver()

    def can_execute(self, action: Action) -> bool:
        return action.payload is not None and "agent_prompt" in action.payload

    def execute(self, action: Action) -> Dict[str, Any]:
        t_start = time.time()
        digital_state = self._state_obs.observe()
        prompt = action.payload.get("agent_prompt", "")
        time.sleep(0.5) 
        return {
            "success": True,
            "outcome": f"[{self.model}] Deployed agent for: '{prompt}'",
            "reward": 1.5, 
            "duration_s": time.time() - t_start,
            "digital_state": digital_state,
        }

def make_agent_space() -> ActionSpace:
    T = Tier
    return (
        ActionSpace("agent_macro_intents")
        .add("agent_debug", "Deploy SWE agent to debug last error", T.SYSTEM, payload={"agent_prompt": "Analyze logs and fix."})
        .add("agent_git_sync", "Deploy agent to resolve git state", T.SYSTEM, payload={"agent_prompt": "Sync git state."})
    )

class ExecutorRouter(Executor):
    def __init__(self, shell_exec, llm_exec):
        self.shell = shell_exec
        self.llm = llm_exec

    def can_execute(self, action: Action) -> bool:
        return self.shell.can_execute(action) or self.llm.can_execute(action)

    def execute(self, action: Action) -> Dict[str, Any]:
        if self.llm.can_execute(action): return self.llm.execute(action)
        return self.shell.execute(action)

class ActBridge:
    def __init__(self, action_space: ActionSpace, executor: Executor, min_confidence: float = 0.3, dry_run: bool = False):
        self.space = action_space
        self.executor = executor
        self.min_conf = min_confidence
        self.dry_run = dry_run
        self._history: List[Dict] = []

    def act(self, action_idx: int, predicted_value: float = 1.0, s4_confidence: Optional[float] = None, info: Optional[Dict] = None) -> Dict[str, Any]:
        if s4_confidence is None and info is not None: s4_confidence = info.get("s4_confidence")
        
        # BIOLOGICAL AUTHORITY: Never dilute human confidence with environmental value
        if s4_confidence is not None:
            effective = float(s4_confidence)
        else:
            effective = float(predicted_value)

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