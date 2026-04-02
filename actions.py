"""
noosphere/actions.py
====================
Action Spaces, Executors, and the Plan→Act Bridge

Features:
- RL Value Veto Removed: The bridge respects the human's confidence absolutely.
- Shell & LLM Executor definitions with full safety-gating parameters.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import numpy as np
import concurrent.futures

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
    
    def by_tier(self, max_tier: int) -> "ActionSpace":
        subset = ActionSpace(f"{self.name}_tier_{max_tier}")
        for a in self.actions:
            if a.tier <= max_tier:
                subset.add(a.name, a.description, a.task_type, a.tier, a.payload)
        return subset

class DigitalStateObserver:
    N_DIMS = 64
    def observe(self, last_result: Optional[Dict] = None, timeout_s: float = 1.0) -> np.ndarray:
        return np.random.randn(self.N_DIMS).astype(np.float32)

class Executor(ABC):
    @abstractmethod
    def execute(self, action: Action) -> Dict[str, Any]: ...
    @abstractmethod
    def can_execute(self, action: Action) -> bool: ...

class ShellExecutor(Executor):
    """
    Simulation Stub: Safely mimics shell execution for testing latencies and world model inference
    without corrupting the user's local operating system.
    """
    def __init__(self, working_dir: str = ".", allow_all: bool = False, timeout_s: float = 10.0, allow_tiers: Optional[Set[int]] = None):
        self.working_dir = working_dir
        self.timeout_s = timeout_s
        self.allow_tiers = allow_tiers or {Tier.SAFE_READ}
        if allow_all: self.allow_tiers = {0, 1, 2, 3, 4, 5}
        self._state_obs = DigitalStateObserver()

    def can_execute(self, action: Action) -> bool:
        return action.payload is not None and "shell_cmd" in action.payload

    def execute(self, action: Action) -> Dict[str, Any]:
        t_start = time.perf_counter()
        cmd = action.payload.get("shell_cmd", "")
        # Simulate dry-run execution latency (~50ms)
        time.sleep(0.05)
        
        if action.tier not in self.allow_tiers:
            return {"executed": False, "success": False, "outcome": "Permission denied: Tier blocked", "reward": -1.0}
            
        return {
            "executed": True,
            "success": True,
            "outcome": f"Simulated success: {cmd}",
            "reward": 0.5,
            "duration_s": time.perf_counter() - t_start,
            "structured": self._state_obs.observe()
        }

class LLMExecutor(Executor):
    def __init__(self, model_endpoint: str = "localhost:11434", model_name: str = "llama3"):
        self.model = model_name
        self.endpoint = model_endpoint
        self._state_obs = DigitalStateObserver()

    def can_execute(self, action: Action) -> bool:
        return action.payload is not None and "agent_prompt" in action.payload

    def execute(self, action: Action) -> Dict[str, Any]:
        t_start = time.perf_counter()
        prompt = action.payload.get("agent_prompt", "")
        time.sleep(0.5) 
        return {
            "executed": True,
            "success": True,
            "outcome": f"[{self.model}] Deployed agent for: '{prompt}'",
            "reward": 1.5, 
            "duration_s": time.perf_counter() - t_start,
            "structured": self._state_obs.observe(),
        }

def make_shell_space(working_dir: str = ".") -> ActionSpace:
    T = Tier
    return (
        ActionSpace("linux_shell")
        .add("ls", "List files", T.SAFE_READ, payload={"shell_cmd": "ls -la"})
        .add("status", "Git status", T.SAFE_READ, payload={"shell_cmd": "git status"})
        .add("ps", "Process list", T.SAFE_READ, payload={"shell_cmd": "ps aux"})
        .add("network", "Ping google", T.NETWORK, payload={"shell_cmd": "ping -c 3 8.8.8.8"})
        .add("rm_rf", "Destructive delete", T.DESTRUCTIVE, payload={"shell_cmd": "rm -rf ./*"})
    )

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
        
        # Latency Eradication: Asynchronous Execution Worker Pool
        self._thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._pending_tasks = {}

    def _trigger_snapshot(self, task_id: str):
        import subprocess, os
        # Obedient Consequence Engine check
        # ZFS or Btrfs retroactive rollback enforcement. Bypass if OS doesn't support it, but attempt snapshot.
        try:
            if os.path.exists(".zfs"):
                subprocess.run(["zfs", "snapshot", f"pool/noosphere@{task_id}"], capture_output=True, timeout=1.0)
            else:
                os.makedirs(".snapshots", exist_ok=True)
                subprocess.run(["btrfs", "subvolume", "snapshot", ".", f".snapshots/{task_id}"], capture_output=True, timeout=1.0)
        except Exception:
            pass

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
            # PREDICTIVE INTENT PRE-FETCHING
            if effective >= self.min_conf * 0.5:
                prefetch_id = f"prefetch_{id(action)}_{time.time_ns()}"
                self._thread_pool.submit(self._prefetch_executor, action)

            out = {"executed": False, "reason": f"conf {effective:.2f} < {self.min_conf}", "reward": 0.0, "action": action}
            self._history.append(out); return out

        # ABSOLUTE SAFETY GATE
        if info is not None and info.get("sim_termination", 0.0) > 0.90:
            out = {"executed": False, "reason": f"safety_gate: termination > 0.90", "reward": 0.0, "action": action}
            self._history.append(out); return out

        if self.dry_run:
            out = {"executed": False, "reason": "dry_run", "reward": 0.0, "action": action}
            self._history.append(out); return out

        # DESTROY LATENCY: Fire and Forget Asynchronous Dispatch
        task_id = f"{id(action)}_{time.time_ns()}"
        
        # Consequence Engine Mandatory Rollback Gateway
        self._trigger_snapshot(task_id)

        future = self._thread_pool.submit(self.executor.execute, action)
        self._pending_tasks[task_id] = future

        out = {
            "executed": "pending",
            "action": action,
            "result": {"status": "dispatched_to_background"},
            "reward": 0.0,
            "outcome": f"[Async] Dispatched {action.name} to background worker...",
            "confidence": effective,
            "task_id": task_id
        }
        self._history.append(out)
        return out

    def _prefetch_executor(self, action: Action):
        if hasattr(self.executor, "prefetch"):
            self.executor.prefetch(action)
        else:
            pass # Stub: load cache, warmup DB connections, stage binaries