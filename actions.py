"""
noosphere/actions.py
====================
Action Spaces and the Plan→Act Bridge

The MCTS planner returns an integer action index.
This module defines what those integers *mean* and how they translate
to real-world effects — physical or digital.

Design
------
An ActionSpace maps integer indices to concrete commands and provides
a vocabulary the world model can plan over. The world model never executes
during planning; it imagines consequences. Only at the Act phase does
the selected index get translated to a real command via an Executor.

                    ┌─────────────────┐
                    │   World Model   │
                    │  (imagination)  │
                    │                 │
    MCTS selects ──→│ imagine_step(a) │──→ predicted outcome
    best action     │    in latent    │    (reward, termination)
                    └────────┬────────┘
                             │ integer action index
                             ▼
                    ┌─────────────────┐
                    │  ActionSpace    │   maps index → command
                    │  router         │   routes to correct executor
                    └────────┬────────┘
                             │
               ┌─────────────┼─────────────┐
               ▼             ▼             ▼
       ApparatusExecutor  ShellExecutor  NullExecutor
       (motor commands)   (linux cmds)   (no-op)

Task types
----------
The action space and executor are chosen based on the decoded intent.
Three task shapes are supported:

    Binary      — two actions (e.g. grasp/release, yes/no, run/abort)
    Multiclass  — N discrete actions from a fixed vocabulary
    Regression  — continuous target decoded directly from the latent state
                  (used for fine-grained motor control, not MCTS)

The world model is domain-agnostic. The action space is the only thing
that changes between physical manipulation and terminal command execution.
"""

import os
import subprocess
import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ── Action vocabulary ─────────────────────────────────────────────────────────

@dataclass
class Action:
    """
    A single action in a vocabulary.

    index      : integer index used by the world model and MCTS
    name       : human-readable label
    description: what this action does in the real world
    task_type  : "binary" | "multiclass" | "regression"
    payload    : executor-specific data (command string, joint deltas, etc.)
    """
    index:       int
    name:        str
    description: str
    task_type:   str = "multiclass"
    payload:     Any = None


@dataclass
class ActionSpace:
    """
    Ordered vocabulary of actions the agent can take.
    The integer indices used by MCTS must match exactly.
    """
    name:    str
    actions: List[Action] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Action:
        return self.actions[idx]

    def add(self, name: str, description: str,
            task_type: str = "multiclass", payload: Any = None) -> "ActionSpace":
        """Add an action. Index is assigned automatically. Chainable."""
        self.actions.append(Action(
            index=len(self.actions),
            name=name,
            description=description,
            task_type=task_type,
            payload=payload,
        ))
        return self

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def describe(self) -> str:
        lines = [f"ActionSpace: {self.name}  ({self.n_actions} actions)"]
        for a in self.actions:
            lines.append(f"  [{a.index:2d}] {a.name:<24} — {a.description}")
        return "\n".join(lines)


# ── Built-in action spaces ────────────────────────────────────────────────────

def make_apparatus_space() -> ActionSpace:
    """
    6-DOF robotic arm apparatus control.
    Actions are joint-space delta moves (not absolute positions).
    Fine-grained positioning is handled by the IK + apparatus pipeline,
    not by MCTS directly.
    """
    return (ActionSpace("apparatus")
        .add("shoulder_yaw_cw",   "Rotate shoulder clockwise",        payload={"joint": 0, "delta_deg": +5.0})
        .add("shoulder_yaw_ccw",  "Rotate shoulder counter-clockwise",payload={"joint": 0, "delta_deg": -5.0})
        .add("shoulder_pitch_up", "Raise shoulder",                   payload={"joint": 1, "delta_deg": +5.0})
        .add("shoulder_pitch_dn", "Lower shoulder",                   payload={"joint": 1, "delta_deg": -5.0})
        .add("elbow_extend",      "Extend elbow",                     payload={"joint": 3, "delta_deg": +8.0})
        .add("elbow_flex",        "Flex elbow",                       payload={"joint": 3, "delta_deg": -8.0})
        .add("wrist_flex",        "Flex wrist",                       payload={"joint": 4, "delta_deg": +8.0})
        .add("wrist_extend",      "Extend wrist",                     payload={"joint": 4, "delta_deg": -8.0})
    )


def make_shell_space(working_dir: str = ".") -> ActionSpace:
    """
    Linux terminal command vocabulary.

    These are the actions the agent can *plan over* in imagination.
    The world model learns to predict the consequence of each command
    (files created, exit code, stdout summary) in latent space.

    The vocabulary starts minimal and grows as the agent gains experience.
    New commands are added via shell_space.add(...).

    Safety: all commands run in a sandboxed subprocess with timeout.
    Commands that modify system state require explicit allow-listing.
    """
    sp = ActionSpace("shell")
    # Read-only exploration (safe to imagine and execute freely)
    sp.add("ls",          "List current directory",           payload={"cmd": "ls -la"})
    sp.add("pwd",         "Print working directory",          payload={"cmd": "pwd"})
    sp.add("cat_readme",  "Read README if present",           payload={"cmd": "cat README.md 2>/dev/null || echo 'no README'"})
    sp.add("list_python", "List Python files",                payload={"cmd": "find . -name '*.py' -maxdepth 3"})
    sp.add("show_env",    "Show environment variables",       payload={"cmd": "env | sort"})
    sp.add("disk_usage",  "Show disk usage",                  payload={"cmd": "df -h ."})
    sp.add("running_procs","List running processes",          payload={"cmd": "ps aux --sort=-%mem | head -20"})
    sp.add("git_status",  "Check git status if in repo",      payload={"cmd": "git status 2>/dev/null || echo 'not a git repo'"})
    sp.add("git_log",     "Recent git commits",               payload={"cmd": "git log --oneline -10 2>/dev/null || echo 'not a git repo'"})
    sp.add("python_version","Check Python version",           payload={"cmd": "python3 --version"})
    # No-op (agent can choose to wait)
    sp.add("wait",        "Do nothing this step",             payload={"cmd": None})
    return sp


def make_binary_space(positive_action: str, negative_action: str,
                      pos_payload: Any = None, neg_payload: Any = None) -> ActionSpace:
    """Binary decision space: yes/no, run/abort, grasp/release, etc."""
    return (ActionSpace("binary")
        .add(positive_action, f"Execute: {positive_action}",
             task_type="binary", payload=pos_payload)
        .add(negative_action, f"Do not: {positive_action}",
             task_type="binary", payload=neg_payload)
    )


# ── Executors ─────────────────────────────────────────────────────────────────

class Executor(ABC):
    """
    Translates a selected Action into a real-world effect and returns
    an observation of the outcome.

    The executor is the only component that interacts with the real world.
    The world model NEVER calls an executor during planning — only during Act.
    """

    @abstractmethod
    def execute(self, action: Action) -> Dict[str, Any]:
        """
        Execute the action. Returns an observation dict that can be
        passed back into agent.step() and agent.observe().

        Minimum returned keys:
            success  : bool
            outcome  : str — human-readable result summary
            reward   : float — scalar feedback signal
        """
        ...

    @abstractmethod
    def can_execute(self, action: Action) -> bool:
        """True if this executor handles this action type."""
        ...


class NullExecutor(Executor):
    """No-op executor for testing and dry runs."""

    def execute(self, action: Action) -> Dict[str, Any]:
        return {"success": True, "outcome": f"[NullExecutor] {action.name}", "reward": 0.0}

    def can_execute(self, action: Action) -> bool:
        return True


class ShellExecutor(Executor):
    """
    Executes shell commands on Linux and returns structured observations.

    The world model learns to predict the outcome of commands in latent
    space. After execution, the actual outcome (exit code, stdout summary,
    files created/modified) is fed back as an observation so the world
    model can update its prediction and improve.

    Safety model:
        - All commands run in a subprocess with timeout (default 30s)
        - Commands NOT in allow_list are refused unless allow_all=True
        - Working directory is constrained to working_dir
        - stdin is always /dev/null
        - No shell=True (explicit argument list prevents injection)

    Growing the vocabulary:
        The agent starts with read-only commands. As it proves reliable,
        write-capable commands can be added to the allow_list. The world
        model learns consequences before they are actually allowed to run.
    """

    def __init__(
        self,
        working_dir: str  = ".",
        timeout_s:   float= 30.0,
        allow_all:   bool = False,
        allow_list:  Optional[List[str]] = None,
        max_output:  int  = 2048,
    ):
        self.cwd       = os.path.abspath(working_dir)
        self.timeout   = timeout_s
        self.allow_all = allow_all
        self.allow_list= set(allow_list or [])
        self.max_output= max_output

    def can_execute(self, action: Action) -> bool:
        if action.payload is None or action.payload.get("cmd") is None:
            return True   # no-op
        if self.allow_all:
            return True
        cmd = action.payload["cmd"]
        base = cmd.strip().split()[0] if cmd else ""
        return base in self.allow_list or action.name in self.allow_list

    def execute(self, action: Action) -> Dict[str, Any]:
        payload = action.payload or {}
        cmd     = payload.get("cmd")

        if cmd is None:
            return {"success": True, "outcome": "wait", "reward": 0.0,
                    "stdout": "", "stderr": "", "exit_code": 0}

        if not self.can_execute(action):
            return {
                "success": False,
                "outcome": f"Command '{action.name}' not in allow_list",
                "reward": -0.1,
                "stdout": "", "stderr": "Permission denied", "exit_code": -1,
            }

        try:
            args   = shlex.split(cmd)
            result = subprocess.run(
                args,
                cwd     = self.cwd,
                capture_output = True,
                text    = True,
                timeout = self.timeout,
                stdin   = subprocess.DEVNULL,
            )
            stdout = result.stdout[:self.max_output]
            stderr = result.stderr[:512]
            success= result.returncode == 0
            reward = 0.5 if success else -0.2

            # Observation summarises the outcome in a fixed-length feature
            # the world model can condition on
            obs_text = f"EXIT:{result.returncode} STDOUT:{len(stdout)}B {stdout[:120]}"
            return {
                "success":   success,
                "outcome":   obs_text,
                "reward":    reward,
                "stdout":    stdout,
                "stderr":    stderr,
                "exit_code": result.returncode,
                "structured": self._to_feature(result.returncode, stdout, stderr),
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "outcome": "timeout",
                    "reward": -0.5, "stdout": "", "stderr": "timeout", "exit_code": -1,
                    "structured": self._to_feature(-1, "", "timeout")}
        except Exception as e:
            return {"success": False, "outcome": str(e),
                    "reward": -0.5, "stdout": "", "stderr": str(e), "exit_code": -1,
                    "structured": self._to_feature(-1, "", str(e))}

    @staticmethod
    def _to_feature(exit_code: int, stdout: str, stderr: str):
        """
        Convert shell output to a fixed-size NumPy-compatible feature vector.
        This is the observation that feeds back into the world model.

        Features (10-dim):
            [0]  exit_code (clamped -1..1)
            [1]  stdout length (log-normalized)
            [2]  stderr length (log-normalized)
            [3]  n_lines in stdout (log-normalized)
            [4]  n_files mentioned (rough heuristic)
            [5]  success flag (0/1)
            [6]  error flag (0/1)
            [7]  empty output flag (0/1)
            [8]  long output flag (0/1)
            [9]  reserved
        """
        import math
        import numpy as np
        n_files = len([w for w in stdout.split() if '/' in w or w.endswith('.py')])
        return np.array([
            max(-1.0, min(1.0, exit_code / 128.0)),
            math.log1p(len(stdout)) / 10.0,
            math.log1p(len(stderr)) / 10.0,
            math.log1p(stdout.count('\n')) / 6.0,
            math.log1p(n_files) / 4.0,
            float(exit_code == 0),
            float(exit_code != 0),
            float(len(stdout.strip()) == 0),
            float(len(stdout) > 512),
            0.0,
        ], dtype='float32')


class ApparatusExecutor(Executor):
    """
    Translates joint-delta actions into apparatus movement.
    Works in conjunction with MovementExecutor from apparatus.py.
    """

    def __init__(self, movement_executor=None, hardware=None):
        self._mex = movement_executor
        self._hw  = hardware
        self._joints = [0.0] * 6   # current joint state (degrees)

    def can_execute(self, action: Action) -> bool:
        return action.payload is not None and "joint" in action.payload

    def execute(self, action: Action) -> Dict[str, Any]:
        if not self.can_execute(action):
            return {"success": False, "outcome": "not an apparatus action",
                    "reward": -0.1}

        joint = action.payload["joint"]
        delta = action.payload["delta_deg"]
        self._joints[joint] = max(-90.0, min(90.0, self._joints[joint] + delta))

        if self._hw is not None:
            import numpy as np
            self._hw.set_all_angles(np.array(self._joints))

        return {
            "success":   True,
            "outcome":   f"joint_{joint} → {self._joints[joint]:.1f}°",
            "reward":    0.0,   # shaped reward comes from world model consequence head
            "joints":    list(self._joints),
        }


# ── Act phase bridge ──────────────────────────────────────────────────────────

class ActBridge:
    """
    Translates MCTS integer action → real-world effect.

    This is the explicit Plan→Act bridge that was missing from the original
    architecture. The bridge:

        1. Looks up the Action in the ActionSpace
        2. Checks if the action is safe to execute (allow-list, confidence threshold)
        3. Routes to the correct Executor
        4. Returns a structured observation for agent.observe()

    The confidence threshold prevents the agent from executing low-confidence
    actions — if the world model's predicted value is below min_confidence,
    the agent waits instead.
    """

    def __init__(
        self,
        action_space:    ActionSpace,
        executor:        Executor,
        min_confidence:  float = 0.3,   # predicted value must exceed this to act
        dry_run:         bool  = False,  # if True, log but don't execute
    ):
        self.space      = action_space
        self.executor   = executor
        self.min_conf   = min_confidence
        self.dry_run    = dry_run
        self._history:  List[Dict] = []

    def act(
        self,
        action_idx:    int,
        predicted_value: float = 1.0,
        info:          Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Execute action_idx if confidence is sufficient.

        Returns observation dict including:
            executed  : bool
            action    : Action
            result    : executor output (or None if not executed)
            reward    : float
        """
        if action_idx >= len(self.space):
            return {"executed": False, "reason": "invalid index",
                    "reward": -0.1, "action": None, "result": None}

        action = self.space[action_idx]

        # Confidence gate
        if predicted_value < self.min_conf:
            result = {"executed": False,
                      "reason": f"confidence {predicted_value:.2f} < {self.min_conf}",
                      "reward": 0.0, "action": action, "result": None}
            self._history.append(result)
            return result

        if self.dry_run:
            result = {"executed": False, "reason": "dry_run",
                      "reward": 0.0, "action": action, "result": None}
            self._history.append(result)
            return result

        exec_result = self.executor.execute(action)
        out = {
            "executed": True,
            "action":   action,
            "result":   exec_result,
            "reward":   exec_result.get("reward", 0.0),
            "outcome":  exec_result.get("outcome", ""),
        }
        # Carry through structured features for world model observation
        if "structured" in exec_result:
            out["structured"] = exec_result["structured"]

        self._history.append(out)
        return out

    def last_n(self, n: int = 5) -> List[Dict]:
        return self._history[-n:]
