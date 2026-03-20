"""
noosphere/trainer.py
====================
Continuous Training Loop

Runs the Perceive→Model→Plan→Act→Observe→Learn cycle continuously,
persisting checkpoints and updating all three learning phases.

The trainer is intentionally domain-agnostic. The same loop drives:
    - Physical apparatus control (EEG → motor commands)
    - Digital task execution (EEG → shell commands)
    - Any environment that implements the Env interface

Usage
-----
    from noosphere.trainer import Trainer, TrainerConfig, Env

    class MyEnv(Env):
        def reset(self): return initial_obs
        def step(self, action, act_result=None): return obs, reward, done

    trainer = Trainer(agent, MyEnv(), TrainerConfig())
    trainer.run()          # runs forever, Ctrl-C to stop
    trainer.run(n_steps=1000)   # fixed budget
"""

import os
import time
import signal
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Environment interface ─────────────────────────────────────────────────────

class Env(ABC):
    """
    Minimal environment interface.

    Implement reset() and step() for any physical or digital task.
    The reward signal is the only supervision the world model requires.
    """

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Return initial observation dict."""
        ...

    @abstractmethod
    def step(
        self,
        action:     int,
        act_result: Optional[Dict] = None,
    ) -> tuple:
        """
        Apply action. act_result carries executor output if an ActBridge ran.

        Returns (obs, reward, done, info).
        obs    : dict of sensor arrays
        reward : float
        done   : bool
        info   : dict (optional metadata)
        """
        ...

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass


# ── Checkpoint utilities ──────────────────────────────────────────────────────

def save_checkpoint(agent, path: str, step: int, metrics: Dict):
    """Save agent weights, optimizer states, and training metrics."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":          step,
        "model_state":   agent.state_dict(),
        "opt_wm_state":  agent.opt_wm.state_dict(),
        "opt_ac_state":  agent.opt_ac.state_dict(),
        "metrics":       metrics,
        "config":        agent.cfg,
    }, path)
    logger.info(f"Checkpoint saved → {path}  (step {step})")


def load_checkpoint(agent, path: str) -> int:
    """Load checkpoint. Returns the step at which it was saved."""
    ckpt = torch.load(path, map_location=agent.device)
    agent.load_state_dict(ckpt["model_state"])
    agent.opt_wm.load_state_dict(ckpt["opt_wm_state"])
    agent.opt_ac.load_state_dict(ckpt["opt_ac_state"])
    step = ckpt.get("step", 0)
    logger.info(f"Checkpoint loaded ← {path}  (step {step})")
    return step


# ── Metrics logger ────────────────────────────────────────────────────────────

class MetricsLog:
    """Rolling metrics with periodic JSON flush."""

    def __init__(self, log_dir: str = "logs", flush_every: int = 100):
        self.dir        = Path(log_dir)
        self.flush_every= flush_every
        self.dir.mkdir(parents=True, exist_ok=True)
        self._buf: List[Dict] = []
        self._step = 0

    def record(self, step: int, metrics: Dict):
        entry = {"step": step, "t": time.time(), **metrics}
        self._buf.append(entry)
        self._step += 1
        if self._step % self.flush_every == 0:
            self.flush()

    def flush(self):
        if not self._buf: return
        path = self.dir / f"metrics_{int(time.time())}.jsonl"
        with open(path, "a") as f:
            for entry in self._buf:
                f.write(json.dumps(entry) + "\n")
        self._buf.clear()


# ── Trainer config ────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    checkpoint_dir:   str   = "checkpoints"
    checkpoint_every: int   = 500    # steps between saves
    log_dir:          str   = "logs"
    log_every:        int   = 10     # steps between console logs
    max_episode_steps:int   = 1000   # max steps per episode before forced reset
    eval_every:       int   = 1000   # steps between eval runs (0 = disabled)
    eval_episodes:    int   = 5
    resume:           bool  = True   # auto-resume from latest checkpoint if found
    render:           bool  = False


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Continuous training loop.

    The loop:
        1. Reset environment → initial observation
        2. Loop:
            a. Perceive: encode obs → latent
            b. Plan:     MCTS → action index
            c. Act:      ActBridge → real-world command
            d. Observe:  env.step() → next obs, reward, done
            e. Learn:    replay.add_step(), periodic update()
        3. On done: log episode, save checkpoint if due

    Training signal sources:
        - World model: Phase A on replay buffer sequences
        - Policy:      Phase B on imagined rollouts (TD-λ or π-StepNFT)
        - Contrastive: Phase C on EEG stream (when EEG is present)

    Continuous training means the model never stops learning.
    Episodes are not barriers — the replay buffer accumulates across
    all episodes and the world model trains on the entire experience.
    """

    def __init__(self, agent, env: Env, cfg: TrainerConfig = TrainerConfig()):
        self.agent   = agent
        self.env     = env
        self.cfg     = cfg
        self.metrics = MetricsLog(cfg.log_dir, flush_every=cfg.log_every * 5)
        self._stop   = False
        self._step   = 0
        self._episode= 0

        # Resume from latest checkpoint if available
        if cfg.resume:
            self._try_resume()

        # Graceful shutdown on SIGINT / SIGTERM
        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, sig, frame):
        logger.info(f"Signal {sig} received — finishing current step then stopping.")
        self._stop = True

    def _try_resume(self):
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        if not ckpt_dir.exists(): return
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        if ckpts:
            self._step = load_checkpoint(self.agent, str(ckpts[-1]))

    def run(self, n_steps: Optional[int] = None):
        """
        Run the training loop.
        n_steps=None runs until Ctrl-C. n_steps=N runs exactly N agent steps.
        """
        agent = self.agent
        cfg   = self.cfg
        max_steps = (self._step + n_steps) if n_steps else float("inf")

        obs    = self.env.reset()
        agent.reset_latent()
        ep_step= 0; ep_reward = 0.0; prev_action = None

        while self._step < max_steps and not self._stop:
            t0 = time.perf_counter()

            # ── Perceive + Plan ───────────────────────────────────────────────
            action, info = agent.step(obs, prev_action)

            # ── Act ───────────────────────────────────────────────────────────
            act_result = info.get("act_result")  # from ActBridge if attached

            # ── Observe ───────────────────────────────────────────────────────
            next_obs, reward, done, env_info = self.env.step(action, act_result)

            # If ActBridge provided a reward, blend it with env reward
            if info.get("act_reward"):
                reward = 0.7 * reward + 0.3 * info["act_reward"]

            agent.observe(next_obs, action, reward, done, info=info)

            ep_reward  += reward
            ep_step    += 1
            prev_action = action

            # ── Learn ─────────────────────────────────────────────────────────
            train_metrics = {}
            if self._step % agent.cfg.train_every == 0:
                train_metrics = agent.update()

            # ── Log ───────────────────────────────────────────────────────────
            if self._step % cfg.log_every == 0:
                m = {
                    "step":          self._step,
                    "episode":       self._episode,
                    "reward":        reward,
                    "pred_reward":   info.get("pred_reward", 0),
                    "physics_E":     info.get("physics_energy", 0),
                    "n_mcts_sims":   info.get("n_mcts_sims", 0),
                    "step_ms":       (time.perf_counter() - t0) * 1000,
                    **{k: v for k, v in train_metrics.items()},
                    **{f"bci_{k}": info.get(f"bci_{k}", 0)
                       for k in ["workload","fatigue","attention"]},
                }
                self.metrics.record(self._step, m)
                if cfg.log_every > 0:
                    wl  = m.get("bci_workload", 0)
                    fat = m.get("bci_fatigue", 0)
                    wm  = m.get("wm/loss", 0)
                    logger.info(
                        f"step {self._step:6d} | ep {self._episode:4d} | "
                        f"r={reward:+.3f} | pred_r={m['pred_reward']:+.3f} | "
                        f"wl={wl:.2f} fat={fat:.2f} | "
                        f"wm_loss={wm:.4f} | {m['step_ms']:.1f}ms"
                    )

            # ── Checkpoint ────────────────────────────────────────────────────
            if self._step % cfg.checkpoint_every == 0 and self._step > 0:
                path = os.path.join(cfg.checkpoint_dir, f"step_{self._step:07d}.pt")
                save_checkpoint(agent, path, self._step, train_metrics)

            obs = next_obs
            self._step += 1

            if done or ep_step >= cfg.max_episode_steps:
                logger.info(
                    f"Episode {self._episode} done — "
                    f"{ep_step} steps, total_reward={ep_reward:.3f}"
                )
                obs    = self.env.reset()
                agent.reset_latent()
                ep_step = 0; ep_reward = 0.0; prev_action = None
                self._episode += 1

        # Final checkpoint on stop
        path = os.path.join(cfg.checkpoint_dir, f"step_{self._step:07d}_final.pt")
        save_checkpoint(agent, path, self._step, {})
        self.metrics.flush()
        self.env.close()
        logger.info(f"Training stopped at step {self._step}.")
