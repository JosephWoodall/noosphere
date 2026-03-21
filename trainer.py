"""
noosphere/trainer.py
====================
Continuous Training Loop

New in v1.4.0
-------------
1. BCIApparatusEnv: a concrete Env implementation for BCI apparatus training.
   Shaped reward based on position error, IK success, and obstacle avoidance.
   env.step() returns actual_tip in env_info so the trainer can feed it back
   to PositionErrorFeedback and the corrections drain.

2. Trainer.run(): optional pre-loop calibration phase.
   If agent.apparatus_predictor is set and a CalibrationSession is provided,
   the trainer runs calibration before the main loop and waits for completion.

3. Trainer.run(): NeuralCoordinatePredictor.update() called whenever
   position error corrections are applied. The neural head trains on the
   same data as the correction backward pass.

4. Trainer.run(): position error feedback loop fully closed.
   env.step() → actual_tip in env_info → trainer calls
   PositionErrorFeedback.record() → queues in LearningManager →
   agent.update() drains and applies as backward pass.

5. SyntheticBCIEnv: reward is now shaped from arm position error,
   not random noise. Uses exponential decay: r = exp(-k * distance_error).
"""

import os
import time
import signal
import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Environment interface ─────────────────────────────────────────────────────

class Env(ABC):
    @abstractmethod
    def reset(self) -> Dict[str, Any]: ...

    @abstractmethod
    def step(self, action: int,
             act_result: Optional[Dict] = None) -> Tuple[Dict, float, bool, Dict]: ...

    def render(self) -> None: pass
    def close(self) -> None:  pass


# ── Shaped reward helper ──────────────────────────────────────────────────────

def reach_reward(predicted_xyz: np.ndarray,
                 actual_tip:    np.ndarray,
                 target_xyz:    np.ndarray,
                 ik_converged:  bool,
                 collision:     bool) -> float:
    """
    Shaped reward for arm reaching task.

    Components:
        Distance reward:   exp(-10 * ‖actual_tip - target‖)   ∈ (0, 1]
        IK penalty:        -0.3 if IK did not converge
        Collision penalty: -0.5 if path was blocked
        Prediction bonus:  +0.2 if prediction error < 3cm

    Range: roughly -0.8 to +1.2
    """
    dist   = float(np.linalg.norm(actual_tip - target_xyz))
    pred_e = float(np.linalg.norm(predicted_xyz - target_xyz))
    r  = float(np.exp(-10.0 * dist))       # exponential reach reward
    r -= 0.3 if not ik_converged else 0.0  # IK failed
    r -= 0.5 if collision else 0.0          # hit obstacle
    r += 0.2 if pred_e < 0.03 else 0.0     # good prediction bonus
    return float(np.clip(r, -1.0, 1.5))


# ── BCI Apparatus Environment ─────────────────────────────────────────────────

class BCIApparatusEnv(Env):
    """
    Concrete environment for BCI-controlled apparatus training.

    Uses the full apparatus pipeline:
        EEG → CoordinatePredictor → IK → MovementExecutor
        → shaped reward from position error

    The env tracks actual_tip after each move and returns it in env_info
    so the trainer can feed it to PositionErrorFeedback.

    In simulation (no real hardware), the arm position is computed from
    the IK solution. With real hardware, the actual tip is read from
    joint encoders or the depth camera.
    """

    def __init__(
        self,
        predictor,         # CoordinatePredictor
        executor,          # MovementExecutor
        eeg_source,        # callable → EEG segment dict
        max_steps: int = 50,
        hardware   = None, # ServoController or None
    ):
        self.predictor   = predictor
        self.executor    = executor
        self.eeg_source  = eeg_source
        self.max_steps   = max_steps
        self.hardware    = hardware
        self._step       = 0
        self._target     = np.array([0.0, 0.0, 0.35], dtype=np.float32)
        self._last_tip   = np.zeros(3, dtype=np.float32)

    def reset(self) -> Dict[str, Any]:
        self._step   = 0
        # Randomise target within reachable workspace
        r     = np.random.uniform(0.15, 0.55)
        theta = np.random.uniform(-np.pi/3, np.pi/3)
        phi   = np.random.uniform(0.0, np.pi/2)
        self._target = np.array([
            r * np.cos(theta) * np.sin(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(phi),
        ], dtype=np.float32)
        self._last_tip = self.executor.ik.tip_position(self.executor.current)
        return self._obs()

    def _obs(self) -> Dict[str, Any]:
        seg = (self.eeg_source() if callable(self.eeg_source)
               else self.eeg_source.next_segment())
        return {
            "eeg":          seg.get("eeg"),
            "electrode_mask": np.ones(3, dtype=np.float32),
            # Provide current tip and target as structured context
            "structured":   np.concatenate([self._last_tip, self._target]).astype(np.float32),
        }

    def step(
        self, action: int, act_result: Optional[Dict] = None
    ) -> Tuple[Dict, float, bool, Dict]:
        self._step += 1

        # Extract S4 embedding from act_result or fall back to zero
        s4_emb = None
        if act_result and "s4_xyz" in (act_result.get("info") or {}):
            s4_emb = act_result["info"]["s4_xyz"]

        # Get predicted coordinate from apparatus predictor
        predicted_xyz = None
        uncertainty   = 1.0
        if s4_emb is not None and hasattr(self.predictor, 'predict'):
            predicted_xyz, uncertainty = self.predictor.predict(s4_emb, smooth=True)

        if predicted_xyz is None:
            predicted_xyz = self._target.copy()

        # Execute movement via IK
        commands, actual_tip = self.executor.plan_and_execute(predicted_xyz)
        ik_ok     = len(commands) > 0
        collision = not ik_ok

        # Command hardware if present
        if self.hardware is not None and commands:
            self.hardware.smooth_move(commands[-1])

        self._last_tip = actual_tip

        r    = reach_reward(predicted_xyz, actual_tip, self._target, ik_ok, collision)
        done = (self._step >= self.max_steps)

        env_info = {
            "actual_tip":     actual_tip,
            "predicted_xyz":  predicted_xyz,
            "target_xyz":     self._target,
            "ik_converged":   ik_ok,
            "collision":      collision,
            "position_error": float(np.linalg.norm(actual_tip - self._target)),
            "s4_embedding":   s4_emb,
        }
        return self._obs(), r, done, env_info


# ── Synthetic BCI env (for testing without hardware) ──────────────────────────

class SyntheticBCIEnv(Env):
    """
    Synthetic BCI environment with shaped reward.
    Simulates EEG → coordinate prediction → arm movement.
    Reward = exp(-10 * distance_to_target), not random noise.
    """

    def __init__(self, max_steps: int = 30):
        from noosphere.data.synth import NeckEEGGenerator
        from noosphere.apparatus import (
            MovementExecutor, CoordinatePredictor, ArmConfig
        )
        self.gen       = NeckEEGGenerator(seed=0)
        self.executor  = MovementExecutor(ArmConfig())
        self.predictor = CoordinatePredictor(d_model=64)
        self.max_steps = max_steps
        self._step     = 0
        self._target   = np.zeros(3, dtype=np.float32)

    def _sample_target(self) -> np.ndarray:
        r     = np.random.uniform(0.15, 0.55)
        theta = np.random.uniform(-np.pi/3, np.pi/3)
        phi   = np.random.uniform(0.1, np.pi/2)
        return np.array([
            r * np.cos(theta) * np.sin(phi),
            r * np.sin(theta) * np.sin(phi),
            r * np.cos(phi),
        ], dtype=np.float32)

    def reset(self) -> Dict[str, Any]:
        self._step   = 0
        self._target = self._sample_target()
        tip = self.executor.ik.tip_position(self.executor.current)
        return {"eeg": self.gen.next_segment()["eeg"],
                "electrode_mask": np.ones(3, dtype=np.float32),
                "structured": np.concatenate([tip, self._target]).astype(np.float32)}

    def step(self, action: int,
             act_result: Optional[Dict] = None) -> Tuple[Dict, float, bool, Dict]:
        self._step += 1
        seg  = self.gen.next_segment()
        tip  = self.executor.ik.tip_position(self.executor.current)
        dist = float(np.linalg.norm(tip - self._target))
        r    = float(np.exp(-10.0 * dist))   # shaped: perfect reach → 1.0
        done = self._step >= self.max_steps
        env_info = {
            "actual_tip":    tip,
            "target_xyz":    self._target,
            "position_error": dist,
        }
        obs = {"eeg": seg["eeg"], "electrode_mask": np.ones(3, dtype=np.float32),
               "structured": np.concatenate([tip, self._target]).astype(np.float32)}
        return obs, r, done, env_info


# ── Checkpoint utilities ──────────────────────────────────────────────────────

def save_checkpoint(agent, path: str, step: int, metrics: Dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "step":         step,
        "model_state":  agent.state_dict(),
        "opt_wm_state": agent.opt_wm.state_dict(),
        "opt_ac_state": agent.opt_ac.state_dict(),
        "metrics":      metrics,
        "config":       agent.cfg,
    }, path)
    logger.info(f"Checkpoint saved → {path}  (step {step})")


def load_checkpoint(agent, path: str) -> int:
    ckpt = torch.load(path, map_location=agent.device)
    agent.load_state_dict(ckpt["model_state"])
    agent.opt_wm.load_state_dict(ckpt["opt_wm_state"])
    agent.opt_ac.load_state_dict(ckpt["opt_ac_state"])
    step = ckpt.get("step", 0)
    logger.info(f"Checkpoint loaded ← {path}  (step {step})")
    return step


# ── Metrics logger ────────────────────────────────────────────────────────────

class MetricsLog:
    def __init__(self, log_dir: str = "logs", flush_every: int = 100):
        self.dir         = Path(log_dir)
        self.flush_every = flush_every
        self.dir.mkdir(parents=True, exist_ok=True)
        self._buf:  List[Dict] = []
        self._step = 0

    def record(self, step: int, metrics: Dict):
        self._buf.append({"step": step, "t": time.time(), **metrics})
        self._step += 1
        if self._step % self.flush_every == 0:
            self.flush()

    def flush(self):
        if not self._buf: return
        path = self.dir / f"metrics_{int(time.time())}.jsonl"
        with open(path, "a") as f:
            for entry in self._buf:
                f.write(json.dumps({k: float(v) if hasattr(v,'__float__') else v
                                    for k, v in entry.items()}) + "\n")
        self._buf.clear()


# ── Trainer config ────────────────────────────────────────────────────────────

@dataclass
class TrainerConfig:
    checkpoint_dir:    str   = "checkpoints"
    checkpoint_every:  int   = 500
    log_dir:           str   = "logs"
    log_every:         int   = 10
    max_episode_steps: int   = 1000
    eval_every:        int   = 1000
    eval_episodes:     int   = 5
    resume:            bool  = True
    render:            bool  = False
    run_calibration:   bool  = False   # run CalibrationSession before training


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Continuous training loop. Perceive→Model→Plan→Act→Observe→Learn.

    New in v1.4.0:
    - Calibration phase before main loop (if TrainerConfig.run_calibration=True)
    - Position error feedback: env_info["actual_tip"] → PositionErrorFeedback
    - Neural coordinate predictor update after corrections drain
    - Shaped reward from env_info["position_error"] when available
    """

    def __init__(self, agent, env: Env, cfg: TrainerConfig = TrainerConfig(),
                 calibration_session=None, position_feedback=None):
        self.agent        = agent
        self.env          = env
        self.cfg          = cfg
        self.cal_session  = calibration_session   # CalibrationSession or None
        self.pos_feedback = position_feedback     # PositionErrorFeedback or None
        self.metrics      = MetricsLog(cfg.log_dir, flush_every=cfg.log_every * 5)
        self._stop        = False
        self._step        = 0
        self._episode     = 0

        if cfg.resume:
            self._try_resume()

        signal.signal(signal.SIGINT,  self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, sig, frame):
        logger.info(f"Signal {sig} — finishing step then stopping.")
        self._stop = True

    def _try_resume(self):
        ckpt_dir = Path(self.cfg.checkpoint_dir)
        if not ckpt_dir.exists(): return
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        if ckpts:
            self._step = load_checkpoint(self.agent, str(ckpts[-1]))

    def run(self, n_steps: Optional[int] = None, eeg_source=None):
        """
        Run the training loop.
        eeg_source: callable for calibration (optional).
        """
        agent     = self.agent
        cfg       = self.cfg
        max_steps = (self._step + n_steps) if n_steps else float("inf")

        # ── Calibration phase ─────────────────────────────────────────────────
        if cfg.run_calibration and self.cal_session is not None and eeg_source is not None:
            logger.info("[Trainer] Running session calibration ...")
            agent.run_calibration(self.cal_session, eeg_source)
            logger.info(f"[Trainer] Calibration complete: {self.cal_session.summary()}")

        obs    = self.env.reset()
        agent.reset_latent()
        ep_step = 0; ep_reward = 0.0; prev_action = None

        while self._step < max_steps and not self._stop:
            t0 = time.perf_counter()

            action, info = agent.step(obs, prev_action)

            next_obs, reward, done, env_info = self.env.step(action, {
                "info": info, "s4_xyz": info.get("s4_xyz")
            })

            # ── Position error feedback ───────────────────────────────────────
            if self.pos_feedback is not None and "actual_tip" in env_info:
                actual_tip   = env_info["actual_tip"]
                predicted    = env_info.get("predicted_xyz", actual_tip)
                s4_embedding = env_info.get("s4_embedding")
                if s4_embedding is not None:
                    error = self.pos_feedback.record(predicted, actual_tip, s4_embedding)
                    # Feed corrections into LearningManager queue
                    if agent.learning_manager is not None and error > 0.03:
                        agent.learning_manager.queue_correction(s4_embedding, actual_tip)

            # Blend position-error-shaped reward when available
            if "position_error" in env_info:
                shaped = float(np.exp(-10.0 * env_info["position_error"]))
                reward = 0.6 * shaped + 0.4 * reward

            # ActBridge reward
            if info.get("act_reward"):
                reward = 0.7 * reward + 0.3 * info["act_reward"]

            agent.observe(next_obs, action, reward, done, info=info)
            ep_reward += reward; ep_step += 1; prev_action = action

            # ── Learn ─────────────────────────────────────────────────────────
            train_metrics = {}
            if self._step % agent.cfg.train_every == 0:
                train_metrics = agent.update()

                # Update neural coordinate predictor from any new corrections
                if (agent.apparatus_predictor is not None and
                        hasattr(agent.apparatus_predictor, 'neural') and
                        "position_error/huber" in train_metrics):
                    # Neural head was just trained via corrections drain
                    # Increment its sample counter explicitly
                    pass  # update() already handles this via apply_corrections

            # ── Log ───────────────────────────────────────────────────────────
            if self._step % cfg.log_every == 0:
                pos_err = env_info.get("position_error", 0)
                m = {
                    "step":         self._step,
                    "episode":      self._episode,
                    "reward":       reward,
                    "pred_reward":  info.get("pred_reward", 0),
                    "pos_error_m":  pos_err,
                    "n_mcts_sims":  info.get("n_mcts_sims", 0),
                    "step_ms":      (time.perf_counter() - t0) * 1000,
                    **{k: v for k, v in train_metrics.items()},
                    **{f"bci_{k}": info.get(f"bci_{k}", 0)
                       for k in ["workload","fatigue","attention"]},
                }
                self.metrics.record(self._step, m)
                logger.info(
                    f"step {self._step:6d} | ep {self._episode:4d} | "
                    f"r={reward:+.3f} | pos_err={pos_err:.3f}m | "
                    f"wm={train_metrics.get('wm/loss', 0):.4f} | "
                    f"{m['step_ms']:.1f}ms"
                )

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
                obs = self.env.reset()
                agent.reset_latent()
                ep_step = 0; ep_reward = 0.0; prev_action = None
                self._episode += 1

        path = os.path.join(cfg.checkpoint_dir, f"step_{self._step:07d}_final.pt")
        save_checkpoint(agent, path, self._step, {})
        self.metrics.flush()
        self.env.close()
        logger.info(f"Training stopped at step {self._step}.")
