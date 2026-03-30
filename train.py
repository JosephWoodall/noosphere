"""
noosphere/train.py
==================
Continuous Training Loop & Metrics Logger

Orchestrates the environment interaction, calls the agent updates, 
and tracks the newly introduced prosthetic alignment metrics:
- SIGReg Loss (Representation stability)
- Behavioral Cloning Loss (Digital Twin Imitation)
- Evidential Confidence (Safety Gate bounding)
- Position Error Huber Loss (Kinematic feedback)
"""

import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import torch

from noosphere.agent import NoosphereAgent
from noosphere.s4_eeg import DirichletEDLLoss

logger = logging.getLogger(__name__)

@dataclass
class TrainerConfig:
    total_steps: int = 100_000
    log_every_n_steps: int = 50
    save_every_n_steps: int = 5000
    eval_every_n_steps: int = 1000
    checkpoint_dir: str = "./checkpoints"
    env_name: str = "synthetic_bci"
    # Annealing for the Evidential Deep Learning (EDL) Intent Loss
    edl_annealing_steps: int = 5000 

class Trainer:
    """
    Main execution loop for the Obedient Consequence Engine.
    Handles experience collection, agent updating, and rich telemetry logging.
    """
    def __init__(self, agent: NoosphereAgent, env: Any, cfg: TrainerConfig):
        self.agent = agent
        self.env = env
        self.cfg = cfg
        self.device = agent.device
        
        # Loss function for explicit intent supervision (when ground truth is known)
        self.edl_loss_fn = DirichletEDLLoss(
            n_classes=agent.cfg.n_actions, 
            annealing_step=cfg.edl_annealing_steps
        )
        
        # Metric tracking windows
        self._recent_rewards = deque(maxlen=100)
        self._recent_confidences = deque(maxlen=100)
        self._recent_bc_loss = deque(maxlen=100)
        self._recent_sigreg = deque(maxlen=100)
        
        self.global_step = 0

    def run(self):
        """Standard continuous control training loop."""
        logger.info(f"Starting Noosphere Training Loop for {self.cfg.total_steps} steps.")
        
        obs = self.env.reset()
        self.agent.reset_latent()
        prev_action = None
        
        t_start = time.time()

        while self.global_step < self.cfg.total_steps:
            self.agent.train()
            
            # 1. Agent observes and selects an action (Intent + Safety Verification)
            with torch.no_grad():
                action, info = self.agent.step(obs, prev_action=prev_action)
            
            # Track Evidential Confidence (from S4 Module)
            conf = info.get("s4_confidence", 0.0)
            self._recent_confidences.append(conf)

            # 2. Environment steps forward based on the physical/digital action
            next_obs, reward, done, env_info = self.env.step(action)
            self._recent_rewards.append(reward)
            
            # 3. Store experience in ReplayBuffer & WorkingMemory
            self.agent.observe(obs, action, reward, done, info)
            
            # 4. Optional: Inject supervised intent targets if the env provides them
            # This trains the EDL intent head to map EEG -> Evidential Probabilities
            self._apply_supervised_intent(obs, env_info)

            # 5. Background updates: World Model (SIGReg) & Actor (BC)
            metrics = self.agent.update()
            self._track_update_metrics(metrics)

            if self.global_step % self.cfg.log_every_n_steps == 0:
                self._log_progress(t_start, metrics)

            if self.global_step > 0 and self.global_step % self.cfg.save_every_n_steps == 0:
                self._save_checkpoint()

            obs = self.env.reset() if done else next_obs
            prev_action = action
            self.global_step += 1
            
        logger.info("Training complete.")

    def _apply_supervised_intent(self, obs: Dict, env_info: Dict):
        """
        If the environment knows the 'true' user intent (e.g., during calibration 
        or synthetic data generation), we explicitly penalize the S4 head using EDL.
        """
        if "true_intent" in env_info and "eeg" in obs:
            true_intent = env_info["true_intent"] # integer class index
            
            target_one_hot = torch.zeros(1, self.agent.cfg.n_actions, device=self.device)
            target_one_hot[0, true_intent] = 1.0

            # Forward pass through perception only to get Dirichlet alphas
            eeg_tensor = torch.tensor(obs["eeg"], device=self.device).unsqueeze(0)
            mask = torch.ones(3, device=self.device)
            
            perc_out = self.agent.perception({"eeg": eeg_tensor, "electrode_mask": mask})
            s4_out = perc_out["s4_out"]
            
            alpha = s4_out["alpha"]
            
            # Compute Sum of Squares Evidential Loss
            loss_edl = self.edl_loss_fn(alpha, target_one_hot, self.global_step)
            
            # Backpropagate through the S4 encoder
            self.agent.opt_wm.zero_grad()
            loss_edl.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.perception.parameters(), self.agent.cfg.grad_clip)
            self.agent.opt_wm.step()

    def _track_update_metrics(self, metrics: Dict[str, float]):
        if "wm/sigreg" in metrics:
            self._recent_sigreg.append(metrics["wm/sigreg"])
        if "ac/bc_loss" in metrics:
            self._recent_bc_loss.append(metrics["ac/bc_loss"])

    def _log_progress(self, t_start: float, latest_metrics: Dict[str, float]):
        fps = self.global_step / (time.time() - t_start)
        avg_reward = np.mean(self._recent_rewards) if self._recent_rewards else 0.0
        avg_conf = np.mean(self._recent_confidences) if self._recent_confidences else 0.0
        avg_bc = np.mean(self._recent_bc_loss) if self._recent_bc_loss else 0.0
        avg_sigreg = np.mean(self._recent_sigreg) if self._recent_sigreg else 0.0

        log_str = (
            f"Step: {self.global_step:06d} | FPS: {fps:.1f} | "
            f"Reward: {avg_reward:+.2f} | "
            f"S4_Conf: {avg_conf:.2f} | "
        )
        
        # Highlight the new alignment metrics
        align_str = f"[SIGReg: {avg_sigreg:.3f} | BC_Loss: {avg_bc:.3f}]"
        
        # Add basic WM metrics if available
        wm_kl = latest_metrics.get("wm/kl", 0.0)
        wm_phys = latest_metrics.get("wm/physics", 0.0)
        
        logger.info(f"{log_str} {align_str} | KL: {wm_kl:.2f} | Phys: {wm_phys:.3f}")

    def _save_checkpoint(self):
        path = f"{self.cfg.checkpoint_dir}/noosphere_step_{self.global_step}.pt"
        logger.info(f"Exporting World Model Bundle to {path}...")
        
        # Utilizing the export_bundle logic introduced in v1.6.0
        train_metrics = {
            "avg_reward": np.mean(self._recent_rewards) if self._recent_rewards else 0.0,
            "avg_confidence": np.mean(self._recent_confidences) if self._recent_confidences else 0.0,
            "bc_loss": np.mean(self._recent_bc_loss) if self._recent_bc_loss else 0.0,
        }
        
        self.agent.export_bundle(
            path=path,
            description=f"Automated checkpoint at step {self.global_step}",
            n_training_steps=self.global_step,
            train_metrics=train_metrics
        )