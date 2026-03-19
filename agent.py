"""
noosphere/agent.py
==================
Noosphere Agent

Closes the perception → world model → planning → action → observation loop.

    step(obs)   encode → RSSM observe → MCTS/actor plan → return action
    observe()   store experience → episodic memory
    update()    Phase A: world model training
                Phase B: actor-critic training in imagination

Training alternates between two phases:

    Phase A  —  world model
        Minimises L_WM = λ_KL·L_KL + λ_r·L_recon + λ_reward·L_reward
                       + L_term + λ_phys·L_physics
        on sequences from the replay buffer.
        Updates: perception, RSSM, physics, consequence model.

    Phase B  —  actor-critic (world model frozen)
        Rolls out actor for H steps in imagination.
        Computes TD(λ) returns, updates actor + critic only.
        World model receives no gradients in this phase.

The 1000-step warmup trains only Phase A before Phase B begins,
ensuring the world model is functional before the actor exploits it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass

from noosphere.perception import HybridPerceptionModel
from noosphere.physics    import PhysicsAugmentedRSSM
from noosphere.rssm       import ConsequenceModel, ObservationDecoder
from noosphere.planner    import Actor, Critic, ActionEncoder, MCTSPlanner, ImaginationBuffer
from noosphere.memory     import SequenceReplayBuffer, EpisodicMemory, WorkingMemory

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    # Dimensions
    d_model:    int = 256
    det_dim:    int = 512
    stoch_cats: int = 32
    stoch_cls:  int = 32
    action_dim: int = 64
    hidden_dim: int = 256

    # Perception
    n_eeg_ch:      int   = 64
    eeg_sfreq:     float = 256.0
    n_nodes:       int   = 20
    node_feat_dim: int   = 12
    patch_size:    int   = 8

    # Physics
    n_bodies:    int   = 4
    fluid_grid:  int   = 4
    dt:          float = 1/60

    # Planning
    n_actions:   int  = 8
    use_mcts:    bool = True
    n_mcts_sims: int  = 30
    mcts_horizon:int  = 10
    imag_horizon:int  = 15

    # Training
    batch_size:         int   = 16
    seq_len:            int   = 50
    lr_perception:      float = 1e-4
    lr_world_model:     float = 3e-4
    lr_actor_critic:    float = 3e-4
    grad_clip:          float = 100.0
    gamma:              float = 0.99
    lam:                float = 0.95
    lambda_kl:          float = 1.0
    lambda_recon:       float = 0.5
    lambda_reward:      float = 1.0
    lambda_physics:     float = 0.5
    entropy_scale:      float = 3e-4
    free_nats:          float = 1.0
    wm_updates:         int   = 5
    ac_updates:         int   = 5
    train_every:        int   = 10
    warmup_steps:       int   = 1000

    # Memory
    replay_capacity:    int   = 500
    episodic_capacity:  int   = 5000


# ── Observation preprocessor ──────────────────────────────────────────────────

class _Prep:
    def __init__(self, device: torch.device):
        self.dev = device

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        out = {}
        def _t(k, shape_fn):
            if obs.get(k) is None: return
            x = np.array(obs[k], dtype=np.float32)
            if k == "rgb" or k == "depth" or k == "rgb_right":
                if x.max() > 1.0 and k != "depth": x /= 255.0
                if x.ndim == 3: x = x.transpose(2,0,1)
                x = x[None]
            elif k == "eeg":
                if x.ndim == 2: x = x[None]
            elif k == "structured" or k == "kinematics":
                if x.ndim == 1: x = x[None, None]
                elif x.ndim == 2: x = x[None]
            out[k] = torch.tensor(x, device=self.dev)
        for k in ["rgb","depth","rgb_right","eeg","structured","kinematics"]:
            _t(k, None)
        if "electrode_mask" in obs and obs["electrode_mask"] is not None:
            out["electrode_mask"] = torch.tensor(
                np.array(obs["electrode_mask"], dtype=np.float32)[None], device=self.dev)
        return out


# ── Agent ──────────────────────────────────────────────────────────────────────

class NoosphereAgent(nn.Module):
    """
    Full Noosphere agent.

    Extend by:
        — registering new sensor modalities via agent.perception.tokenizer.register_modality()
        — subclassing and overriding _encode_obs() for custom preprocessing
        — implementing a domain-specific environment and passing it to AgentTrainer
    """

    def __init__(self, cfg: AgentConfig, device: torch.device):
        super().__init__()
        self.cfg    = cfg
        self.device = device
        C = cfg

        # Perception
        self.perception = HybridPerceptionModel(
            d_model=C.d_model, n_heads=8, n_layers=6,
            n_eeg_channels=C.n_eeg_ch,
            s4_d_state=64, s4_n_blocks=4, s4_downsample=4,
            n_kinematic_nodes=C.n_nodes, node_feature_dim=C.node_feat_dim,
            gnn_n_layers=3, patch_size=C.patch_size,
        )

        # World model
        self.rssm = PhysicsAugmentedRSSM(
            embed_dim=C.d_model, action_dim=C.action_dim,
            n_bodies=C.n_bodies, G=C.fluid_grid,
            det_dim=C.det_dim, stoch_cats=C.stoch_cats, stoch_classes=C.stoch_cls,
            hidden_dim=C.hidden_dim, dt=C.dt,
        )
        state_dim = self.rssm.state_dim

        self.consequence  = ConsequenceModel(state_dim, C.hidden_dim)
        self.obs_decoder  = ObservationDecoder(state_dim, C.d_model, C.hidden_dim)
        self.action_enc   = ActionEncoder(C.n_actions, C.action_dim)

        # Planning
        self.actor  = Actor(state_dim, C.n_actions, C.hidden_dim)
        self.critic = Critic(state_dim, C.hidden_dim)
        if C.use_mcts:
            self.planner = MCTSPlanner(
                self.rssm.rssm, self.consequence, self.actor, self.action_enc,
                C.n_actions, C.n_mcts_sims, C.mcts_horizon, C.gamma, device=device,
            )

        # Memory
        self.replay   = SequenceReplayBuffer(C.replay_capacity, C.seq_len)
        self.episodic = EpisodicMemory(state_dim, 64, C.episodic_capacity)
        self.working  = WorkingMemory(20)

        # Optimizers — store param lists explicitly so clip_grad_norm_
        # in each phase targets only that phase's parameters.
        self._wm_params = (
            list(self.perception.parameters()) +
            list(self.rssm.parameters()) +
            list(self.consequence.parameters()) +
            list(self.obs_decoder.parameters()) +
            list(self.action_enc.parameters())
        )
        self._ac_params = (
            list(self.actor.parameters()) +
            list(self.critic.parameters())
        )
        self.opt_wm = optim.AdamW([
            {"params": self.perception.parameters(), "lr": C.lr_perception},
            {"params": [p for m in [self.rssm, self.consequence,
                                     self.obs_decoder, self.action_enc]
                        for p in m.parameters()], "lr": C.lr_world_model},
        ], eps=1e-8)
        self.opt_ac = optim.AdamW(self._ac_params, lr=C.lr_actor_critic, eps=1e-8)

        # State
        self._h: Optional[torch.Tensor] = None
        self._z: Optional[torch.Tensor] = None
        self._step = 0
        self._prep = _Prep(device)
        self.to(device)

    # ── Latent state ──────────────────────────────────────────────────────────

    def reset_latent(self):
        init = self.rssm.initial_state(1, self.device)
        self._h = init["h"]; self._z = init["z"]
        self.rssm.reset_episode()

    # ── Perception ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_obs(self, obs: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Encode observation once, return (embed, full_perception_output).
        Avoids redundant forward passes — perception is the most expensive call.
        """
        tensors = self._prep(obs)
        if not tensors:
            return torch.zeros(1, self.cfg.d_model, device=self.device), {}
        out = self.perception(tensors)
        return out["embed"], out

    # ── Step ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(
        self,
        obs:          Dict,
        prev_action:  Optional[int] = None,
        deterministic:bool          = False,
    ) -> Tuple[int, Dict]:
        """
        Run one agent step.

        Returns (action, info_dict).
        info_dict contains predicted reward, value, termination probability,
        BCI cognitive state (if EEG present), and physics energy.
        """
        # Single perception forward pass — result cached for BCI budget and info
        obs_embed, perc_out = self._encode_obs(obs)
        s4_out = perc_out.get("s4_out")

        a_embed = (torch.zeros(1, self.cfg.action_dim, device=self.device)
                   if prev_action is None
                   else self.action_enc(torch.tensor([prev_action], device=self.device)))

        if self._h is None: self.reset_latent()

        self._h, self._z, pp, qp, ps, phys_losses = self.rssm.observe_step(
            self._h, self._z, a_embed, obs_embed)

        state = torch.cat([self._h, self._z], -1)
        cons  = self.consequence(state)

        # BCI cognitive budget (from S4 output — no extra forward pass needed)
        n_sims = self.cfg.n_mcts_sims
        if s4_out is not None:
            budget = s4_out["planning_budget"].mean().item()
            n_sims = max(5, int(n_sims * budget))

        # Plan
        if self.cfg.use_mcts and not deterministic:
            self.planner.n_simulations = n_sims
            action = self.planner.search(self._h, self._z)
        else:
            action = int(self.actor.act(state, deterministic).item())

        info = {
            "pred_reward":     cons["reward"].item(),
            "pred_value":      cons["value"].item(),
            "termination_prob":cons["termination"].item(),
            "physics_energy":  ps.energy.mean().item(),
            "n_mcts_sims":     n_sims,
        }
        if s4_out is not None:
            cog = s4_out["cognitive"]
            info.update({f"bci_{k}": v.mean().item() for k, v in cog.items()})

        self._step += 1
        return action, info

    # ── Observe ───────────────────────────────────────────────────────────────

    def observe(self, obs: Dict, action: int, reward: float, done: bool):
        self.replay.add_step(obs, action, reward, done)
        self.working.push(np.zeros(1), action, reward)
        if self._h is not None and self._step % 10 == 0:
            state = torch.cat([self._h, self._z], -1)
            val   = torch.zeros(1, 64, device=self.device)
            self.episodic.write(state, val)
        if done:
            self.reset_latent()
            self.working.clear()

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        metrics = {}
        if len(self.replay) < self.cfg.batch_size: return metrics
        for _ in range(self.cfg.wm_updates):
            m = self._update_wm(); metrics.update(m)
        if self._step >= self.cfg.warmup_steps:
            for _ in range(self.cfg.ac_updates):
                m = self._update_ac(); metrics.update(m)
        return metrics

    def _update_wm(self) -> Dict[str, float]:
        C     = self.cfg
        batch = self.replay.sample(C.batch_size, self.device)
        if not batch or "actions" not in batch: return {}
        B, T  = batch["actions"].shape
        T_eff = min(T, 20)

        # Pre-encode all timesteps in one batched call per modality.
        # This is significantly faster than calling perception T times in the loop.
        obs_embeds = torch.zeros(B, T_eff, C.d_model, device=self.device)
        for t in range(T_eff):
            for mod in ("structured", "eeg", "kinematics", "rgb", "depth"):
                if mod in batch:
                    try:
                        inp = {mod: batch[mod][:, t]}
                        obs_embeds[:, t] = self.perception(inp)["embed"]
                        break
                    except Exception:
                        pass

        h = torch.zeros(B, self.rssm.rssm.det_dim,   device=self.device)
        z = torch.zeros(B, self.rssm.rssm.stoch_dim,  device=self.device)
        L_kl = L_r = L_rew = L_t = L_p = torch.tensor(0., device=self.device)

        for t in range(T_eff):
            e = obs_embeds[:, t]
            a = self.action_enc(batch["actions"][:, t])
            h, z, pp, qp, ps, pl = self.rssm.observe_step(h, z, a, e)
            s    = torch.cat([h, z], -1)
            cons = self.consequence(s)
            L_kl  = L_kl  + self.rssm.kl_loss(pp, qp, C.free_nats)
            L_r   = L_r   + F.mse_loss(self.obs_decoder(s), e.detach())
            L_rew = L_rew + F.mse_loss(cons["reward"], batch["rewards"][:, t])
            L_t   = L_t   + F.binary_cross_entropy(cons["termination"], batch["dones"][:, t])
            if "physics/total_loss" in pl:
                L_p = L_p + pl["physics/total_loss"]

        f    = float(T_eff)
        loss = (C.lambda_kl*L_kl + C.lambda_recon*L_r +
                C.lambda_reward*L_rew + L_t + C.lambda_physics*L_p) / f
        self.opt_wm.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self._wm_params, C.grad_clip)
        self.opt_wm.step()
        return {"wm/loss": loss.item(), "wm/kl": (L_kl/f).item(),
                "wm/reward_pred": (L_rew/f).item(), "wm/physics": (L_p/f).item()}

    def _update_ac(self) -> Dict[str, float]:
        C  = self.cfg
        B  = C.batch_size; H = C.imag_horizon
        h  = torch.zeros(B, self.rssm.rssm.det_dim,  device=self.device)
        z  = torch.zeros(B, self.rssm.rssm.stoch_dim, device=self.device)
        buf = ImaginationBuffer(C.gamma, C.lam)
        for _ in range(H):
            s    = torch.cat([h, z], -1)
            dist = self.actor(s)
            a    = dist.sample()
            lp   = dist.log_prob(a)
            v    = self.critic.min_value(s)
            with torch.no_grad():
                h2, z2, _ = self.rssm.rssm.imagine_step(h, z, self.action_enc(a))
                s2   = torch.cat([h2, z2], -1)
                cons = self.consequence(s2)
            buf.add(s, a, cons["reward"], v, lp, (cons["termination"] > 0.5).float())
            h, z = h2.detach(), z2.detach()
        G  = buf.lambda_returns().to(self.device)
        st = torch.stack(buf.states)
        lp = torch.stack(buf.log_probs)
        v1, v2 = self.critic(st.detach().view(-1, st.shape[-1]))
        L_v = F.mse_loss(v1.view(H,B), G) + F.mse_loss(v2.view(H,B), G)
        A   = (G - G.mean()) / (G.std() + 1e-8)
        L_p = -(lp * A.detach()).mean()
        ent = self.actor(st.view(-1, st.shape[-1]).detach()).entropy().mean()
        loss = L_p - C.entropy_scale * ent + L_v
        self.opt_ac.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self._ac_params, C.grad_clip)
        self.opt_ac.step()
        return {"ac/actor": L_p.item(), "ac/critic": L_v.item(),
                "ac/entropy": ent.item(), "ac/return": G.mean().item()}
