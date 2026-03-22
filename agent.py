"""
noosphere/agent.py
==================
Noosphere Agent — v1.4.0

Wired in this version
---------------------
1. Physics conservation loss: observe_step now returns a tensor (not float),
   so gradient flows through the conservation penalties to the residual
   corrector and state estimator.

2. GNN sparsity loss: gnn_out["sparsity_loss"] is added to the world model
   training loss, driving learned adjacency toward sparse physical topology.

3. Episodic memory read: before MCTS search, the agent retrieves the top-K
   most similar past states from episodic memory and uses their stored values
   to bias the root prior (optimistic initialisation from past experience).

4. Working memory → budget: recent reward trend supplements the BCI cognitive
   load signal. If recent rewards are consistently negative (failures), the
   agent increases simulation budget rather than reducing it.

5. Correction drain: position error corrections queued in the LearningManager
   are applied as an additional supervised backward pass after each Phase A
   world model update.

6. s4_confidence → ActBridge: the S4 encoder's calibrated uncertainty is
   passed to ActBridge.act() so the dual confidence gate works correctly.

7. Calibration: agent exposes run_calibration() so the trainer can call it
   before the main loop when an apparatus predictor is attached.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from noosphere.actions import ActBridge, ActionSpace, Executor, NullExecutor
from noosphere.bundle import BundleMetadata, export_bundle, inspect_bundle, load_bundle
from noosphere.learning import LearningConfig, LearningManager, S4XYZSupervisionLoss
from noosphere.memory import EpisodicMemory, SequenceReplayBuffer, WorkingMemory
from noosphere.perception import HybridPerceptionModel
from noosphere.physics import PhysicsAugmentedRSSM
from noosphere.planner import (
    ActionEncoder,
    Actor,
    Critic,
    ImaginationBuffer,
    MCTSPlanner,
)
from noosphere.rssm import ConsequenceModel, ObservationDecoder

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    d_model: int = 256
    det_dim: int = 512
    stoch_cats: int = 32
    stoch_cls: int = 32
    action_dim: int = 64
    hidden_dim: int = 256

    n_eeg_ch: int = 3
    eeg_sfreq: float = 256.0
    n_nodes: int = 20
    node_feat_dim: int = 12
    patch_size: int = 8

    n_bodies: int = 4
    fluid_grid: int = 4
    dt: float = 1 / 60

    n_actions: int = 8
    use_mcts: bool = True
    n_mcts_sims: int = 30
    mcts_horizon: int = 10
    imag_horizon: int = 15

    batch_size: int = 16
    seq_len: int = 50
    lr_perception: float = 1e-4
    lr_world_model: float = 3e-4
    lr_actor_critic: float = 3e-4
    grad_clip: float = 100.0
    gamma: float = 0.99
    lam: float = 0.95
    lambda_kl: float = 1.0
    lambda_recon: float = 0.5
    lambda_reward: float = 1.0
    lambda_physics: float = 0.5
    lambda_xyz: float = 2.0
    lambda_gnn_sparse: float = 0.01
    entropy_scale: float = 3e-4
    free_nats: float = 1.0
    wm_updates: int = 5
    ac_updates: int = 5
    train_every: int = 10
    warmup_steps: int = 1000

    task_type: str = "multiclass"
    min_act_confidence: float = 0.3
    dry_run: bool = False

    replay_capacity: int = 500
    episodic_capacity: int = 5000


# ── Observation preprocessor ──────────────────────────────────────────────────


class _Prep:
    def __init__(self, device: torch.device):
        self.dev = device

    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        out = {}

        def _t(k):
            if obs.get(k) is None:
                return
            x = np.array(obs[k], dtype=np.float32)
            if k in ("rgb", "depth", "rgb_right"):
                if x.max() > 1.0 and k != "depth":
                    x /= 255.0
                if x.ndim == 3:
                    x = x.transpose(2, 0, 1)
                x = x[None]
            elif k == "eeg":
                if x.ndim == 2:
                    x = x[None]
            elif k in ("structured", "kinematics"):
                if x.ndim == 1:
                    x = x[None, None]
                elif x.ndim == 2:
                    x = x[None]
            out[k] = torch.tensor(x, device=self.dev)

        for k in ["rgb", "depth", "rgb_right", "eeg", "structured", "kinematics"]:
            _t(k)
        if "electrode_mask" in obs and obs["electrode_mask"] is not None:
            out["electrode_mask"] = torch.tensor(
                np.array(obs["electrode_mask"], dtype=np.float32)[None], device=self.dev
            )
        return out


# ── Agent ─────────────────────────────────────────────────────────────────────


class NoosphereAgent(nn.Module):
    def __init__(self, cfg: AgentConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        C = cfg

        self.perception = HybridPerceptionModel(
            d_model=C.d_model,
            n_heads=8,
            n_layers=6,
            n_eeg_channels=C.n_eeg_ch,
            s4_d_state=64,
            s4_n_blocks=4,
            s4_downsample=4,
            n_kinematic_nodes=C.n_nodes,
            node_feature_dim=C.node_feat_dim,
            gnn_n_layers=3,
            patch_size=C.patch_size,
        )
        self.rssm = PhysicsAugmentedRSSM(
            embed_dim=C.d_model,
            action_dim=C.action_dim,
            n_bodies=C.n_bodies,
            G=C.fluid_grid,
            det_dim=C.det_dim,
            stoch_cats=C.stoch_cats,
            stoch_classes=C.stoch_cls,
            hidden_dim=C.hidden_dim,
            dt=C.dt,
        )
        state_dim = self.rssm.state_dim

        self.consequence = ConsequenceModel(state_dim, C.hidden_dim)
        self.obs_decoder = ObservationDecoder(state_dim, C.d_model, C.hidden_dim)
        self.action_enc = ActionEncoder(C.n_actions, C.action_dim)
        self.s4_xyz_loss = S4XYZSupervisionLoss(delta=0.05, max_reach=0.70)

        self.actor = Actor(state_dim, C.n_actions, C.hidden_dim)
        self.critic = Critic(state_dim, C.hidden_dim)
        if C.use_mcts:
            self.planner = MCTSPlanner(
                self.rssm.rssm,
                self.consequence,
                self.actor,
                self.action_enc,
                C.n_actions,
                C.n_mcts_sims,
                C.mcts_horizon,
                C.gamma,
                device=device,
            )
        self._imag_buf = ImaginationBuffer(C.gamma, C.lam)

        self.replay = SequenceReplayBuffer(C.replay_capacity, C.seq_len)
        self.episodic = EpisodicMemory(state_dim, 64, C.episodic_capacity)
        self.working = WorkingMemory(20)

        # Optional apparatus pipeline attachment
        self.act_bridge: Optional[ActBridge] = None
        self.learning_manager: Optional[LearningManager] = None
        self.apparatus_predictor: Any = None  # CoordinatePredictor

        self._wm_params = (
            list(self.perception.parameters())
            + list(self.rssm.parameters())
            + list(self.consequence.parameters())
            + list(self.obs_decoder.parameters())
            + list(self.action_enc.parameters())
        )
        self._ac_params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt_wm = optim.AdamW(
            [
                {"params": list(self.perception.parameters()), "lr": C.lr_perception},
                {
                    "params": list(
                        p
                        for m in [
                            self.rssm,
                            self.consequence,
                            self.obs_decoder,
                            self.action_enc,
                        ]
                        for p in m.parameters()
                    ),
                    "lr": C.lr_world_model,
                },
            ],
            eps=1e-8,
        )
        self.opt_ac = optim.AdamW(self._ac_params, lr=C.lr_actor_critic, eps=1e-8)

        self._h: Optional[torch.Tensor] = None
        self._z: Optional[torch.Tensor] = None
        self._step = 0
        self._prep = _Prep(device)
        self.to(device)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _det_dim(self) -> int:
        return self.rssm.rssm.det_dim

    @property
    def _stoch_dim(self) -> int:
        return self.rssm.rssm.stoch_dim

    def reset_latent(self):
        init = self.rssm.initial_state(1, self.device)
        self._h = init["h"]
        self._z = init["z"]
        self.rssm.reset_episode()

    @torch.no_grad()
    def _encode_obs(self, obs: Dict) -> Tuple[torch.Tensor, Dict]:
        tensors = self._prep(obs)
        if not tensors:
            return torch.zeros(1, self.cfg.d_model, device=self.device), {}
        out = self.perception(tensors)
        return out["embed"], out

    # ── Step ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(
        self,
        obs: Dict,
        prev_action: Optional[int] = None,
        deterministic: bool = False,
    ) -> Tuple[int, Dict]:
        obs_embed, perc_out = self._encode_obs(obs)
        s4_out = perc_out.get("s4_out")
        gnn_out = perc_out.get("gnn_out")

        a_embed = (
            torch.zeros(1, self.cfg.action_dim, device=self.device)
            if prev_action is None
            else self.action_enc(torch.tensor([prev_action], device=self.device))
        )

        if self._h is None:
            self.reset_latent()

        self._h, self._z, pp, qp, ps, phys_tensor, phys_log = self.rssm.observe_step(
            self._h, self._z, a_embed, obs_embed
        )

        state = torch.cat([self._h, self._z], -1)
        cons = self.consequence(state)

        # ── Planning budget ───────────────────────────────────────────────────
        n_sims = self.cfg.n_mcts_sims

        # BCI cognitive load scaling
        if s4_out is not None:
            budget = s4_out["planning_budget"].mean().item()
            n_sims = max(5, int(n_sims * budget))

        # Working memory failure correction:
        # if recent rewards are consistently negative, increase search budget
        recent_r = self.working.recent_rewards(10)
        if recent_r:
            avg_r = float(np.mean(recent_r))
            if avg_r < -0.1:  # failing — search harder
                n_sims = min(n_sims * 2, self.cfg.n_mcts_sims * 3)

        # ── Episodic context → MCTS prior bias ───────────────────────────────
        # Retrieve top-K similar past states; use their stored values to
        # nudge the actor's prior before MCTS search begins.
        episodic_value_bonus: Optional[torch.Tensor] = None
        try:
            ep_vals, ep_attn = self.episodic.read(state)
            # Weighted average of past values → scalar bias
            ep_bonus = (ep_vals[:, 0] * ep_attn).sum()  # first dim is value
            episodic_value_bonus = ep_bonus.detach()
        except Exception:
            pass

        # ── Action selection ──────────────────────────────────────────────────
        if self.cfg.use_mcts and not deterministic:
            # Apply episodic value bonus to the consequence model value for root
            # by temporarily biasing the planner's evaluation horizon
            if episodic_value_bonus is not None and episodic_value_bonus.abs() > 0.01:
                # Patch planner's consequence model value with episodic bonus
                self.planner.episodic_bonus = episodic_value_bonus.item()
            else:
                self.planner.episodic_bonus = 0.0
            self.planner.n_simulations = n_sims
            action = self.planner.search(self._h, self._z)
        else:
            action = int(self.actor.act(state, deterministic).item())

        info = {
            "pred_reward": cons["reward"].item(),
            "pred_value": cons["value"].item(),
            "termination_prob": cons["termination"].item(),
            "physics_energy": ps.energy.mean().item(),
            "n_mcts_sims": n_sims,
        }
        if s4_out is not None:
            cog = s4_out["cognitive"]
            info.update({f"bci_{k}": v.mean().item() for k, v in cog.items()})
            if "continuous_xyz" in s4_out:
                info["s4_xyz"] = s4_out["continuous_xyz"][0].cpu().numpy()
                info["s4_confidence"] = s4_out["confidence"][0].item()

        # ── Act bridge — now passes s4_confidence for dual gate ───────────────
        if self.act_bridge is not None:
            act_result = self.act_bridge.act(
                action,
                predicted_value=cons["value"].item(),
                s4_confidence=info.get("s4_confidence"),
                info=info,
            )
            info["act_executed"] = act_result["executed"]
            info["act_outcome"] = act_result.get("outcome", "")
            info["act_reward"] = act_result.get("reward", 0.0)
            if "structured" in act_result:
                info["_exec_structured"] = act_result["structured"]

        self._step += 1
        return action, info

    # ── Observe ───────────────────────────────────────────────────────────────

    def observe(
        self,
        obs: Dict,
        action: int,
        reward: float,
        done: bool,
        info: Optional[Dict] = None,
    ):
        if info and "_exec_structured" in info:
            obs = dict(obs)
            obs["structured"] = info["_exec_structured"]

        self.replay.add_step(obs, action, reward, done)
        self.working.push(np.zeros(1), action, reward)

        if self._h is not None and self._step % 10 == 0:
            state = torch.cat([self._h, self._z], -1)
            with torch.no_grad():
                val = self.consequence(state)["value"].unsqueeze(-1)
                val_padded = F.pad(val, (0, self.episodic.values.shape[-1] - 1))
            self.episodic.write(state, val_padded)

        if done:
            self.reset_latent()
            self.working.clear()

    # ── Apply corrections (position error feedback) ───────────────────────────

    def apply_corrections(self, corrections: List[Dict]):
        """
        Apply position error corrections from PositionErrorFeedback.drain().
        Each correction is a dict with "embedding" (numpy) and "actual_tip" (numpy).
        Runs a supervised backward pass on the S4 xyz head.
        Called by the trainer after env.step() returns actual_tip.
        """
        if not corrections or self.learning_manager is None:
            return {}
        import torch

        embeddings = torch.tensor(
            np.stack([c["embedding"] for c in corrections]),
            dtype=torch.float32,
            device=self.device,
        )
        actual_tips = torch.tensor(
            np.stack([c["actual_tip"] for c in corrections]),
            dtype=torch.float32,
            device=self.device,
        )
        # Re-encode through the current perception model to get fresh gradients
        # (embeddings from apparatus were numpy snapshots at step time)
        # Use them as supervision targets on the S4 xyz head directly
        predicted_xyz = (
            self.perception.s4.xyz_head(embeddings) * self.perception.s4.max_reach
        )
        loss, metrics = self.learning_manager.compute_position_error_loss(
            predicted_xyz, actual_tips
        )
        self.opt_wm.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._wm_params, self.cfg.grad_clip)
        self.opt_wm.step()
        return metrics

    # ── Calibration ───────────────────────────────────────────────────────────

    def run_calibration(self, calibration_session, eeg_source) -> bool:
        """
        Run session-start calibration.
        calibration_session: CalibrationSession instance
        eeg_source: callable() → EEG segment dict with "s4_embedding" key,
                    or a NeckEEGGenerator for synthetic runs

        Returns True when all reference movements are collected.
        """
        for name, target, prompt in calibration_session.MOVEMENTS:
            logger.info(f"[Calibration] {prompt} and hold ...")
            seg = eeg_source() if callable(eeg_source) else eeg_source.next_segment()
            # Extract S4 embedding from the segment
            if "s4_embedding" in seg:
                embedding = np.array(seg["s4_embedding"], dtype=np.float32)
            else:
                # Fall back: encode through S4 encoder
                import torch

                eeg_t = torch.tensor(
                    seg["eeg"][None], dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    s4_out = self.perception.s4(eeg_t)
                embedding = s4_out["summary"][0].cpu().numpy()
            calibration_session.add_movement(name, embedding, target)
        return calibration_session.complete

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self) -> Dict[str, float]:
        metrics = {}
        if len(self.replay) < self.cfg.batch_size:
            return metrics
        for _ in range(self.cfg.wm_updates):
            m = self._update_wm()
            metrics.update(m)
        if self._step >= self.cfg.warmup_steps:
            for _ in range(self.cfg.ac_updates):
                m = self._update_ac()
                metrics.update(m)
        # Drain and apply any queued position error corrections
        if self.learning_manager is not None:
            corrections = self.learning_manager.drain_corrections()
            if corrections:
                m = self.apply_corrections(corrections)
                metrics.update(m)
        return metrics

    def _update_wm(self) -> Dict[str, float]:
        C = self.cfg
        batch = self.replay.sample(C.batch_size, self.device)
        if not batch or "actions" not in batch:
            return {}

        B, T = batch["actions"].shape
        T_eff = min(T, 20)

        obs_embed_list = []
        s4_xyz_preds = []
        gnn_sparse = torch.tensor(0.0, device=self.device)

        for t in range(T_eff):
            inp = {}
            for mod in ("eeg", "kinematics", "structured", "rgb", "depth"):
                if mod in batch:
                    inp[mod] = batch[mod][:, t]
            if not inp:
                obs_embed_list.append(torch.zeros(B, C.d_model, device=self.device))
                continue
            with torch.set_grad_enabled(True):
                perc_out = self.perception(inp)
                obs_embed_list.append(perc_out["embed"])
                s4 = perc_out.get("s4_out")
                if s4 is not None and "continuous_xyz" in s4:
                    s4_xyz_preds.append(s4["continuous_xyz"])
                # Accumulate GNN sparsity loss
                gnn = perc_out.get("gnn_out")
                if gnn is not None and "sparsity_loss" in gnn:
                    gnn_sparse = gnn_sparse + gnn["sparsity_loss"]

        h = torch.zeros(B, self._det_dim, device=self.device)
        z = torch.zeros(B, self._stoch_dim, device=self.device)

        L_kl = torch.tensor(0.0, device=self.device)
        L_r = torch.tensor(0.0, device=self.device)
        L_rew = torch.tensor(0.0, device=self.device)
        L_t = torch.tensor(0.0, device=self.device)
        L_p = torch.tensor(0.0, device=self.device)

        for t in range(T_eff):
            e = obs_embed_list[t]
            a = self.action_enc(batch["actions"][:, t])
            # physics.observe_step now returns tensor loss (7th value)
            h, z, pp, qp, ps, phys_tensor, phys_log = self.rssm.observe_step(h, z, a, e)
            s = torch.cat([h, z], -1)
            cons = self.consequence(s)

            L_kl = L_kl + self.rssm.kl_loss(pp, qp, C.free_nats)
            L_r = L_r + F.mse_loss(self.obs_decoder(s), e.detach())
            L_rew = L_rew + torch.nan_to_num(
                F.mse_loss(cons["reward"], batch["rewards"][:, t]), nan=0.0
            )
            # Clamp termination to [0,1] before BCE — NaN/out-of-range values
            # can appear when obs_embed is a zero-padded fallback tensor
            term_clamped = cons["termination"].clamp(0.0, 1.0)
            term_clamped = torch.nan_to_num(term_clamped, nan=0.5)
            L_t = L_t + F.binary_cross_entropy(term_clamped, batch["dones"][:, t])
            # Physics loss now stays in graph — gradient flows to residual corrector
            L_p = L_p + torch.nan_to_num(phys_tensor, nan=0.0)

        L_xyz = torch.tensor(0.0, device=self.device)
        if s4_xyz_preds and "kinematics" in batch:
            xyz_pred = torch.stack(s4_xyz_preds[:T_eff]).mean(0)
            xyz_label = batch["kinematics"][:, 0, :3]
            L_xyz, _ = self.s4_xyz_loss(xyz_pred, xyz_label)
            L_xyz = C.lambda_xyz * L_xyz

        f = float(T_eff)
        loss = (
            C.lambda_kl * L_kl
            + C.lambda_recon * L_r
            + C.lambda_reward * L_rew
            + L_t
            + C.lambda_physics * L_p
            + C.lambda_gnn_sparse * gnn_sparse
            + L_xyz
        ) / f

        self.opt_wm.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._wm_params, C.grad_clip)
        self.opt_wm.step()

        return {
            "wm/loss": loss.item(),
            "wm/kl": (L_kl / f).item(),
            "wm/reward_pred": (L_rew / f).item(),
            "wm/physics": (L_p / f).item(),
            "wm/gnn_sparse": (gnn_sparse / f).item(),
            "wm/xyz": L_xyz.item(),
        }

    def _update_ac(self) -> Dict[str, float]:
        C = self.cfg
        B = C.batch_size
        H = C.imag_horizon

        h = torch.zeros(B, self._det_dim, device=self.device)
        z = torch.zeros(B, self._stoch_dim, device=self.device)
        self._imag_buf.clear()

        for _ in range(H):
            s = torch.cat([h, z], -1)
            dist = self.actor(s)
            a = dist.sample()
            lp = dist.log_prob(a)
            v = self.critic.min_value(s)
            with torch.no_grad():
                h2, z2, _ = self.rssm.rssm.imagine_step(h, z, self.action_enc(a))
                s2 = torch.cat([h2, z2], -1)
                cons = self.consequence(s2)
            self._imag_buf.add(
                s, a, cons["reward"], v, lp, (cons["termination"] > 0.5).float()
            )
            h, z = h2.detach(), z2.detach()

        G = self._imag_buf.lambda_returns()
        st = torch.stack(self._imag_buf.states)
        lp = torch.stack(self._imag_buf.log_probs)

        v1, v2 = self.critic(st.detach().view(-1, st.shape[-1]))
        L_v = F.mse_loss(v1.view(H, B), G) + F.mse_loss(v2.view(H, B), G)
        A = (G - G.mean()) / (G.std() + 1e-8)
        L_p = -(lp * A.detach()).mean()
        ent = self.actor.entropy(st.view(-1, st.shape[-1]).detach()).mean()
        loss = L_p - C.entropy_scale * ent + L_v

        self.opt_ac.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._ac_params, C.grad_clip)
        self.opt_ac.step()

        return {
            "ac/actor": L_p.item(),
            "ac/critic": L_v.item(),
            "ac/entropy": ent.item(),
            "ac/return": G.mean().item(),
        }

    # ── Bundle convenience methods ────────────────────────────────────────────

    def export_bundle(
        self,
        path: str,
        domain_tags: Optional[List[str]] = None,
        description: str = "",
        author: str = "",
        n_training_steps: int = 0,
        train_metrics: Optional[Dict[str, float]] = None,
    ):
        """Export transferable world dynamics. Personal components excluded."""
        meta = BundleMetadata(
            domain_tags=domain_tags or [],
            description=description,
            author=author,
            n_training_steps=n_training_steps or self._step,
        )
        return export_bundle(self, path, meta, train_metrics)

    def load_bundle(
        self,
        path: str,
        strict_arch: bool = True,
        modules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Load community bundle into world dynamics. Personal components untouched."""
        return load_bundle(self, path, strict_arch, modules)
