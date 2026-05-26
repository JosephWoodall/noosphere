"""
noosphere/agent.py
==================
Noosphere Agent

Features:
- Intent Momentum: Resolves the "Dead-Man's Switch" bug by gracefully decaying 
  biological intent over ~300ms, bridging transient EEG signal gaps.
- Differentiable Kinematic Clamp: Prevents imagination teleportation during AC updates.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from noosphere.actions import ActBridge
from noosphere.bundle import BundleMetadata, export_bundle, load_bundle
from noosphere.learning import LearningConfig, LearningManager, SpatialTopologyLoss
from noosphere.memory import EpisodicMemory, SequenceReplayBuffer, WorkingMemory
from noosphere.perception import HybridPerceptionModel
from noosphere.physics import PhysicsAugmentedRSSM
from noosphere.planner import ActionEncoder, Actor, Critic, ImaginationBuffer, MCTSPlanner
from noosphere.rssm import ConsequenceModel, ObservationDecoder

logger = logging.getLogger(__name__)

NO_OP_ACTION_INDEX = 0 

@dataclass
class AgentConfig:
    d_model: int = 256; det_dim: int = 512; stoch_cats: int = 32; stoch_cls: int = 32
    action_dim: int = 64; hidden_dim: int = 256; n_eeg_ch: int = 3; eeg_sfreq: float = 256.0
    n_nodes: int = 20; node_feat_dim: int = 12; patch_size: int = 8; n_bodies: int = 4
    fluid_grid: int = 4; dt: float = 1 / 60; n_actions: int = 8; use_mcts: bool = True
    n_mcts_sims: int = 30; mcts_horizon: int = 10; imag_horizon: int = 15
    batch_size: int = 16; seq_len: int = 50; lr_perception: float = 1e-4
    lr_world_model: float = 3e-4; lr_actor_critic: float = 3e-4; grad_clip: float = 100.0
    gamma: float = 0.99; lam: float = 0.95; lambda_kl: float = 1.0; lambda_recon: float = 0.5
    lambda_reward: float = 1.0; lambda_physics: float = 0.5; lambda_gnn_sparse: float = 0.01
    entropy_scale: float = 3e-4; bc_weight: float = 2.0; min_act_confidence: float = 0.3
    fast_path_threshold: float = 0.85; max_reach: float = 0.70; max_velocity: float = 1.0
    wm_updates: int = 5; ac_updates: int = 5; warmup_steps: int = 1000; dry_run: bool = False
    replay_capacity: int = 500; episodic_capacity: int = 5000

class _Prep:
    def __init__(self, device: torch.device): self.dev = device
    def __call__(self, obs: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        out = {}
        for k in ["rgb", "depth", "rgb_right", "eeg", "structured", "kinematics"]:
            if obs.get(k) is not None:
                x = np.asarray(obs[k], dtype=np.float32)
                if k in ("rgb", "depth", "rgb_right"):
                    if x.max() > 1.0 and k != "depth": x /= 255.0
                    if x.ndim == 2: x = np.expand_dims(x, axis=-1)
                    if x.ndim == 3: x = x.transpose(2, 0, 1)
                    x = x[None]
                elif k == "eeg" and x.ndim == 2: x = x[None]
                elif k in ("structured", "kinematics"):
                    if x.ndim == 1: x = x[None, None]
                    elif x.ndim == 2: x = x[None]
                out[k] = torch.tensor(x, device=self.dev)
        if "electrode_mask" in obs and obs["electrode_mask"] is not None:
            out["electrode_mask"] = torch.tensor(np.array(obs["electrode_mask"], dtype=np.float32)[None], device=self.dev)
        return out

class NoosphereAgent(nn.Module):
    def __init__(self, cfg: AgentConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg; self.device = device; C = cfg

        self.perception = HybridPerceptionModel(
            d_model=C.d_model, n_heads=8, n_layers=6, n_eeg_channels=C.n_eeg_ch, s4_d_state=64, s4_n_blocks=4, s4_downsample=4, 
            n_kinematic_nodes=C.n_nodes, node_feature_dim=C.node_feat_dim, gnn_n_layers=3, patch_size=C.patch_size, max_reach=C.max_reach
        )
        self.rssm = PhysicsAugmentedRSSM(
            embed_dim=C.d_model, action_dim=C.action_dim, n_bodies=C.n_bodies, G=C.fluid_grid,
            det_dim=C.det_dim, stoch_cats=C.stoch_cats, stoch_classes=C.stoch_cls, hidden_dim=C.hidden_dim, dt=C.dt,
        )
        state_dim = self.rssm.state_dim
        
        self.consequence = ConsequenceModel(state_dim, C.hidden_dim)
        self.obs_decoder = ObservationDecoder(state_dim, C.d_model, C.hidden_dim)
        self.action_enc = ActionEncoder(C.n_actions, C.action_dim)
        self.actor = Actor(state_dim, C.n_actions, C.hidden_dim, max_reach=C.max_reach)
        self.critic = Critic(state_dim, C.hidden_dim)
        
        if C.use_mcts:
            self.planner = MCTSPlanner(
                self.rssm.rssm, self.consequence, self.actor, self.action_enc, C.n_actions, 
                C.n_mcts_sims, C.mcts_horizon, C.gamma, max_vel=C.max_velocity, dt=C.dt, device=device
            )
            
        self._imag_buf = ImaginationBuffer(C.gamma, C.lam)
        self.replay = SequenceReplayBuffer(C.replay_capacity, C.seq_len)
        self.episodic = EpisodicMemory(state_dim, 64, C.episodic_capacity)
        self.working = WorkingMemory(20)
        self.act_bridge: Optional[ActBridge] = None
        self.learning_manager: Optional[LearningManager] = None
        self.apparatus_predictor: Any = None 

        self._wm_params = list(self.perception.parameters()) + list(self.rssm.parameters()) + list(self.consequence.parameters()) + list(self.obs_decoder.parameters()) + list(self.action_enc.parameters())
        self._ac_params = list(self.actor.parameters()) + list(self.critic.parameters())
        
        self.opt_wm = optim.AdamW([
            {"params": list(self.perception.parameters()), "lr": C.lr_perception},
            {"params": list(p for m in [self.rssm, self.consequence, self.obs_decoder, self.action_enc] for p in m.parameters()), "lr": C.lr_world_model},
        ], eps=1e-8)
        self.opt_ac = optim.AdamW(self._ac_params, lr=C.lr_actor_critic, eps=1e-8)

        self._h = None; self._z = None; self._step = 0; self._prep = _Prep(device); self.to(device)
        
        # SOTA INFERENCE COMPRESSION
        if hasattr(torch, "compile") and device.type == "cuda":
            import os
            if os.name != 'nt': # compile unsupported on Windows
                logger.info("Initializing TorchInductor JIT Compilation for latency eradication...")
                self.perception = torch.compile(self.perception, mode="reduce-overhead")
                self.rssm = torch.compile(self.rssm, mode="reduce-overhead")
                self.actor = torch.compile(self.actor, mode="reduce-overhead")
                self.action_enc = torch.compile(self.action_enc, mode="reduce-overhead")
                self.consequence = torch.compile(self.consequence, mode="reduce-overhead")
        
        # State tracking for Intent Momentum
        self._bci_momentum = 0.0
        self._last_bci_discrete = None
        self._last_bci_cont = None

    @property
    def _det_dim(self) -> int: return self.rssm.rssm.det_dim
    @property
    def _stoch_dim(self) -> int: return self.rssm.rssm.stoch_dim
    def reset_latent(self):
        init = self.rssm.initial_state(1, self.device)
        self._h, self._z = init["h"], init["z"]
        self.rssm.reset_episode()
        self._bci_momentum = 0.0

    @torch.no_grad()
    def _encode_obs(self, obs: Dict) -> Tuple[torch.Tensor, Dict]:
        tensors = self._prep(obs)
        if not tensors: return torch.zeros(1, self.cfg.d_model, device=self.device), {}
        return self.perception(tensors)["embed"], self.perception(tensors)

    @torch.no_grad()
    def step(self, obs: Dict, prev_action: Optional[int] = None, prev_cont_exec: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[int, np.ndarray, Dict]:
        # Latency Eradication: Mixed precision FP16 blocks mathematical bloat
        autocast_dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        if hasattr(torch.compiler, "cudagraph_mark_step_begin"): torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self.device.type in ["cuda", "mps"]):
            obs_embed, perc_out = self._encode_obs(obs)
            s4_out = perc_out.get("s4_out")

            prev_act_t = torch.tensor([prev_action if prev_action is not None else 0], device=self.device)
            prev_cont_t = torch.tensor(prev_cont_exec, device=self.device).unsqueeze(0) if prev_cont_exec is not None else torch.zeros(1, 3, device=self.device)
            a_embed = self.action_enc(prev_act_t, prev_cont_t) if prev_action is not None else torch.zeros(1, self.cfg.action_dim, device=self.device)

            if self._h is None: self.reset_latent()
            self._h, self._z, pp, qp, ps, phys_tensor, phys_log = self.rssm.observe_step(self._h, self._z, a_embed, obs_embed)
            state = torch.cat([self._h, self._z], -1)

            raw_confidence = s4_out["confidence"][0].item() if s4_out is not None else 0.0

        # INTENT MOMENTUM: Decay gracefully to bridge transient EEG gaps (~333ms at 60Hz)
        self._bci_momentum = max(0.0, self._bci_momentum - 0.05) 
        
        if raw_confidence >= self.cfg.min_act_confidence:
            self._bci_momentum = raw_confidence
            self._last_bci_discrete = s4_out["intent_probs"][0]
            self._last_bci_cont = perc_out["xyz_pred"][0] if "xyz_pred" in perc_out else None

        bci_active = self._bci_momentum > 0.0
        effective_confidence = max(raw_confidence, self._bci_momentum)
        
        p_bci = self._last_bci_discrete if bci_active else None
        cont_bci = self._last_bci_cont if bci_active else None

        n_sims = 0
        fast_path_active = False

        if effective_confidence >= self.cfg.fast_path_threshold:
            action = int(p_bci.argmax().item())
            p_final = p_bci
            cont_final = cont_bci
            fast_path_active = True
        elif not bci_active and not self.cfg.use_mcts:
            action = NO_OP_ACTION_INDEX
            p_final = torch.zeros(self.cfg.n_actions, device=self.device)
            p_final[NO_OP_ACTION_INDEX] = 1.0
            _, actor_cont = self.actor(state)
            cont_final = actor_cont.squeeze(0)
        else:
            uncertainty = 1.0 - effective_confidence
            
            # HAMILTONIAN COGNITIVE DYNAMICS
            hamiltonian_budget = 1.0
            if s4_out is not None and "cognitive_state" in s4_out:
                cog = s4_out["cognitive_state"]
                # Mathematical energy mapping: E = 0.5(W^2) + F
                W = float(cog.get("workload", 0.0))
                F = float(cog.get("fatigue", 0.0))
                E = 0.5 * (W ** 2) + F
                hamiltonian_budget = max(0.1, 1.0 - E)
                
            n_sims = max(2, int(self.cfg.n_mcts_sims * hamiltonian_budget * uncertainty))
            
            if self.cfg.use_mcts and not deterministic:
                self.planner.n_simulations = n_sims
                action, p_final, cont_final = self.planner.search(self._h, self._z, bci_discrete=p_bci, bci_continuous=cont_bci)
            else:
                actor_dist, actor_cont = self.actor(state)
                p_final = p_bci if bci_active else actor_dist.probs.squeeze(0)
                cont_final = cont_bci if bci_active else actor_cont.squeeze(0)
                action = int(p_final.argmax().item()) if deterministic else int(torch.distributions.Categorical(probs=p_final).sample().item())

            a_test_emb = self.action_enc(torch.tensor([action], device=self.device), cont_final.unsqueeze(0))
            h2, z2, _ = self.rssm.imagine_step(self._h, self._z, a_test_emb)
            s2 = torch.cat([h2, z2], -1)
            cons2 = self.consequence(s2)
            
            sim_termination = cons2["termination"].item()

            if sim_termination > 0.90:
                cont_final = prev_cont_t.squeeze(0) if prev_cont_exec is not None else torch.zeros_like(cont_final)

            info = {
                "sim_termination": sim_termination, 
                "physics_energy": ps.energy.mean().item(),
                "n_mcts_sims": n_sims,
                "raw_continuous_intent": cont_final.detach().float().cpu().numpy(),
                "fast_path_active": fast_path_active,
                "bci_active": bci_active
            }
        
        if s4_out is not None:
            info["s4_confidence"] = effective_confidence
            info["p_bci"] = p_bci.detach().cpu().numpy() if p_bci is not None else None

        if self.act_bridge is not None:
            act_result = self.act_bridge.act(action, s4_confidence=info.get("s4_confidence"), info=info)
            info["act_executed"] = act_result["executed"]
            info["act_outcome"] = act_result.get("outcome", "")
            if "structured" in act_result: info["_exec_structured"] = act_result["structured"]

        self._step += 1
        return action, info["raw_continuous_intent"], info

    def observe(self, obs: Dict, action: int, raw_cont: np.ndarray, exec_cont: np.ndarray, reward: float, done: bool, info: Optional[Dict] = None):
        if info and "_exec_structured" in info: obs = dict(obs); obs["structured"] = info["_exec_structured"]
        bci_active = info.get("bci_active", False) if info else False
        self.replay.add_step(obs, action, raw_cont, exec_cont, bci_active, reward, done)
        self.working.push(np.zeros(1), action, reward)

        if self._h is not None and self._step % 10 == 0:
            state = torch.cat([self._h, self._z], -1)
            with torch.no_grad():
                val = self.critic.min_value(state).unsqueeze(-1)
                val_padded = F.pad(val, (0, self.episodic.values.shape[-1] - 1))
            self.episodic.write(state, val_padded)

        if done: self.reset_latent(); self.working.clear()

    def update(self) -> Dict[str, float]:
        metrics = {}
        if len(self.replay) < self.cfg.batch_size: return metrics
        for _ in range(self.cfg.wm_updates): metrics.update(self._update_wm())
        if self._step >= self.cfg.warmup_steps:
            for _ in range(self.cfg.ac_updates): metrics.update(self._update_ac())
        return metrics

    def _update_wm(self) -> Dict[str, float]:
        C = self.cfg; batch = self.replay.sample(C.batch_size, self.device)
        if not batch or "actions" not in batch: return {}

        B, T = batch["actions"].shape
        T_eff = min(T, 20)
        obs_embed_list = []
        gnn_sparse, L_sigreg = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        st_list, act_list, raw_cont_list = [], [], []

        for t in range(T_eff):
            inp = {mod: batch[mod][:, t] for mod in ("eeg", "kinematics", "structured", "rgb", "depth") if mod in batch}
            if not inp:
                obs_embed_list.append(torch.zeros(B, C.d_model, device=self.device))
                continue
            with torch.set_grad_enabled(True):
                perc_out = self.perception(inp)
                obs_embed_list.append(perc_out["embed"])
                if "eeg" in inp and self.learning_manager is not None:
                    def recon_encoder_fn(x): return self.perception.s4(x, inp.get("electrode_mask"))["embed"]
                    sigreg_loss, _ = self.learning_manager.compute_unsupervised_loss(inp["eeg"], recon_encoder_fn)
                    L_sigreg = L_sigreg + sigreg_loss
                    
                    eeg = inp["eeg"]
                    out = self.perception.s4(eeg)
                    topo_embed = out["topological"] + out["embed"]
                    spatial_topo_loss, _ = SpatialTopologyLoss(d_model=self.cfg.d_model, n_channels=self.cfg.n_eeg_ch).to(self.device)(topo_embed, eeg)
                    L_sigreg = L_sigreg + spatial_topo_loss

        h = torch.zeros(B, self._det_dim, device=self.device); z = torch.zeros(B, self._stoch_dim, device=self.device)
        L_kl, L_r, L_rew, L_t, L_p = [torch.tensor(0.0, device=self.device) for _ in range(5)]

        for t in range(T_eff):
            e = obs_embed_list[t]
            a = self.action_enc(batch["actions"][:, t], batch["exec_cont"][:, t])
            h, z, pp, qp, ps, phys_tensor, phys_log = self.rssm.observe_step(h, z, a, e)
            s = torch.cat([h, z], -1)
            cons = self.consequence(s)
            
            st_list.append(s.detach()); act_list.append(batch["actions"][:, t]); raw_cont_list.append(batch["raw_cont"][:, t])

            L_kl = L_kl + self.rssm.kl_loss(pp, qp, 1.0)
            L_r = L_r + F.mse_loss(self.obs_decoder(s), e.detach())
            reward_pred = cons.get("reward", torch.zeros_like(batch["rewards"][:, t].unsqueeze(-1)))
            L_rew = L_rew + torch.nan_to_num(F.mse_loss(reward_pred.squeeze(-1), batch["rewards"][:, t]), nan=0.0)
            term_clamped = torch.nan_to_num(cons["termination"].clamp(0.0, 1.0).squeeze(-1), nan=0.5)
            L_t = L_t + F.binary_cross_entropy(term_clamped, batch["dones"][:, t])
            L_p = L_p + torch.nan_to_num(phys_tensor, nan=0.0)

        f = float(T_eff)
        loss = (C.lambda_kl * L_kl + C.lambda_recon * L_r + C.lambda_reward * L_rew + L_t + C.lambda_physics * L_p + L_sigreg) / f

        self._last_st = torch.stack(st_list, dim=1)  
        self._last_act = torch.stack(act_list, dim=1) 
        self._last_raw_cont = torch.stack(raw_cont_list, dim=1)

        self.opt_wm.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._wm_params, C.grad_clip)
        self.opt_wm.step()
        self.perception.s4.update_momentum(m=0.996)

        return {"wm/loss": loss.item()}

    def _update_ac(self) -> Dict[str, float]:
        C = self.cfg; B = C.batch_size; H = C.imag_horizon
        batch = self.replay.sample(B, self.device)
        
        if not batch or "actions" not in batch: return {}

        if hasattr(self, "_last_st") and self._last_st is not None:
            flat_st = self._last_st.view(-1, self._last_st.shape[-1])
            idx = torch.randint(0, flat_st.shape[0], (B,), device=self.device)
            s = flat_st[idx]
            h, z = s[..., :self._det_dim], s[..., self._det_dim:]
            
            flat_act = self._last_act.view(-1)
            flat_raw_cont = self._last_raw_cont.view(-1, 3)
            dist_bc, cont_bc = self.actor(flat_st)
            mask = batch["bci_active"][:, :self._last_st.shape[1]].contiguous().view(-1)
            L_bc_discrete = -dist_bc.log_prob(flat_act)
            L_bc_cont = F.mse_loss(cont_bc, flat_raw_cont, reduction='none').mean(dim=-1)
            L_bc_base = ((L_bc_discrete + L_bc_cont) * mask).sum() / (mask.sum() + 1e-8)
        else:
            h = torch.zeros(B, self._det_dim, device=self.device); z = torch.zeros(B, self._stoch_dim, device=self.device)
            L_bc_base = torch.tensor(0.0, device=self.device)

        self._imag_buf.clear()
        
        # Track previous cont for differentiable kinematic clamping
        prev_cont = flat_raw_cont if hasattr(self, "_last_st") else torch.zeros(B, 3, device=self.device)
        max_delta = C.max_velocity * C.dt

        for step_i in range(H):
            s = torch.cat([h, z], -1)
            dist, cont = self.actor(s)
            
            # DIFFERENTIABLE KINEMATIC CLAMP
            delta = cont - prev_cont
            norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
            clamped_delta = torch.where(norm > max_delta, delta * (max_delta / (norm + 1e-8)), delta)
            clamped_cont = prev_cont + clamped_delta
            prev_cont = clamped_cont
            
            a = dist.sample()
            lp = dist.log_prob(a)
            v = self.critic.min_value(s)
            
            with torch.no_grad():
                a_emb = self.action_enc(a, clamped_cont)
                h2, z2, _ = self.rssm.imagine_step(h, z, a_emb)
                s2 = torch.cat([h2, z2], -1)
                cons = self.consequence(s2)
            
            self._imag_buf.add(s, a, clamped_cont, cons.get("reward", torch.zeros_like(v)), v, lp, (cons["termination"] > 0.5).float())
            h, z = h2.detach(), z2.detach()

        G = self._imag_buf.lambda_returns()
        st = torch.stack(self._imag_buf.states)
        lp = torch.stack(self._imag_buf.log_probs)
        
        v1, v2 = self.critic(st.detach().view(-1, st.shape[-1]))
        L_v = F.mse_loss(v1.view(H, B), G) + F.mse_loss(v2.view(H, B), G)
        
        A = (G - G.mean()) / (G.std() + 1e-8)
        L_p = -(lp * A.detach()).mean()
        dist_ent, _ = self.actor(st.view(-1, st.shape[-1]).detach())
        ent = dist_ent.entropy().mean()

        dynamic_bc_weight = C.bc_weight * torch.exp(-ent.detach())
        loss = L_p - C.entropy_scale * ent + L_v + (dynamic_bc_weight * L_bc_base)

        self.opt_ac.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._ac_params, C.grad_clip)
        self.opt_ac.step()

        return {"ac/actor": L_p.item(), "ac/critic": L_v.item(), "ac/bc_loss": L_bc_base.item()}

    def run_sleep_phase(self, n_dreams: int = 16) -> Dict[str, float]:
        """Adversarial Sleep-Phase Consolidation (Dream Daemon)
        Actively targets high-uncertainty latents for deep MCTS search and consolidation."""
        if not hasattr(self, "_last_st") or self._last_st is None: return {}
        if not self.cfg.use_mcts or self.planner is None: return {}
        
        flat_st = self._last_st.view(-1, self._last_st.shape[-1])
        if flat_st.shape[0] == 0: return {}
        
        # Identify Epistemic Uncertainty using Critic Disagreement
        with torch.no_grad():
            v1, v2 = self.critic(flat_st)
            disagreement = (v1 - v2).abs().squeeze(-1)
        
        k = min(n_dreams, disagreement.shape[0])
        _, topk_idx = torch.topk(disagreement, k)
        
        uncertain_states = flat_st[topk_idx]
        h_roots = uncertain_states[..., :self._det_dim]
        z_roots = uncertain_states[..., self._det_dim:]
        
        total_loss = torch.tensor(0.0, device=self.device)
        orig_sims = self.planner.n_simulations
        self.planner.n_simulations = orig_sims * 2  # Deep search for Dream Daemon
        
        for i in range(k):
            h_i, z_i = h_roots[i:i+1], z_roots[i:i+1]
            # TD-MPC2 / Go-Explore target search
            _, p_target, cont_target = self.planner.search(h_i, z_i)
            
            dist, cont = self.actor(uncertain_states[i:i+1])
            target_p_tensor = torch.tensor(p_target, device=self.device)
            target_cont_tensor = cont_target.to(self.device).squeeze(0)
            
            # InfoNCE contrast against improved MCTS target
            l_pi = -(target_p_tensor * F.log_softmax(dist.logits, dim=-1)).sum(dim=-1).mean()
            l_cont = F.mse_loss(cont.squeeze(0), target_cont_tensor)
            total_loss = total_loss + l_pi + l_cont
            
        self.planner.n_simulations = orig_sims
        
        if k > 0:
            self.opt_ac.zero_grad()
            (total_loss / k).backward()
            nn.utils.clip_grad_norm_(self._ac_params, self.cfg.grad_clip)
            self.opt_ac.step()
            
        return {"sleep/dreams": float(k), "sleep/consolidation_loss": (total_loss / max(k, 1)).item()}

    def export_bundle(self, path: str, domain_tags: Optional[List[str]] = None, description: str = "", author: str = "", n_training_steps: int = 0, train_metrics: Optional[Dict[str, float]] = None):
        meta = BundleMetadata(domain_tags=domain_tags or [], description=description, author=author, n_training_steps=n_training_steps or self._step)
        return export_bundle(self, path, meta, train_metrics)

    def load_bundle(self, path: str, strict_arch: bool = True, modules: Optional[List[str]] = None) -> Dict[str, Any]:
        return load_bundle(self, path, strict_arch, modules)