"""
noosphere/agent.py
==================
Noosphere Agent - Refactored for Modularity and Performance

Features:
- Clean, decoupled architecture with separate Intent and Preprocessing modules.
- Torch-compiled inference paths for sub-20ms latency.
- Enhanced shared autonomy with probabilistic blending.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from noosphere.actions import ActBridge
from noosphere.bundle import BundleMetadata, export_bundle, load_bundle
from noosphere.configs import AgentConfig
from noosphere.intent import IntentArbiter
from noosphere.learning import LearningManager, SpatialTopologyLoss
from noosphere.memory import EpisodicMemory, SequenceReplayBuffer, WorkingMemory
from noosphere.perception import HybridPerceptionModel
from noosphere.physics import PhysicsAugmentedRSSM
from noosphere.planner import ActionEncoder, Actor, Critic, ImaginationBuffer, MCTSPlanner
from noosphere.preprocessing import ObservationPreprocessor
from noosphere.rssm import EnhancedConsequenceModel, ObservationDecoder

logger = logging.getLogger(__name__)

NO_OP_ACTION_INDEX = 0

class NoosphereAgent(nn.Module):
    def __init__(self, cfg: AgentConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        self._init_models()
        self._init_optimizers()
        self._init_memory()
        self._init_bci()
        
        self._step = 0
        self.to(device)
        self._jit_compile()

    def _init_models(self):
        C = self.cfg
        P = C.perception
        W = C.world_model

        self.perception = HybridPerceptionModel(
            d_model=P.d_model, n_heads=P.n_heads, n_layers=P.n_layers, 
            n_eeg_channels=P.n_eeg_ch, s4_d_state=P.s4_d_state, 
            s4_n_blocks=P.s4_n_blocks, s4_downsample=P.s4_downsample, 
            n_kinematic_nodes=P.n_nodes, node_feature_dim=P.node_feat_dim, 
            gnn_n_layers=P.gnn_n_layers, patch_size=P.patch_size, max_reach=P.max_reach
        )
        
        self.rssm = PhysicsAugmentedRSSM(
            embed_dim=P.d_model, action_dim=W.action_dim, n_bodies=W.n_bodies, G=W.fluid_grid,
            det_dim=W.det_dim, stoch_cats=W.stoch_cats, stoch_classes=W.stoch_cls, 
            hidden_dim=W.hidden_dim, dt=W.dt, model_type=W.type
        )
        
        state_dim = self.rssm.state_dim
        self.consequence = EnhancedConsequenceModel(state_dim, W.hidden_dim)
        self.obs_decoder = ObservationDecoder(state_dim, P.d_model, W.hidden_dim)
        self.action_enc = ActionEncoder(C.bci.n_actions, W.action_dim)
        self.actor = Actor(state_dim, C.bci.n_actions, W.hidden_dim, max_reach=P.max_reach)
        self.critic = Critic(state_dim, W.hidden_dim)
        
        if C.planning.use_mcts:
            self.planner = MCTSPlanner(
                self.rssm.rssm, self.consequence, self.actor, self.action_enc, C.bci.n_actions, 
                C.planning.n_mcts_sims, C.planning.mcts_horizon, C.ac.gamma, 
                max_vel=C.planning.max_velocity, dt=W.dt, device=self.device
            )

    def _init_optimizers(self):
        C = self.cfg
        self.opt_wm = optim.AdamW([
            {"params": list(self.perception.parameters()), "lr": C.training.lr_perception},
            {"params": list(p for m in [self.rssm, self.consequence, self.obs_decoder, self.action_enc] for p in m.parameters()), "lr": C.training.lr_world_model},
        ], eps=1e-8)
        self.opt_ac = optim.AdamW(list(self.actor.parameters()) + list(self.critic.parameters()), lr=C.ac.lr_actor_critic, eps=1e-8)

    def _init_memory(self):
        C = self.cfg
        self.replay = SequenceReplayBuffer(C.training.replay_capacity, C.training.seq_len)
        self.episodic = EpisodicMemory(self.rssm.state_dim, 64, C.training.episodic_capacity)
        self.working = WorkingMemory(20)
        self._imag_buf = ImaginationBuffer(C.ac.gamma, C.ac.lam)

    def _init_bci(self):
        C = self.cfg
        self.prep = ObservationPreprocessor(self.device)
        self.intent = IntentArbiter(
            n_actions=C.bci.n_actions, 
            min_confidence=C.bci.min_act_confidence,
            fast_path_threshold=C.bci.fast_path_threshold,
            momentum_decay=C.bci.momentum_decay
        )
        self.act_bridge: Optional[ActBridge] = None
        self.learning_manager: Optional[LearningManager] = None
        
        from noosphere.apparatus_iot import IoTApparatus
        self.iot_apparatus = IoTApparatus()
        
        from noosphere.proto import NCPTransport
        self.transport = NCPTransport.inproc(node_id="local_operator")
        
        from noosphere.network import NetworkSessionManager, NetworkUI
        self.network_manager = NetworkSessionManager("local_operator", self.transport)
        self.network_ui = NetworkUI(self.network_manager)
        
        self.network_manager.listen_for_insights(self._on_insight_received)
        self.received_insights = []

        from noosphere.monitor import Monitor, MonitorConfig
        self.monitor = Monitor(MonitorConfig(), ncp_transport=self.transport)
        self.monitor.start()
        
        self._h = None; self._z = None

    def _on_insight_received(self, insight_type: str, data: Dict):
        """Callback for incoming 'Whisper' insights from other nodes."""
        logger.info(f"[Agent] Received '{insight_type}' insight from network")
        self.received_insights.append({"type": insight_type, "data": data, "ts": time.time()})
        if len(self.received_insights) > 10: self.received_insights.pop(0)

    def _jit_compile(self):
        if hasattr(torch, "compile") and self.device.type == "cuda":
            import os
            if os.name != 'nt':
                logger.info("Initializing JIT Compilation...")
                self.perception = torch.compile(self.perception, mode="reduce-overhead")
                self.rssm = torch.compile(self.rssm, mode="reduce-overhead")
                self.actor = torch.compile(self.actor, mode="reduce-overhead")

    def reset_latent(self):
        init = self.rssm.initial_state(1, self.device)
        self._h, self._z = init["h"], init["z"]
        self.rssm.reset_episode()
        self.intent.reset()

    @torch.no_grad()
    def step(self, obs: Dict, prev_action: Optional[int] = None, prev_cont_exec: Optional[np.ndarray] = None, deterministic: bool = False) -> Tuple[int, np.ndarray, Dict]:
        # Inject IoT state if modality is missing
        if "structured" not in obs and hasattr(self, "iot_apparatus"):
            obs = dict(obs)
            obs["structured"] = np.pad(self.iot_apparatus.get_state_vector(), (0, 64 - 3))

        autocast_dtype = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype, enabled=self.device.type in ["cuda", "mps"]):
            # 1. Perceive & Encode
            tensors = self.prep(obs)
            perc_out = self.perception(tensors)
            obs_embed = perc_out["embed"]
            
            # 2. Update World Model Latent
            prev_act_t = torch.tensor([prev_action or 0], device=self.device)
            prev_cont_t = torch.tensor(prev_cont_exec, device=self.device).unsqueeze(0) if prev_cont_exec is not None else torch.zeros(1, 3, device=self.device)
            a_embed = self.action_enc(prev_act_t, prev_cont_t)
            
            if self._h is None: self.reset_latent()
            self._h, self._z, _, _, ps, _, _ = self.rssm.observe_step(self._h, self._z, a_embed, obs_embed)
            state = torch.cat([self._h, self._z], -1)

            # 3. Process BCI Intent
            bci_out = self.intent.update(perc_out.get("s4_out"), perc_out.get("xyz_pred"))
            
            # Update Network Sessions
            if self.cfg.bci.enable_inter_agent_comms:
                self.network_manager.update(bci_out["contact_id"], bci_out["identity_confidence"])
                self.network_ui.render()

            # 4. Plan & Sample
            actor_dist, actor_cont = self.actor(state)
            p_final, cont_final = self.intent.blend(bci_out, actor_dist.probs.squeeze(0), actor_cont.squeeze(0))
            
            if bci_out["fast_path"]:
                action = int(p_final.argmax().item())
                n_sims = 0
            else:
                n_sims = self._calculate_mcts_budget(perc_out.get("s4_out"), bci_out["confidence"])
                if self.cfg.planning.use_mcts and n_sims > 0:
                    self.planner.n_simulations = n_sims
                    action, p_final, cont_final = self.planner.search(self._h, self._z, bci_discrete=bci_out["discrete"], bci_continuous=bci_out["continuous"])
                else:
                    action = int(p_final.argmax().item()) if deterministic else int(torch.distributions.Categorical(probs=p_final).sample().item())

            # 5. Network Routing Bypass
            # If a session is active and it's a discrete intent, route to network
            if self.cfg.bci.enable_inter_agent_comms and self.network_manager.active_session:
                if bci_out["active"] and action < 6: # Maps 0-5 to our Macros
                    self.network_manager.send_message(action)
                    # We continue to execute locally too, but the Network UI will show it

            # 6. Safety Gating
            sim_termination = self._safety_check(action, cont_final)
            if sim_termination > 0.90:
                cont_final = prev_cont_t.squeeze(0)

            info = {
                "sim_termination": sim_termination,
                "n_mcts_sims": n_sims,
                "bci_active": bci_out["active"],
                "bci_confidence": bci_out["confidence"],
                "contact_id": bci_out["contact_id"],
                "identity_confidence": bci_out["identity_confidence"],
                "raw_continuous_intent": cont_final.detach().float().cpu().numpy()
            }
            
            if self.act_bridge is not None:
                # Digital Consequence Safety Gate Check
                if action < len(self.act_bridge.space):
                    action_obj = self.act_bridge.space[action]
                    if action_obj.payload and "shell_cmd" in action_obj.payload:
                        if self.intent.predict_critical_failure(action_obj.payload["shell_cmd"]):
                            info["sim_termination"] = 1.0 # Force block
                            logger.warning(f"Safety Gate Blocked Destructive Command: {action_obj.payload['shell_cmd']}")

                act_result = self.act_bridge.act(action, s4_confidence=bci_out["confidence"], info=info)
                info["act_executed"] = act_result["executed"]
                if "structured" in act_result: info["_exec_structured"] = act_result["structured"]

        self._step += 1
        return action, info["raw_continuous_intent"], info

    def _calculate_mcts_budget(self, s4_out: Optional[Dict], bci_conf: float) -> int:
        if not self.cfg.planning.use_mcts: return 0
        budget = 1.0
        if s4_out and "cognitive_state" in s4_out:
            cog = s4_out["cognitive_state"]
            E = 0.5 * (cog.get("workload", 0.0)**2) + cog.get("fatigue", 0.0)
            budget = max(0.1, 1.0 - E)
        return max(2, int(self.cfg.planning.n_mcts_sims * budget * (1.0 - bci_conf)))

    def _safety_check(self, action: int, cont: torch.Tensor) -> float:
        with torch.no_grad():
            a_emb = self.action_enc(torch.tensor([action], device=self.device), cont.unsqueeze(0))
            h2, z2, _ = self.rssm.imagine_step(self._h, self._z, a_emb)
            cons = self.consequence(torch.cat([h2, z2], -1))
            return cons["termination"].item()

    def observe(self, obs: Dict, action: int, raw_cont: np.ndarray, exec_cont: np.ndarray, reward: float, done: bool, info: Optional[Dict] = None):
        if info and "_exec_structured" in info: obs = dict(obs); obs["structured"] = info["_exec_structured"]
        bci_active = info.get("bci_active", False) if info else False
        self.replay.add_step(obs, action, raw_cont, exec_cont, bci_active, reward, done, info=info)
        
        # Monitor: Record step
        if hasattr(self, "monitor"):
            self.monitor.record_step(self._step, info or {}, {}, env_info=info)

        if self._h is not None and self._step % 10 == 0:
            state = torch.cat([self._h, self._z], -1)
            with torch.no_grad():
                val = self.critic.min_value(state).unsqueeze(-1)
                val_padded = F.pad(val, (0, self.episodic.values.shape[-1] - 1))
            self.episodic.write(state, val_padded)
        if done: self.reset_latent()

    def update(self) -> Dict[str, float]:
        metrics = {}
        if len(self.replay) < self.cfg.training.batch_size: return metrics
        for _ in range(self.cfg.training.wm_updates): metrics.update(self._update_wm())
        if self._step >= self.cfg.training.warmup_steps:
            for _ in range(self.cfg.ac.ac_updates): metrics.update(self._update_ac())
            
        # Monitor: Pass metrics and drain alerts
        if hasattr(self, "monitor"):
            self.monitor.record_step(self._step, {}, metrics)
            alerts = self.monitor.drain_alerts()
            for a in alerts:
                logger.warning(f"[AGENT MONITOR] {a}")
                # Self-healing: Respond to critical failures
                if a.source in ("kl_explosion", "wm_loss_spike"):
                    logger.warning("[Agent] Stabilizing world model: reducing learning rate by 50%")
                    for param_group in self.opt_wm.param_groups:
                        param_group['lr'] *= 0.5

        # Collaborative Learning: The "Whisper" Protocol
        if self.cfg.bci.allow_collective_learning and self._step % 100 == 0:
            # We share abstract dynamics (RSSM parameters related to physics)
            weights = {k: v for k, v in self.rssm.named_parameters() if "residual" in k or "physics" in k}
            if weights:
                self.network_manager.share_insights(weights)
                
        return metrics

    def _update_wm(self) -> Dict[str, float]:
        C = self.cfg; B, T = C.training.batch_size, C.training.seq_len
        batch = self.replay.sample(B, self.device)
        if not batch: return {}

        T_eff = min(T, batch["actions"].shape[1])
        obs_embeds, st_list, act_list, raw_cont_list = [], [], [], []
        L_sigreg = torch.tensor(0.0, device=self.device)

        # 1. Encode Sequence
        for t in range(T_eff):
            inp = self.prep({mod: batch[mod][:, t] for mod in batch if mod in ("eeg", "kinematics", "structured", "rgb", "depth")})
            perc = self.perception(inp)
            obs_embeds.append(perc["embed"])
            
            if "eeg" in inp and self.learning_manager:
                def enc_fn(x): return self.perception.s4(x, inp.get("electrode_mask"))["embed"]
                sig, _ = self.learning_manager.compute_unsupervised_loss(inp["eeg"], enc_fn)
                L_sigreg += sig
                
                # Spatial Topology Loss
                s4_out = perc["s4_out"]
                topo_loss, _ = SpatialTopologyLoss(C.perception.d_model, C.perception.n_eeg_ch).to(self.device)(s4_out["sequence"], inp["eeg"])
                L_sigreg += topo_loss

        # 2. RSSM Backprop
        h = torch.zeros(B, self.rssm.rssm.det_dim, device=self.device)
        z = torch.zeros(B, self.rssm.rssm.stoch_dim, device=self.device)
        L_wm = torch.tensor(0.0, device=self.device)

        for t in range(T_eff):
            a = self.action_enc(batch["actions"][:, t], batch["exec_cont"][:, t])
            h, z, pp, qp, ps, phys_tensor, _ = self.rssm.observe_step(h, z, a, obs_embeds[t])
            s = torch.cat([h, z], -1)
            cons = self.consequence(s)
            
            st_list.append(s.detach()); act_list.append(batch["actions"][:, t]); raw_cont_list.append(batch["raw_cont"][:, t])

            kl = self.rssm.kl_loss(pp, qp, 1.0)
            recon = F.mse_loss(self.obs_decoder(s), obs_embeds[t].detach())
            rew = F.mse_loss(cons.get("reward", torch.zeros_like(batch["rewards"][:, t])).squeeze(), batch["rewards"][:, t])
            term = F.binary_cross_entropy(cons["termination"].clamp(0, 1).squeeze(), batch["dones"][:, t])
            
            L_wm += C.world_model.lambda_kl * kl + C.world_model.lambda_recon * recon + C.world_model.lambda_reward * rew + term + C.world_model.lambda_physics * phys_tensor

            # Digital Consequence Supervision (v1.7.0)
            if hasattr(self.consequence, "digital_loss") and "digital_feedback" in batch:
                df = batch["digital_feedback"]
                L_wm += self.consequence.digital_loss(
                    s, 
                    actual_exit=df["exit_code"][:, t],
                    actual_stdout_len=df["stdout_len"][:, t],
                    actual_state_change=df["state_change"][:, t],
                    actual_next_state=df["next_digital"][:, t]
                )

        loss = (L_wm + L_sigreg) / T_eff
        self.opt_wm.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.opt_wm.param_groups[0]['params'] + self.opt_wm.param_groups[1]['params'], C.training.grad_clip)
        self.opt_wm.step()
        
        self._last_st, self._last_act, self._last_raw_cont = torch.stack(st_list, 1), torch.stack(act_list, 1), torch.stack(raw_cont_list, 1)
        return {"wm/loss": loss.item()}

    def _update_ac(self) -> Dict[str, float]:
        C = self.cfg; B = C.training.batch_size; H = C.ac.imag_horizon
        if not hasattr(self, "_last_st"): return {}
        
        # Sample starting states from WM rollout
        flat_st = self._last_st.view(-1, self.rssm.state_dim)
        idx = torch.randint(0, flat_st.shape[0], (B,), device=self.device)
        s = flat_st[idx]
        h, z = s[..., :self.rssm.rssm.det_dim], s[..., self.rssm.rssm.det_dim:]
        
        # BC Loss from history
        dist_bc, cont_bc = self.actor(flat_st)
        mask = self.replay.sample(B, self.device)["bci_active"].view(-1)[:flat_st.shape[0]] # approximation
        L_bc = ((-dist_bc.log_prob(self._last_act.view(-1)) + F.mse_loss(cont_bc, self._last_raw_cont.view(-1, 3), reduction='none').mean(-1)) * mask).sum() / (mask.sum() + 1e-8)

        self._imag_buf.clear()
        prev_cont = self._last_raw_cont.view(-1, 3)[idx]
        
        for _ in range(H):
            s = torch.cat([h, z], -1)
            dist, cont = self.actor(s)
            
            # Kinematic clamp
            delta = cont - prev_cont
            norm = torch.norm(delta, p=2, dim=-1, keepdim=True)
            clamped_cont = prev_cont + torch.where(norm > C.planning.max_velocity*W.dt, delta * (C.planning.max_velocity*W.dt/(norm+1e-8)), delta)
            
            a = dist.sample()
            v = self.critic.min_value(s)
            
            with torch.no_grad():
                h, z, _ = self.rssm.imagine_step(h, z, self.action_enc(a, clamped_cont))
                cons = self.consequence(torch.cat([h, z], -1))
            
            self._imag_buf.add(s, a, clamped_cont, cons.get("reward", torch.zeros_like(v)), v, dist.log_prob(a), (cons["termination"] > 0.5).float())
            prev_cont = clamped_cont

        G = self._imag_buf.lambda_returns()
        st_imag = torch.stack(self._imag_buf.states)
        lp_imag = torch.stack(self._imag_buf.log_probs)
        
        v1, v2 = self.critic(st_imag.detach().view(-1, self.rssm.state_dim))
        L_v = F.mse_loss(v1.view(H, B), G) + F.mse_loss(v2.view(H, B), G)
        
        A = (G - G.mean()) / (G.std() + 1e-8)
        L_p = -(lp_imag * A.detach()).mean()
        ent = dist_bc.entropy().mean()
        
        loss = L_p - C.ac.entropy_scale * ent + L_v + C.ac.bc_weight * L_bc
        self.opt_ac.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), C.training.grad_clip)
        self.opt_ac.step()
        
        return {"ac/actor": L_p.item(), "ac/critic": L_v.item(), "ac/bc_loss": L_bc.item()}

    def export_bundle(self, path: str, **kwargs):
        meta = BundleMetadata(domain_tags=kwargs.get("domain_tags", []), description=kwargs.get("description", ""), author=kwargs.get("author", ""), n_training_steps=self._step)
        return export_bundle(self, path, meta, kwargs.get("train_metrics"))

    def load_bundle(self, path: str, strict_arch: bool = True, modules: Optional[List[str]] = None):
        return load_bundle(self, path, strict_arch, modules)
