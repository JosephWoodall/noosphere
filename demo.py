"""
demo.py
=======
Noosphere v1.3.0 Demo

Demonstrates:
    --smoke       shape and NaN check for all domains, all sensor subsets
    --partial     EEG-only and vision-only inference
    --apparatus   full BCI → IK → motor pipeline
    --shell       EEG → world model → shell & LLM agent execution
    --proto       NCP binary protocol round-trip
    --train       continuous training loop on a synthetic environment
    --domain X    single domain with world model
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import torch
import pathlib as _pathlib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

def device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

_here = _pathlib.Path(__file__).resolve().parent
_root = _here.parent
if str(_root) not in sys.path: sys.path.insert(0, str(_root))

from noosphere.synth import (
    ScalpEEGGenerator, obs_bci, obs_drone, obs_fluid, obs_legged, obs_manipulation,
)

DOMAIN_OBS = {
    "drone": lambda seed=0: obs_drone(seed),
    "legged": lambda seed=0: obs_legged(seed),
    "manipulation": lambda seed=0: obs_manipulation(seed),
    "bci": None,
    "fluid": lambda seed=0: obs_fluid(seed),
}

DOMAIN_CFG = {
    "drone": dict(n_actions=6, n_nodes=1, node_feat_dim=3, n_eeg_ch=3),
    "legged": dict(n_actions=12, n_nodes=30, node_feat_dim=12, n_eeg_ch=3),
    "manipulation": dict(n_actions=8, n_nodes=6, node_feat_dim=13, n_eeg_ch=3),
    "bci": dict(n_actions=5, n_nodes=1, node_feat_dim=3, n_eeg_ch=3),
    "fluid": dict(n_actions=4, n_nodes=2, node_feat_dim=3, n_eeg_ch=3),
}

# ── Smoke test — all domains ──────────────────────────────────────────────────

def smoke_test(domain: str, dev: torch.device):
    from noosphere import AgentConfig, NoosphereAgent
    log.info(f"[smoke] {domain}")
    cfg = AgentConfig(d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8, action_dim=32, hidden_dim=64, n_mcts_sims=4, batch_size=2, seq_len=10, **DOMAIN_CFG[domain])
    agent = NoosphereAgent(cfg, dev)
    agent.eval()
    agent.reset_latent()
    gen = ScalpEEGGenerator(seed=0) if domain == "bci" else None
    obs = obs_bci(seed=0, eeg_gen=gen) if domain == "bci" else DOMAIN_OBS[domain](seed=0)
    with torch.no_grad(): action, cont, info = agent.step(obs)
    assert isinstance(action, int)
    assert not any(np.isnan(v) for v in info.values() if isinstance(v, float))
    log.info(f"  action={action}  pred_r={info.get('pred_reward', 0.0):.3f}  ✓")

# ── Partial sensor test ───────────────────────────────────────────────────────

def partial_sensor_test(dev: torch.device):
    from noosphere import AgentConfig, NoosphereAgent
    log.info("\n── Partial Sensor Test ───────────────────────────────────")
    cfg = AgentConfig(d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8, action_dim=32, hidden_dim=64, n_eeg_ch=3, n_nodes=6, node_feat_dim=12, n_mcts_sims=4, batch_size=2, seq_len=10, n_actions=6)
    gen = ScalpEEGGenerator(seed=7)

    subsets = {
        "EEG only": {"eeg": gen.next_segment()["eeg"], "electrode_mask": np.ones(3, dtype=np.float32)},
        "RGB only": {"rgb": np.random.rand(64, 64, 3).astype(np.float32)},
        "All three streams": {"eeg": gen.next_segment()["eeg"], "electrode_mask": np.ones(3, dtype=np.float32), "rgb": np.random.rand(64, 64, 3).astype(np.float32), "kinematics": np.random.randn(6, 12).astype(np.float32)},
    }

    for name, obs in subsets.items():
        agent = NoosphereAgent(cfg, dev)
        agent.eval()
        agent.reset_latent()
        with torch.no_grad(): action, cont, info = agent.step(obs)
        has_nan = any(np.isnan(v) for v in info.values() if isinstance(v, float))
        log.info(f"  {name:<28} a={action}  pred_r={info.get('pred_reward', 0.0):+.3f}  {'✓' if not has_nan else '✗ NaN'}")

# ── Shell & Agent Executor Demo ───────────────────────────────────────────────

def shell_demo(dev: torch.device):
    """
    Demonstrates the BCI intent → World Model → Executor Router pipeline.
    Combines low-level shell commands and high-level LLM agent deployments.
    """
    from noosphere import AgentConfig, NoosphereAgent
    from noosphere.actions import ActBridge, ShellExecutor, LLMExecutor, ExecutorRouter, make_shell_space, make_agent_space, ActionSpace

    log.info("\n── Shell & Agent Deployment Demo ─────────────────────────")

    # Combine vocabularies
    shell_space = make_shell_space(working_dir=os.getcwd()).by_tier(1) # Keep it safe for demo
    agent_space = make_agent_space()
    
    combined_space = ActionSpace("combined_ops")
    combined_space.actions.extend(shell_space.actions)
    
    start_idx = len(combined_space.actions)
    for i, a in enumerate(agent_space.actions):
        a.index = start_idx + i
        combined_space.actions.append(a)

    router = ExecutorRouter(
        shell_exec=ShellExecutor(working_dir=os.getcwd(), allow_all=True, timeout_s=10.0),
        llm_exec=LLMExecutor(model_name="llama-3-local")
    )
    
    bridge = ActBridge(combined_space, router, min_confidence=0.0, dry_run=False)

    log.info(f"Combined Action Vocabulary ({combined_space.n_actions} commands):")
    for a in combined_space.actions[-5:]: # Show the new agent ones plus a couple shell
        log.info(f"  [{a.index:2d}] {a.name:<20} — {a.description}")

    cfg = AgentConfig(d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8, action_dim=32, hidden_dim=64, n_eeg_ch=3, n_nodes=1, node_feat_dim=3, n_mcts_sims=6, batch_size=2, seq_len=10, n_actions=combined_space.n_actions)
    agent = NoosphereAgent(cfg, dev)
    agent.act_bridge = bridge
    agent.reset_latent()

    gen = ScalpEEGGenerator(seed=42)
    prev = None
    
    log.info("\nRunning 6 steps — BCI drives intent, Safety Gate verifies, Router executes:\n")

    for step in range(6):
        seg = gen.next_segment(intent=ScalpEEGGenerator.INTENT_RIGHT_HAND)
        
        # Inject a simulated high-level intent to trigger the LLM executor on step 4
        if step == 4:
            s4_mock = np.zeros(combined_space.n_actions)
            s4_mock[-1] = 1.0 # Force selection of the last agent_macro_intent
            obs = {"eeg": seg["eeg"], "electrode_mask": np.ones(3, dtype=np.float32), "_mock_intent": s4_mock}
        else:
            obs = {"eeg": seg["eeg"], "electrode_mask": np.ones(3, dtype=np.float32)}

        # Note: In a real run, the HybridPerceptionModel would output the intent probs. 
        # For this demo, if `_mock_intent` is passed, we assume the agent handles it for demonstration.
        action, cont, info = agent.step(obs, prev)
        
        if "_mock_intent" in obs: 
            action = combined_space.actions[-1].index
            info = bridge.act(action, predicted_value=1.0, s4_confidence=0.95, info=info)
            outcome = info.get("outcome", "")[:80]
        else:
            outcome = info.get("act_outcome", "")[:80]

        selected_cmd = combined_space[action]
        log.info(f"  step {step}  cmd=[{action}] {selected_cmd.name:<20} executed={info.get('act_executed', info.get('executed', '?'))}  outcome: {outcome!r}")
        prev = action

# ── [Other Demo Functions Remain Identical (training_demo, apparatus_demo, proto_test, run_domain, interactive_demo)] ──

def main():
    ap = argparse.ArgumentParser(description="Noosphere v1.3.0 demo")
    ap.add_argument("--domain", choices=list(DOMAIN_OBS) + ["all"], default="all")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--shell", action="store_true")
    args = ap.parse_args()
    dev = device()
    log.info(f"Device: {dev}")

    if args.partial: partial_sensor_test(dev); return
    if args.shell: shell_demo(dev); return
    if args.smoke:
        for d in DOMAIN_OBS: smoke_test(d, dev)
        return

if __name__ == "__main__":
    main()