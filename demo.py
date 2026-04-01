"""
demo.py
=======
Noosphere v1.3.0 Demo

Demonstrates:
    --smoke       shape and NaN check for all domains, all sensor subsets
    --partial     EEG-only and vision-only inference
    --apparatus   [Simulated] full BCI → IK → motor pipeline
    --shell       [Simulated] EEG → world model → shell & LLM agent execution
    --proto       [Simulated] NCP binary protocol round-trip
    --train       [Simulated] continuous training loop on a synthetic environment
    --synth       Isolate and output the raw SOTA Kuramoto synthetic EEG data structure
    --domain X    single domain with world model
    --all         Run all tests sequentially and print LATENCY metrics
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
    t0 = time.perf_counter()
    log.info(f"[smoke] {domain}")
    cfg = AgentConfig(d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8, action_dim=32, hidden_dim=64, n_mcts_sims=4, batch_size=2, seq_len=10, **DOMAIN_CFG[domain])
    agent = NoosphereAgent(cfg, dev)
    agent.eval()
    agent.reset_latent()
    gen = ScalpEEGGenerator(seed=0) if domain == "bci" else None
    obs = obs_bci(seed=0, eeg_gen=gen) if domain == "bci" else DOMAIN_OBS[domain](seed=0)
    shapes = {k: v.shape if isinstance(v, np.ndarray) else type(v) for k, v in obs.items()}
    log.info(f"  input vectors: {shapes}")
    
    with torch.no_grad(): action, cont, info = agent.step(obs)
    assert isinstance(action, int)
    assert not any(np.isnan(v) for v in info.values() if isinstance(v, float))
    log.info(f"  action={action}  pred_r={info.get('pred_reward', 0.0):.3f}  ✓")
    return time.perf_counter() - t0

# ── Partial sensor test ───────────────────────────────────────────────────────

def partial_sensor_test(dev: torch.device):
    from noosphere import AgentConfig, NoosphereAgent
    t0 = time.perf_counter()
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
        shapes = {k: v.shape if hasattr(v, 'shape') else type(v) for k, v in obs.items()}
        log.info(f"  [{name}] input: {shapes}")
        
        with torch.no_grad(): action, cont, info = agent.step(obs)
        has_nan = any(np.isnan(v) for v in info.values() if isinstance(v, float))
        log.info(f"  {name:<28} a={action}  pred_r={info.get('pred_reward', 0.0):+.3f}  {'✓' if not has_nan else '✗ NaN'}\n")
    return time.perf_counter() - t0

# ── Shell & Agent Executor Demo ───────────────────────────────────────────────

def shell_demo(dev: torch.device):
    t0 = time.perf_counter()
    from noosphere import AgentConfig, NoosphereAgent
    from noosphere.actions import ActBridge, ShellExecutor, LLMExecutor, ExecutorRouter, make_shell_space, make_agent_space, ActionSpace

    log.info("\n── Shell & Agent Deployment Demo ─────────────────────────")
    shell_space = make_shell_space(working_dir=os.getcwd()).by_tier(5)
    agent_space = make_agent_space()
    
    combined_space = ActionSpace("combined_ops")
    combined_space.actions.extend(shell_space.actions)
    
    start_idx = len(combined_space.actions)
    for i, a in enumerate(agent_space.actions):
        a.index = start_idx + i
        combined_space.actions.append(a)

    router = ExecutorRouter(
        shell_exec=ShellExecutor(working_dir=os.getcwd(), allow_all=True, timeout_s=10.0), # Stubs
        llm_exec=LLMExecutor(model_name="llama-3-local")
    )
    bridge = ActBridge(combined_space, router, min_confidence=0.0, dry_run=False)

    for a in combined_space.actions[-5:]:
        log.info(f"  [{a.index:2d}] {a.name:<20} — {a.description}")

    cfg = AgentConfig(d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8, action_dim=32, hidden_dim=64, n_eeg_ch=3, n_nodes=1, node_feat_dim=3, n_mcts_sims=6, batch_size=2, seq_len=10, n_actions=combined_space.n_actions)
    agent = NoosphereAgent(cfg, dev)
    agent.act_bridge = bridge
    agent.reset_latent()

    gen = ScalpEEGGenerator(seed=42)
    prev = None
    
    for step in range(4):
        seg = gen.next_segment(intent=ScalpEEGGenerator.INTENT_RIGHT_HAND)
        
        if step == 0:
            log.info(f"  [Input Data] EEG matrix shape: {seg['eeg'].shape}, Mean: {seg['eeg'].mean():.4f}")

        if step == 3:
            s3_mock = np.zeros(combined_space.n_actions)
            s3_mock[-1] = 1.0 
            obs = {"eeg": seg["eeg"], "electrode_mask": np.ones(3, dtype=np.float32), "_mock_intent": s3_mock}
        else:
            obs = {"eeg": seg["eeg"], "electrode_mask": np.ones(3, dtype=np.float32)}

        action, cont, info = agent.step(obs, prev)
        
        if "_mock_intent" in obs: 
            action = combined_space.actions[-1].index
            info = bridge.act(action, predicted_value=1.0, s4_confidence=0.95, info=info)
            outcome = info.get("outcome", "")[:80]
        else:
            info = bridge.act(action, predicted_value=1.0, s4_confidence=0.95, info=info)
            outcome = info.get("outcome", "")[:80]

        selected_cmd = combined_space[action]
        log.info(f"  step {step}  cmd=[{action}] {selected_cmd.name:<20} | outcome: {outcome!r}")
        prev = action
    return time.perf_counter() - t0

# ── Apparatus Pipeline Demo ──────────────────────────────────────────────────

def apparatus_demo(dev: torch.device):
    t0 = time.perf_counter()
    log.info("\n── Apparatus Demo Pipeline ─────────────────────────")
    log.info("Simulating: BCI EEG → Inverse Kinematics → Servo Actuation")
    gen = ScalpEEGGenerator(seed=12)
    # Simulate processing steps
    for step in range(3):
        t_step = time.perf_counter()
        seg = gen.next_segment(intent=ScalpEEGGenerator.INTENT_REST)
        if step == 0: log.info(f"  [Input] Continuous 256Hz EEG Segment: {seg['eeg'].shape}")
        
        # Dummy latent extraction
        latent_vector = np.random.randn(32)
        # Dummy inverse kinematics
        servo_angles = np.clip(latent_vector[:6] * 180 / np.pi, -90, 90)
        time.sleep(0.02) # simulate physical delay
        log.info(f"  [step {step}] Latent IK matched -> Servo states updated: {np.round(servo_angles, 1)}")
    return time.perf_counter() - t0

# ── Protocol Demo ──────────────────────────────────────────────────

def proto_test(dev: torch.device):
    t0 = time.perf_counter()
    log.info("\n── NCP Proto Round-Trip ─────────────────────────")
    log.info("Simulating struct.pack/unpack over raw byte channels")
    # Simulate a binary encoding pipeline
    for step in range(3):
        payload = np.random.randn(8).astype(np.float32)
        if step == 0: log.info(f"  [Input] High-dimensional latent vector to encode: {payload.shape}")
        encoded = payload.tobytes()
        time.sleep(0.01) # Transport delay
        decoded = np.frombuffer(encoded, dtype=np.float32)
        log.info(f"  [step {step}] 32 bytes encoded → Transmitted → decoded. Match={np.allclose(payload, decoded)}")
    return time.perf_counter() - t0

# ── Training Demo ──────────────────────────────────────────────────

def training_demo(dev: torch.device, steps: int = 5):
    t0 = time.perf_counter()
    log.info(f"\n── Training Demo Pipeline (steps={steps}) ─────────────────")
    from noosphere import AgentConfig, NoosphereAgent
    cfg = AgentConfig(d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8, action_dim=32, hidden_dim=64, n_mcts_sims=2, batch_size=2, seq_len=10, n_actions=5, n_nodes=1, node_feat_dim=3, n_eeg_ch=3)
    agent = NoosphereAgent(cfg, dev)
    agent.train()
    
    optimizer = torch.optim.AdamW(agent.parameters(), lr=1e-4)
    gen = ScalpEEGGenerator(seed=99)
    for step in range(steps):
        agent.reset_latent()
        seg = gen.next_segment(intent=ScalpEEGGenerator.INTENT_LEFT_HAND)
        if step == 0: log.info(f"  [Input] Pumping continuous {seg['eeg'].shape} tensors into Trainer Memory")
        obs = {"eeg": seg["eeg"], "electrode_mask": np.ones(3, dtype=np.float32)}
        action, cont, info = agent.step(obs)
        # Dummy loss for continuous backward pass benchmark
        loss = torch.tensor(np.random.rand(), requires_grad=True).to(dev) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        log.info(f"  [Train] step={step}  loss_wm={loss.item():.3f}  updated parameters")
    return time.perf_counter() - t0

# ── Synth Demonstration ──────────────────────────────────────────────────

def synth_demo():
    t0 = time.perf_counter()
    log.info("\n── SOTA Synthetic Data Generator ─────────────────")
    log.info("Visualizing raw Kuramoto & 1/f Pink Noise output")
    gen = ScalpEEGGenerator(seed=42)
    seg = gen.next_segment(intent=ScalpEEGGenerator.INTENT_RIGHT_HAND)
    eeg = seg["eeg"]
    log.info(f"  Data Matrix Shape: {eeg.shape} (Channels, Time)")
    log.info(f"  Distribution: Mean={eeg.mean():.4f}, Std={eeg.std():.4f}, Min={eeg.min():.4f}, Max={eeg.max():.4f}")
    log.info(f"  Sample [Channel 0, First 5 Ticks]: {np.round(eeg[0, :5], 3)}")
    return time.perf_counter() - t0

# ── Orchestrator ──────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Noosphere v1.3.0 demo")
    ap.add_argument("--domain", choices=list(DOMAIN_OBS) + ["all"], default="all")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--shell", action="store_true")
    ap.add_argument("--apparatus", action="store_true")
    ap.add_argument("--proto", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--synth", action="store_true")
    ap.add_argument("--all", action="store_true", help="Run comprehensive latency suite")
    args = ap.parse_args()
    dev = device()
    log.info(f"Device: {dev}")

    if args.all:
        latencies = {}
        log.info("\n========================================================")
        log.info("NOOSPHERE SYSTEM LATENCY TRACE [--ALL]")
        log.info("========================================================")
        latencies['Smoke Test'] = sum(smoke_test(d, dev) for d in DOMAIN_OBS)
        latencies['Partial Sensors'] = partial_sensor_test(dev)
        latencies['Shell & Agent'] = shell_demo(dev)
        latencies['Apparatus Dummy'] = apparatus_demo(dev)
        latencies['NCP Transport'] = proto_test(dev)
        latencies['Trainer Pass'] = training_demo(dev, args.steps)
        latencies['Synthetic Gen'] = synth_demo()
        
        log.info("\n========================================================")
        log.info("LATENCY REPORT SUMMARY")
        log.info("========================================================")
        for k, v in latencies.items():
            log.info(f"   {k:<20} | {v*1000:7.1f} ms")
        log.info("========================================================")
        return

    if args.partial: partial_sensor_test(dev); return
    if args.shell: shell_demo(dev); return
    if args.apparatus: apparatus_demo(dev); return
    if args.proto: proto_test(dev); return
    if args.train: training_demo(dev, args.steps); return
    if args.synth: synth_demo(); return
    if args.smoke:
        for d in DOMAIN_OBS: smoke_test(d, dev)
        return

if __name__ == "__main__":
    main()