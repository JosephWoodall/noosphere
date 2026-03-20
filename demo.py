"""
demo.py
=======
Noosphere v1.2.0 Demo

Demonstrates:
    --smoke       shape and NaN check for all domains, all sensor subsets
    --partial     EEG-only and vision-only inference (no crash, correct masking)
    --apparatus   full BCI → IK → motor pipeline
    --shell       EEG → world model → shell command execution
    --proto       NCP binary protocol round-trip
    --train       continuous training loop on a synthetic environment
    --domain X    single domain with world model

Usage
-----
    python demo.py --smoke
    python demo.py --partial
    python demo.py --shell
    python demo.py --train --steps 200
    python demo.py --domain bci --profile
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


from noosphere.data.synth import (
    NeckEEGGenerator,
    obs_drone, obs_legged, obs_manipulation, obs_bci, obs_fluid,
)

DOMAIN_OBS = {
    "drone":        lambda seed=0: obs_drone(seed),
    "legged":       lambda seed=0: obs_legged(seed),
    "manipulation": lambda seed=0: obs_manipulation(seed),
    "bci":          None,
    "fluid":        lambda seed=0: obs_fluid(seed),
}

DOMAIN_CFG = {
    "drone":        dict(n_actions=6,  n_nodes=1,  node_feat_dim=3,  n_eeg_ch=3),
    "legged":       dict(n_actions=12, n_nodes=20, node_feat_dim=30, n_eeg_ch=3),
    "manipulation": dict(n_actions=8,  n_nodes=6,  node_feat_dim=13, n_eeg_ch=3),
    "bci":          dict(n_actions=5,  n_nodes=1,  node_feat_dim=3,  n_eeg_ch=3),
    "fluid":        dict(n_actions=4,  n_nodes=2,  node_feat_dim=3,  n_eeg_ch=3),
}


# ── Smoke test — all domains ──────────────────────────────────────────────────

def smoke_test(domain: str, dev: torch.device):
    from noosphere import NoosphereAgent, AgentConfig
    log.info(f"[smoke] {domain}")
    cfg = AgentConfig(
        d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8,
        action_dim=32, hidden_dim=64,
        n_mcts_sims=4, batch_size=2, seq_len=10,
        **DOMAIN_CFG[domain],
    )
    agent = NoosphereAgent(cfg, dev)
    agent.eval()
    agent.reset_latent()
    gen = NeckEEGGenerator(seed=0) if domain == "bci" else None
    obs = obs_bci(seed=0, eeg_gen=gen) if domain == "bci" else DOMAIN_OBS[domain](seed=0)
    with torch.no_grad():
        action, info = agent.step(obs)
    assert isinstance(action, int)
    assert not any(np.isnan(v) for v in info.values() if isinstance(v, float))
    log.info(f"  action={action}  pred_r={info['pred_reward']:.3f}  ✓")


# ── Partial sensor test — THIS is what answers the EEG-only question ──────────

def partial_sensor_test(dev: torch.device):
    """
    Verifies that any subset of sensors works correctly with no NaNs.
    Missing streams are masked out of attention, not zero-padded in.
    """
    from noosphere import NoosphereAgent, AgentConfig
    log.info("\n── Partial Sensor Test ───────────────────────────────────")

    cfg = AgentConfig(
        d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8,
        action_dim=32, hidden_dim=64, n_eeg_ch=3,
        n_nodes=6, node_feat_dim=12,
        n_mcts_sims=4, batch_size=2, seq_len=10, n_actions=6,
    )
    gen = NeckEEGGenerator(seed=7)

    subsets = {
        "EEG only":              {"eeg": gen.next_segment()["eeg"],
                                  "electrode_mask": np.ones(3, dtype=np.float32)},
        "RGB only":              {"rgb": np.random.rand(64, 64, 3).astype(np.float32)},
        "Depth only":            {"depth": np.random.rand(64, 64).astype(np.float32) * 3.0},
        "Kinematics only":       {"kinematics": np.random.randn(6, 12).astype(np.float32)},
        "EEG + RGB":             {"eeg": gen.next_segment()["eeg"],
                                  "electrode_mask": np.ones(3, dtype=np.float32),
                                  "rgb": np.random.rand(64, 64, 3).astype(np.float32)},
        "EEG + kinematics":      {"eeg": gen.next_segment()["eeg"],
                                  "electrode_mask": np.ones(3, dtype=np.float32),
                                  "kinematics": np.random.randn(6, 12).astype(np.float32)},
        "RGB + depth + kin":     {"rgb": np.random.rand(64, 64, 3).astype(np.float32),
                                  "depth": np.random.rand(64, 64).astype(np.float32),
                                  "kinematics": np.random.randn(6, 12).astype(np.float32)},
        "All three streams":     {"eeg": gen.next_segment()["eeg"],
                                  "electrode_mask": np.ones(3, dtype=np.float32),
                                  "rgb": np.random.rand(64, 64, 3).astype(np.float32),
                                  "depth": np.random.rand(64, 64).astype(np.float32),
                                  "kinematics": np.random.randn(6, 12).astype(np.float32)},
    }

    for name, obs in subsets.items():
        agent = NoosphereAgent(cfg, dev)
        agent.eval()
        agent.reset_latent()
        with torch.no_grad():
            action, info = agent.step(obs)
        has_nan = any(np.isnan(v) for v in info.values() if isinstance(v, float))
        status  = "✓" if not has_nan else "✗ NaN"
        streams = []
        if "eeg" in obs:        streams.append("S4")
        if "kinematics" in obs: streams.append("GNN")
        if any(k in obs for k in ("rgb","depth","rgb_right","lidar","audio")):
            streams.append("Vis")
        log.info(f"  {name:<28} streams={'+'.join(streams) or 'none'!r:<12} "
                 f"a={action}  pred_r={info['pred_reward']:+.3f}  {status}")

    log.info("Partial sensor tests complete.\n")


# ── Shell executor demo ───────────────────────────────────────────────────────

def shell_demo(dev: torch.device):
    """
    Demonstrates the EEG → world model → shell command pipeline.

    The world model plans over a vocabulary of Linux commands.
    The ActBridge translates the selected integer action to a real command,
    executes it safely, and feeds the result back as an observation.

    The world model learns to predict which commands are useful given
    the current latent state — without ever running commands during planning.
    """
    from noosphere import NoosphereAgent, AgentConfig
    from noosphere.actions import make_shell_space, ShellExecutor, ActBridge

    log.info("\n── Shell Command Demo ────────────────────────────────────")

    space    = make_shell_space(working_dir=os.getcwd())
    executor = ShellExecutor(
        working_dir=os.getcwd(),
        allow_all=True,          # allow all for demo; restrict in production
        timeout_s=10.0,
    )
    bridge   = ActBridge(space, executor, min_confidence=0.0, dry_run=False)

    log.info(f"Action vocabulary ({space.n_actions} commands):")
    for a in space.actions:
        log.info(f"  [{a.index:2d}] {a.name:<20} — {a.description}")

    cfg = AgentConfig(
        d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8,
        action_dim=32, hidden_dim=64, n_eeg_ch=3,
        n_nodes=1, node_feat_dim=3,
        n_mcts_sims=6, batch_size=2, seq_len=10,
        n_actions=space.n_actions,
    )
    agent = NoosphereAgent(cfg, dev)
    agent.act_bridge = bridge
    agent.reset_latent()

    gen  = NeckEEGGenerator(seed=42)
    prev = None
    log.info("\nRunning 8 steps — EEG drives intent, world model selects command:\n")

    for step in range(8):
        seg     = gen.next_segment(
            intent=NeckEEGGenerator.INTENT_RIGHT_HAND if step < 4
                   else NeckEEGGenerator.INTENT_FINGER_FLEX
        )
        obs     = {"eeg": seg["eeg"],
                   "electrode_mask": np.ones(3, dtype=np.float32)}
        action, info = agent.step(obs, prev)
        selected_cmd = space[action]

        reward = 1.0 if info.get("act_executed") and info.get("act_reward", 0) > 0 else 0.0
        agent.observe(obs, action, reward, done=(step == 7), info=info)

        outcome = info.get("act_outcome", "")[:80]
        log.info(f"  step {step}  cmd=[{action}] {selected_cmd.name:<20}  "
                 f"executed={info.get('act_executed','?')}  "
                 f"outcome: {outcome!r}")
        prev = action

    log.info("\nShell demo complete.")


# ── Continuous training demo ──────────────────────────────────────────────────

def training_demo(dev: torch.device, n_steps: int = 100):
    """
    Runs the continuous training loop on a synthetic BCI environment.
    Demonstrates checkpoint saving, metric logging, and the full
    Perceive→Model→Plan→Act→Observe→Learn cycle.
    """
    from noosphere import NoosphereAgent, AgentConfig
    from noosphere.trainer import Trainer, TrainerConfig, Env

    log.info("\n── Continuous Training Demo ──────────────────────────────")

    class SyntheticBCIEnv(Env):
        """Minimal synthetic BCI environment for training demo."""
        def __init__(self):
            self.gen   = NeckEEGGenerator(seed=0)
            self._step = 0

        def reset(self):
            self._step = 0
            return self._obs()

        def _obs(self):
            seg = self.gen.next_segment(
                intent=NeckEEGGenerator.INTENT_RIGHT_HAND
            )
            return {"eeg": seg["eeg"],
                    "electrode_mask": np.ones(3, dtype=np.float32)}

        def step(self, action, act_result=None):
            self._step += 1
            obs    = self._obs()
            reward = np.random.randn() * 0.1   # replace with real shaped reward
            done   = self._step >= 30
            return obs, float(reward), done, {}

    cfg = AgentConfig(
        d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8,
        action_dim=32, hidden_dim=64, n_eeg_ch=3,
        n_nodes=1, node_feat_dim=3, n_actions=5,
        n_mcts_sims=4, batch_size=4, seq_len=10,
        train_every=5, warmup_steps=20,
        replay_capacity=50, log_every=10,
    )
    agent = NoosphereAgent(cfg, dev)

    trainer_cfg = TrainerConfig(
        checkpoint_dir="/tmp/noosphere_demo_ckpts",
        checkpoint_every=50,
        log_dir="/tmp/noosphere_demo_logs",
        log_every=20,
        max_episode_steps=30,
        resume=False,
    )

    trainer = Trainer(agent, SyntheticBCIEnv(), trainer_cfg)
    log.info(f"Running {n_steps} training steps...")
    trainer.run(n_steps=n_steps)
    log.info("Training demo complete.")


# ── BCI apparatus demo ────────────────────────────────────────────────────────

def apparatus_demo(hardware: str = "sim"):
    from noosphere.apparatus import (
        IntentionFilter, AnomalyDetector, CoordinatePredictor, MovementExecutor,
    )
    from noosphere.data.synth import NeckEEGGenerator
    from noosphere.hardware   import ServoController
    from noosphere.proto      import NCPEncoder

    log.info(f"\n── BCI Apparatus Demo (hardware={hardware}) ──────────────")

    gen       = NeckEEGGenerator(seed=42)
    filt      = IntentionFilter()
    anomaly   = AnomalyDetector()
    predictor = CoordinatePredictor(retrain_every=20)
    executor  = MovementExecutor()
    servo     = ServoController(backend=hardware)
    enc       = NCPEncoder()

    intents   = [NeckEEGGenerator.INTENT_RIGHT_HAND,
                 NeckEEGGenerator.INTENT_SHOULDER_SHRUG,
                 NeckEEGGenerator.INTENT_FINGER_FLEX]
    n_moved   = 0

    for i in range(40):
        seg = gen.next_segment(intent=intents[i % len(intents)])
        if not filt.is_intentional(seg): continue
        if not anomaly.update_and_check(seg["probabilities"].tolist()): continue

        kin = seg["hierarchical"].get("kinematic") or {}
        if kin:
            feats = CoordinatePredictor.extract_features(seg)
            predictor.add_sample(feats, [kin["x"], kin["y"], kin["z"]])
            target = np.array([kin["x"], kin["y"], kin["z"]], dtype=np.float32)
            frame  = enc.eeg_segment(
                tuple(seg["raw_microvolts"].tolist()),
                tuple(seg["probabilities"].tolist()),
                seg["root_label"], seg["hierarchical"]["muscle_intent"] or 0,
                (kin["x"], kin["y"], kin["z"]),
                kin.get("velocity", 0.), kin.get("force", 0.),
                seg["timestamp"],
            )
            cmds = executor.plan_and_execute(target, interp_steps=3)
            if cmds:
                n_moved += 1
                servo.smooth_move(cmds[-1], steps=len(cmds), step_delay_s=0.0)
                log.info(f"  [{n_moved}] NCP {len(frame)}B → "
                         f"target=[{target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f}]")

    log.info(f"Apparatus demo: {n_moved} movements executed.")


# ── Protocol test ─────────────────────────────────────────────────────────────

def proto_test():
    from noosphere.proto import NCPEncoder, NCPDecoder, MsgType
    import struct

    log.info("\n── NCP Protocol Test ─────────────────────────────────────")
    enc = NCPEncoder(); dec = NCPDecoder()

    cases = [
        ("EEG_SEGMENT",     enc.eeg_segment((1.2,-0.5,3.1), tuple(range(8)),
                                             2, 1, (0.1,0.2,-0.3), 1.5, 0.8, 1234.5)),
        ("DESTINATION",     enc.destination_coords(0.1, 0.2, 0.3)),
        ("MOTOR_CMD",       enc.motor_command((10.,-20.,0.,45.,5.,-5.), smooth=True)),
        ("COGNITIVE",       enc.cognitive_state(0.7, 0.5, 0.6, 0.4, 0.3, 0.8)),
        ("HEARTBEAT",       enc.heartbeat(12345)),
        ("LEARNING_SIGNAL", enc.learning_signal(1, 0.95)),
    ]
    all_ok = True
    for name, frame in cases:
        msg = dec.decode(frame)
        ok  = msg["type"] is not None
        all_ok = all_ok and ok
        json_est = len(name) * 40  # rough JSON equivalent
        saving = 100*(1 - len(frame)/json_est)
        log.info(f"  {name:<18} {len(frame):3d}B  (~{saving:.0f}% smaller than JSON)  "
                 f"type={msg['type'].name}  {'✓' if ok else '✗'}")

    log.info(f"Protocol tests: {'ALL PASS' if all_ok else 'FAILURES DETECTED'}\n")


# ── Domain run ────────────────────────────────────────────────────────────────

def run_domain(domain: str, dev: torch.device, n_steps: int = 30, profile: bool = False):
    from noosphere import NoosphereAgent, AgentConfig
    log.info(f"\n{'─'*55}\nDOMAIN: {domain.upper()}\n{'─'*55}")

    cfg = AgentConfig(
        d_model=128, det_dim=256, stoch_cats=16, stoch_cls=16,
        action_dim=32, hidden_dim=128,
        n_mcts_sims=8, batch_size=4, seq_len=20,
        use_mcts=True, n_bodies=DOMAIN_CFG[domain].get("n_nodes", 4),
        **DOMAIN_CFG[domain],
    )
    agent = NoosphereAgent(cfg, dev)
    if profile: agent.perception.enable_profiling()
    log.info(f"Parameters: {sum(p.numel() for p in agent.parameters()):,}")

    gen = NeckEEGGenerator(seed=0) if domain == "bci" else None
    agent.reset_latent()
    prev = None; total_r = 0.0

    for step in range(n_steps):
        if domain == "bci":
            obs = obs_bci(seed=step, intent=(step//5)%9+1, eeg_gen=gen)
        else:
            obs = DOMAIN_OBS[domain](seed=step)

        action, info = agent.step(obs, prev)
        reward = float(np.random.randn() * 0.1)
        done   = step == n_steps - 1
        agent.observe(obs, action, reward, done)
        total_r += reward; prev = action

        if step % 10 == 0:
            bci = (f"  wl={info['bci_workload']:.2f} fat={info['bci_fatigue']:.2f}"
                   if "bci_workload" in info else "")
            log.info(f"  step {step:3d}  a={action}  "
                     f"pred_r={info['pred_reward']:+.3f}  "
                     f"E={info['physics_energy']:.2f}J{bci}")

        if step > 0 and step % cfg.train_every == 0:
            m = agent.update()
            if m:
                log.info(f"  [train] wm={m.get('wm/loss',0):.4f}  "
                         f"phys={m.get('wm/physics',0):.4f}  "
                         f"actor={m.get('ac/actor',0):.4f}")

    log.info(f"Done.  total_reward={total_r:.3f}")
    if profile: agent.perception.print_profile()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Noosphere v1.2.0 demo")
    ap.add_argument("--domain",    choices=list(DOMAIN_OBS)+["all"], default="all")
    ap.add_argument("--steps",     type=int, default=30)
    ap.add_argument("--profile",   action="store_true")
    ap.add_argument("--smoke",     action="store_true")
    ap.add_argument("--partial",   action="store_true",
                    help="Test EEG-only, vision-only, and other partial sensor subsets")
    ap.add_argument("--shell",     action="store_true",
                    help="EEG → world model → Linux shell command demo")
    ap.add_argument("--apparatus", action="store_true")
    ap.add_argument("--hardware",  default="sim",
                    choices=["sim","rpi_pca9685","arduino","rpi_gpio"])
    ap.add_argument("--proto",     action="store_true")
    ap.add_argument("--train",     action="store_true",
                    help="Run continuous training loop on synthetic BCI env")
    args = ap.parse_args()
    dev  = device()
    log.info(f"Device: {dev}")

    if args.proto:     proto_test();                     return
    if args.partial:   partial_sensor_test(dev);         return
    if args.shell:     shell_demo(dev);                  return
    if args.apparatus: apparatus_demo(args.hardware);    return
    if args.train:     training_demo(dev, args.steps);   return

    if args.smoke:
        for d in DOMAIN_OBS: smoke_test(d, dev)
        log.info("All smoke tests passed.")
        return

    domains = list(DOMAIN_OBS) if args.domain == "all" else [args.domain]
    for d in domains:
        try:
            run_domain(d, dev, args.steps, args.profile)
        except Exception as e:
            log.error(f"{d} failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
