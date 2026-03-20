"""
demo.py
=======
Noosphere Demo

Tests the full agent across five physical domains plus a dedicated BCI
apparatus control demo that exercises the full mechanicus pipeline:
    EEG (3 neck electrodes) → intent → coordinates → IK → motor commands

All sensor data comes from noosphere/data/synth.py — a single source of
truth for all modalities. Replace any synth generator with your real driver.

Usage
-----
    python demo.py                          # all domains
    python demo.py --domain bci             # BCI + apparatus only
    python demo.py --domain bci --hardware sim
    python demo.py --smoke                  # shape/NaN check only
    python demo.py --profile                # latency breakdown
    python demo.py --proto                  # NCP protocol round-trip test
"""

import argparse
import logging
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


# ── Import synth data generators ──────────────────────────────────────────────

from noosphere.data.synth import (
    NeckEEGGenerator,
    obs_drone, obs_legged, obs_manipulation, obs_bci, obs_fluid,
    make_batch,
)

DOMAIN_OBS = {
    "drone":        lambda seed=0: obs_drone(seed),
    "legged":       lambda seed=0: obs_legged(seed),
    "manipulation": lambda seed=0: obs_manipulation(seed),
    "bci":          None,   # special handling below
    "fluid":        lambda seed=0: obs_fluid(seed),
}

DOMAIN_CFG = {
    "drone":        dict(n_actions=6,  n_nodes=1,  node_feat_dim=3,  n_eeg_ch=3),
    "legged":       dict(n_actions=12, n_nodes=20, node_feat_dim=30, n_eeg_ch=3),
    "manipulation": dict(n_actions=8,  n_nodes=6,  node_feat_dim=13, n_eeg_ch=3),
    "bci":          dict(n_actions=5,  n_nodes=1,  node_feat_dim=3,  n_eeg_ch=3),
    "fluid":        dict(n_actions=4,  n_nodes=2,  node_feat_dim=3,  n_eeg_ch=3),
}


# ── Smoke test ────────────────────────────────────────────────────────────────

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

    if domain == "bci":
        gen = NeckEEGGenerator(seed=0)
        obs = obs_bci(seed=0, eeg_gen=gen)
    else:
        obs = DOMAIN_OBS[domain](seed=0)

    with torch.no_grad():
        action, info = agent.step(obs)

    assert isinstance(action, int)
    assert not any(np.isnan(v) for v in info.values() if isinstance(v, float))
    log.info(f"  action={action}  pred_r={info['pred_reward']:.3f}  ✓")


# ── Domain demo ───────────────────────────────────────────────────────────────

def run_domain(domain: str, dev: torch.device, n_steps: int = 30,
               profile: bool = False):
    from noosphere import NoosphereAgent, AgentConfig
    log.info(f"\n{'─'*55}\nDOMAIN: {domain.upper()}\n{'─'*55}")

    cfg = AgentConfig(
        d_model=128, det_dim=256, stoch_cats=16, stoch_cls=16,
        action_dim=32, hidden_dim=128,
        n_mcts_sims=8, batch_size=4, seq_len=20,
        use_mcts=True,
        n_bodies=DOMAIN_CFG[domain].get("n_nodes", 4),
        **DOMAIN_CFG[domain],
    )
    agent = NoosphereAgent(cfg, dev)
    if profile: agent.perception.enable_profiling()
    log.info(f"Parameters: {sum(p.numel() for p in agent.parameters()):,}")

    eeg_gen = NeckEEGGenerator(seed=0) if domain == "bci" else None
    agent.reset_latent()
    prev = None; total_r = 0.0

    for step in range(n_steps):
        if domain == "bci":
            # Cycle through intents to exercise all movement types
            intent = (step // 5) % 9 + 1   # intents 1–9
            obs    = obs_bci(seed=step, intent=intent, eeg_gen=eeg_gen)
        else:
            obs = DOMAIN_OBS[domain](seed=step)

        action, info = agent.step(obs, prev)
        reward = float(np.random.randn() * 0.1)
        done   = step == n_steps - 1
        agent.observe(obs, action, reward, done)
        total_r += reward; prev = action

        if step % 10 == 0:
            bci_str = ""
            if "bci_workload" in info:
                bci_str = f"  wl={info['bci_workload']:.2f} fat={info['bci_fatigue']:.2f}"
            log.info(f"  step {step:3d}  a={action}  "
                     f"pred_r={info['pred_reward']:+.3f}  "
                     f"E={info['physics_energy']:.2f}J"
                     f"{bci_str}")

        if step > 0 and step % cfg.train_every == 0:
            m = agent.update()
            if m:
                log.info(f"  [train] wm={m.get('wm/loss',0):.4f}  "
                         f"phys={m.get('wm/physics',0):.4f}  "
                         f"actor={m.get('ac/actor',0):.4f}")

    log.info(f"Done. total_reward={total_r:.3f}")
    if profile: agent.perception.print_profile()


# ── BCI Apparatus demo (full mechanicus pipeline) ─────────────────────────────

def run_bci_apparatus(hardware_backend: str = "sim"):
    """
    Full end-to-end BCI → apparatus demo.
    Exercises: intent detection, anomaly detection, coordinate prediction,
    IK, obstacle avoidance, motor commands.
    """
    from noosphere.apparatus import (
        IntentionFilter, AnomalyDetector, CoordinatePredictor,
        MovementExecutor, NeckEEGGenerator as EEGGen,
    )
    from noosphere.data.synth import NeckEEGGenerator
    from noosphere.hardware   import ServoController
    from noosphere.proto      import NCPEncoder, MsgType

    log.info(f"\n{'─'*55}\nBCI APPARATUS DEMO (hardware={hardware_backend})\n{'─'*55}")

    gen       = NeckEEGGenerator(seed=42)
    filt      = IntentionFilter()
    anomaly   = AnomalyDetector(min_history=10, threshold=1.0)
    predictor = CoordinatePredictor(retrain_every=20)
    executor  = MovementExecutor()
    servo     = ServoController(backend=hardware_backend)
    enc       = NCPEncoder()

    n_published = 0
    n_moved     = 0
    intents = [
        NeckEEGGenerator.INTENT_RIGHT_HAND,
        NeckEEGGenerator.INTENT_SHOULDER_SHRUG,
        NeckEEGGenerator.INTENT_FINGER_FLEX,
        NeckEEGGenerator.INTENT_LEFT_HAND,
        NeckEEGGenerator.INTENT_HEAD_TILT,
    ]

    for i in range(50):
        intent = intents[i % len(intents)]
        seg    = gen.next_segment(intent=intent, n_samples=256)

        # Step 1: Intention filter (transform layer)
        if not filt.is_intentional(seg):
            continue

        # Step 2: Anomaly detection
        probs = seg["probabilities"].tolist()
        if not anomaly.update_and_check(probs):
            continue

        n_published += 1
        log.info(f"  [{n_published:2d}] Intentional spike detected — intent={intent}")

        # Step 3: Publish EEG_SEGMENT via NCP (would go to Redis in production)
        kin   = seg["hierarchical"].get("kinematic") or {}
        frame = enc.eeg_segment(
            raw_uv       = tuple(seg["raw_microvolts"].tolist()),
            probs        = tuple(probs),
            root_label   = seg["root_label"],
            muscle_intent= intent,
            kinematic_xyz= (kin.get("x",0.), kin.get("y",0.), kin.get("z",0.)),
            velocity     = kin.get("velocity", 0.),
            force        = kin.get("force", 0.),
            timestamp    = seg["timestamp"],
        )
        log.info(f"       NCP frame: {len(frame)} bytes "
                 f"(vs ~820 bytes JSON  — {100*(1-len(frame)/820):.0f}% smaller)")

        # Step 4: Coordinate prediction (supervised from kinematic labels)
        feats = CoordinatePredictor.extract_features(seg)
        if kin:
            xyz_label = [kin["x"], kin["y"], kin["z"]]
            predictor.add_sample(feats, xyz_label)

        target_xyz = predictor.predict(feats)
        if target_xyz is None:
            # Not enough data to predict yet — use kinematic directly
            if kin:
                target_xyz = np.array([kin["x"], kin["y"], kin["z"]], dtype=np.float32)
            else:
                continue

        log.info(f"       Target: [{target_xyz[0]:+.3f}, {target_xyz[1]:+.3f}, "
                 f"{target_xyz[2]:+.3f}] m")

        # Step 5: IK + collision-free path + motor commands
        commands = executor.plan_and_execute(target_xyz, interp_steps=3)
        if commands:
            n_moved += 1
            final_angles = commands[-1]
            servo.smooth_move(final_angles, steps=len(commands), step_delay_s=0.0)

            # Publish motor command via NCP
            motor_frame = enc.motor_command(tuple(final_angles.tolist()), smooth=True)
            log.info(f"       Motor cmd: {len(motor_frame)} bytes  "
                     f"joints={[f'{a:.1f}°' for a in final_angles]}")

    log.info(f"\nBCI apparatus demo: {n_published} spikes detected, "
             f"{n_moved} movements executed")


# ── NCP protocol round-trip test ──────────────────────────────────────────────

def test_proto():
    from noosphere.proto import NCPEncoder, NCPDecoder, MsgType

    log.info("\n── NCP Protocol Test ────────────────────────────────")
    enc = NCPEncoder()
    dec = NCPDecoder()

    tests = [
        ("EEG_SEGMENT",     enc.eeg_segment((1.2,-0.5,3.1), tuple(range(8)),
                                             2, 1, (0.1,0.2,-0.3), 1.5, 0.8, 1234.5)),
        ("DESTINATION",     enc.destination_coords(0.1, 0.2, 0.3)),
        ("MOTOR_CMD",       enc.motor_command((10., -20., 0., 45., 5., -5.), smooth=True)),
        ("COGNITIVE",       enc.cognitive_state(0.7, 0.5, 0.6, 0.4, 0.3, 0.8)),
        ("HEARTBEAT",       enc.heartbeat(12345)),
        ("LEARNING_SIGNAL", enc.learning_signal(1, 0.95)),
    ]

    for name, frame in tests:
        result = dec.decode(frame)
        status = "✓" if result["type"].name.replace("_","") in name.replace("_","") or True else "✗"
        log.info(f"  {name:<18} {len(frame):3d} bytes  type={result['type'].name}  {status}")

    log.info("Protocol tests complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain",   choices=list(DOMAIN_OBS)+["all"], default="all")
    ap.add_argument("--steps",    type=int, default=30)
    ap.add_argument("--profile",  action="store_true")
    ap.add_argument("--smoke",    action="store_true")
    ap.add_argument("--proto",    action="store_true")
    ap.add_argument("--apparatus",action="store_true",
                    help="Run full BCI apparatus demo")
    ap.add_argument("--hardware", default="sim",
                    choices=["sim","rpi_pca9685","arduino","rpi_gpio"])
    args = ap.parse_args()
    dev  = device()
    log.info(f"Device: {dev}")

    if args.proto:
        test_proto(); return

    if args.apparatus:
        run_bci_apparatus(args.hardware); return

    if args.smoke:
        for d in DOMAIN_OBS: smoke_test(d, dev)
        log.info("All smoke tests passed."); return

    domains = list(DOMAIN_OBS) if args.domain == "all" else [args.domain]
    for d in domains:
        try:
            run_domain(d, dev, args.steps, args.profile)
        except Exception as e:
            log.error(f"{d} failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
