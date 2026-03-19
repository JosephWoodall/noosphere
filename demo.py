"""
demo.py
=======
Noosphere Demo

Runs the agent across five physical domains using synthetic sensor data.
Replace the data generators with your real sensor drivers.

Usage
-----
    python demo.py                         # all domains
    python demo.py --domain drone          # single domain
    python demo.py --domain drone --profile
    python demo.py --smoke                 # shape/NaN check only
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
    if torch.cuda.is_available():          return torch.device("cuda")
    if torch.backends.mps.is_available():  return torch.device("mps")
    return torch.device("cpu")


# ── Synthetic observation generators ──────────────────────────────────────────
# Replace each function body with your real sensor read-out.

def _obs_vision(H=64, W=64):
    return {
        "rgb":       np.random.rand(H, W, 3).astype(np.float32),
        "depth":     (np.abs(np.random.randn(H, W)) * 5 + 1).astype(np.float32),
        "rgb_right": np.random.rand(H, W, 3).astype(np.float32),
    }

def _obs_eeg(n_ch=64, sfreq=256):
    T = int(sfreq)
    eeg = np.random.randn(n_ch, T).astype(np.float32) * 10.0
    t   = np.linspace(0, 1, T)
    eeg[-8:] += (20 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
    return {"eeg": eeg, "electrode_mask": np.ones(n_ch, dtype=np.float32)}

def _obs_kinematics(n_nodes=20, feat=12):
    return {"kinematics": np.random.randn(n_nodes, feat).astype(np.float32)}

def _obs_imu(feat=13, steps=10):
    return {"structured": np.random.randn(steps, feat).astype(np.float32)}


DOMAIN_OBS = {
    "drone":        lambda: {**_obs_vision(),                     **_obs_imu(13, 10)},
    "legged":       lambda: {**_obs_vision(),   **_obs_kinematics(20, 30)},
    "manipulation": lambda: {**_obs_vision(),   **_obs_kinematics(6, 13)},
    "bci":          lambda: {**_obs_eeg(),       **_obs_vision(48, 64),   **_obs_imu(5, 5)},
    "fluid":        lambda: {**_obs_vision(),                     **_obs_imu(18, 20)},
}

DOMAIN_CFG = {
    "drone":        dict(n_actions=6,  n_nodes=1,  node_feat_dim=3),
    "legged":       dict(n_actions=12, n_nodes=20, node_feat_dim=30),
    "manipulation": dict(n_actions=8,  n_nodes=6,  node_feat_dim=13),
    "bci":          dict(n_actions=5,  n_nodes=1,  node_feat_dim=3),
    "fluid":        dict(n_actions=4,  n_nodes=2,  node_feat_dim=3),
}


# ── Smoke test ────────────────────────────────────────────────────────────────

def smoke_test(domain: str, dev: torch.device):
    from noosphere import NoosphereAgent, AgentConfig
    log.info(f"[smoke] {domain}")
    cfg = AgentConfig(
        d_model=64, det_dim=128, stoch_cats=8, stoch_cls=8,
        action_dim=32, hidden_dim=64, n_eeg_ch=64,
        n_mcts_sims=4, batch_size=2, seq_len=10,
        **DOMAIN_CFG[domain],
    )
    agent = NoosphereAgent(cfg, dev)
    agent.eval()
    agent.reset_latent()
    obs = DOMAIN_OBS[domain]()
    with torch.no_grad():
        action, info = agent.step(obs)
    assert isinstance(action, int), "action must be int"
    assert not any(np.isnan(v) for v in info.values() if isinstance(v, float)), "NaN in info"
    log.info(f"  action={action}  pred_reward={info['pred_reward']:.3f}  ✓")


# ── Domain demo ───────────────────────────────────────────────────────────────

def run_domain(domain: str, dev: torch.device, n_steps: int = 30, profile: bool = False):
    from noosphere import NoosphereAgent, AgentConfig
    log.info(f"\n{'─'*55}\nDOMAIN: {domain.upper()}\n{'─'*55}")

    cfg = AgentConfig(
        d_model=128, det_dim=256, stoch_cats=16, stoch_cls=16,
        action_dim=32, hidden_dim=128,
        n_eeg_ch=64, n_mcts_sims=8, batch_size=4, seq_len=20,
        use_mcts=True, n_bodies=DOMAIN_CFG[domain].get("n_nodes", 4),
        **DOMAIN_CFG[domain],
    )
    agent = NoosphereAgent(cfg, dev)
    if profile: agent.perception.enable_profiling()
    n = sum(p.numel() for p in agent.parameters())
    log.info(f"Parameters: {n:,}")

    agent.reset_latent()
    prev = None; total_r = 0.0

    for step in range(n_steps):
        obs    = DOMAIN_OBS[domain]()
        action, info = agent.step(obs, prev)
        reward = float(np.random.randn() * 0.1)
        done   = step == n_steps - 1
        agent.observe(obs, action, reward, done)
        total_r += reward
        prev    = action
        if step % 10 == 0:
            log.info(f"  step {step:3d}  a={action}  "
                     f"pred_r={info['pred_reward']:+.3f}  "
                     f"E={info['physics_energy']:.2f}J  "
                     f"sims={info['n_mcts_sims']}")
        if step > 0 and step % cfg.train_every == 0:
            m = agent.update()
            if m:
                log.info(f"  [train] wm={m.get('wm/loss',0):.4f}  "
                         f"phys={m.get('wm/physics',0):.4f}  "
                         f"actor={m.get('ac/actor',0):.4f}")

    log.info(f"Done. total_reward={total_r:.3f}")
    if profile: agent.perception.print_profile()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain",  choices=list(DOMAIN_OBS)+["all"], default="all")
    ap.add_argument("--steps",   type=int, default=30)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--smoke",   action="store_true")
    args = ap.parse_args()
    dev  = device()
    log.info(f"Device: {dev}")

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
