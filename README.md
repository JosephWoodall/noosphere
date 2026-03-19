# Noosphere

A physics-informed world model agent with multimodal perception and brain-computer interface support.

---

## What it is

A single model that learns to perceive, predict, and act across physical systems — drones, legged robots, manipulation arms, fluid/soft-body systems, and BCI-controlled interfaces — using a unified multimodal architecture grounded in hard-coded physical laws.

The core loop:

```
Perceive → Model → Plan → Act → Observe → Learn → Repeat
```

A person wearing EEG electrodes thinks about what they want to do. The system decodes that intent, observes the environment through cameras and sensors, simulates what will happen if it acts using physics-constrained world model dynamics, plans the best action via MCTS in latent space, and learns continuously from the consequences.

---

## Architecture

### Three perception streams

| Stream | Encoder | Modalities |
|---|---|---|
| A | Patch Tokenizer → Transformer | RGB · depth · stereo · LiDAR · audio |
| B | S4 Structured State Space Model | EEG (64ch, continuous — no windowing) |
| C | Learned-Adjacency GNN | Joint angles · IMU · force/torque |

### Four fusion strategies

```
[CLS | S4_tok | GNN_tok | vis_patch_1 ... vis_patch_N]
```

1. **Single injection** — S4 + GNN summaries prepended as tokens at layer 0
2. **Multi-scale injection** — summaries re-injected at transformer layers 2 and 4
3. **Cross-attention** — transformer tokens attend into S4 sequence and GNN nodes at layers 1, 3, 5
4. **Gated fusion** — `γ = σ(W[pool_transformer; pool_external])` — dynamic per-modality gate

### Physics-augmented RSSM

World model dynamics use RK4 integration of hard-coded Newtonian physics:

```
Linear:     v̇ = (F_ext + F_grav + F_drag + F_contact) / m
Rotational: ω̇ = I⁻¹(τ - ω × Iω)
Quaternion: q̇ = ½ q ⊗ [0, ω]
Fluid:      ∂u/∂t ≈ ν∇²u  (coarse Navier-Stokes)
```

A neural residual corrector learns `Δs = s_actual - s_physics`. Conservation law losses enforce energy, momentum, angular momentum, and incompressibility constraints during training.

### Planning

MCTS operates entirely in latent space — the world model is the simulator. BCI motor intent seeds the root policy prior. Cognitive workload and fatigue dynamically scale the simulation budget. Actor-critic trains on H=15 step imagined rollouts via TD(λ).

---

## Quick start

```bash
pip install torch numpy scipy
python demo.py --domain drone
python demo.py --domain bci
python demo.py --domain all --profile
python demo.py --smoke
```

---

## Installation

```bash
git clone https://github.com/yourhandle/noosphere
cd noosphere
pip install -r requirements.txt
```

---

## Usage

```python
import torch
from noosphere import NoosphereAgent, AgentConfig

cfg   = AgentConfig(n_actions=6, n_eeg_ch=64, n_nodes=20)
agent = NoosphereAgent(cfg, device=torch.device("cuda"))

obs = {
    "rgb":        rgb_array,        # (H, W, 3) float32
    "depth":      depth_array,      # (H, W) float32, metres
    "eeg":        eeg_array,        # (64, T_samples) float32, μV
    "kinematics": joints_array,     # (20, 12) float32
}

action, info = agent.step(obs)
agent.observe(obs, action, reward, done)
metrics = agent.update()
```

### Adding a new sensor modality

```python
from noosphere.tokenizer import ImagePatchTokenizer

agent.perception.tokenizer.register_modality(
    "thermal",
    ImagePatchTokenizer(in_channels=1, d_model=cfg.d_model, patch_size=8)
)

# Pass "thermal": tensor in your obs dict — no other changes needed.
```

### Real-world sensor integration

Replace the synthetic generators in `demo.py` with your actual drivers. The agent interface expects plain NumPy arrays; the preprocessor handles conversion to tensors.

```python
# Example: Alpaca live trading as an environment
class AlpacaEnv:
    def step(self, action):
        # submit order, read bar data...
        obs = {"structured": bar_data}     # (T, F) float32
        return obs, reward, done, info
```

---

## Supported domains

| Domain | Actions | Primary sensors |
|---|---|---|
| Drone | thrust, roll, pitch, yaw, up, down | RGB, depth, IMU |
| Legged locomotion | 12 joint torques | Stereo RGB, joint state |
| Manipulation | 7-DOF arm + gripper | RGBD, force-torque |
| BCI control | 5 intent classes | EEG, visual feedback |
| Fluid / soft-body | pump, valve, heater | RGB, pressure array |

---

## Training

Two alternating phases:

**Phase A — World model** (on real sensor data)
```
L = λ_KL · KL(q ‖ p)  +  λ_r · ‖Dec(s) - e‖²  +  λ_rew · ‖f_r(s) - r‖²
  + BCE(f_d(s), done)  +  λ_phys · L_conservation
```

**Phase B — Actor-critic** (in imagination, world model frozen)
```
TD(λ): Gₜ = rₜ + γ[(1-λ)Vₜ₊₁ + λGₜ₊₁]
L_actor  = -E[log π(a|s) · Â] - α·H[π]
L_critic = MSE(V₁(s), G) + MSE(V₂(s), G)
```

---

## Project structure

```
noosphere/
├── __init__.py      public API
├── agent.py         NoosphereAgent — main perception→model→plan→act loop
├── perception.py    HybridPerceptionModel — three streams, four fusion strategies
├── tokenizer.py     UnifiedTokenizer — Stream A (vision, LiDAR, audio)
├── s4_eeg.py        S4EEGEncoder — Stream B (continuous EEG)
├── gnn.py           KinematicGNN — Stream C (learned-adjacency kinematics)
├── physics.py       PhysicsAugmentedRSSM + conservation laws
├── rssm.py          RSSM + ConsequenceModel + ObservationDecoder
├── planner.py       MCTSPlanner + Actor + Critic + ImaginationBuffer
└── memory.py        SequenceReplayBuffer + EpisodicMemory + WorkingMemory
demo.py              runnable demo — five domains, profiler, smoke test
requirements.txt
```

---

## Citation

```bibtex
@software{noosphere2025,
  title  = {Noosphere: Physics-Informed World Model Agent},
  year   = {2025},
  url    = {https://github.com/yourhandle/noosphere}
}
```
