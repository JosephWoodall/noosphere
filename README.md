# Noosphere

**Physics-informed world model agent with multimodal perception and brain-computer interface apparatus control.**

---

## What it is

Noosphere is a system that closes the loop between human thought and physical action.

A person wearing three EEG electrodes on the back of their neck thinks about moving. The system reads the resulting neural signal, decodes the intent, observes the physical environment through cameras and sensors, builds an internal model of how the world works — including hard-coded physics — simulates what will happen if it acts, plans the best action via search in latent space, executes it, and learns continuously from the outcome.

Everything runs onboard. No data leaves the device.

**Core loop:**

```
Perceive → Model → Plan → Act → Observe → Learn → Repeat
```

At each cycle: sensors are encoded into a unified embedding, the world model updates its latent state, MCTS searches for the best action in imagination, the action is executed, the outcome is observed, and all three learning systems update — supervised where labels are available, contrastive on unlabeled EEG, and reinforcement on the aggregate outcome signal.

---

## Architecture

### Three perception streams, always early-fused

The central insight is that EEG, vision, and kinematics are not independent. The person's neural state should influence how the visual scene is interpreted, and vice versa. Noosphere fuses all three streams into a single shared transformer from layer 1 onward — not at the end.

```
Raw sensors
    │
    ├── RGB · depth · stereo · LiDAR · audio
    │       ↓ patch tokenizer
    │       Stream A tokens  (B, N, d_model)
    │
    ├── EEG — 3 electrodes, posterior neck, 256 Hz
    │       ↓ S4 structured state space model
    │       Stream B token + sequence  (B, 1+T', d_model)
    │
    └── Joint angles · IMU · force/torque
            ↓ learned-adjacency GNN
            Stream C token + nodes  (B, 1+N_nodes, d_model)

Token sequence entering the shared transformer:
[CLS | S4_tok | GNN_tok | vis_patch_1 ... vis_patch_N]

Every token attends to every other from layer 1.
EEG features directly influence visual interpretation. No information barrier.
```

**Stream A — Patch Tokenizer**

RGB, depth, stereo, LiDAR, and audio are split into non-overlapping patches and projected to `d_model`. ViT-style patch embedding with factored 2D positional encoding. New modalities register at runtime via `register_modality()` — no architectural changes.

**Stream B — S4 Structured State Space Model (EEG)**

EEG is processed sample-by-sample at full temporal resolution. No windowing.

Why this matters: a P300 ERP spans approximately 50ms at 256Hz — about 12 samples. A patch boundary that falls midway through this event splits it across two tokens, destroying the shape that carries the classification label. The S4 continuous-time ODE has no boundaries. Every sample is integrated into the state.

The S4 state space: `ẋ(t) = Ax(t) + Bu(t)`, `y(t) = Cx(t) + Du(t)`. The HiPPO-LegS initialization of `A` is designed to optimally memorize continuous signals — aligned with the oscillatory structure of neural data. Training runs as a parallel FFT convolution `O(L log L)`. Inference runs as a single recurrence step `O(N)` — real-time capable.

**EEG electrode placement:** Three electrodes on the posterior neck (C7 level). At this site, neck muscle EMG is the signal, not the noise. Standard EEG artifact rejection is inverted: `MuscleArtifact` with `action=Intentional` is what gets published downstream. `CleanBrain` segments are discarded. The S4 encoder includes a motor intent decoder (5 classes) and cognitive state estimator (workload, attention, arousal, valence, fatigue) as auxiliary heads.

**Stream C — Learned-Adjacency GNN (Kinematics)**

Joint states are embedded as graph nodes. Edges are learned from data — not hardcoded from a skeleton. A flat transformer treats all joint pairs as equally related at initialization. The GNN starts with a data-driven bias toward physically coupled joints. Each message-passing layer has its own independent adjacency matrix, learned with sparsity regularization that drives the topology toward actual physical coupling structure.

---

## Fusion Strategies

All four strategies run simultaneously, alternating across transformer layers so no layer double-counts any stream:

**Strategy 1 — Single Injection (layer 0)**
S4 and GNN summaries are each compressed to one token and prepended to the vision sequence before the first attention computation. The transformer sees the full external context from layer 0. Cost: 2 extra tokens.

**Strategy 2 — Multi-scale Injection (layers 2, 4)**
Updated summaries re-injected at intermediate layers via gated residual addition. The gate `γ = σ(W[token_state; external_summary])` learns when the external signal is relevant at that abstraction level.

**Strategy 3 — Cross-Attention (layers 1, 3, 5)**
Transformer tokens (queries) attend into the full S4 temporal sequence and GNN node embeddings (keys/values). Any transformer token can pull from any EEG timestep or any joint node on demand — not just from the compressed summary token.

```
Q = transformer_tokens · W_Q
K = s4_sequence · W_K,   V = s4_sequence · W_V
out = softmax(QKᵀ / √d_k) · V
output = transformer + γ · cross_out
γ = σ(W[mean(transformer); mean(s4_sequence)])
```

**Strategy 4 — Gated Fusion (all injection points)**
The gate is conditioned on the pooled state of both the transformer and the external stream. If visual tokens already explain the current state, the gate suppresses EEG/GNN contributions. If EEG carries novel information not in vision, the gate opens. The model learns which sensors are informative at each layer and timestep.

---

## Physics-Augmented RSSM

The world model maintains a structured latent state `sₜ = (hₜ, zₜ)`:

- `hₜ` — deterministic GRU state capturing long-range temporal dependencies
- `zₜ` — stochastic discrete categorical latent capturing uncertainty and multimodal futures

Two forward modes: `observe_step` (uses real observations, trains the posterior) and `imagine_step` (runs the prior for planning without accessing reality).

**Physics transition prior (hard-coded RK4 ODE)**

```
Linear:     v̇ = (F_ext + F_grav + F_drag + F_contact) / m
               F_drag    = -½ρ·Cd·A·|v|·v
               F_contact = k·δ·n̂ + b·v_rel   (Kelvin-Voigt)
Rotational: ω̇ = I⁻¹(τ - ω × Iω)
Quaternion: q̇ = ½ q ⊗ [0, ω]
Fluid:      ∂u/∂t ≈ ν∇²u
```

The neural **residual corrector** learns only `Δs = s_actual - s_physics` — what the physics gets wrong. It is initialized near-zero. The network never needs to rediscover Newton's second law.

**Conservation law losses** enforce five mathematical identities as differentiable constraints during training: energy (work-energy theorem), momentum (impulse-momentum), angular momentum, quaternion unit norm, and fluid incompressibility. These ensure physically consistent predictions on out-of-distribution observations.

**KL loss (DreamerV3 balanced):**
```
L_KL = 0.8 · KL(sg(q) ‖ p)  +  0.2 · KL(q ‖ sg(p))
clamp(min=free_nats=1.0)
```

---

## Planning

The world model is frozen during planning and acts as a perfect simulator — no real environment calls during search.

**MCTS in latent space**

```
Select   → UCB: Q(s,a) + c · P(a|s) · √(Σn_parent) / (1 + n_a)
Expand   → imagine_step(h, z, a) for each action
Evaluate → imagined rollout to horizon H, bootstrap with value head
Backup   → propagate value up the search path
```

BCI motor intent seeds the root policy prior. Cognitive workload and fatigue scale the simulation budget:
```
n_sims = max(5, n_sims_base × (1 - 0.4·workload - 0.4·fatigue))
```

**Actor-Critic** trains on H=15 imagined rollouts via TD(λ) with clipped double-Q critic.

**π-StepNFT** (arxiv:2603.02083): critic-free policy fine-tuning using step-wise negative-aware contrastive loss. Labels imagined trajectories positive (clean reach) or negative (collision / IK failure). No value network required. Single forward pass per update. Better OOD generalization for sparse-reward apparatus control.

```
L = -β · Σₜ wₜ · [log π(aₜ|sₜ)⁺ - log π(aₜ|sₜ)⁻]
wₜ = t/H  (linearly increasing toward horizon)
```

---

## Quick start

```bash
pip install torch numpy scipy scikit-learn
python demo.py --smoke              # verify all shapes, no NaNs
python demo.py --proto              # NCP protocol round-trip test
python demo.py --apparatus          # full BCI → IK → motor pipeline
python demo.py --domain bci         # BCI domain with world model
python demo.py --domain all         # all five domains
python demo.py --profile            # per-stream latency breakdown
```

---

## Installation

```bash
git clone https://github.com/yourhandle/noosphere
cd noosphere
pip install -r requirements.txt
```

Optional hardware backends:
```bash
pip install rppal pwm-pca9685    # Raspberry Pi + PCA9685 (recommended, 6+ servos)
pip install pyserial             # Arduino serial
pip install redis                # Redis transport for NCP
```

---

## Usage

### Basic agent

```python
import torch
from noosphere import NoosphereAgent, AgentConfig

cfg   = AgentConfig(n_actions=6, n_eeg_ch=3, n_nodes=6)
agent = NoosphereAgent(cfg, device=torch.device("cpu"))

obs = {
    "eeg":        eeg_array,        # (3, 256) float32, μV — 3 neck electrodes
    "rgb":        rgb_array,        # (H, W, 3) float32 [0, 1]
    "depth":      depth_array,      # (H, W) float32, metres
    "kinematics": joints_array,     # (n_nodes, node_feat_dim) float32
}

action, info = agent.step(obs)
agent.observe(obs, action, reward, done)
metrics = agent.update()
```

`info` returns: `pred_reward`, `pred_value`, `termination_prob`, `physics_energy`, `n_mcts_sims`, and — when EEG is present — `bci_workload`, `bci_fatigue`, `bci_attention`, `bci_arousal`, `bci_valence`.

### BCI apparatus control

```python
from noosphere.apparatus import (
    IntentionFilter, AnomalyDetector, CoordinatePredictor, MovementExecutor
)
from noosphere.hardware import ServoController

filt      = IntentionFilter()
anomaly   = AnomalyDetector()
predictor = CoordinatePredictor()
executor  = MovementExecutor()
servo     = ServoController(backend="rpi_pca9685")  # or "sim"

# For each 1-second EEG segment:
if filt.is_intentional(segment) and anomaly.update_and_check(segment["probabilities"]):
    feats  = CoordinatePredictor.extract_features(segment)
    if segment["hierarchical"]["kinematic"]:
        kin = segment["hierarchical"]["kinematic"]
        predictor.add_sample(feats, [kin["x"], kin["y"], kin["z"]])
    target = predictor.predict(feats)
    if target is not None:
        for angles_deg in executor.plan_and_execute(target):
            servo.smooth_move(angles_deg)
```

### Obstacle avoidance

The arm's range of motion is a continuous 3D vector space bounded by arm reach. The depth camera scans the environment and populates this space with obstacle points. Before each movement, the planner routes a collision-free arc:

```python
executor.obstacles.update_from_depth(
    depth_map=depth_array,    # (H, W) float32, metres
    K=camera_intrinsics,      # (3, 3)
    T_cam_world=camera_pose,  # (4, 4) SE3, optional
)
commands = executor.plan_and_execute(target_xyz=np.array([0.2, 0.1, 0.3]))
for angles_deg in commands:
    servo.set_all_angles(angles_deg)
```

### NCP communication

```python
from noosphere.proto import NCPEncoder, NCPDecoder, Channel

enc   = NCPEncoder()
dec   = NCPDecoder()

frame = enc.eeg_segment(raw_uv, probs, root_label, intent, xyz, vel, force, ts)
r.publish(Channel.EEG_SOURCE, frame)      # Redis, or any transport

msg = dec.decode(frame)
# msg["type"], msg["payload"], msg["seq"]
```

NCP frame size comparison at 256 Hz:

| Message | NCP | JSON | Reduction |
|---|---|---|---|
| EEG_SEGMENT | 84 B | ~820 B | 90% |
| DESTINATION | 22 B | ~120 B | 82% |
| MOTOR_CMD | 35 B | ~200 B | 82% |

---

## Adding a new sensor modality

```python
from noosphere.tokenizer import ImagePatchTokenizer

agent.perception.tokenizer.register_modality(
    "thermal",
    ImagePatchTokenizer(in_channels=1, d_model=cfg.d_model, patch_size=8)
)

# Pass "thermal": array in obs. Nothing else changes.
obs = {"rgb": ..., "thermal": thermal_array, "eeg": ...}
```

---

## Real-world sensor integration

Replace the generators in `noosphere/data/synth.py` with your hardware drivers. The agent interface expects plain NumPy arrays — all tensor conversion is handled internally.

```python
# EEG — any 256 Hz amplifier
raw = amp.read_samples(256)           # (3, 256) float32, microvolts
obs = {"eeg": raw, "electrode_mask": np.ones(3)}

# Depth camera — RealSense, Azure Kinect, etc.
depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * 0.001  # mm → m
obs["depth"] = depth

# Anything that returns (T, F) arrays works as "structured"
obs["structured"] = imu.read()        # (T, F) float32
```

---

## Supported domains

| Domain | Actions | Primary sensors |
|---|---|---|
| Drone | 6 (thrust, roll, pitch, yaw, up, down) | RGB, depth, IMU (13-dim) |
| Legged locomotion | 12 joint torques | Stereo RGB, joint state (30 DOF) |
| Manipulation | 8 (7-DOF arm + gripper) | RGBD, force-torque |
| BCI apparatus | 5 intent classes | 3-ch neck EEG, visual feedback |
| Fluid / soft-body | 4 (pump, valve, heater, nozzle) | RGB, pressure array |

---

## Training

Three phases, all sharing one agent object.

**Phase A — World model** (real data from replay buffer)
```
L = λ_KL · KL(q‖p)  +  λ_r · ‖Dec(s) - e‖²  +  λ_rew · ‖f_r(s) - r‖²
  + BCE(f_d(s), done)  +  λ_phys · L_conservation
```
Updates: perception, RSSM, physics, consequence model.

**Phase B — Policy** (imagination, world model frozen)

Actor-Critic (TD(λ)):
```
Gₜ = rₜ + γ[(1-λ)Vₜ₊₁ + λGₜ₊₁]
L_actor  = -E[log π(a|s) · Â] - α·H[π]
L_critic = MSE(V₁, G) + MSE(V₂, G)
```

π-StepNFT (recommended for apparatus control, no critic):
```
L = -β · Σₜ wₜ · [log π(aₜ|sₜ)⁺ - log π(aₜ|sₜ)⁻],   wₜ = t/H
```

**Phase C — Contrastive EEG** (always running, no labels needed)
```
L = NT-Xent(encoder(aug₁(eeg)), encoder(aug₂(eeg)))
```

Warmup: 1000 Phase A steps before Phase B begins.

---

## Project structure

```
noosphere/
├── __init__.py       public API
├── agent.py          NoosphereAgent — step / observe / update
├── perception.py     HybridPerceptionModel — 3 streams, 4 fusion strategies
├── tokenizer.py      UnifiedTokenizer — Stream A (vision, LiDAR, audio)
├── s4_eeg.py         S4EEGEncoder — Stream B (continuous 3-ch neck EEG)
├── gnn.py            KinematicGNN — Stream C (learned-adjacency)
├── physics.py        PhysicsAugmentedRSSM + conservation laws
├── rssm.py           RSSM + ConsequenceModel + ObservationDecoder
├── planner.py        MCTSPlanner + Actor + Critic + ImaginationBuffer
├── memory.py         SequenceReplayBuffer + EpisodicMemory + WorkingMemory
├── apparatus.py      IntentionFilter + IK + ObstacleSphere + MovementExecutor
├── hardware.py       ServoController (sim / PCA9685 / Arduino / GPIO)
├── proto.py          NCP binary protocol — encoder, decoder, channel names
├── learning.py       Supervised + NT-Xent + π-StepNFT
└── data/
    └── synth.py      All synthetic test data — one file, all modalities

demo.py               Entry point (--smoke, --proto, --apparatus, --domain, --profile)
requirements.txt
```

17 files · 4,650 lines · pure Python + PyTorch

---

## Data privacy

All computation runs onboard. No data leaves the device. No network connection required.

---

## Citation

```bibtex
@software{noosphere2025,
  title  = {Noosphere: Physics-Informed World Model Agent with BCI Apparatus Control},
  year   = {2025},
  url    = {https://github.com/yourhandle/noosphere}
}
```
