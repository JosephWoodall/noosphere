# Noosphere v1.6.0

**Physics-informed world model agent with brain-computer interface control, Linux shell integration, community-shareable dynamics, and internal monitoring.**

---

## What it is

**Noosphere does not adapt to the world; it forces the world to adapt to the operator.** It is not an interface to a generic external AI. It is the Foundation Model itself, locally and idiosyncratically aligned to the single human operator.

Noosphere is an **Obedient consequence Engine** and a **Personalized Cognitive Foundation Model** that translates human intent — captured as electrical signals from three electrodes on the scalp — into physical or digital action, using an Autonomous RL Agent purely as a *Subjugated Path-Planner*.

When you think about moving your arm, or running a command, your motor cortex fires a pattern. Noosphere reads true neural signals at the scalp at 256 Hz, infers your macro-intent, explicitly executes your intention, and uses imagination strictly to predict and verify the consequence of your command before execution. 

**Zero-to-One Autogenous Independence:**
The system is explicitly forbidden from outsourcing biological mapping or representation learning to external third-party foundation models. All bootstrapping is strictly autogenous: Noosphere forges its own topological manifolds through continuous resting-state predictive coding and offline adversarial imagination. It guarantees total independence using strictly local data.

**Core loop:**

```
Perceive (Intent & Context) → Simulate Consequence → Act (Obey & Gate) → Observe → Learn → Repeat
```

The same trained world model can drive a physical robotic arm, execute Linux shell commands, or train entirely synthetically without hardware via SOTA physics simulation. Only the action vocabulary and executor change.

---

## Architecture

### Overview
```
1. Neural Models & Roles
Noosphere is composed of six interlinked neural modules:
- S4EEGEncoder (State-Space Signal Processor): 
    - Processes raw, noisy EEG time-series data. Uses **Evidential Deep Learning (EDL)** to map biological brain activity into Dirichlet-bounded intent probabilities (`intent_probs`), mathematically rigorous epistemic uncertainty, and cognitive states (fatigue, workload). EDL replaces uncalibrated logits with probabilities that are provably bounded and carry principled uncertainty estimates.
- NoosphereGNN (Graph Neural Network): 
    - Handles variable brain electrode topologies. It routes the extracted spatial features across the brain graph.    
- PhysicsAugmentedRSSM (World Model): 
    - A Recurrent State-Space Model fused with strict Hamiltonian physics priors. Its job is to simulate the environment and predict the future latent state based on the current state and a given action.
- ConsequenceModel (The Critic):
    - Evaluates a latent state produced by the World Model, predicting the expected reward, value, and termination (the probability of a catastrophic collision/failure).
- Actor (Digital Twin Policy):
    - A neural policy trained inside the World Model's imagination. It learns to avoid fatal states while explicitly mimicking the human's historical intent.
- MCTSPlanner (Monte Carlo Tree Search):
    - Rapidly simulates multiple future trajectories using the World Model to evaluate complex sequential actions before executing them.
- HardwareDiscoveryDaemon (Plug-and-Play Extremity Detector):
    - Scans USB/Serial/Bluetooth LE interfaces at runtime to detect newly attached physical extremities (arms, grippers, wheels). Automatically calls `KinematicGNN.inject_nodes()` to expand the spatial graph topology — no code changes required when adding new hardware.

2. Intent Translation (Shared Autonomy)
Human intent is translated into action through a Probabilistic Blending Circuit, rather than a rigid deterministic switch:
- Decoding: 
    - The S4 module decodes the raw EEG into a Dirichlet-distributed biological probability distribution (`p_bci`) via **Evidential Deep Learning (EDL)** and a mathematically rigorous epistemic uncertainty estimate (`1 - confidence`). High uncertainty → lower alpha blend → AI stabilises the action.
- AI Prior: 
    - Simultaneously, the Actor / MCTSPlanner evaluates the environment and proposes a digital twin probability distribution (p_ai).
- Blending: 
    - The two distributions are fused mathematically: p_final = (confidence * p_bci) + ((1 - confidence) * p_ai) When the user is focused and the BCI signal is strong, the human dictates the action. When the signal degrades, the AI stabilizes the choice using its historical understanding of the user.
- Safety Gating: 
    - Before execution, the system imagines executing the highest probability action. If the ConsequenceModel predicts a fatal outcome (sim_termination > 0.90), the ActBridge intercepts and blocks the translation, acting as an involuntary safety reflex.

3. Continuous Learning Loop
- Noosphere improves automatically while the user operates it. It does not require pausing for "calibration sessions."
- Experience Replay: 
    - Every action, observation, and biological intent is appended to the ReplayBuffer.
- World Model Updates: 
    - A background thread continuously updates the RSSM by minimizing reconstruction loss, Kullback-Leibler (KL) divergence, and physics conservation errors. It learns how the environment works.
- Imagination Training (TD-λ): 
    - The Actor and Critic train rapidly inside the frozen World Model's imagination.   
- The Imitation Prior (Behavioral Cloning): 
    - During imagination training, the Actor is penalized if it diverges from the actual commands the human previously executed in the same states. This ensures the AI culturally aligns with the user rather than becoming an autonomous explorer.

4. Performance Metics (Monitoring Export)
To build a comprehensive monitoring dashboard, hook into the noosphere.monitor.SystemMonitor or standard logging outputs and extract the following telemetry:
    - Learning Stability Metrics
        - wm/loss: World Model total loss. Spikes indicate the environment is exhibiting novel, unpredicted behavior.
        - wm/kl_loss: Predictability error. High KL divergence means the system is surprised.
        - wm/phys_loss: Physics conservation violations (useful for debugging synthetic environments).
        - ac/value_loss (Critic error): How badly the Consequence model mispredicted the outcome.
        - ac/bc_loss (Behavioral Cloning loss): Critical Metric. Measures how well the AI digital twin models the user. If this rises, the AI is losing its alignment with the human.
    -  Biological Telemetry
        - bci/confidence: Real-time signal-to-noise ratio. Drops indicate sensor degradation or user distraction.
        - bci/cognitive_workload: Mental strain. Triggers automatic increases to MCTS search budget when the user is overwhelmed.
        - bci/fatigue: Tracks neural drift and user exhaustion over the session.
    - Operational & Safety Metrics
        - safety/sim_termination: Expected catastrophic probability for the chosen action.
        - safety/interventions: A counter of how many commands the ActBridge successfully intercepted.
        - perf/gnn_ms & perf/s4_ms: Latency markers for the neural forward passes (must remain <50ms for smooth control).
```
### Three perception streams, always early-fused

```
Any subset of sensors is valid. Missing streams are masked — not zero-padded.
EEG-only, vision-only, kinematics-only all work without contaminating the CLS token.

Sensors
   │
   ├── RGB · depth · stereo · LiDAR · audio    → patch tokenizer   Stream A  tokenizer.py
   ├── EEG — 3 scalp electrodes @ 256 Hz       → S4 SSM            Stream B  s4_eeg.py
   └── joints · IMU · force/torque             → learned-adj GNN   Stream C  gnn.py

Token sequence entering the shared transformer:
   [CLS | S4_tok | GNN_tok | vis_patch_1 ... vis_patch_N]

All tokens attend to all other tokens from layer 1.
EEG directly modulates visual interpretation and vice versa.
```

**Stream A — Patch Tokenizer:** ViT-style patch embedding for spatial data. New modalities register at runtime via `register_modality()` with no architectural changes required.

**Stream B — S4 EEG Encoder:** EEG processed sample-by-sample with no windowing. The S4 continuous-time ODE uses HiPPO-LegS initialisation, which is mathematically optimal for memorising oscillatory signals. Outputs eleven values: a `summary` embedding, a full `sequence` for cross-attention, coarse `intent_logits` (5 classes), a continuous `continuous_xyz` coordinate prediction, a calibrated `confidence` scalar, five cognitive state dimensions (workload, attention, arousal, valence, fatigue), and a `planning_budget` that scales MCTS simulation count.

**Synthetic Validation Engine:** Representation learning and architectural continuous integration do not require a physical headset. The environment integrates a SOTA Synthetic Generator featuring a non-linear phase-coupled **Kuramoto Oscillator Network**, 1/f aperiodic spatial pink noise, and a mathematically rigorous **Leadfield Volume Conduction Matrix** that projects 5 cortical source topologies onto the 3 external C3/Cz/C4 sensors.

**EEG electrode placement:** Three electrodes on the scalp targeting the motor cortex (C3, Cz, C4). True neural activity (such as Event-Related Desynchronization during motor imagery) is the intentional signal. Standard EEG artifact rejection is applied.

**Stream C — Learned-Adjacency GNN:** Joint states as graph nodes. Adjacency is learned per layer from data with L1 sparsity regularisation. Normalisation uses element-wise diagonal scaling — O(N²) not O(N³). Topology converges toward actual physical coupling structure over training.

### Four fusion strategies

```
Layer 0: single injection    — S4 + GNN summaries prepended as tokens
Layer 1: cross-attention     — transformer Q attends into S4 sequence and GNN nodes
Layer 2: multi-scale inject  — summaries re-injected via gated residual
Layer 3: cross-attention
Layer 4: multi-scale inject
Layer 5: cross-attention
```

A dynamic gate `γ = σ(W[pool_transformer; pool_external])` at every injection point learns when external streams carry new information and suppresses them when they are redundant.

### Physics-Augmented RSSM

Latent state `sₜ = (hₜ, zₜ)`: deterministic GRU state for temporal memory, stochastic 32×32 discrete categorical latent for uncertainty and multimodal futures.

**Physics transition prior — hard-coded RK4 ODE:**
```
v̇ = (F_ext + F_grav + F_drag + F_contact) / m     Kelvin-Voigt contact
ω̇ = I⁻¹(τ − ω × Iω)                               Euler rotation
q̇ = ½ q ⊗ [0, ω]                                   quaternion, no gimbal lock
∂u/∂t ≈ ν∇²u                                        coarse Navier-Stokes
```

Gravity is cached as a `register_buffer` (not allocated per call). All six state components including fluid velocity are properly RK4-combined. The `ResidualCorrector` is a deep continuous-time MLP using `SwishResBlock` residual blocks — it learns `Δs = s_actual − s_physics` for unmodelled dynamics (soft-body deformations, turbulent flows, complex contacts), initialised to zero so analytical physics dominates at the start of training. Five conservation law penalties (energy, momentum, angular momentum, quaternion unit norm, incompressibility) are tensor losses — gradient flows through them to the residual corrector.

**KL loss (DreamerV3 balanced, NaN-safe):**
```
L_KL = 0.8 · KL(sg(q) ‖ p)  +  0.2 · KL(q ‖ sg(p))
probs clamped to [1e-6, 1] before log — prevents log(0) = −inf NaN propagation
```

### Digital consequence model

`EnhancedConsequenceModel` extends the base consequence model with `DigitalConsequenceHead`, which predicts structured digital outcomes from the latent state:

```
exit_logits   (B, 3)   — success / error / timeout classification
stdout_len    (B,)     — predicted stdout length (normalised)
state_change  (B,)     — predicted magnitude of environment change [0, 1]
next_digital  (B, 64)  — predicted next digital state vector
```

These heads are supervised directly from `ShellExecutor` output so the world model learns the texture of what commands do — not just whether they get reward.

### Intent Translation (Shared Autonomy)

Human intent is translated into action through a **Probabilistic Blending Circuit**, rather than a rigid deterministic switch:

1. **Decoding:** The `S4` module decodes the raw EEG into a biological probability distribution (`p_bci`) and an estimate of signal clarity (`confidence`).
2. **AI Prior:** Simultaneously, the `Actor`/`MCTSPlanner` evaluates the environment and proposes a digital twin probability distribution (`p_ai`).
3. **Blending:** The two distributions are fused mathematically: `p_final = (confidence * p_bci) + ((1 - confidence) * p_ai)`. When the signal is strong, the human dictates the action. When weak, the AI stabilizes the choice.
4. **Safety Gating:** If the `ConsequenceModel` predicts a fatal outcome (`sim_termination > 0.90`) for the highest probability action, the `ActBridge` blocks the translation, acting as an involuntary safety reflex.

**Budget scaling for MCTS:**
```
n_sims = max(5, n_sims_base × budget)
  where budget = 1 − 0.4·workload − 0.4·fatigue    (from S4 cognitive heads)
  and   recent failure streak → n_sims × 2          (from WorkingMemory)
```

### Continuous Learning Loop

Noosphere improves automatically while the user operates it. It does not require pausing for "calibration sessions."

- **Experience Replay:** Every action, observation, and biological intent is appended to the `ReplayBuffer`.
- **World Model Updates (Phase A):** A background thread continuously updates the `RSSM` by minimizing reconstruction loss, Kullback-Leibler (KL) divergence, and physics conservation errors.
  ```
  L = λ_kl · KL + λ_r · reconstruction + λ_rew · reward_prediction + termination + λ_phys · conservation + λ_gnn · sparsity
  ```
- **Imagination Training & Imitation Prior (Phase B):** The `Actor` and `Critic` train rapidly *inside* the frozen World Model's imagination. The `Actor` is heavily penalized (`L_bc`) if it diverges from the actual commands the human historically executed. This ensures the AI culturally aligns with the user rather than drifting.
- **Contrastive Learning (Phase C):** Always running, no labels. NT-Xent on four EEG augmentations to cluster neural embeddings continuously.

---

## Quick start

```bash
pip install torch numpy scipy scikit-learn
python demo.py --smoke              # shapes, NaN check, all domains
python demo.py --partial            # EEG-only, vision-only, mixed sensor subsets
python demo.py --shell              # EEG → world model → Linux commands
python demo.py --train --steps 200  # continuous training on synthetic BCI env
python demo.py --apparatus          # full BCI → IK → motor pipeline
python demo.py --proto              # NCP round-trip test
python demo.py --profile            # per-stream latency breakdown
```

---

## Installation

```bash
git clone https://github.com/JosephWoodall/noosphere
cd noosphere
pip install -r requirements.txt
```

Optional hardware backends (Bluetooth LE is natively supported out-of-the-box):
```bash
pip install rppal pwm-pca9685    # Raspberry Pi + PCA9685 (≥6 servos)
pip install pyserial             # Arduino serial
pip install redis                # Redis transport for NCP
```

---

## Usage

### Any subset of sensors works

```python
from noosphere import NoosphereAgent, AgentConfig

cfg   = AgentConfig(n_actions=6, n_eeg_ch=3, n_nodes=6)
agent = NoosphereAgent(cfg, device=torch.device("cpu"))

# All valid — missing streams are masked, not zero-padded
agent.step({"eeg": eeg_array})
agent.step({"rgb": rgb, "depth": depth})
agent.step({"eeg": eeg, "rgb": rgb, "kinematics": joints})
```

### Linux shell domain

```python
from noosphere import make_shell_space, ShellExecutor, ActBridge, Tier

space = make_shell_space(working_dir=".")
print(f"{space.n_actions} commands across 6 tiers")  # 148 commands

# Start read-only, expand as the world model proves reliable
executor = ShellExecutor(allow_tiers={Tier.SAFE_READ})
bridge   = ActBridge(space, executor, min_confidence=0.4)

# Expand to file writes when ready
executor.allow_tiers = {Tier.SAFE_READ, Tier.SAFE_WRITE}

# Inspect a tier-filtered vocabulary
safe_space = space.by_tier(Tier.SAFE_READ)   # 102 commands

cfg   = AgentConfig(n_actions=space.n_actions, n_eeg_ch=3)
agent = NoosphereAgent(cfg, device)
agent.act_bridge = bridge
```

### Continuous training with monitoring

```python
from noosphere import Trainer, TrainerConfig, SyntheticBCIEnv, Monitor, MonitorConfig

monitor = Monitor(MonitorConfig())
monitor.start()

agent   = NoosphereAgent(cfg, device)
trainer = Trainer(agent, SyntheticBCIEnv(), TrainerConfig())

# In the loop (or subclass Trainer):
for step in ...:
    action, info = agent.step(obs)
    ...
    monitor.record_step(step, info, train_metrics, env_info)
    for alert in monitor.drain_alerts():
        print(alert)   # coloured console, file, desktop notify, NCP
```

### WorldModelBundle — share dynamics with others

```python
from noosphere import export_bundle, load_bundle, inspect_bundle, check_compatibility

# Export after training — only world dynamics, no personal data
agent.export_bundle(
    "my_linux_dynamics.pt",
    domain_tags=["shell", "linux", "git", "docker"],
    description="300k steps on NixOS — full shell vocabulary",
    author="yourname",
    train_metrics=latest_metrics,
)

# Inspect any bundle before loading
print(inspect_bundle("community_dynamics.pt"))

# Check compatibility without loading
info = check_compatibility(agent, "community_dynamics.pt")
if info["compatible"]:
    result = agent.load_bundle("community_dynamics.pt")
    print(result["loaded"])   # ["rssm", "consequence", "obs_decoder"]
    # Your S4 encoder and calibration are untouched

# Load with strict=False to skip mismatched modules instead of raising
agent.load_bundle("community_dynamics.pt", strict_arch=False)
```

### Physical apparatus control

```python
from noosphere import (
    IntentionFilter, AnomalyDetector, CoordinatePredictor,
    CalibrationSession, MovementExecutor, ServoController
)

filt      = IntentionFilter()
anomaly   = AnomalyDetector()
predictor = CoordinatePredictor(d_model=256)
executor  = MovementExecutor()
servo     = ServoController(backend="bluetooth") # or "rpi_pca9685", "arduino", "sim"

# Session calibration — anchors GP to today's electrode placement
cal = CalibrationSession(predictor.gp)
# ... collect 5 reference movements ...

for segment in eeg_stream:
    if filt.is_intentional(segment) and anomaly.update_and_check(segment["probabilities"]):
        embedding = agent.perception.s4(eeg_tensor)["summary"][0].numpy()
        xyz, uncertainty = predictor.predict(embedding)   # GP + neural + smoother
        if xyz is not None:
            commands, actual_tip = executor.plan_and_execute(xyz)
            for angles in commands:
                servo.smooth_move(angles)
            # Feed error back to improve future predictions
            predictor.gp.add_sample(embedding, actual_tip)
```

### NCP inter-process communication

```python
from noosphere import NCPEncoder, NCPDecoder, NCPTransport, Channel

transport = NCPTransport.redis(host="127.0.0.1", port=6379)
# Falls back to in-process queue automatically if Redis unavailable

enc = NCPEncoder()
dec = NCPDecoder()

frame = enc.eeg_segment(raw_uv, probs, root_label, intent, xyz, vel, force, ts)
transport.publish(Channel.EEG_SOURCE, frame)

msg = dec.decode(frame)   # {"type": MsgType.EEG_SEGMENT, "payload": {...}}
```

### Adding a new sensor

```python
from noosphere.tokenizer import ImagePatchTokenizer

agent.perception.tokenizer.register_modality(
    "thermal",
    ImagePatchTokenizer(in_channels=1, d_model=cfg.d_model, patch_size=8)
)
# Pass "thermal": array in obs — nothing else changes
```

---

## Shell vocabulary

138 commands organised in six tiers. The world model trains from the bottom up; expand tiers as consequence prediction accuracy improves.

| Tier | Name | Count | Examples |
|---|---|---|---|
| 0 | SAFE_READ | 102 | `ls`, `ps`, `git status`, `nvidia-smi`, `docker ps`, `ss`, `lsblk`, `sensors`, `sqlite3 .tables`, `journalctl` |
| 1 | SAFE_WRITE | 14 | `make`, `cargo build`, `pytest`, `git commit`, `pip install -r` |
| 2 | PROCESS | 5 | `systemctl restart`, `docker restart`, `nohup` |
| 3 | NETWORK | 9 | `ping`, `curl`, `git pull`, `pip outdated`, `nix-channel --update` |
| 4 | SYSTEM | 5 | `pip upgrade`, `nixos-rebuild dry-build`, `nix-env` |
| 5 | DESTRUCTIVE | 3 | `rm -rf build/`, `git reset --hard`, `docker system prune` |

Command output is encoded as a 32-dimensional feature vector: exit code, stdout/stderr length, numeric value statistics, file path count, IP address count, JSON/table detection, error keyword density, change detection vs previous run of the same command.

Digital system state is captured as a 64-dimensional observation at each step: memory pressure, CPU load, disk usage, process counts, network connections, GPU utilisation, git working tree state, Docker container counts, Python environment, and file type distribution in the current directory.

---

## Supported domains

| Domain | Actions | Primary sensors |
|---|---|---|
| BCI apparatus | 5 intent classes | 3-ch scalp EEG, depth camera |
| Linux shell | 138 commands (6 tiers) | EEG, structured (64-dim system state) |
| Drone | 6 | RGB, depth, IMU |
| Legged locomotion | 12 | Stereo RGB, joint state |
| Manipulation | 8 | RGBD, force-torque |
| Fluid / soft-body | 4 | RGB, pressure array |

---

## Monitoring (Metrics & Telemetry)

`Monitor` runs as a background thread to track the continuous learning loop and the user's biological state. For an external dashboard (e.g., Prometheus/Grafana), extract the following metrics:

### A. Learning Stability Metrics
- **`wm/loss`**: World Model total loss. Spikes indicate the environment is exhibiting novel, unpredicted behavior.
- **`wm/kl_loss`**: Predictability error. High KL divergence means the system is surprised.
- **`wm/phys_loss`**: Physics conservation violations.
- **`ac/value_loss`**: How badly the consequence model mispredicted the outcome.
- **`ac/bc_loss` (Imitation Prior):** *Critical Metric.* Measures how well the AI models the user. If this rises, the AI is losing cultural alignment and shared autonomy is degrading.

### B. Biological Telemetry
- **`bci/confidence`**: Real-time signal-to-noise ratio. Controls the `alpha` blend between the biological brain and digital physics brain.
- **`bci/cognitive_workload`**: Mental strain. Triggers automatic increases to MCTS search budget when the user is overwhelmed.
- **`bci/fatigue`**: Tracks neural drift and user exhaustion over the session.

### C. Operational & Safety Metrics
- **`safety/sim_termination`**: Expected catastrophic probability for the chosen trajectory.
- **`safety/interventions`**: Counter of how many commands `ActBridge` intercepted.
- **`perf/gnn_ms` & `perf/s4_ms`**: Extracted latency markers to ensure inference remains <50ms.

**API Usage:**
```python
monitor = Monitor(MonitorConfig(
    mem_pct_warn=85.0,      # warn at 85% memory
    gpu_mem_pct_crit=97.0,  # critical at 97% GPU memory
    kl_max=20.0,            # alert on KL explosion
    cooldown_s=30.0,        # don't re-fire same alert within 30s
    desktop_notify=True,    # notify-send / osascript
    alert_file="alerts.jsonl",
))
monitor.start()
```

Alert channels: coloured console logging, `alerts.jsonl` append, desktop notification (`notify-send` on Linux, `osascript` on macOS), NCP frame on `ncp:alert` channel.

---

## Training

### Phases

**Phase A — World model** (real data from replay buffer):
```
L = λ_kl · KL  +  λ_r · reconstruction  +  λ_rew · reward prediction
  + termination BCE  +  λ_phys · conservation laws
  + λ_xyz · S4 coordinate supervision  +  λ_gnn · adjacency sparsity
```
All physics losses are tensor losses — gradient flows to the residual corrector.

**Phase B — Policy** (in imagination, world model frozen):
Actor-Critic with TD(λ), or π-StepNFT (critic-free, step-wise negative-aware fine-tuning from arxiv:2603.02083).

**Phase C — Contrastive EEG** (always running, no labels):
NT-Xent on four EEG augmentations. Single batched forward pass through both views.

**Reward shaping:**
```
reach_reward = exp(−10 · ‖actual_tip − target‖)   ∈ (0, 1]
             − 0.3  if IK did not converge
             − 0.5  if obstacle collision
             + 0.2  if prediction error < 3cm

shell_reward = (1.0 − tier × 0.1)  if exit code 0    (higher tier = lower base reward)
             + 0.1                  if stdout non-empty
             − 0.2 − 0.1·tier      if non-zero exit
             − 0.5                  if permission denied
```

### Warmup

1000 Phase A steps run before Phase B begins, ensuring the world model is functional before the policy starts exploiting it.

---

## WorldModelBundle format

A bundle is a `.pt` file containing:

```python
{
    "metadata": {
        "bundle_format":    "1.0",
        "noosphere_version": "1.6.0",
        "created_at":       "2025-11-14T09:30:00",
        "author":           "...",
        "domain_tags":      ["shell", "linux"],
        "description":      "...",
        "n_training_steps": 500000,
        "state_dim":        1536,
        "det_dim":          512,
        "stoch_cats":       32,
        "stoch_classes":    32,
        "consequence_type": "enhanced",
        "digital_state_dim": 64,
        "wm_loss":          0.0312,
        "kl_loss":          1.14,
        "reward_avg":       0.71,
    },
    "state_dicts": {
        "rssm":        {...},   # PhysicsAugmentedRSSM + all sub-modules
        "consequence": {...},   # ConsequenceModel or EnhancedConsequenceModel
        "obs_decoder": {...},   # ObservationDecoder
    }
}
```

**What is included:** RSSM (GRU, prior/posterior MLPs, physics state estimator, RK4 transition prior, residual corrector, conservation laws, physics projection), consequence model (reward/value/termination heads, digital prediction heads if present), observation decoder.

**What is excluded:** S4 EEG encoder (calibrated to one person's neural patterns), GNN topology (task-specific learned adjacency), patch tokenizer (fine structurally but trained with personal data), apparatus predictor (GP calibration data, neural head kinematic labels).

---

## Project structure

```
noosphere/
├── __init__.py       public API — all exports
├── agent.py          NoosphereAgent — step / observe / update
│                     export_bundle() · load_bundle() · run_calibration()
├── perception.py     HybridPerceptionModel — 3 streams, 4 fusion strategies
├── tokenizer.py      UnifiedTokenizer — Stream A
├── s4_eeg.py         S4EEGEncoder — Stream B (continuous 3-ch neck EEG)
├── gnn.py            KinematicGNN — Stream C (learned-adjacency)
├── physics.py        PhysicsAugmentedRSSM + conservation laws
├── rssm.py           RSSM + ConsequenceModel + DigitalConsequenceHead
├── planner.py        MCTSPlanner + Actor + Critic + ImaginationBuffer
├── memory.py         SequenceReplayBuffer + EpisodicMemory + WorkingMemory
├── apparatus.py      Full BCI → motor pipeline
│                     IntentionFilter · AnomalyDetector · SparseGPPredictor
│                     NeuralCoordinatePredictor · TemporalSmoother
│                     CalibrationSession · PositionErrorFeedback
│                     KinematicSolver · ObstacleSphere · MovementExecutor
├── hardware.py       ServoController (sim / Bluetooth / PCA9685 / Arduino / GPIO)
├── actions.py        138-command shell vocabulary across 6 tiers
│                     ShellExecutor · DigitalStateObserver · ShellOutputEncoder
│                     ApparatusExecutor · ActBridge (dual confidence gate)
├── bundle.py         WorldModelBundle — shareable world dynamics
│                     export_bundle · load_bundle · inspect_bundle
│                     check_compatibility · BundleMetadata
├── trainer.py        Trainer + TrainerConfig + Env
│                     BCIApparatusEnv (shaped reward) · SyntheticBCIEnv
│                     save_checkpoint · load_checkpoint
├── monitor.py        Monitor (background thread, 15 alert conditions)
│                     Console · file · desktop · NCP alert channels
├── proto.py          NCP binary protocol + NCPTransport (Redis / in-process)
├── learning.py       5 loss classes + LearningManager + EEGAugment
├── discovery.py      Hardware & Bluetooth DiscoveryDaemons — plug-and-play extremity detection
└── data/
    └── synth.py      ScalpEEGGenerator + obs_* builders + make_batch()

demo.py               Entry point
requirements.txt
```

22 files · 7,800+ lines · pure Python + PyTorch

---

## Data privacy

All computation runs onboard. No data leaves the device. No network connection required during operation. The bundle format explicitly excludes personal data by design — the architecture boundary between transferable dynamics and personal calibration is enforced in code, not convention.

---

## Citation

```bibtex
@software{noosphere2026,
  title  = {Noosphere: Physics-Informed World Model Agent with BCI Control},
  year   = {2026},
  url    = {https://github.com/JosephWoodall/noosphere},
  author = {Joseph Woodall},
}
```
