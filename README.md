# Noosphere — From BCI Classification to Continuous Human-Machine Symbiosis

**Author:** Joseph Woodall

This repository documents the full evolution of a Brain-Computer Interface system for prosthetic arm control — from an academic classifier (`v1`) to a continuously learning digital twin (`v2`). Both versions are preserved here as active codebases.

---

## Project Overview

The central goal is a zero-latency interface between biological thought and mechanical action. An EEG signal captured from the scalp contains information about the user's motor intent. The question this project answers is: *how do you convert that information into continuous, reliable control of a robotic arm?*

Version 1 answered that question with a discriminative classifier trained on labeled datasets. Version 2 rejects that answer and replaces it with a generative agent that interprets intent continuously, adapts in real-time, and never runs out of output categories because it never had any.

---

## v1 — Riemannian-S4 Encoder (Discriminative BCI Classifier)

**Directory:** `v1/`

### What it does

v1 is a multi-stage neural architecture that classifies EEG signals into discrete motor imagery categories (rest, left hand, right hand, both hands). It was benchmarked on five MOABB datasets across 193 subjects and submitted to IEEE JBHI as the paper *"Riemannian Selective State Space (RS-S4) Encoder for Subject-Invariant Brain-Computer Interfaces."*

### Architecture

```
Raw EEG (C × T)
    ↓ Euclidean Alignment (EA) — whitens to a reference covariance, removes subject drift
    ↓ RiemannianStem — log-covariance vectorization → subject embedding token
    ↓ WaveletStem — parametric Morlet wavelets (multi-scale alpha/beta decomposition)
    ↓ SelectiveS4Block × N — Mamba-style data-dependent SSM, symplectic recurrence,
                              FiLM conditioning, GLU output
    ↓ MultiHeadAttentionPooling — sequence → fixed-length representation
    ↓ Dirichlet EDL Head — class probabilities + aleatoric uncertainty (α / Σα)
    → Discrete class label (0 = rest, 1 = right, 2 = left, 3 = both)
```

Supporting systems:
- **Mode-Selective Mamba RSSM** (`rssm.py`) — world model for consequence prediction before execution
- **Physics-Augmented RSSM** (`physics.py`) — RK4 Hamiltonian integration enforcing conservation laws
- **MCTS Planner** (`planner.py`) — imagined rollouts before committing to an action
- **IntentArbiter** (`intent.py`) — blends neural signal with AI safety policy: `α·brain + (1-α)·policy`
- **Sinkhorn-Knopp Attractor Resonance (SKAR)** (`riemann.py`) — optimal transport alignment to a learned cognitive attractor manifold
- **P2P Network** (`network.py`, `proto.py`) — decentralized Dynamics Insight sharing (no raw neural data transmitted)

### Benchmarks

Evaluated using LOSO (Leave-One-Subject-Out) and 75/25 chronological within-subject split protocols on:
- Schirrmeister2017, Lee2019, Cho2017, Weibo2014, PhysioNetMI

Baseline comparisons: CSP+LDA, EEGNet-8,2, ShallowConvNet.

### The Fundamental Limitation

v1 works. The problem is what it *cannot* do.

**Fixed category ceiling.** The output is one of N pre-defined labels. You define the label set at training time. To add a new movement type, you collect new labeled data and retrain from scratch. The prosthetic cannot do anything you did not anticipate when you designed the experiment.

**No continuous control.** A 6-DOF robotic arm requires smooth, simultaneous joint commands — not sequential button presses. Mapping "right hand" to a pre-programmed servo trajectory is not control; it is a macro. Real prosthetic use requires continuous, proportional output for every degree of freedom at once.

**Static training, static user.** v1 trains offline on a frozen dataset. Every new user requires a full calibration session with labeled trials. The model does not update from proprioceptive feedback; it has no mechanism to notice when its predictions are wrong and correct itself.

**The core insight, in the user's words:**
> *"With a discriminative process, I am limited to the categories in which I can create actions from... I think with an agentic one I can somehow disclose [arbitrary continuous intent]."*

That insight is what v2 is built to deliver.

---

## Why v1 → v2: The Structural Argument

The shift from v1 to v2 is not a performance improvement. It is a change in what the system fundamentally *is*.

| Dimension | v1 | v2 |
|---|---|---|
| Output space | Discrete: 4 classes | Continuous: R^6 (6-DOF joint commands) |
| Intent model | Discriminative posterior P(class \| EEG) | Generative prediction of next motor state |
| Adaptation | Offline retraining | Online continual learning from proprioceptive feedback |
| Category limit | Hard ceiling at N labels | No ceiling — output is a vector in [-1,1]^6 |
| Safety | EDL uncertainty + consequence model | ERN detection + adaptive Kalman + watchdog gate |
| Pretraining | Supervised on labeled MOABB data | JEPA self-supervised — no labels needed |
| Input modalities | EEG only | EEG + HRV + GSR + proprioception |
| Latency model | Batch inference on fixed-length epochs | Single-sample streaming via O(1) decode_step() |
| Hardware interface | Not implemented | ZeroMQ bridge + SimulatedHardware mock |

The discriminative framing of v1 is correct for a research paper. It is wrong for a prosthetic. A prosthetic arm does not ask "which of four movements are you doing?" It responds to a continuous, composable, time-varying intent signal. v2 produces that signal.

---

## v2 — Digital Twin (Continuous Intent Agent)

**Directory:** `v2_digital_self_replication/`

### What it does

v2 is a continuously learning agent that streams EEG (plus optional physiological sensors) and outputs a 6-DOF motor command vector in real time. It adapts from proprioceptive feedback — no labeled data required after JEPA pretraining. The long-term goal is complete zero-latency symbiosis between biological intent and mechanical action.

The design philosophy is taken directly from the user's stated vision:
> *"An additional consciousness that I can leverage as a supplement to my own... complete interaction and zero latency between what my biological form tells me that I should have and what the mechanical apparatus would be able to do."*

### Architecture

```
EEG (21ch, 256 Hz) + HRV (1ch) + GSR (1ch) + Proprioception (6ch)
    ↓
MultiModalFusion
    Linear projection per modality → sum → LayerNorm
    ↓
StreamEncoder  (N × BiosignalSSMBlock)
    ZOH discretization:  bar_A = exp(dt · A),  A initialized at -5 → provably stable
    Bilinear binding:    h = bar_A·h + bar_B·x + bar_W·tanh(h·x)
    GLU output + residual per block
    decode_step() → O(1) per sample streaming inference
    ↓
IntentDecoder
    → mu    ∈ [-1, 1]^6  — continuous normalized joint command
    → sigma ∈ (0, ∞)^6  — aleatoric uncertainty per DOF
    → ern_prob ∈ (0, 1)  — error-related negativity probability
    ↓
AdaptiveKalmanFilter
    State: [position (6), velocity (6)]  Constant-velocity model
    Measurement noise R = diag(sigma²)  ← uncertainty-adaptive
    High sigma → trust momentum; low sigma → trust prediction
    ↓
SafetyGate
    ERN gate:     P(ERN) > 0.7 → halt 500ms
    Uncertainty:  max(sigma) > 1.5 → halt 100ms
    Watchdog:     no input for > 2s → halt
    Emergency:    explicit call → halt until reset()
    ↓
ZMQBridge (port 5555 PUB / 5556 SUB)
    → Prosthetic hardware
```

### 6-DOF Output Space

| Index | DOF | Range |
|---|---|---|
| 0 | shoulder_yaw | [-1, 1] |
| 1 | shoulder_pitch | [-1, 1] |
| 2 | shoulder_roll | [-1, 1] |
| 3 | elbow_flex | [-1, 1] |
| 4 | wrist_rotate | [-1, 1] |
| 5 | grip_aperture | [-1, 1]  (−1 = closed, +1 = open) |

### Pretraining: JEPA (No Labels Required)

```
EEG window (B, T, 21)
    Split: context = x[:, :0.75T, :]   target = x[:, 0.75T:, :]

Student encoder f_θ:
    encode(context) → mean-pool → z_context (B, d_model)
    predictor g_φ(z_context) → z_pred (B, d_model)

Teacher encoder f_ψ (EMA of f_θ, no gradient):
    encode(target) → mean-pool → z_target (B, d_model)

Loss: MSE(z_pred, stop_gradient(z_target))
Teacher update: ψ ← 0.99·ψ + 0.01·θ
```

This teaches the encoder to build representations that can predict the future — directly useful for a proactive prosthetic controller.

### Online Adaptation

After JEPA pretraining and optional supervised fine-tuning, the twin continues to learn in the field:

1. Every step: proprioceptive feedback (actual arm position) is stored as `Experience(eeg, cmd_pred, cmd_actual)`
2. Every 100 steps: 5 gradient steps on 32 randomly sampled experiences
3. EMA smoothing (`decay=0.995`) after each update prevents catastrophic forgetting
4. The safety gate's ERN detector signals when a prediction was wrong — the optimizer can use this signal

### Synthetic Data

No physical sensors are required for development. All training data is generated synthetically:

**EEG** (`data/synthetic_eeg.py`): 21-channel, 256 Hz. Kuramoto oscillator networks model alpha (8-13 Hz), beta (13-30 Hz), gamma (30-60 Hz), and theta (4-8 Hz) bands. ERD (alpha/beta suppression) at C3 and ERS at C4 are scaled to the total motor activity level of the 6-DOF intent vector. ERN (error-related negativity, negative deflection at FCz) is triggered by rapid intent reversals.

**HRV** (`data/synthetic_physio.py`): Integral Pulse Frequency Modulation model with LF (Mayer waves, 0.07-0.12 Hz) and HF (respiratory sinus arrhythmia, 0.18-0.35 Hz) components. Arousal profile from motor activity modulates mean HR.

**GSR** (`data/synthetic_physio.py`): Double-exponential SCR kernel convolved with a Poisson event train. Tonic drift at 0.01-0.05 Hz. Phasic event rate scales with arousal.

### Memory System (4-Tier)

| Tier | Storage | Purpose |
|---|---|---|
| Short-term | RAM deque (512 cap) | Online adaptation replay buffer |
| Episodic | SQLite | Session summaries: duration, mean error, halt count |
| Semantic | FAISS flat index | Nearest-neighbour episode retrieval (optional) |
| Procedural | JSON key-value | Learned behavioral rules (e.g., grip threshold for subject) |

### File Structure

```
v2_digital_self_replication/
├── config.py                    # V2Config dataclass — all hyperparameters
├── core/
│   ├── stream_encoder.py        # ZOH SSMCell, BiosignalSSMBlock, MultiModalFusion, StreamEncoder
│   ├── intent_decoder.py        # IntentDecoder → (mu, sigma, ern_prob), IntentLoss
│   ├── kalman_filter.py         # AdaptiveKalmanFilter (R = diag(sigma²))
│   └── safety_gate.py           # SafetyGate (ERN + uncertainty + watchdog + emergency)
├── data/
│   ├── synthetic_eeg.py         # EEGStreamGenerator (21ch Kuramoto + ERD/ERS/ERN)
│   ├── synthetic_physio.py      # HRV (IPFM) + GSR (SCR convolution)
│   └── stream_buffer.py         # Thread-safe ring buffer for streaming
├── agent/
│   ├── digital_twin.py          # DigitalTwin — full inference + online adaptation loop
│   └── memory_store.py          # MemoryStore (4-tier: short-term, episodic, semantic, procedural)
├── training/
│   ├── pretrain_jepa.py         # JEPATrainer — EMA teacher, context/target split
│   └── online_train.py          # SupervisedTrainer — frozen backbone + last-block fine-tuning
├── comms/
│   └── zmq_bridge.py            # ZMQBridge (PUB/SUB) + SimulatedHardware mock
└── tests/                       # 28 passing tests
```

### Quick Start

```python
from v2_digital_self_replication.agent.digital_twin import DigitalTwin
from v2_digital_self_replication.data.synthetic_eeg import EEGStreamGenerator
from v2_digital_self_replication.comms.zmq_bridge import SimulatedHardware

# Initialize
twin = DigitalTwin()
twin.reset_state()

gen = EEGStreamGenerator(seed=42, subject_id=1)
hw  = SimulatedHardware()
intent = [0.5, 0.3, 0.0, 0.2, 0.0, 0.4]  # continuous 6-DOF intent

for t in range(1000):
    eeg = gen.step(intent)
    cmd = twin.step(eeg, hrv=[70.0], gsr=[5.0], prop=hw.position)
    if cmd is not None:
        hw.step(cmd)
        twin.observe_outcome(hw.position)
    if twin.should_adapt():
        twin.adapt()
```

**Run JEPA pretraining:**

```python
from v2_digital_self_replication.training.pretrain_jepa import JEPATrainer
from v2_digital_self_replication.data.synthetic_eeg import make_training_batch

trainer = JEPATrainer()
data = make_training_batch(n_subjects=10, n_trials=50)
trainer.train(data)
```

**Run tests:**

```bash
.venv/bin/python -m pytest v2_digital_self_replication/tests/ -v
```

---

## Running v2

All v2 operations run through shell scripts in `v2_digital_self_replication/scripts/`.  
**Python interpreter used:** `.venv/bin/python` (auto-detected from repo root).

### Prerequisites

```bash
# One-time setup (already done if .venv exists)
python3 -m venv .venv
.venv/bin/pip install -r v2_digital_self_replication/requirements.txt
```

### Full pipeline — cold start to inference

```bash
# Run all four steps with default settings (10 subjects, 50 epochs JEPA, 20 epochs FT)
bash v2_digital_self_replication/scripts/run_pipeline.sh

# Quick smoke test (2 subjects, 3 epochs JEPA, 2 epochs FT, 512 inference steps)
bash v2_digital_self_replication/scripts/run_pipeline.sh --quick

# Resume from an existing checkpoint — skip data generation and retraining
bash v2_digital_self_replication/scripts/run_pipeline.sh --skip-data --skip-pretrain --skip-finetune
```

### Individual steps

Each script is self-contained and can be run independently. Default paths and hyperparameters are overridable via environment variables (see the header of each script for the full list).

#### Step 1 — Generate synthetic training data

```bash
bash v2_digital_self_replication/scripts/01_generate_data.sh

# Override: smaller dataset for fast iteration
N_SUBJECTS=3 N_TRIALS=10 DURATION=2.0 \
    bash v2_digital_self_replication/scripts/01_generate_data.sh
```

Writes per-subject `.npy` archives + `metadata.json` to `v2_digital_self_replication/data/generated/`.

#### Step 2 — JEPA self-supervised pretraining

```bash
bash v2_digital_self_replication/scripts/02_pretrain_jepa.sh

# Override: reduce epochs and learning rate
JEPA_EPOCHS=10 LR=1e-3 \
    bash v2_digital_self_replication/scripts/02_pretrain_jepa.sh
```

Saves `v2_digital_self_replication/checkpoints/jepa_encoder_final.pt`.  
No class labels needed — learns to predict future EEG latents from past context.

#### Step 3 — Supervised fine-tuning

```bash
bash v2_digital_self_replication/scripts/03_finetune.sh

# Override: unfreeze the full encoder (more adaptation, risk of forgetting)
FREEZE_ENCODER=false FT_EPOCHS=30 \
    bash v2_digital_self_replication/scripts/03_finetune.sh
```

Saves `v2_digital_self_replication/checkpoints/supervised_best.pt`.  
By default the pretrained encoder backbone is frozen; only the decoder and last encoder block adapt.

#### Step 4 — Inference loop

```bash
bash v2_digital_self_replication/scripts/04_run_twin.sh

# Override: different subject profile, longer run
SUBJECT_ID=3 N_STEPS=5120 INTENT="0.8 0.6 0.0 0.4 0.0 0.7" \
    bash v2_digital_self_replication/scripts/04_run_twin.sh
```

Streams synthetic EEG through the trained twin, drives a simulated arm, and adapts online from proprioceptive feedback every 100 steps.  
Prints a live status line every 256 steps (configurable via `LOG_INTERVAL`).  
Writes `v2_digital_self_replication/logs/session_latest.json` on exit.

### Pipeline flags summary

| Flag | Effect |
|---|---|
| `--quick` | Smoke-test mode: 2 subjects, 3 JEPA epochs, 2 FT epochs, 512 inference steps |
| `--skip-data` | Reuse existing `data/generated/` (requires `metadata.json`) |
| `--skip-pretrain` | Skip JEPA training (requires `checkpoints/jepa_encoder_final.pt`) |
| `--skip-finetune` | Skip supervised FT (requires `checkpoints/supervised_best.pt`) |

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `v2_digital_self_replication/data/generated` | Where data is read/written |
| `CHECKPOINT_DIR` | `v2_digital_self_replication/checkpoints` | Where checkpoints are saved |
| `LOG_DIR` | `v2_digital_self_replication/logs` | Where log files are written |
| `DEVICE` | `cpu` | Set to `cuda` if a GPU is available |
| `N_SUBJECTS` | `10` | Synthetic subjects to generate |
| `N_TRIALS` | `50` | Trials per subject |
| `JEPA_EPOCHS` | `50` | JEPA pretraining epochs |
| `FT_EPOCHS` | `20` | Supervised fine-tuning epochs |
| `N_STEPS` | `2560` | Inference loop steps (0 = run until Ctrl-C) |

### Output files

```
v2_digital_self_replication/
├── data/generated/
│   ├── metadata.json               # dataset parameters
│   ├── sub00_eeg.npy               # (n_trials, T, 21) per subject
│   ├── sub00_commands.npy          # (n_trials, T, 6) per subject
│   └── sub00_ern.npy               # (n_trials, T) per subject
├── checkpoints/
│   ├── jepa_encoder_best.pt        # best JEPA encoder
│   ├── jepa_encoder_final.pt       # final JEPA encoder (used by step 3)
│   └── supervised_best.pt          # full twin checkpoint (used by step 4)
└── logs/
    ├── 01_generate_data.log
    ├── 02_pretrain_jepa.log
    ├── 03_finetune.log
    ├── 04_run_twin.log
    ├── pipeline_YYYYMMDD_HHMMSS.log  # full pipeline transcript
    └── session_latest.json           # inference session summary
```

---

## Development Roadmap

| Phase | Description | Status |
|---|---|---|
| v1 — Academic baseline | RS-S4 discriminative encoder, MOABB benchmarks, JBHI paper | Complete |
| v2 — Architecture | ZOH SSM encoder, continuous decoder, Kalman filter, safety gate | Complete (28/28 tests) |
| v2 — Shell scripts | Single-command pipeline entry points with skip flags and env overrides | Complete |
| v2 — Pretraining | JEPA on synthetic 21-channel EEG | Implemented |
| v2 — Supervised FT | Fine-tune on labeled synthetic data | Implemented |
| v2 — Hardware | ZMQ bridge to physical prosthetic controller | Implemented (sim only) |
| v2 — Live EEG | Integrate real EEG headset (Bluetooth/LSL) | Planned |
| v2 — Article | Write methodology paper for v2 architecture | Planned |

---

## Citation

```bibtex
@software{noosphere2026,
  title  = {Noosphere: From BCI Classification to Continuous Human-Machine Symbiosis},
  year   = {2026},
  author = {Joseph Woodall},
}
```
