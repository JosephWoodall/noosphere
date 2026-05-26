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
│   ├── zmq_bridge.py            # ZMQBridge (PUB/SUB) + SimulatedHardware mock
│   └── arduino_bridge.py        # ZMQ SUB → pyserial → Arduino motor controller
├── arduino/
│   └── motor_control.ino        # Arduino sketch: receives "M<idx> <pwm>\n", drives PWM pins
├── cli/
│   ├── calibrate.py             # Step 4 CLI — countdown capture + fine-tune on subject data
│   └── run_twin.py              # Step 5 CLI — inference loop with ZMQ + rich live dashboard
└── tests/                       # 32 passing tests
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
# Run all five steps with default settings
bash v2_digital_self_replication/scripts/run_pipeline.sh

# Quick smoke test (2 subjects, 3 JEPA epochs, 2 FT epochs, 1 calibration rep, 512 inference steps)
bash v2_digital_self_replication/scripts/run_pipeline.sh --quick

# Resume from an existing checkpoint — skip data generation and retraining
bash v2_digital_self_replication/scripts/run_pipeline.sh --skip-data --skip-pretrain --skip-finetune

# Skip only calibration (reuse previous calibrated.pt or fall back to supervised_best.pt)
bash v2_digital_self_replication/scripts/run_pipeline.sh --skip-calibrate
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

Saves `checkpoints/jepa_encoder_final.pt`.  
No class labels needed — learns to predict future EEG latents from past context.

#### Step 3 — Supervised fine-tuning

```bash
bash v2_digital_self_replication/scripts/03_finetune.sh

# Override: unfreeze the full encoder (more adaptation, higher risk of forgetting)
FREEZE_ENCODER=false FT_EPOCHS=30 \
    bash v2_digital_self_replication/scripts/03_finetune.sh
```

Saves `checkpoints/supervised_best.pt`.  
The pretrained encoder backbone is frozen by default; only the decoder and last encoder block adapt.

#### Step 4 — Subject calibration

```bash
bash v2_digital_self_replication/scripts/04_calibrate.sh

# Override: more reps, longer capture, more fine-tune epochs
N_REPS=5 CAPTURE_S=3.0 FT_EPOCHS=10 \
    bash v2_digital_self_replication/scripts/04_calibrate.sh
```

Runs a countdown-based capture loop across 7 movement targets (reach, elbow flex, grip close/open, wrist rotate CW/CCW, rest). Captures synthetic EEG per target, fine-tunes the decoder on the captured data, and saves `checkpoints/calibrated.pt`.

This step personalizes the generic `supervised_best.pt` model to the specific subject's neural signatures. It runs every time the pipeline runs by default.

#### Step 5 — Inference loop

```bash
bash v2_digital_self_replication/scripts/05_run_twin.sh

# Override: different subject profile, longer run, enable ZMQ
SUBJECT_ID=3 N_STEPS=5120 INTENT="0.8 0.6 0.0 0.4 0.0 0.7" \
    bash v2_digital_self_replication/scripts/05_run_twin.sh
```

Streams synthetic EEG through the calibrated twin, drives a simulated arm, and adapts online from proprioceptive feedback every 100 steps. Displays a rich live dashboard showing 6-DOF arm state, throughput (Hz), safety status, and ZMQ state. Pass `--no-dashboard` for plain log output.

Writes `logs/session_latest.json` on exit.

#### Arduino bridge (optional hardware)

```bash
# Terminal 1: run inference with ZMQ enabled
bash v2_digital_self_replication/scripts/05_run_twin.sh  # add --zmq flag

# Terminal 2: forward commands to Arduino
python -m v2_digital_self_replication.comms.arduino_bridge --port /dev/ttyACM0
```

Flash `arduino/motor_control.ino` to the Arduino first. It receives `M<index> <pwm>\n` commands and drives PWM on pins 9 (shoulder_yaw) and 10 (grip_aperture). A 500 ms watchdog zeros all motors if the serial stream goes silent.

### Pipeline flags summary

| Flag | Effect |
|---|---|
| `--quick` | Smoke-test: 2 subjects, 3 JEPA epochs, 2 FT epochs, 1 calibration rep, 512 inference steps |
| `--skip-data` | Reuse existing `data/generated/` (requires `metadata.json`) |
| `--skip-pretrain` | Skip JEPA training (requires `checkpoints/jepa_encoder_final.pt`) |
| `--skip-finetune` | Skip supervised FT (requires `checkpoints/supervised_best.pt`) |
| `--skip-calibrate` | Skip subject calibration (uses existing `calibrated.pt` or falls back to `supervised_best.pt`) |

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `DATA_DIR` | `v2_digital_self_replication/data/generated` | Where data is read/written |
| `CHECKPOINT_DIR` | `v2_digital_self_replication/checkpoints` | Where checkpoints are saved |
| `LOG_DIR` | `v2_digital_self_replication/logs` | Where log files are written |
| `DEVICE` | auto | `cuda` if a GPU is detected, otherwise `cpu` |
| `N_SUBJECTS` | `10` | Synthetic subjects to generate |
| `N_TRIALS` | `50` | Trials per subject |
| `JEPA_EPOCHS` | `15` | JEPA pretraining epochs (GPU users: export to 50) |
| `FT_EPOCHS` | `5` | Supervised fine-tuning epochs |
| `N_REPS` | `3` | Calibration repetitions per movement target |
| `CAPTURE_S` | `2.0` | Calibration capture duration per rep (seconds) |
| `CAL_FT_EPOCHS` | `5` | Fine-tune epochs during calibration (independent of `FT_EPOCHS`) |
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
│   ├── jepa_encoder_best.pt        # best JEPA encoder (by validation loss)
│   ├── jepa_encoder_final.pt       # final JEPA encoder → input to step 3
│   ├── supervised_best.pt          # generic fine-tuned twin → input to step 4
│   └── calibrated.pt               # subject-calibrated twin → used by step 5
└── logs/
    ├── 01_generate_data.log
    ├── 02_pretrain_jepa.log
    ├── 03_finetune.log
    ├── 04_calibrate.log
    ├── 05_run_twin.log
    ├── pipeline_YYYYMMDD_HHMMSS.log  # full pipeline transcript
    ├── session_latest.json           # inference session summary (JSON)
    └── episodes.db                   # episodic memory (SQLite)
```

---

## Development Roadmap

| Phase | Description | Status |
|---|---|---|
| v1 — Academic baseline | RS-S4 discriminative encoder, MOABB benchmarks, JBHI paper | Complete |
| v2 — Architecture | ZOH SSM encoder, continuous decoder, Kalman filter, safety gate | Complete (32/32 tests) |
| v2 — Pipeline | 5-step shell pipeline: generate → pretrain → finetune → calibrate → infer | Complete |
| v2 — Pretraining | JEPA self-supervised on synthetic 21-channel EEG | Complete |
| v2 — Supervised FT | Labeled synthetic data → continuous 6-DOF decoder | Complete |
| v2 — Subject calibration | Countdown capture loop → per-subject fine-tune → `calibrated.pt` | Complete |
| v2 — Live dashboard | `rich.live` panel: 6-DOF bars, Hz, safety status, ZMQ state | Complete |
| v2 — Hardware (ZMQ) | ZMQ PUB/SUB bridge + SimulatedHardware first-order lag mock | Complete |
| v2 — Hardware (Arduino) | ZMQ → pyserial → Arduino PWM bridge + `.ino` motor sketch | Complete |
| v2 — Semantic memory | FAISS episode index; populated from mean latent at session end | Complete |
| v2 — Live EEG | Integrate real EEG headset via LSL or USB | Planned |
| v2 — Article | Methodology paper for v2 architecture | Planned |

---

## Citation

```bibtex
@software{noosphere2026,
  title  = {Noosphere: From BCI Classification to Continuous Human-Machine Symbiosis},
  year   = {2026},
  author = {Joseph Woodall},
}
```
