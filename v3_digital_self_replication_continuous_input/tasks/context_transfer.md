# Context Transfer Document — Noosphere v3 / v4

**Author:** Joseph Woodall (josephlw4@gmail.com)
**Date written:** 2026-05-31
**Purpose:** Complete briefing for a new Claude session to continue v3/v4 implementation
**Repo root:** `/home/redleadr/workspace/noosphere/`

---

## 1. The Project

**Name:** Noosphere
**Goal:** A closed-loop brain-computer interface where the user's neural signals continuously
drive both physical (prosthetic arm) and digital (terminal / Claude API) actions through one
unified world model — a digital twin of the user's intent.

**The user's exact words on the endgame:**
> "An additional consciousness that I can leverage as a supplement to my own...
> complete interaction and zero latency between what my biological form tells me
> that I should have and what the mechanical apparatus would be able to do."

**The core architectural principle (invariant across ALL versions):**
The ZOH-SSM world model (encoder + action-conditioned transition model) is the
permanent interpreter. It reads streaming EEG at 37 Hz, maintains a latent
brain-state representation h_t, and conditions each decoding step on the previous
motor command a_{t-1}. The DECODER changes between v3 (physical arm velocity) and
v4 (digital intent). The world model does not change.

---

## 2. Hardware Status (Critical — Read Before Implementing Anything)

**EEG:** No real hardware. All EEG is simulated (existing `EEGStreamGenerator` in v2)
or pre-recorded (PhysioNet, used for LOSO evaluation only). v3 must extend the
simulator to generate realistic ERD for continuous arbitrary 6-DOF arm movement.
Real EEG device integration is a future milestone, not a v3 requirement.

**Physical arm:** No physical hardware. Everything runs against `SimulatedHardware`
(first-order lag model in `v2_digital_self_replication/comms/zmq_bridge.py`).
Pico + one 28BYJ-48 stepper motor EXISTS but is not connected. Physical hardware
integration is a future milestone.

**Implication:** v3 and v4 are entirely in simulation. The outputs that matter are:
- Correct continuous decoder behavior (arm tracks simulated intent continuously)
- Closed-loop convergence metrics (Fitts' Law throughput on SimulatedHardware)
- Paper-worthy evaluation numbers from simulated experiments
- Architecture that plugs into real hardware without redesign when hardware arrives

---

## 3. Version History Summary

### v1 (complete)
Discriminative classifier. Riemannian-S4 encoder → 4-class EEG label. Submitted to
IEEE JBHI. Desk-rejected. The limitation: fixed output categories, no continuous control.

### v2 (complete — article ready for TNSRE submission)

**Article:** `v2_digital_self_replication/articles/riemannian_s4_v2.tex`
**Status:** Ready for IEEE TNSRE submission. All numbers verified, all tables filled,
all dashes removed, figure rewritten with absolute coordinates.

**What v2 proved works:**
- ZOH-SSM encoder (`stream_encoder.py`): O(1) streaming at 37 Hz, stable
- AC-SSM action conditioning: a_{t-1} injected into first SSM block via SiLU gate
- Adaptive Kalman filter: R = diag(sigma²), smooths noisy decoder output
- Safety gate: ERN + sigma + watchdog + emergency stop
- ZMQ bridge: PUB/SUB protocol, bidirectional hardware loop
- Online adaptation: every 100 steps, MSE on proprioceptive feedback, EMA smoothing

**What v2 proved broken:**
- IntentDecoder: trained on 3 discrete class labels → near-zero 6-DOF commands
- Closed-loop convergence with continuous decoder: 0%
- Workaround used in paper: classifier + discrete intent lookup = 33.8% convergence
- This workaround is NOT v3 — it is a measurement of the decoder bottleneck

**Key article results (for v3 paper to cite):**
- AC-SSM vs JEPA decoder: 33.8% vs 0% closed-loop convergence (p<0.0001)
- AC-SSM vs CSP: 33.8% vs 26.3% (p=0.21, not significant — competitive)
- Static LOSO: AC-SSM 34.7%, JEPA 36.5%, EEGConformer 41.1%, CSP 60.1%
- Gap is data-limited: consistent across 3 window settings, 2 datasets, all methods

### v3 (to build — this document)
Continuous ERD population vector decoder → arbitrary 6-DOF arm position in simulation.
Replaces IntentDecoder. All other v2 components unchanged.

### v4 (to build after v3 — this document)
Same world model. Second decoder pathway: EEG → digital intent → Claude API / shell.
Closed-loop: execution result → ERN confirmation/rejection gate.

---

## 4. What "Closed-Loop" Means — Precisely

The user explicitly specified: "this is supposed to be a closed-loop system, and the
world model should be doing the interpretation."

Closed-loop means the system's output affects the input in a feedback cycle:

```
EEG (t) → world model → command(t) → execution → outcome(t)
                ↑                                      |
                └──────── feedback (proprioception) ───┘
                          modifies EEG (t+1)
```

In v2 this exists but is incomplete:
- a_{t-1} feeds back into the encoder ✓ (action conditioning)
- Arm position feeds into observe_outcome() → adaptation ✓
- Kalman filter maintains momentum from prior commands ✓
- But: the decoder ignores the arm position entirely ✗
- And: the world model (transition T) was never in the hot path ✗

In v3 the loop closes completely:
- ERD decoder outputs VELOCITY, not position
- Position INTEGRATES from velocity: pos_t = pos_{t-1} + dt × cmd_t
- The integrated position feeds back as proprioception into the next EEG step
- The simulated EEG generator updates ERD based on whether intent is satisfied
  (if arm is at target, ERD diminishes; if not, ERD sustains)
- This is the natural motor control closed loop: intend → move → sense → update intent

In v4 the loop extends to digital:
- Digital command executes → result displayed on screen
- Visual feedback → ERN if wrong (existing ERN gate) → rollback
- No ERN → command confirmed → continue
- The same ZOH-SSM encoder reads the ERN in the EEG stream

---

## 5. v3 Architecture — Full Specification

### 5.1 What Changes from v2

**Remove:** IntentDecoder (MLP → near-zero 6-DOF commands)
**Add:** ERDDecoder (population vector → continuous 6-DOF velocity)
**Add:** Extended EEGStreamGenerator (continuous arbitrary-DOF ERD synthesis)
**Add:** Position integrator (velocity → position state)
**Add:** Calibration system (fits per-user scaling matrix W, 5 minutes)
**Unchanged:** StreamEncoder (AC-SSM), AdaptiveKalmanFilter, SafetyGate,
  ActionConditionedTransition, LatencyPlanner, ZMQBridge, SimulatedHardware,
  MemoryStore, online adaptation loop

### 5.2 The ERD Population Vector Decoder

**Neuroscience basis (must cite these):**
- Georgopoulos et al. (1986): motor cortex encodes direction as population vector
  doi:10.1126/science.3749885
- Moran & Schwartz (1999): population vector predicts hand velocity continuously
  doi:10.1152/jn.1999.82.5.2207
- Pfurtscheller & Lopes da Silva (1999): ERD/ERS fundamentals
  doi:10.1016/S1388-2457(99)00141-8

**Algorithm:**

```python
# Step 1: Extract band power from the AC-SSM latent h_t
# h_t encodes frequency structure because the SSM was pretrained on EEG oscillations
# Alpha = 8-13 Hz, Beta = 13-30 Hz — both suppress during motor imagery (ERD)

def extract_erd(h_t, alpha_proj, beta_proj):
    # alpha_proj, beta_proj: (d_model,) → (6,) learned projections per channel
    # Returns: erd (6,) — one value per motor channel
    alpha_power = F.softplus(alpha_proj @ h_t)  # always positive
    beta_power  = F.softplus(beta_proj  @ h_t)
    erd = -(alpha_power + beta_power)            # negative = desynchronization
    return erd

# Step 2: Population vector
# preferred_directions: (6, 6) — rows are channel preferred directions in DOF space
PREFERRED_DIRECTIONS = torch.tensor([
    [ 1.,  0.,  0.,  0.,  0.,  0.],  # C3:  right lateral (left motor cortex)
    [-1.,  0.,  0.,  0.,  0.,  0.],  # C4:  left lateral (right motor cortex)
    [ 0.,  1.,  0.,  0.,  0.,  0.],  # Cz:  vertical (bilateral vertex)
    [ 0.,  0.,  1.,  0.,  0.,  0.],  # T7:  internal rotation
    [ 0.,  0., -1.,  0.,  0.,  0.],  # T8:  external rotation
    [ 0.,  0.,  0.,  0.,  0.,  1.],  # FCz: grip aperture
])

def population_vector(erd):
    return (erd.unsqueeze(1) * PREFERRED_DIRECTIONS).sum(0)  # (6,) velocity

# Step 3: Per-user calibration scaling
def decode(h_t, W, alpha_proj, beta_proj):
    erd = extract_erd(h_t, alpha_proj, beta_proj)
    v   = population_vector(erd)
    cmd = W @ v               # (6,) — scaled velocity in [-1, 1]^6
    return torch.tanh(cmd)    # bounded
```

**W calibration:**

```python
def calibrate(eeg_windows, arm_positions, alpha_proj, beta_proj):
    # eeg_windows: (N, d_model) — encoded latents from calibration session
    # arm_positions: (N, 6) — simultaneous arm positions (ground truth velocity targets)
    V = torch.stack([population_vector(extract_erd(h, alpha_proj, beta_proj))
                     for h in eeg_windows])       # (N, 6)
    # Least-squares: V @ W.T ≈ arm_positions
    W, _, _, _ = torch.linalg.lstsq(V, arm_positions)  # (6, 6)
    return W.T
```

### 5.3 Extended EEG Simulator

The existing `EEGStreamGenerator` generates ERD based on "total motor activity."
For v3, it must generate per-DOF spatially specific ERD:

```python
# Existing (v2): scale total ERD by norm of intent vector
# New (v3): project intent vector onto channel topography

def generate_eeg_v3(intent_velocity, channel_topography=PREFERRED_DIRECTIONS):
    # intent_velocity: (6,) continuous velocity command
    # channel_topography: (6, 6) same as PREFERRED_DIRECTIONS
    
    # ERD at each motor channel = dot product of intent with channel's preferred direction
    erd_amplitudes = (channel_topography @ intent_velocity)  # (6,)
    
    # Map to EEG channels: C3=idx8, C4=idx10, Cz=idx9, T7=idx7, T8=idx11, FCz=idx20
    MOTOR_CHANNEL_IDX = [8, 10, 9, 7, 11, 20]
    eeg = generate_background_eeg()  # existing: alpha + beta oscillations at all channels
    
    for i, ch_idx in enumerate(MOTOR_CHANNEL_IDX):
        # ERD = amplitude suppression in alpha/beta bands
        eeg[:, ch_idx] *= (1.0 - 0.6 * abs(erd_amplitudes[i]))
    
    return eeg
```

**Closed-loop simulator behavior:**
The EEG generator must know where the arm is to know whether intent is satisfied:
- If `||arm_pos - target_pos|| < threshold`: user intent relaxes → ERD diminishes
- If `||arm_pos - target_pos|| > threshold`: user intent sustains → ERD continues
- This is the proprioceptive feedback closing the loop through the EEG generator

### 5.4 Position Integration

```python
class ArmPositionIntegrator:
    def __init__(self, n_dof=6, dt=1/37.0):
        self.pos = np.zeros(n_dof)
        self.dt  = dt
        self.limits = (-1.0, 1.0)
    
    def step(self, velocity_cmd):
        self.pos = np.clip(self.pos + self.dt * velocity_cmd,
                           self.limits[0], self.limits[1])
        return self.pos.copy()
    
    def reset(self):
        self.pos[:] = 0.
```

### 5.5 Calibration Session (5 minutes, no labels)

```
Session design:
  - Virtual arm executes trajectory through 6-DOF space (pre-choreographed)
  - User watches and imagines following ("mirror imagery")
  - System records (EEG_latent, arm_velocity) pairs at 37 Hz for 5 minutes = 11,100 pairs
  - Fit W via least squares: W maps population vectors to arm velocities
  - Save W to checkpoints/calibration_{user_id}.npy

Trajectory design:
  - Phase 1 (1 min): pure left/right movements (isolates C3/C4 ERD)
  - Phase 2 (1 min): pure vertical movements (isolates Cz)
  - Phase 3 (1 min): rotation movements (isolates T7/T8)
  - Phase 4 (1 min): grip open/close (isolates FCz)
  - Phase 5 (1 min): complex multi-DOF trajectories (fits cross-coupling in W)
```

### 5.6 Full v3 Inference Loop

```python
# digital_twin_v3.py — wraps v2 DigitalTwin, replaces only the decoder step

class DigitalTwinV3:
    def __init__(self):
        # Reuse ALL v2 components
        self.encoder    = StreamEncoder(d_dof=6, n_eeg_recon=21)  # AC-SSM
        self.transition = ActionConditionedTransition()
        self.planner    = LatencyPlanner(self.transition, erd_decoder)  # NOTE: planner now uses ERD decoder
        self.kalman     = AdaptiveKalmanFilter()
        self.safety     = SafetyGate()
        self.memory     = MemoryStore()
        self.integrator = ArmPositionIntegrator()
        
        # NEW: ERD decoder instead of IntentDecoder
        self.erd_decoder  = ERDDecoder()      # alpha_proj, beta_proj, W
        self._hidden      = None
        self._cmd_prev    = np.zeros(6)

    def step(self, eeg_sample, arm_feedback=None):
        # 1. Encode with action conditioning (v2 AC-SSM, unchanged)
        h_t, self._hidden = self.encoder.decode_step(
            eeg_sample, hidden_states=self._hidden,
            a_prev=torch.from_numpy(self._cmd_prev).unsqueeze(0)
        )
        
        # 2. Decode: ERD → velocity (NEW in v3)
        velocity, sigma = self.erd_decoder(h_t, return_uncertainty=True)
        
        # 3. Kalman smooth (v2, unchanged)
        smooth_vel = self.kalman.step(velocity, sigma)
        
        # 4. Safety gate (v2, unchanged — ERN gate still applies)
        safe_vel, reason = self.safety.check(smooth_vel, sigma, ern_prob=0.0)
        
        # 5. Position integration (NEW in v3)
        if safe_vel is not None:
            pos = self.integrator.step(safe_vel)
        else:
            pos = self.integrator.pos.copy()  # hold position on HALT
        
        # 6. Store for next step (v2 action conditioning)
        self._cmd_prev = safe_vel if safe_vel is not None else np.zeros(6)
        
        return pos  # (6,) arm position
```

---

## 6. v4 Architecture — Full Specification

### 6.1 The Core Principle

**Same ZOH-SSM world model. Different decoder output. Closed-loop via ERN.**

v4 adds a second decoder pathway to the existing v3 pipeline. The encoder and world
model run identically. At each step, the system decides: is this EEG moment encoding
a physical intent (→ v3 arm pathway) or a digital intent (→ v4 command pathway)?

The mode classifier is itself a lightweight head on top of h_t:
- Motor ERD present → route to v3 (arm velocity)
- Motor ERD absent + cognitive signal → route to v4 (digital command)

### 6.2 v4 Closed-Loop

```
User thinks "run tests"
    → Motor ERD suppressed (not imagining arm movement)
    → Cognitive signal in EEG (sustained attention, preparatory negativity)
    → Mode classifier: DIGITAL mode
    → Intent classifier: COMMAND_EXECUTE (vs NAVIGATE, WRITE, SEARCH, etc.)
    → Command decoder: "run tests" category detected
    → Executor: subprocess.run(["pytest", ...])
    → Result displayed on screen
    → User evaluates result
    → If wrong: ERN in EEG → ERN gate fires → undo/rollback
    → If correct: no ERN → command confirmed → next intent cycle begins
```

**The ERN validation loop is already implemented in v2.** The SafetyGate already:
- Monitors ern_prob from the decoder
- Halts on ern_prob > 0.7 for 500ms

For v4, this becomes the CONFIRMATION mechanism: after a digital command executes,
the gate monitors for 300-500ms (the P300/ERN window). ERN detected → undo.
No ERN → confirmed. This is free — it uses existing safety infrastructure.

### 6.3 v4 Decoder Components

```python
# v4 adds three components on top of v3

class ModeClassifier(nn.Module):
    """Lightweight head on h_t: physical vs. digital mode."""
    def forward(self, h_t):
        # Returns P(digital | h_t) ∈ [0, 1]
        # High during sustained attention, absent motor ERD
        ...

class IntentClassifier(nn.Module):
    """When in digital mode: what category of digital action?"""
    CATEGORIES = [
        "EXECUTE_COMMAND",    # run shell command
        "QUERY_AI",           # ask Claude
        "NAVIGATE",           # cursor/file navigation
        "WRITE_TEXT",         # type/dictate
        "CONFIRM",            # yes/accept
        "REJECT",             # no/undo
    ]
    def forward(self, h_t):
        # Returns softmax over CATEGORIES
        ...

class CommandExecutor:
    """Routes decoded intent to the appropriate execution backend."""
    def execute(self, category, context, history):
        if category == "EXECUTE_COMMAND":
            return self._shell_execute(context)
        elif category == "QUERY_AI":
            return self._claude_query(context)
        # ...
    
    def undo_last(self):
        """Called when ERN gate fires after execution."""
        ...
```

### 6.4 v4 Training Signal

This is the open research question for v4. The physical decoder (v3) has:
- Ground truth: continuous arm position from SimulatedHardware
- Loss: MSE between decoded velocity and actual velocity

The digital decoder (v4) does not have continuous ground truth. Training options:
1. **Simulated P300/ERN signals**: generate synthetic ERP waveforms for known
   command intentions. Train mode and intent classifiers on synthetic cognitive EEG.
2. **Self-supervised from ERN gate**: log which commands produced ERN (wrong) vs.
   no ERN (correct). Use as a noisy binary training signal.
3. **Active interface**: display a command grid (P300 paradigm). User attends to
   target. Record P300. Use P300 amplitude to label cognitive state.

**Recommendation for v4 paper:** Start with option 3 (P300 paradigm). It is the most
established, has the strongest signal, and requires only synthetic simulation in v4's
scope. The neural signals are well-characterized (positive deflection at Pz, ~300ms).

---

## 7. Paper Plans

### v3 Paper

**Title (draft):** "Continuous 6-DOF Prosthetic Arm Control from Streaming EEG via
Action-Conditioned SSM World Model and ERD Population Vector Decoding"

**Venue:** IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE)
(same as v2 submission)

**Key claims:**
1. First demonstration of continuous arbitrary-position arm control from simulated EEG
   using a population vector decoder on a streaming SSM world model
2. No discrete class labels required at any stage — calibration only (5 minutes, simulated)
3. AC-SSM world model improves closed-loop trajectory smoothness vs. standard JEPA
4. Fitts' Law throughput competitive with simulated discrete-class baselines

**Evaluation (primary metric: Fitts' Law throughput):**
- 8 targets at 3 amplitudes × 3 widths in 6-DOF space
- Throughput T = log₂(1 + A/W) / MT (bits/second)
- Baselines: v3 continuous vs v2 discrete vs CSP discrete
- Multiple simulated "subjects" (different random seeds for EEG generator)
- Report: throughput mean ± SD, % targets reachable (v3: all; v2/CSP: only 3)

**What makes it novel vs. v2:**
- v2 showed world model CAN work (33.8% convergence) but only to 3 discrete targets
- v3 shows continuous control to ANY target using ERD population vector
- The world model (not changed) is the reason it works — same encoder, different decoder
- v2 cited as foundation; v3 is the clinical contribution

**Cite v2 as:** Woodall (2026) — the architecture foundation paper

### v4 Paper

**Title (draft):** "A Unified Neural Interface for Physical and Digital Control via
Closed-Loop Streaming EEG World Model"

**Venue:** Nature Biomedical Engineering or IEEE TNSRE Special Issue on BCI

**Key claims:**
1. Single EEG stream drives both physical arm (v3) and digital execution (terminal / AI)
2. ERN gate closes the digital feedback loop: wrong command → automatic undo
3. World model (ZOH-SSM) is the unified interpreter without retraining between domains
4. Proof of concept: user completes physical + digital tasks in one session, same headset

**Cite v2 and v3 as foundation papers**

---

## 8. Lessons from v2 (Read Before Implementing)

### L001 — Never report results before you have them
v2's article had to be substantially rewritten because claims outran data. For v3:
write the evaluation protocol FIRST, implement it, get results, THEN write the paper.

### L002 — The decoder is the bottleneck, not the world model
v2 spent months proving the world model is correct. It is. The bottleneck was always
the decoder. For v3: get a working continuous decoder in Week 1. Everything else is
already built.

### L003 — Simulated EEG must reflect the actual signal being decoded
v2's synthetic EEG did not accurately simulate ERD for continuous arbitrary positions.
For v3: extend EEGStreamGenerator to generate spatially correct ERD per DOF before
testing the ERD decoder. Test the simulator before testing the decoder.

### L004 — The closed-loop must actually close
In v2, observe_outcome() stored experiences for adaptation but the EEG generator
did not respond to arm position. The loop was not closed — the simulator was open-loop.
For v3: the EEG generator MUST reduce ERD when the arm reaches the target. If not,
the system gets infinite intent signal and never knows it succeeded.

### L005 — torch.jit.script is deprecated in Python 3.14+
Use torch.compile(fn) instead. Already done in v2 for _bilinear_scan.

### L006 — WINDOW_LEN=256 is infeasible on CPU for long training
Default WINDOW_LEN=128 for CPU runs. Override for GPU.

### L007 — Run the full eval before updating the article
Run all evals to completion before touching the LaTeX. Partial results in the article
caused multiple rewrite cycles.

### L008 — Label the figures with absolute coordinates
The TikZ figure's relative positioning caused text overlap. v2's figure was rewritten
with absolute coordinates. Always use absolute coordinates in TikZ for figures with
multiple crossing paths.

### L009 — The 69-trial data limit is real and fundamental
No amount of architecture changes overcome 55 training trials for classification.
For v3: the ERD decoder uses ZERO labels. The calibration uses 5 minutes of data.
This is the correct solution to the data problem, not more labelled trials.

---

## 9. Implementation Order

Do NOT deviate from this order. Each phase must be verified before the next begins.

### Phase 0: Verify Starting Point (Day 1)
```bash
cd /home/redleadr/workspace/noosphere
.venv/bin/python -m pytest v2_digital_self_replication/tests/ -v
# All 18 tests must pass before touching anything
```

### Phase 1: Extended EEG Simulator (Week 1)
**File:** `v3_digital_self_replication_continuous_input/data/eeg_stream_v3.py`

Goal: Given a continuous (6,) velocity intent, generate realistic ERD at the correct
motor channels. The simulator must respond to arm position feedback (close the loop).

Tests to write BEFORE implementing:
- `test_c3_erd_on_right_intent`: positive right-DOF intent → ERD at C3, ERS at C4
- `test_erd_proportional_to_speed`: double the intent magnitude → double the ERD
- `test_erd_diminishes_at_target`: arm at target → ERD diminishes to baseline
- `test_population_vector_recovers_direction`: ERD → population vector → direction

### Phase 2: ERD Decoder (Week 1-2)
**File:** `v3_digital_self_replication_continuous_input/core/erd_decoder.py`

Goal: `h_t → (6,) velocity command`. Calibratable via W matrix.

Tests to write BEFORE implementing:
- `test_preferred_directions_orthogonal`: no cross-DOF contamination
- `test_calibration_recovers_W`: synthetic calibration data → W matches ground truth
- `test_zero_erd_zero_velocity`: resting EEG → near-zero velocity
- `test_velocity_magnitude_bounded`: tanh output always in [-1, 1]

### Phase 3: Calibration System (Week 2)
**File:** `v3_digital_self_replication_continuous_input/cli/calibrate_v3.py`

Goal: 5-minute simulated calibration session → W saved to disk.

Tests:
- `test_calibration_session_runs_5min`: session completes without error
- `test_W_loads_and_applies`: saved W restores correctly
- `test_calibration_improves_accuracy`: post-calibration decoder accuracy > pre

### Phase 4: DigitalTwin v3 (Week 3)
**File:** `v3_digital_self_replication_continuous_input/agent/digital_twin_v3.py`

Import ALL v2 components. Replace ONLY IntentDecoder with ERDDecoder.
Add ArmPositionIntegrator.

Tests:
- `test_closed_loop_converges`: SimulatedHardware reaches arbitrary target
- `test_chaining_smooth`: two sequential targets, smooth trajectory
- `test_halt_holds_position`: safety gate HALT → arm position held, not reset

### Phase 5: Evaluation (Week 4-5)
**File:** `v3_digital_self_replication_continuous_input/cli/eval_fitts.py`

Fitts' Law paradigm. 8 targets. Report throughput vs. v2 and CSP baselines.
Run 20 simulated subjects (different EEG noise seeds).

### Phase 6: v4 Mode Classifier (After v3 paper draft)
**File:** `v3_digital_self_replication_continuous_input/core/mode_classifier.py`

Trained on synthetic cognitive EEG (P300 waveforms) vs. motor ERD.
Binary: physical mode vs. digital mode.

### Phase 7: v4 Intent + Executor (After Phase 6 verified)
**Files:** `core/intent_classifier.py`, `agent/command_executor.py`

P300-based category selection → shell / Claude API execution.
ERN gate for confirmation/rejection.

---

## 10. File Locations Reference

```
/home/redleadr/workspace/noosphere/
├── v2_digital_self_replication/         ← DO NOT MODIFY (v2 is complete)
│   ├── articles/riemannian_s4_v2.tex    ← READY FOR TNSRE SUBMISSION
│   ├── core/stream_encoder.py           ← AC-SSM (import for v3)
│   ├── core/kalman_filter.py            ← Kalman (import for v3)
│   ├── core/safety_gate.py              ← Safety (import for v3)
│   ├── core/transition_model.py         ← Transition T (import for v3)
│   ├── core/latency_planner.py          ← Planner (import for v3)
│   ├── comms/zmq_bridge.py              ← ZMQ + SimulatedHardware (import for v3)
│   ├── agent/digital_twin.py            ← DigitalTwin base class
│   └── checkpoints/jepa_encoder_best.pt ← PRETRAINED WEIGHTS (load for v3)
│
└── v3_digital_self_replication_continuous_input/
    ├── tasks/
    │   ├── context_transfer.md          ← THIS FILE
    │   ├── plan.md                      ← Architecture + v4 preview
    │   ├── core_principle.md            ← North Star
    │   └── v2_what_it_is.md             ← Plain English v2 description
    ├── data/
    │   └── eeg_stream_v3.py             ← Phase 1 (to build)
    ├── core/
    │   ├── erd_decoder.py               ← Phase 2 (to build)
    │   └── mode_classifier.py           ← Phase 6 (to build, after v3 paper)
    ├── cli/
    │   ├── calibrate_v3.py              ← Phase 3 (to build)
    │   └── eval_fitts.py                ← Phase 5 (to build)
    └── agent/
        └── digital_twin_v3.py           ← Phase 4 (to build)
```

---

## 11. Open Questions (Resolved in This Conversation)

| Question | Answer |
|---|---|
| Real EEG hardware? | No. Simulate everything. |
| Physical arm? | No. SimulatedHardware only. |
| v4 "terminal interaction"? | World model interprets EEG. ERN closes loop. Claude API + shell exec. |
| Endgame scope? | v3 (arm) + v4 (digital). Not further for now. |
| Paper venues? | v3 → TNSRE. v4 → TNSRE special issue or Nature BME. |
| v2 article status? | Complete. Ready to submit to TNSRE. Do not modify. |

---

## 12. First Action for the Next Session

```
1. Run: .venv/bin/python -m pytest v2_digital_self_replication/tests/ -v
   Confirm all 18 tests pass. If any fail, fix before touching v3.

2. Read: v3_digital_self_replication_continuous_input/tasks/plan.md
   Understand the full architecture before implementing Phase 1.

3. Write tests for Phase 1 (eeg_stream_v3.py) BEFORE implementing.
   The test names are in Section 9 of this document.

4. Implement Phase 1. Verify all tests pass.

5. Do not touch v2_digital_self_replication/ for any reason.
   It is complete and awaiting submission.
```

---

## 13. The User

**Name:** Joseph Woodall
**Email:** josephlw4@gmail.com
**Role:** Independent researcher in neural engineering and BCI
**Working style:** Ambitious long-term vision, practical incremental implementation.
  Values honesty about what works vs. what doesn't. Prefers the Destructor protocol
  (first-principles, no comfort-zone assumptions, compress timelines).
**Key preference:** Build in stages. Each stage cites the prior as foundation.
  Do not skip stages. Do not claim results before they exist.
