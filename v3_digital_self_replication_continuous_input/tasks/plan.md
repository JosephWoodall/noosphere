# v3: Continuous Streaming Motor Intent
# v4: Neural Interface to Digital Systems

## The Roadmap

```
v2 (done):   Discrete 3-class MI → fixed target lookup → 33.8% convergence
v3 (this):   Continuous ERD → population vector velocity → any position, fluid chaining
v4 (next):   Extend continuous stream to digital commands → terminal, editor, OS
```

Each stage builds directly on the previous. v3 replaces only the decoder component of v2.
v4 adds a parallel pathway to v3's existing stream. Nothing is thrown away.

---

## v3 Goal

The user imagines moving their arm to coordinate [x, y, z, ...].
The system reads the continuous flow of motor intent from streaming EEG,
produces a continuous velocity command, integrates it to position,
and the arm moves there — to any arbitrary position, not just 3 fixed targets.

Chaining is automatic: the system streams at 37 Hz continuously. When intent shifts
from "move right" to "now up," the arm curves smoothly through the transition.
There is no explicit "end command A, begin command B." It is one continuous stream.

---

## Why v2's Architecture Already Supports This

The v2 pipeline was designed for continuous control:
- AC-SSM encoder processes EEG at 37 Hz → h_t updates every sample
- a_{t-1} action conditioning feeds the previous command back into the encoder →
  each decode is contextually aware of the arm's recent trajectory (the chain)
- Adaptive Kalman filter with constant-velocity model → smooth trajectory integration
- ZMQ bridge streams at 100ms heartbeat → hardware receives commands continuously
- Safety gate runs every step → can halt mid-trajectory without latching the system

The only component that does not support continuous control is the IntentDecoder.
It was trained on 3 discrete class labels → 3 fixed output directions → arm can only
reach 3 points. Replace it with the ERD population vector decoder and every other
component works unchanged.

---

## The Signal: Event-Related Desynchronization (ERD)

When you imagine moving your arm, alpha (8-13 Hz) and beta (13-30 Hz) oscillations at
motor cortex electrodes decrease in amplitude. This is ERD — Event-Related Desynchronization.

Key properties:
- **Continuous and graded:** ERD amplitude is proportional to imagined effort.
  Think harder → larger ERD → faster arm movement.
  Relax → ERD returns to baseline → arm decelerates and stops.
- **Directional:** C3 (left motor cortex) encodes right-arm movement.
  C4 (right motor cortex) encodes left-arm movement. Cz encodes vertical.
- **Already in your EEG:** No new hardware. No new recording protocol.
  The AC-SSM encoder already processes these channels.

Georgopoulos (1986) showed motor cortex neurons encode movement direction as a
population vector: the weighted sum of each neuron's "preferred direction" scaled
by its firing rate. ERD at scalp electrodes is the non-invasive proxy.
Moran & Schwartz (1999) showed this encodes velocity continuously, not just direction.

---

## The v3 Decoder: ERD Population Vector

```
h_t (128-dim latent from AC-SSM, 37 Hz)
        │
        ▼
ERD Extraction
  alpha/beta band power at 6 motor channels:
  C3, C4, Cz, T7, T8, FCz
        │
        ▼
Population Vector
  v_t = Σᵢ (ERD_i(t) × preferred_direction_i)

  Motor topography priors (no labels needed):
  C3   → [ 1,  0,  0,  0,  0,  0]  right lateral (left motor cortex)
  C4   → [-1,  0,  0,  0,  0,  0]  left lateral (right motor cortex)
  Cz   → [ 0,  1,  0,  0,  0,  0]  vertical (bilateral vertex)
  T7   → [ 0,  0,  1,  0,  0,  0]  internal rotation (temporal)
  T8   → [ 0,  0, -1,  0,  0,  0]  external rotation
  FCz  → [ 0,  0,  0,  0,  0,  1]  grip aperture (fronto-central)
        │
  (5-minute per-user calibration fits scaling matrix W)
        │
        ▼
cmd_t = W @ v_t          (continuous 6-DOF velocity)
        │
        ▼
Kalman filter (unchanged from v2)
  pos_t = pos_{t-1} + dt × smooth(cmd_t, σ_t)
        │
        ▼
Safety gate (unchanged from v2)
        │
        ▼
ZMQ → Pico → physical arm
```

---

## Calibration Protocol (5 minutes, no discrete labels)

The calibration session fits the per-user scaling matrix W:

1. User watches a virtual arm execute a trajectory through 6-DOF space
   (a choreographed sequence of movements spanning all axes)
2. User imagines following the virtual arm ("mirror imagery")
3. System records simultaneous (EEG, arm_position) pairs at 37 Hz
4. Regression: W = argmin_W ||W @ V - P||²
   where V = (N, 6) ERD population vectors, P = (N, 6) arm positions
5. W is stored as a (6×6) matrix, loaded at session start

This is entirely label-free. No "left hand" or "right hand" categories.
The user just imagines following a moving arm. W maps their individual ERD
topography to the arm's position space.

---

## How Chaining Works

No discrete triggers. No "end command A, begin command B."

The system runs this loop at 37 Hz forever:

```
while running:
    eeg_sample = read_eeg_channel()               # one 21-channel sample
    h_t, hidden = ac_ssm.decode_step(eeg_sample,
                                      hidden,
                                      a_prev=cmd_prev)   # action context
    cmd_t = erd_decoder(h_t, W)                   # continuous velocity
    smooth_cmd = kalman.step(cmd_t, sigma_t)
    safe_cmd = safety.check(smooth_cmd, ...)
    pos_t = pos_prev + dt * safe_cmd              # position integration
    send_to_hardware(pos_t)
    cmd_prev = safe_cmd                            # feeds next step
```

When you shift intent mid-motion:
- ERD amplitude changes as your motor imagery shifts
- cmd_t changes direction at the next 37 Hz tick
- Kalman filter constant-velocity model blends the transition smoothly
- a_prev (AC-SSM conditioning) ensures the encoder knows the arm just changed direction
- The arm curves naturally through the intent transition

This is what makes it feel like your own arm: the commands are always live,
always current, always chaining from the previous moment.

---

## Implementation Plan

### Phase 1: ERD Population Decoder
**File:** `core/erd_decoder.py`

- Band-pass filter h_t at alpha/beta frequencies (or use SSM frequency features)
- Extract per-channel power at 6 motor channels
- Compute population vector using topography priors
- Apply calibration matrix W
- Output: (6,) velocity command
- Unit test: synthetic ERD with known direction → correct population vector

### Phase 2: Calibration System
**File:** `cli/calibrate.py`

- Virtual arm trajectory generator covering 6-DOF space in 5 minutes
- Real-time ERD extraction during calibration
- Least-squares fit of W: `np.linalg.lstsq(V, P)`
- Per-user W saved to `checkpoints/calibration_{user_id}.npy`
- Integration test: calibrate on synthetic data, verify W recovers ground truth

### Phase 3: DigitalTwin v3 Agent
**File:** `agent/digital_twin_v3.py`

- Subclass or compose with v2 DigitalTwin
- Replace IntentDecoder.step() with ERDDecoder.step()
- Add position state integration
- Add position bounds (joint limits)
- Load/apply calibration W at session start
- Online W refinement: update W using proprioceptive error every 100 steps
- Integration test: SimulatedHardware → target acquisition with continuous commands

### Phase 4: Hardware
**File:** `pico/main_v3.py`

- Start with existing single-axis stepper (1-DOF proof of concept)
- Verify continuous velocity → position integration on hardware
- Add axes incrementally as hardware expands

### Phase 5: Evaluation
**File:** `cli/eval_fitts.py`

- Fitts' Law paradigm: 8 targets at varying distances and widths in 3D space
- Metric: throughput (bits/s) = log₂(1 + A/W) / MT (movement time)
- Baselines: v3 continuous vs. v2 discrete vs. CSP discrete
- 5 subjects, 3 sessions each (learn curve across sessions)
- Primary result: v3 can reach targets v2 cannot (arbitrary positions)

---

## Expected Results

| Metric | v3 (continuous) | v2 (discrete) | CSP (discrete) |
|---|---|---|---|
| Reachable positions | Any [x,y,z] | 3 fixed | 3 fixed |
| Convergence rate | 50-70%* | 33.8% | 26.3% |
| Throughput (bits/s) | 0.5-1.5 | N/A | ~0.3 |
| Calibration time | 5 min (no labels) | 20 min + labels | 5 min |
| Fluid chaining | Yes (continuous) | No (discrete) | No (discrete) |

*Convergence improves because: (a) calibration personalizes to user's ERD topography,
(b) continuous velocity allows the arm to correct mid-trajectory, (c) any error in
direction self-corrects as long as ERD is sustained in the right direction.

---

## v4 Preview: Neural Interface to Digital Systems

v3 gives you continuous physical control. v4 extends the same stream to digital space.

The "digital twin" vision: one neural interface drives both your physical prosthetic
and your computing environment. Your brain commands physical actions (arm moves to position X)
and digital actions (terminal executes command Y) from the same continuous EEG stream.

**v4 adds a second pathway to the v3 stream:**

```
Continuous EEG stream
        │
        ├── Motor ERD pathway (v3 unchanged) ──→ physical arm position
        │
        └── Cognitive signal pathway (v4):
                    │
                    ├── Intent mode classifier:
                    │       Motor ERD present → route to v3
                    │       Motor ERD absent + cognitive signal → route to v4
                    │
                    ├── Command decoder:
                    │       P300 (300ms response to stimuli) → selection
                    │       SSVEP (frequency-coded attention) → navigation
                    │       Imagined speech → text/command generation
                    │
                    └── Execution layer:
                            Claude API → "write a function that does X"
                            Terminal → "run tests / git commit / build"
                            Editor → cursor movement, selection, paste
                            OS → open file, switch window, search
```

**The EEG signals for digital commands:**
- P300: occurs 300ms after seeing a target stimulus. User silently counts targets.
  The system detects which stimulus triggered a P300 → which command was selected.
- SSVEP: flickering interface elements at different Hz. Attending to an element
  produces a matching oscillation in occipital EEG → which element is focused.
- Imagined speech (future): sub-vocal motor programs detectable near Broca's area.
  Willett et al. (2023) decoded full sentences at 62 words/minute invasively.
  Non-invasive is harder but active research area.

**Why v3 first:** The continuous motor control pipeline (streaming, Kalman, safety, hardware)
is the same infrastructure v4's digital pathway runs on. Building v3 correctly means v4
is adding a new decoder to an already-working system, not rebuilding the stack.

**v4 paper target:** "Unified Neural Interface for Physical and Digital Control via
Streaming EEG World Model" — demonstrates both physical arm control (v3) and terminal
command execution from the same session, same electrode setup, same hardware.

---

## File Structure (v3 builds on v2 imports)

```
v3_digital_self_replication_continuous_input/
├── tasks/
│   ├── plan.md               ← this file
│   ├── core_principle.md     ← North Star
│   └── v2_what_it_is.md      ← reference: what we build from
├── core/
│   └── erd_decoder.py        ← ERD population vector decoder (Phase 1)
├── cli/
│   ├── calibrate.py          ← calibration session (Phase 2)
│   └── eval_fitts.py         ← Fitts' Law evaluation (Phase 5)
├── agent/
│   └── digital_twin_v3.py   ← extends v2 DigitalTwin (Phase 3)
└── pico/
    └── main_v3.py            ← continuous position control firmware (Phase 4)

# All v2 components imported unchanged:
# from v2_digital_self_replication.core.stream_encoder import StreamEncoder
# from v2_digital_self_replication.core.kalman_filter import AdaptiveKalmanFilter
# from v2_digital_self_replication.core.safety_gate import SafetyGate
# from v2_digital_self_replication.comms.zmq_bridge import ZMQBridge
```

---

## References

- Georgopoulos et al. (1986). Neuronal population coding of movement direction.
  Science 233(4771):1416-9. doi:10.1126/science.3749885
- Moran & Schwartz (1999). Motor cortical representation of speed and direction
  during reaching. J Neurophysiol 82(5):2207-18. doi:10.1152/jn.1999.82.5.2207
- Pfurtscheller & Lopes da Silva (1999). Event-related EEG/MEG synchronization
  and desynchronization. Clin Neurophysiol 110(11):1842-57.
- Willett et al. (2021). High-performance brain-to-text communication via handwriting.
  Nature 593:249-254. doi:10.1038/s41586-021-03506-2
- Willett et al. (2023). A high-performance speech neuroprosthesis.
  Nature 620:1031-1036. doi:10.1038/s41586-023-06377-x
