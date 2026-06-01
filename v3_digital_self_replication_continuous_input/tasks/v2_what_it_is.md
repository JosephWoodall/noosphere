# What v2 Is — Plain English

## The One-Sentence Version

v2 is a brain-wave reading system that can tell which of three pre-defined arm movements you're
imagining, then drives a virtual (or physical) arm toward the corresponding fixed position —
about a third of the time successfully.

---

## What It Does, Step by Step

**1. It reads your brain waves.**
21 EEG electrodes placed on your scalp pick up the electrical activity of your brain at 256
samples per second. These signals are small — about a millionth of a volt — and very noisy.

**2. It processes the signal in real time.**
A neural network called a ZOH-SSM (Zero-Order-Hold State Space Model) reads the EEG one
sample at a time, 37 times per second. It summarises the last second or so of brain activity
into a compact 128-number "brain state snapshot." Think of it like taking a photograph of what
your brain is currently doing.

**3. It guesses what you're imagining.**
A second neural network (the Intent Decoder) looks at that brain state snapshot and tries to
guess which of three motor movements you're imagining: moving your left hand, your right hand,
or your feet. This guess is packaged as a 6-dimensional number representing arm position.

**4. It smooths the noisy guess.**
A Kalman filter (a standard engineering tool from missile guidance, applied here to brain
signals) smooths out the jitter in the decoder's output. When the decoder is uncertain, it
trusts the arm's current momentum more than the new guess.

**5. It checks for safety.**
A safety gate monitors for two things: (a) your brain producing an "error signal" suggesting
the last command felt wrong, and (b) the decoder being too uncertain to trust. Either triggers
an automatic halt — the arm stops moving.

**6. It sends the command to hardware.**
The smoothed, safety-checked command goes over a network connection (ZMQ protocol) to a
Raspberry Pi Pico microcontroller, which drives a small stepper motor. This is the physical
arm — currently one axis, one motor, proof of concept.

**7. It learns from feedback.**
Every 100 steps, the system compares what it predicted to what the arm actually did and takes
a small gradient step to improve. This online adaptation is designed to personalize the system
to you over a session.

---

## What It Actually Achieves

**In the simulation (virtual arm, no hardware):**

- If the system knew the answer (oracle): arm reaches the correct target 100% of the time.
- Using standard brain-wave classification (CSP, the 40-year-old method): arm reaches the
  target about 26% of the time.
- Using the v2 AC-SSM architecture: arm reaches the target about 34% of the time.
- Using the previous version of the neural decoder: arm reaches the target 0% of the time
  because the decoder was outputting near-zero commands and the arm never moved.

The key result: v2 fixed the 0% problem. The AC-SSM is the first neural architecture in this
codebase that actually moves the arm.

**In static classification (guessing the right label from one EEG window):**

- Chance: 33.3% (random guessing across 3 classes)
- v2 system: ~35% (barely above chance)
- Best traditional method (CSP): 60%

So at guessing the class label, v2 is only slightly better than random. The 22-percentage-point
gap to CSP reflects the fundamental problem: 55 training examples is not enough data for a
neural network to learn reliable EEG-to-class mappings.

---

## What It Is Not

- It is **not** a system that can move an arm to any position you imagine. It can only move
  to one of three pre-defined fixed positions: "left hand position," "right hand position,"
  or "feet position."

- It is **not** decoding continuous motor intent. It is classifying your brain state into a
  category, then looking up a fixed arm position for that category. This is the same strategy
  as the 40-year-old CSP method, just with a neural network doing the classification.

- It is **not** yet connected to a physical 6-DOF arm. The hardware is one stepper motor
  on one axis. The 6-DOF pipeline exists in software.

- It is **not** real-time in the sense that a user could "think left" and immediately see
  the arm move left. The classification requires a full 1-second EEG window before a command
  is issued.

---

## Why It Matters Anyway

v2 is the foundation. It proved:

1. The SSM encoder correctly captures streaming EEG dynamics at 37 Hz on a standard CPU.
2. Injecting the previous motor command into the encoder (action conditioning) makes the
   decoder more consistent in sequential use — this is the architectural novelty.
3. The safety gate, Kalman filter, and hardware communication layer all work correctly.
4. The world model (the component that predicts future brain states given a motor command)
   is correctly implemented — it just needs a better decoder to demonstrate its full value.

v3 builds the better decoder: one that maps brain states to continuous arm velocities using
Georgopoulos's population vector framework (1986) — a neuroscience result that's been proven
in invasive recordings for 40 years, applied non-invasively for the first time to a
streaming SSM world model.

---

## The Honest One-Paragraph Summary

v2 is a working BCI pipeline — encoder, safety system, Kalman filter, hardware interface,
online learning — that correctly reads brain waves and outputs motor commands at 37 Hz. The
architecture is sound. The one component that needs replacing is the intent decoder: it was
trained on three discrete class labels and therefore only knows how to point the arm at three
fixed targets. Replace that with the continuous ERD population vector decoder (v3), and the
system becomes what it was designed to be: a digital twin that continuously interprets motor
intent and executes it on hardware.
