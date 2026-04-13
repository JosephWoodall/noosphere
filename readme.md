# Noosphere v1.8.0 — The Personal Foundation Model

**A physics-informed world model agent with Brain-Computer Interface (BCI) control, decentralized neural networking, and autonomous digital/physical execution.**

---

## 1. What is Noosphere?

Noosphere is **not** an interface to a generic external AI. It is a **Personalized Cognitive Foundation Model** that lives locally on your hardware. It is designed to bridge the gap between human thought and digital/physical action.

### The Core Vision
Noosphere allows an operator to control complex systems (robotic arms, Linux terminals, smart homes) and communicate with others using only three EEG electrodes on the scalp. 

Unlike traditional "Universal" AI that tries to understand everyone, Noosphere **forces the world to adapt to you.** It learns your specific brain patterns, your idiosyncratic way of thinking, and your personal goals to become an "Obedient Consequence Engine."

---

## 2. The Three Pillars of Execution

### I. Intention-to-Action (Physical & Digital)
*   **Robotics:** Drive physical hardware (robotic arms, drones) using continuous XYZ intent decoding.
*   **Linux Shell:** Execute complex terminal commands (git, docker, system management) via discrete intent.
*   **Safety Gating:** A "Digital Twin" simulates the outcome of your thought *before* it happens. If the AI predicts a failure (e.g., "the arm will hit the table"), it intercepts and blocks the action.

### II. Intention-to-Communication (The Noosphere Network)
*   **Neural Messaging:** Send binary-packed `ActionTokens` and digital messages between users on a decentralized P2P network.
*   **Neural Prototyping (The "Mom" Mapping):** While 3-channel EEG cannot decode specific words, it decodes **Neural Prototypes**. By associating a specific mental focus (e.g., visualizing a face) with a contact, the S4 encoder maps this signature into an **Identity Manifold**.
*   **Collaborative Learning:** Users can optionally enable **Inter-Agent Comms**. This allows agents to share "Dynamics Insights"—high-level mathematical patterns of how the world works—facilitating collective growth without sharing private neural data.

### III. Intention-to-State (IoT & Smart Home)
*   **Extension of Will:** Your smart home becomes an extension of your body. 
*   **IoT Apparatus:** Toggle lights, unlock doors, or manage appliances. The world model treats a smart lock exactly like a robotic finger—it predicts the consequence of the state change (`Locked -> Unlocked`) before firing the API call.

---

## 3. Detailed Architecture

### Multi-Sensory Early-Fusion (Dual-Stream S4)
Noosphere v1.8.0 introduces a **Dual-Stream S4EEGEncoder**:
1.  **Global Riemannian Stream:** Uses tangent-space projection for stable covariance manifold features.
2.  **Localized S4 Stream:** Employs deeper State-Space blocks with learned **SpatialAttention** for fine-grained temporal dynamics.
3.  **Kinematic GNN:** Manages device topologies with learned-adjacency Graph Neural Networks.

### Physics-Augmented World Model (RSSM)
The latent state `s_t = (h_t, z_t)` is governed by a strict Hamiltonian physics prior combined with a deep `ResidualCorrector`.
*   **Digital Consequence Head:** Predicts structured digital outcomes (Shell exit codes, IoT state changes) with supervised feedback loops.
*   **Hardened NCP Binary Protocol:** High-density, low-latency binary packing for P2P identity and control frames.

---

## 4. Project Structure

```
noosphere/
├── configs.py        # Centralized settings (Perception, Physics, Planning).
├── intent.py         # Shared autonomy logic; blends brain signals with AI policy.
├── preprocessing.py  # Standardizes input from cameras, sensors, and EEG.
├── agent.py          # The core Perceive -> Simulate -> Act loop.
├── s4_eeg.py         # State-Space signal processor with Evidential Deep Learning (EDL).
├── physics.py        # Physics-Augmented RSSM and conservation laws.
├── rssm.py           # World Model, ConsequenceModel, and Digital Prediction heads.
├── planner.py        # MCTSPlanner, Actor, and Critic architectures.
├── actions.py        # Command vocabulary (Shell, IoT, Robotics).
├── proto.py          # Noosphere Network (NCP) binary protocol for P2P.
├── memory.py         # SequenceReplayBuffer and EpisodicMemory.
├── perception.py     # Multimodal Hybrid Perception Model.
├── gnn.py            # Kinematic GNN with learned adjacency.
├── hardware.py       # Drivers for servos, Bluetooth, and IoT integrations.
├── bundle.py         # WorldModelBundle format for sharing Dynamics Insights.
├── trainer.py        # Continuous Learning / "Sleep-Phase" engine.
├── monitor.py        # System telemetry (fatigue, workload, AI alignment).
├── learning.py       # SIGReg and Spatial Topology loss functions.
├── discovery.py      # Plug-and-play Hardware & Peer discovery.
└── synth.py          # SOTA Synthetic EEG (Kuramoto oscillators + Leadfield matrix).
```

---

## 5. Usage & Benchmarks

### Installation
```bash
pip install -r requirements.txt
```

### Proving Performance (The Academic Standard)
To run the MOABB-based benchmark used for peer review (aiming for **75% accuracy** across neuro-diverse datasets):
```bash
python demo_real_eeg.py --benchmark
```

### Demonstration Commands
*   **`python demo.py --smoke`**: Verifies all modules (Drone, Robot, BCI, Fluids) in one pass.
*   **`python demo.py --train`**: Starts the autogenous learning loop (The "Dreaming" loop).
*   **`python demo.py --partial`**: Tests "Modality Dropout" (e.g., operating with only EEG when cameras are blinded).

---

## 6. Privacy & Ethics
**Zero-to-One Autogenous Independence:** All neural mapping is performed locally. We strictly forbid the use of external "Teacher" models. The Noosphere Network only facilitates the exchange of **Dynamics Insights**—abstract patterns of environment physics—ensuring total cognitive privacy. Inter-agent communication and collective learning are **disabled by default** and must be explicitly toggled in `configs.py`.

---

## 7. Citation
```bibtex
@software{noosphere2026,
  title  = {Noosphere: Physics-Informed World Model Agent with BCI Control},
  year   = {2026},
  url    = {https://github.com/JosephWoodall/noosphere},
  author = {Joseph Woodall},
}
```
