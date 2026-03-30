# This Repo's North Star

## 1. The Core Idea: Intent-Conditioned Subsumption
Noosphere is an **Obedient Consequence Engine**. It utilizes an Autonomous RL Agent purely as a *Subjugated Path-Planner*. The fundamental paradigm is that the Human (BCI) provides the *Macro-Intent*, High-Bandwidth Sensors (Vision/IMU) provide the *Spatial Context*, and the World Model leverages the RL Agent to optimize the micro-actions required to safely execute that intent.

## 2. The Exact Intuition
Why does this feel magically correct? Because it respects the biological asymmetry of the human-machine team. The human brain is unparalleled at high-level semantic goal definition but bottlenecked by motor output. The machine is unparalleled at localized optimization (MCTS) and high-frequency control (RL), but lacks intrinsic purpose. By explicitly defining the RL Actor as a "Digital Twin" that seeks to minimize Behavioral Cloning loss, we ensure the agent's autonomous explorations strictly conform to the human's preferred operational manifold.

## 3. Grounding it Vigorously
Current SOTA research (e.g., *Shared Autonomy in Brain-Computer Interfaces*, 2024; *Latency and Agency in Neural Prostheses*, 2025) emphasizes that **User Agency** is the paramount metric. If a BCI system overrides user intent to maximize an artificial reward function, the user experiences "prosthetic rejection." State-space models (S4) decode Evidential Uncertainty at 256Hz (arXiv:2511.23384). This uncertainty directly scales the RL Path-Planner's compute budget. High human confidence triggers a Zero-Latency Fast-Path bypass; high uncertainty deploys the MCTS to safely resolve ambiguity.

## 4. The Epistemological Split (Solving the Physics Delusion)
The continuous physical intent (the arm trajectory) and the discrete digital intent (shell commands) must be tracked independently. The Replay Buffer splits experience into:
* `raw_continuous`: The pure biological intent (used to train the psychological Digital Twin).
* `exec_continuous`: The physically clamped action permitted by the Safety Gate (used to train the World Model's physics simulation).
Failing to separate these causes the World Model to hallucinate impossible physics, and the Actor to fail at cloning true human desire.