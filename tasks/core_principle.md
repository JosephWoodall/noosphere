# This Repo's North Star

## 1. The Core Idea
Noosphere is an **Obedient Consequence Engine**, not an Autonomous RL Agent. The fundamental paradigm is that the Human (BCI) provides the *Command Intent*, High-Bandwidth Sensors (Vision/IMU) provide the *Spatial Context*, and the World Model predicts the *Physical/Digital Consequence* to assist execution. 

## 2. The Exact Intuition
Why does this feel magically correct? Because it respects the biological asymmetry of the human-machine team. The human brain is unparalleled at high-level semantic goal definition but bottlenecked by motor output (or BCI bandwidth). The machine is unparalleled at localized optimization and parallel simulation, but lacks intrinsic purpose. By restricting the BCI to emitting discrete intents (e.g., "Select", "Grasp", "Execute") and restricting the World Model to predicting the outcome of those intents, we build a true prosthesis—an extension of the human will—rather than a misaligned AI that happens to wear an EEG cap.

## 3. Grounding it Vigorously
Current SOTA research (e.g., *Shared Autonomy in Brain-Computer Interfaces*, 2024; *Latency and Agency in Neural Prostheses*, 2025) emphasizes that **User Agency** is the paramount metric. If a BCI system overrides user intent to maximize an artificial reward function, the user experiences "prosthetic rejection" (the system feels like an adversarial entity rather than a tool). State-space models (S4) decoding intent at 256Hz (arXiv:2511.23384) provide the low-latency command signal; the World Model's job is purely to verify feasibility and resolve continuous-space ambiguities (Inverse Kinematics).