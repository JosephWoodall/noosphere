# This Repo's North Star

## 1. The Core Idea
Noosphere is an **Obedient Consequence Engine**, not an Autonomous RL Agent. The fundamental paradigm is that the Human (BCI) provides the *Command Intent*, High-Bandwidth Sensors (Vision/IMU) provide the *Spatial Context*, and the World Model predicts the *Physical/Digital Consequence* to assist execution. 

## 2. The Exact Intuition
Why does this feel magically correct? Because it respects the biological asymmetry of the human-machine team. The human brain is unparalleled at high-level semantic goal definition but bottlenecked by motor output (or BCI bandwidth). The machine is unparalleled at localized optimization and parallel simulation, but lacks intrinsic purpose. By restricting the BCI to emitting discrete intents (e.g., "Select", "Grasp", "Execute") and restricting the World Model to predicting the outcome of those intents, we build a true prosthesis—an extension of the human will—rather than a misaligned AI that happens to wear an EEG cap.

## 3. Grounding it Vigorously
Current SOTA research (e.g., *Shared Autonomy in Brain-Computer Interfaces*, 2024; *Latency and Agency in Neural Prostheses*, 2025) emphasizes that **User Agency** is the paramount metric. If a BCI system overrides user intent to maximize an artificial reward function, the user experiences "prosthetic rejection" (the system feels like an adversarial entity rather than a tool). State-space models (S4) decoding intent at 256Hz (arXiv:2511.23384) provide the low-latency command signal; the World Model's job is purely to verify feasibility and resolve continuous-space ambiguities (Inverse Kinematics).

## 4. The Fundamental Flaw Discovered
**The current architecture exhibits a catastrophic misalignment: it demotes the human to a passive sensor.** 
In `agent.py`, the `step()` function feeds the EEG observations into the World Model to update its state, but then calls `self.planner.search()` or `self.actor.act()` to select the final action. The agent is choosing its own action to maximize arbitrary environmental rewards (e.g., `shell_reward`), completely ignoring the `intent_logits` decoded from the S4 module! If the human intends to execute a "destructive" command, the RL planner will refuse and execute "ls" instead because it yields a higher RL reward.

### How to Fix the Flaw:
1. **Action Decoding Bypass**: The executed `action` must explicitly be `argmax(s4_out["intent_logits"])` when BCI mode is active, not the output of the Actor/MCTS.
2. **Repurpose MCTS / World Model**: The World Model should only simulate the forward consequence of the *Human's Decoded Action*. If the `ConsequenceModel` predicts `termination_prob > 0.9` (e.g., a catastrophic collision or destructive command), the `ActBridge` acts as a safety gate to block it.
3. **Macro-Expansion (Future)**: If the human intent requires continuous trajectory planning (e.g., "Fetch"), the Mcripts planner is bounded to optimize *only* the micro-actions that fulfill the human's macro-intent, penalizing any deviation from the user's semantic goal.

## 5. The Second Misalignment: Extrinsic RL vs. Imitation Learning
Even with the action decoding bypass fix, `agent.py._update_ac()` still trains the Actor-Critic purely to maximize the generic environmental reward using TD-lambda in imagination. 
**Why this is philosophically broken:** If the human uses the BCI to perform tasks, the Actor should be learning to be a *digital twin* of the human (Behavioral Cloning / Imitation Learning) so it can eventually automate repetitive cognitive workloads exactly as the user prefers. Training the Actor on an extrinsic RL reward causes the agent's internal policy to rapidly drift *away* from the human's preferred behavioral distribution, breaking Prosthetic Alignment when MCTS is used for digital exploration.
**The Fix (Phase 4):**
Augment the Actor training loss in `agent.py`. In addition to the standard RL objective, sample the human's explicitly commanded actions (`argmax(intent_logits)`) from the Replay Buffer and add a Behavioral Cloning (BC) Negative Log-Likelihood loss term. This creates an **Imitation Prior** where the Actor's baseline behavior perfectly mirrors the user's biological intent.
