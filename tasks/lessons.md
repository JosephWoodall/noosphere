# Lessons Learned

## Numerical Stability in Simulation Interacts with Perception Randomness
**Date**: March 23, 2026
**Context**: Added SOTA BCI components (S4D, PLV attention, Temporal Smoothing) which instantiated new neural network parameters.
**Issue**: Upon running the first training step, the World Model physics loss `wm/physics` evaluated to `inf`, cascading NaN gradients across all encoders and crashing the script at the actor inference step.
**Root Cause**: The unconstrained RK4 integration in `PhysicsTransitionPrior` had a dormant stiffness instability. The newly added parameters in `S4EEGEncoder` consumed pseudo-random numbers from the PyTorch RNG, shifting the initialization seqeunce for the `PhysicsStateEstimator`. The new random weights caused the estimator to predict heavily overlapping rigid bodies with immense velocities, causing the RK4 integration forces (spring penalization and aerodynamic drag) to explode to infinity within 4 solver steps.
**Fix**: Clamped the intermediate derivatives (`dv`, `dpos`, `domega`, `drot`) and energy dissipation directly within the RK4 `_deriv` function to guarantee unconditional mathematical stability regardless of parameter initialization. 

**Takeaway**: In end-to-end models with non-differentiable or explicit physics simulators (like RK4), always bound the numerical derivatives. Even if the system appears stable for a long time, a simple change in architecture can shift the parameter RNG seed and expose dormant explosion edge-cases.

### Lesson 1: RK4 Stability in Augmented Physics Simulators
**Issue:** 
Integration of S4D and PLV attention added stochasticity to the perception initialization, which shifted the global random number generator (RNG) sequence. This new RNG sequence spawned an initial state in the `PhysicsAugmentedRSSM` that caused the unconstrained RK4 integrator to diverge, propagating `inf` gradients and resulting in NaN losses.

**Root Cause:**
RK4 is conditionally stable. When observing novel or chaotic representations (like those produced during un-trained epochs or shifted by RNG), the velocities and forces could momentarily spike. `dt` was fixed at `1/60`, meaning large derivatives caused the integration to jump past the stability region.

**Fix:**
Always clamp derivatives (velocities, forces, angular velocities) within the integration steps of a neural-differentiable physics simulator. It's not enough to clamp the states; the intermediate `_deriv` function outputs must be bounded to strictly enforce Lipschitz continuity and prevent numerical explosion regardless of initial seed.

---

### Lesson 2: Prosthetic Alignment vs. Autonomous RL
**Issue:**
The original architecture fed human intent (EEG) into the World Model and then ran an MCTS planner (using an Actor-Critic policy) to select the final action. If the human intended a command that the system's RL agent deemed "suboptimal" (e.g., executing a destructive command with `reward = -0.5`), the system would silently override the human and choose a different action to maximize its own reward score.

**Root Cause:**
Applying standard RL (where the agent discovers actions to maximize environmental reward) to a BCI prosthesis. A prosthesis must be an obedient extension of the user's will. By treating the human's explicit brain command as a passive environmental "observation", the system demoted the human to a sensor and promoted the AI to the commander. Furthermore, training the internal Actor via TD-lambda RL inside imagination causes its behavioral distribution to drift towards generic artificial reward metrics rather than anchoring to the user's biological preferences.

**Fix:**
**Action Decoding Bypass:** We refactored `agent.py` to act as an *Obedient Consequence Engine*. When confident BCI intent is triggered, the system explicitly decodes the action (`argmax(s4_out["intent_logits"])`) and bypasses the MCTS Planner/Actor entirely. 
**Repurposing Simulation:** The World Model still runs, but it only simulates the forward trajectory of the *human's intended action* to predict safety/termination. If the simulated termination probability is critical (>90%), `ActBridge` steps in to block the catastrophic failure, acting as a safety gate rather than an autonomous decision-maker.
**The Imitation Prior (Behavioral Cloning):** To ensure the internal Actor remains perfectly culturally aligned with the user (for when MCTS macro-expansion is required), we added a Negative Log-Likelihood (NLL) Behavioral Cloning loss (`L_bc`) to the `_update_ac()` sequence. The Actor is now forced to mimic the human's explicitly executed commands stored in the replay buffer, making it a personalized digital twin.
