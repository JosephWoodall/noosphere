# Architecture Redesign Plan (True BCI)

## 1. Relocate Sensors & Rename
- [x] `readme.md`: Change "neck/C7" references to "scalp/motor cortex (C3/C4/Cz)".
- [x] `readme.md`: Describe true BCI premise (CleanBrain as intent, Muscle as noise).
- [x] `readme.md`: Document decoupling of spatial targeting (IMU/Eye-tracking) and abstract intent (EEG).
- [x] `data/synth.py`: Rename `NeckEEGGenerator` to `ScalpEEGGenerator`.

## 2. Revert Artifact Rejection Logic
- [x] `data/synth.py`: Update `_next_sample()` and `next_segment()` so `CleanBrain` is the dominant intentional signal (desynchronization) and `MuscleArtifact` is background noise.
- [x] `apparatus.py`: Change `IntentionFilter.is_intentional()` to trigger on `RootArtifactLabel.CLEAN_BRAIN` rather than `MUSCLE`.
- [x] `apparatus.py`: Update `AnomalyDetector` documentation/logic if base probabilities change.

## 3. Separate Abstract Intent from Spatial Continuous Control
- [x] `data/synth.py`: Remove kinematics prediction from the EEG segment builder. Create or use existing IMU builder for continuous coordinates.
- [x] `s4_eeg.py`: Remove `xyz_head` output and `continuous_xyz` from `S4EEGEncoder`. Retain `intent_logits` and `confidence`.
- [x] `apparatus.py`: Refactor `SparseGPPredictor` and `NeuralCoordinatePredictor` to predict coordinates from a separate structural stream (e.g., IMU) rather than the S4 embedding, or bypass the GP and use IMU directly for spatial control.
- [x] `learning.py`: Remove `S4XYZSupervisionLoss` entirely as S4 no longer predicts coordinates.
- [x] `agent.py`: Remove `S4XYZSupervisionLoss` usages in the loop. Route IMU/structured data appropriately for spatial targeting instead of reading `s4_xyz_preds`.
- [x] `agent.py`: Update `apply_corrections` to supervise the appropriate new spatial prediction module, not the S4 encoder.

## Verification
- [ ] Run `python demo.py --smoke`, `python demo.py --partial`, `python demo.py --shell`, `python demo.py --apparatus`, `python demo.py --train --steps 200` to ensure no crashes.
- [ ] Ensure MCTS planner and ActBridge work with the new decoupled intent structure.

## Phase 2: SOTA BCI Enhancements
- [x] `s4_eeg.py`: Implement `S4D` (Diagonalized S4) for sequence modeling. (Already present in codebase as `A_log` diagonal approximation).
- [x] `s4_eeg.py`: Implement `Phase-Locking Value (PLV)` spatial attention before S4 blocks.
- [x] `s4_eeg.py`: Implement `Temporal Smoothing Head` (overlapping window ensemble).
- [x] `learning.py`: Add `EEGReconstructionLoss` to mask and reconstruct channels.
- [x] `agent.py`: Hook up `EEGReconstructionLoss` as an auxiliary objective in `_update_wm`.
- [x] `demo.py`: Verify changes via tests.

## Phase 2 Results Review
- **S4D & PLV Attention**: Successfully integrated Functional Connectivity Attention before the S4 blocks. The S4 sequences are effectively normalized prior to gating.
- **Denoising Loss**: EEGReconstructionLoss now successfully runs, correctly applying a mask to batch data and training the S4 encoder to act as a generative regularizer.
- **Numerical Stability**: Fixed a critical RK4 stiff-system divergence issue in `PhysicsTransitionPrior`. The random initialization of the new subcomponents shifted the global RNG sequence, revealing that the old unconstrained integrator could diverge and propagate infinity gradients across the entire world model parameter block. Clamped velocities and forces in `physics.py` unconditionally fix this for any RNG seed.

## Phase 3: Restoring User Agency
- [x] `agent.py`: Modify `step()` to directly use `argmax(s4_out["intent_logits"])` as the chosen action when BCI intent is confident.
- [x] `agent.py`: Bypass `planner.search()` and `actor.act()` when the S4 intent is triggered.
- [x] `agent.py`: Use `ConsequenceModel` to predict the forward consequence of the *human's decoded action*; pass termination warnings to `ActBridge`.
- [x] `actions.py`: Add safety gating to `ActBridge` when the simulated termination probability exceeds 0.90.
- [x] `demo.py`: Run full tests to verify the human intent executes perfectly without RL corruption.

## Phase 3 Results Review
- **True Prosthetic Alignment:** The Noosphere BCI now acts as an Obedient Consequence Engine. By bypassing the Actor-Critic and MCTS modules during confident BCI decodes, the system is strictly bound to execute the semantic command natively intended by the human brain (`s4_out["intent_logits"]`), preventing the RL reward function from overriding user agency.
- **Safety Gating via Simulation:** `agent.py` was updated to explicitly run `imagine_step` forward on the user's intended action. This allows the `ConsequenceModel` to predict the catastrophic likelihood of the command.
- **ActBridge Defense:** `actions.py` was updated so `ActBridge` intercepts and rejects any execution where the predicted `sim_termination` exceeds 90%, fulfilling the World Model's true purpose (safety verification) without hijacking command authority.

## Phase 4: Imitation Learning (Prosthetic Alignment Phase 2)
- [x] `config.py`/`agent.py`: Add `bc_weight` to the `AgentConfig`.
- [x] `agent.py`: Modify `_update_wm()` to save the deterministic and stochastic posteriors (`st_list`) and corresponding actions (`act_list`) taken by the user.
- [x] `agent.py`: Modify `_update_ac()` to read the buffered posteriors/actions from world model training. Compute the Actor's Negative Log-Likelihood (`L_bc = -dist.log_prob(act).mean()`) on the human's biological intent.
- [x] `agent.py`: Append `bc_weight * L_bc` to the final actor-critic loss equation to establish the Imitation Prior.
- [x] `demo.py`: Verify that `ac/bc_loss` is present, decreasing, and doesn't cause gradient divergence during continuous training.

## Phase 4 Results Review
- **Behavioral Cloning Pipeline:** `_update_ac` now ingests the detached posterior states and actual commanded actions from `_update_wm()`. This cleverly recycles the heavy perception/world-modeling forward pass, allowing us to compute the BC Negative Log-Likelihood loss almost for free.
- **The Imitation Prior:** By fusing `L_bc` into the Actor's loss equation, Noosphere's internal agent is no longer an autonomous explorer. It is a personalized digital twin. When the MCTS is invoked for macro-expansion or when the human is absent, the agent evaluates the environment using the *human's* baseline behavior, preventing the philosophical drift that ruins standard BCI/RL hybrid systems.
