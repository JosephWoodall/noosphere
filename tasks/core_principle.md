# This Repo's North Star

**The single underlying principle:**
Continuous intent is already encoded in the brain's motor cortex as a streaming neural process — the goal is to decode it in real-time as a 6-DOF continuous command, not classify it into discrete categories.

---

## Why This Feels Magically Correct

v1 was a discriminative classifier: EEG → {left, right, rest}. Every discriminative BCI is bounded by the category set. You can never express "grip at 40% aperture while rotating wrist 20°" because the label space doesn't contain it. The user saw this immediately: *"With a discriminative process, I am limited to the categories in which I can create actions from."*

The brain doesn't encode discrete categories. Motor cortex fires continuous population vectors that directly encode movement direction, velocity, and force magnitude (Georgopoulos et al., 1986). ERD/ERS amplitude at C3/C4 is proportional to effort magnitude, not a binary class. FCz encodes error signals (ERN) that are graded by surprise. These are continuous signals, and decoding them continuously is the mathematically correct solution.

---

## State-of-the-Art Grounding

- **ZOH SSM (Mamba/S4):** Discretized SSMs with zero-order hold give provably stable recurrence (`bar_A = exp(dt*A)`, stability guaranteed when `A < 0`). Gu et al., 2021 (arXiv:2111.00396), Gu & Dao, 2023 (arXiv:2312.00752).
- **JEPA pretraining:** Joint-Embedding Predictive Architecture learns latent representations by predicting future EEG states from past context without class labels. LeCun, 2022 (OpenReview). Eliminates the label bottleneck entirely.
- **Adaptive Kalman filter:** Covariance `R = diag(sigma²)` — when the decoder is uncertain (high sigma), the filter trusts momentum over prediction. Provides physically meaningful smoothing matched to model confidence.
- **Bilinear binding:** `h = bar_A*h + bar_B*x + bar_W*tanh(h*x)` — captures nonlinear cross-temporal correlation (neural synchrony, phase-amplitude coupling) within the stable ZOH recurrence framework.

---

## Alternatives Rejected

1. **Transformer-based BCI (e.g., EEGNet-Transformer):** O(T²) attention is incompatible with causal streaming inference. Cannot run decode_step() at O(1). Killed by latency requirements.
2. **LLM fine-tuned on EEG tokens (e.g., SubQ-style):** Text LLMs have no inductive bias for the temporal autocorrelation structure of EEG oscillations. Requires tokenization that destroys phase information. Wrong substrate.
3. **Classical Riemannian geometry classifier (v1):** Works well for N-class discrimination but cannot output a continuous 6-DOF vector. The manifold structure constrains you to a finite label set. Provably insufficient for the stated goal.

---

## Alignment Check (apply to every change)

Every code change must answer: *Does this help the system decode continuous motor intent from streaming EEG with lower latency, higher accuracy, or better safety?* If not, cut it.
