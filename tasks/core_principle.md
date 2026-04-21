# This Repo's North Star

**Core Principle:** The Noosphere RS-S4 Encoder solves subject non-stationarity in neuro-decoding by anchoring a long-range sequence model (the S4/HiPPO state-space layer) to a geometrically stable spatial representation (the Riemannian manifold).

**Why it feels magically correct:** Traditional deep learning (Transformers/CNNs) collapses when applied to BCI because muscle artifacts and electrode shifts destroy the temporal continuity of the signal. By wrapping standard voltage time series in a Symmetric Positive Definite (SPD) covariance manifold before temporal extraction, we render the spatial data invariant to volume conduction. Supplying this stable state to the S4 engine allows it to learn pure neural oscillations. 

**State-of-the-Art Grounding:**
- The spatial geometry is grounded in *Riemannian Geometry applied to BCI* (Yger et al., 2017).
- The temporal engine uses the modern State-Space limit, explicitly the *HiPPO (High-order Polynomial Projection Operators)* framework (Gu et al., 2020) for continuous-time memorization.
- The readout leverages *Evidential Deep Learning* (Sensoy et al., 2018) via Dirichlet distributions, explicitly estimating epistemic uncertainty rather than just softmax confidence.

**Alternatives Rejected:**
1. *Deep ConvNets (EEGNet/ShallowConvNet):* Fail at long-range temporal dependencies and require massive data scales.
2. *Transformers (EEGConformer):* Quadratic complexity $O(L^2)$ chokes on high-sampling-rate EEG (512 Hz), failing to process 10-second trials effectively. 
3. *Pure Riemannian (MDM/FBCSP):* Excellent spatial filters but fundamentally discard complex temporal wave shapes across varying frequency bands.

**Integration Mandate:** Every operation we run must prove that it maximizes the distance (delta) between this theoretically perfect architecture and legacy pipelines. If a dataset behaves fundamentally differently (like PhysionetMI's label noise), we do not cripple the model to account for it; we isolate the data.
