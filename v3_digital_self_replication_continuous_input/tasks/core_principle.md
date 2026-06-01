# This Repo's North Star

**The single underlying principle:**
The brain continuously encodes intended arm movement as an oscillatory population vector across motor cortex channels — decode it continuously in real time, integrate it to position, and execute it on hardware. No discrete classes. No fixed targets. Any position the user imagines, the arm reaches.

---

## Why This Is the Right Problem

v2 proved that architecture is not the bottleneck: the ZOH-SSM encoder, Kalman filter, safety gate, and ZMQ hardware bridge all work correctly. The only broken component was the decoder — it maps brain states to near-zero motor commands because it was trained on three discrete class labels, not on continuous motor trajectories.

The brain does not encode discrete classes. Motor cortex fires continuous population vectors that encode movement direction, velocity, and effort magnitude (Georgopoulos et al., 1986, doi:10.1126/science.3749885). ERD amplitude at C3/C4 is proportional to imagined effort, not a binary on/off. The signal for continuous decoding already exists in raw EEG. What was missing was a decoder that reads it.

---

## State-of-the-Art Grounding

- **Georgopoulos population vector (1986):** Motor cortex population activity encodes movement direction as a vector sum of preferred directions weighted by firing rate. With 21-channel EEG, ERD amplitude at motor channels is the non-invasive proxy. doi:10.1126/science.3749885
- **Moran & Schwartz (1999):** Population vector predicts hand velocity continuously from cortical activity, not just direction. doi:10.1152/jn.1999.82.5.2207
- **Willett et al. (2021):** 90 characters/minute from imagined handwriting decoded continuously from motor cortex — continuous decoding, not classification. doi:10.1038/s41586-021-03506-2
- **AC-SSM (v2, this repo):** Action-conditioned SSM world model, ZOH stability guarantees, 33.8% closed-loop convergence from 55 training trials. The correct encoder backbone — just needs a continuous decoder on top.

---

## Alternatives Rejected

1. **More discrete class labels:** Even at 576 trials (BCI2a max), classification accuracy stays 22pp below CSP. Collecting more discrete labels does not solve the continuous decoding problem.
2. **Larger neural network decoder:** The bottleneck is signal geometry (ERD is continuous), not model capacity. A bigger MLP trained on three class labels still outputs only three directions.
3. **Invasive BCI (ECoG/Utah array):** Higher SNR but not the vision. The non-invasive pathway via scalp EEG is the clinical target.

---

## Alignment Check

Every code change must answer: *Does this help the system continuously decode motor intent from streaming EEG and execute it to hardware with lower latency, higher position accuracy, or better safety?* If not, cut it.
