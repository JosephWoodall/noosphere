# v2 Digital Twin — Active TODO

## JBHI v2 Paper Pipeline — COMPLETE (2026-05-31)

### Article: riemannian_s4_v2.tex (724 lines, branch: joseph-woodall-branch)

All evaluation complete and committed. Article ready for submission.

### Final Results Summary

**PhysioNetMI (20 subjects, 3-class, 8 conditions):**
| Condition | Accuracy | 95% CI |
|---|---|---|
| CSP+LDA | 60.1% | [54.7–65.6%] |
| MDM | 47.5% | [42.5–52.8%] |
| EEGNet (supervised) | 38.4% | [35.6–41.2%] |
| JEPA encoder (e2e FT) | 36.6% | [34.0–39.3%] |
| Ablation (random init) | 35.5% | [33.4–37.7%] |
| World-model cls | 34.2% | [32.9–35.6%] |
| CNS cross-modal | 32.7% | [31.9–33.6%] |

**BCI2a (9 subjects, 4-class):**
| Condition | Accuracy |
|---|---|
| CSP+LDA | 58.7% |
| MDM | 50.4% |
| EEGNet | 36.0% |
| JEPA (cross-domain) | 27.9% ≈ ablation |

**Closed-loop sim (133 trials, 3 controllers):**
| Controller | Convergence |
|---|---|
| Oracle (perfect decoder) | 100% at step 193 |
| CSP (60% accuracy) | 27.8% |
| JEPA decoder | 0% |

**Latency (CPU, post-compile):**
- decode_step(): 26.9 ms → 37 Hz
- self_condition() fast path: 10.9 ms → 92 Hz

### Remaining (optional, not needed for submission)
- [ ] BCI2a native JEPA pretraining (~54h CPU, in background, not blocking)
- [ ] GPU parallelisation via S4D parameterisation (future work)
- [ ] Closed-loop sim with real hardware once Pico is connected

## Backlog
- [ ] Resolve RTX 5070 Blackwell CUDA compile hang (S4D diagonal parameterisation)
- [ ] ZMQ integration test with SimulatedHardware in separate process

## Lessons
- see tasks/lessons.md
