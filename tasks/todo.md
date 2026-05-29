# v2 Digital Twin — Active TODO

## JBHI v2 Paper Pipeline

### Phase 1 — COMPLETE (2026-05-27)
- [x] Fix eval_loso.py bugs (checkpoint leak, window size, T=1025 JIT, CSP numpy-2.x, CUDA hang)
- [x] Run full 20-subject LOSO v1 → loso_results.json
- [x] Diagnose results: zero_shot_probe=33.0% (chance), csp_lda=60.1%, mdm=47.5%
- [x] Write LaTeX article → v2_digital_self_replication/articles/riemannian_s4_v2.tex
- [x] Edge case handled: zero_shot_probe ≈ chance → article reframed around architecture + negative result + CNS motivation

### Phase 2 — World Model + CNS Integration — COMPLETE (2026-05-29)
- [x] Task 4a: data/nlb_loader.py — synthetic MC_Maze (real NLB unavailable without DANDI credentials)
- [x] Task 4b: models/cns_encoder.py — ZOH-SSM on (B,T,182), same arch as EEG encoder
- [x] Task 4c: training/cross_modal_jepa.py — CNS teacher + EEG student + Sinkhorn OT alignment
- [x] Task 4d: ActionConditionedTransition — residual MLP T(h,a)→h', near-identity init, self-consistency loss
- [x] Task 4e: LatencyPlanner — hybrid CEM (K=32 MC rollouts) + gradient descent (12 Adam steps), γ=0.8
- [x] Task 5: Pretrain CNS encoder → checkpoints/jepa_encoder_cns_pretrained.pt
- [x] Task 6: Phase 2 cross-modal training → jepa_encoder_cns_pretrained_final.pt
- [x] Task 7: LOSO v3 (7 conditions, 20 subjects) → loso_results_v3.json
- [x] Task 8: Update article with v3 results + world model architecture sections
- [x] Task 9: Add missing bib entries (chua2018deep, hafner2020dreamer, schrittwieser2020muzero, williams2017information)

### LOSO v3 Final Results (20 subjects, 5-fold CV, 3-class MI)
| Condition | Accuracy | CI 95% |
|---|---|---|
| CSP + LDA | 60.1% | [54.7–65.6%] |
| MDM (Riemannian) | 47.5% | [42.5–52.8%] |
| JEPA encoder + cls (e2e FT) | 34.9% | [32.4–37.8%] |
| Random encoder ablation | 34.3% | [32.4–36.3%] |
| JEPA + world-model cls | 33.8% | [32.5–35.3%] |
| CNS cross-modal + cls | 33.7% | [32.2–35.2%] |
| CNS + world-model cls | 32.8% | [32.1–33.5%] |
| Chance | 33.3% | — |

**Interpretation:** All neural conditions at chance. Gap vs CSP = 26.8 pp.
World model designed for closed-loop latency reduction, not static accuracy.

## Backlog
- [ ] GPU: resolve RTX 5070 Blackwell compile hang (diagonal S4D parameterisation)
- [ ] Closed-loop simulation: drive virtual prosthetic arm with LatencyPlanner, measure ITR vs baseline
- [ ] Benchmark decode_step() latency (target: <10ms at 256Hz)
- [ ] ZMQ integration test with SimulatedHardware in separate process
- [ ] Install tectonic or texlive to enable local PDF compilation

## Lessons
- see tasks/lessons.md
