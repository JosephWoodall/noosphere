# v2 Digital Twin — Active TODO

## JBHI → TNSRE Resubmission Pipeline (2026-05-31)

### Article: riemannian_s4_v2.tex (branch: joseph-woodall-branch)

### Architecture Changes (DONE)
- [x] Action-conditioned SSM: SiLU-gated `a_prev` injection into first SSM block (stream_encoder.py)
- [x] EEG reconstruction head added to StreamEncoder
- [x] New `ac_ssm` LOSO eval condition (eval_loso.py) — curriculum + cosine alignment + reconstruction loss
- [x] New `closed_loop_sim_v2.py` — four-controller sim with AC-SSM classifier-decoder
- [x] All 18 existing tests still pass

### Evaluations
- [x] Closed-loop sim v2 (10 subjects, 133 trials) — DONE
      AC-SSM: 33.8% convergence, TTT=319 (vs JEPA 0%, p<0.0001; vs CSP 26.3%, p=0.21)
- [ ] LOSO v5 (20 subjects, 5 folds, fast mode) — RUNNING (~2h remaining, at subject 7/20)

### Article Update (PENDING — run when LOSO v5 completes)
- [ ] Run: `python v2_digital_self_replication/scripts/update_article.py`
      Updates all tables, per-subject rows, inline numbers, closed-loop section
- [ ] Review diff and verify no stale numbers remain
- [ ] Update abstract with new primary result framing
- [ ] Update Discussion §5.2 (world model value) — now has non-zero convergence
- [ ] Check contributions list (item 3 — transition model; item 4 — planner)

### Submission Prep (AFTER article update)
- [ ] Switch venue framing to IEEE TNSRE (title, cover letter, keywords)
- [ ] Write TNSRE cover letter
- [ ] Compile PDF, verify no LaTeX errors
- [ ] Final read-through for stale JBHI-specific language
- [ ] Submit

## Key Numbers (to verify against article after update)

### Closed-loop sim v2 (FINAL)
| Controller | Convergence | TTT (steps) | Final Err | p vs JEPA |
|---|---|---|---|---|
| Oracle | 100.0% | 193 ± 0 | 0.246 | — |
| CSP | 26.3% | 334 ± 84 | 1.252 | p<0.0001 |
| JEPA | 0.0% | 384 ± 0 | 0.984 | baseline |
| **AC-SSM** | **33.8%** | **319 ± 90** | **1.147** | **p<0.0001** |
AC-SSM vs CSP: p=0.21 (not significant — competitive)

### LOSO v5 (PENDING)
TBD — will fill after eval completes

## Backlog
- [ ] Resolve RTX 5070 Blackwell CUDA compile hang (S4D diagonal parameterisation)
- [ ] ZMQ integration test with SimulatedHardware in separate process
- [ ] BCI2a native AC-SSM evaluation (optional, not blocking)
