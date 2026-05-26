# Lessons Learned

## L001 — torch.jit.script deprecated in Python 3.14
**Pattern:** `@torch.jit.script` works but emits `DeprecationWarning` on Python 3.14+.
**Rule:** Migrate inner scan loops to `torch.compile(fn)` for future-proofing. JIT still works now but is on borrowed time.
**How to apply:** When writing new JIT-accelerated functions, use `torch.compile` instead of `@torch.jit.script`.

## L002 — JEPA WINDOW_LEN=256 is infeasible on CPU
**Pattern:** T=256 with full 50-epoch JEPA on CPU = 26+ hours. Default must be CPU-aware.
**Rule:** Default `WINDOW_LEN=128`, `JEPA_EPOCHS=15` for CPU runs. T=128 (500ms) is scientifically valid for EEG (captures 4-8 alpha cycles) and ~4x faster than T=256.
**How to apply:** Set `WINDOW_LEN` and `JEPA_EPOCHS` defaults in scripts/env.sh to be CPU-feasible; document GPU overrides clearly.

## L003 — test_no_nan_gradients must pass all modalities
**Pattern:** Test originally only passed `eeg`; hrv/gsr/prop projections had None gradients, causing the test to fail.
**Rule:** Any test that checks gradient flow must exercise every parameter in the model — pass all modalities (eeg, hrv, gsr, prop) to MultiModalFusion.
**How to apply:** In gradient tests, always construct minimal tensors for all inputs and verify `.grad is not None` for every parameter.

## L005 — GPU auto-detection: env.sh now defaults to cuda if available
**Pattern:** Original env.sh had `DEVICE=cpu` hardcoded. RTX 5070 was idle while training ran on CPU.
**Rule:** env.sh probes torch.cuda.is_available() at startup. DEVICE=cuda if GPU present, cpu otherwise. Override by setting DEVICE explicitly before calling any script.
**How to apply:** This is now the default in env.sh. No action needed — GPU is used automatically.

## L004 — Pipeline --quick flag is a positional arg, not an env var
**Pattern:** `QUICK=true bash run_pipeline.sh` silently does nothing; the script resets `QUICK=false` at start. The flag is `--quick` passed as an argument.
**Rule:** Pipeline flags are positional arguments, not env vars. To override individual settings, use the specific env vars (JEPA_EPOCHS, WINDOW_LEN, etc.).
**How to apply:** Document this in the README; add a comment in run_pipeline.sh header.
