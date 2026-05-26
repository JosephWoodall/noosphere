# v2 Digital Twin — Active TODO

## In Progress
- [ ] Full cold-start pipeline run (JEPA T=128, 15 epochs, FT 5 epochs) — running now

## Next: Post-pipeline
- [ ] Migrate `torch.jit.script` → `torch.compile` for Python 3.14 compatibility
      (current JIT works but is deprecated; `torch.compile` backend='inductor' is the path)
- [ ] Add per-epoch checkpoint loss curve logging to pretrain_jepa.py
- [ ] Validate session_latest.json output from run_twin

## Backlog
- [ ] GPU smoke test: re-run pipeline with DEVICE=cuda if hardware becomes available
- [ ] Benchmark decode_step() latency end-to-end (target: <10ms per step at 256Hz)
- [ ] ZMQ integration test with SimulatedHardware in a separate process
- [ ] Replace hard-coded channel count (21) with config-driven value throughout

## Lessons
- see tasks/lessons.md
