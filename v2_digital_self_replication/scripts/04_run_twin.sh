#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Run the digital twin inference loop.
#
# Streams synthetic EEG through the trained twin, drives the simulated arm,
# and triggers online adaptation every 100 steps.
# Prints live status every LOG_INTERVAL steps.
# Writes a JSON session summary to LOG_DIR/session_latest.json on exit.
#
# Overridable env vars:
#   CHECKPOINT_DIR    directory with supervised_best.pt  (default: v2/.../checkpoints)
#   TWIN_CKPT         full path to twin checkpoint        (default: CHECKPOINT_DIR/supervised_best.pt)
#   SUBJECT_ID        synthetic subject profile seed      (default: 1)
#   SEED              random seed                         (default: 42)
#   N_STEPS           steps to run (0 = run until Ctrl-C) (default: 2560)
#   INTENT            6-DOF intent vector (space-separated floats) (default: "0.5 0.3 0.0 0.2 0.0 0.4")
#   LOG_INTERVAL      steps between status prints         (default: 256)
#   DEVICE            cpu or cuda                         (default: cpu)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

TWIN_CKPT="${TWIN_CKPT:-$CHECKPOINT_DIR/supervised_best.pt}"
SUBJECT_ID="${SUBJECT_ID:-1}"
SEED="${SEED:-42}"
N_STEPS="${N_STEPS:-2560}"
INTENT="${INTENT:-0.5 0.3 0.0 0.2 0.0 0.4}"
LOG_INTERVAL="${LOG_INTERVAL:-256}"
SESSION_LOG="$LOG_DIR/session_latest.json"
LOG_FILE="$LOG_DIR/04_run_twin.log"

_log "=== Step 4: Run digital twin ==="
_log "  Checkpoint:   $TWIN_CKPT"
_log "  Subject ID:   $SUBJECT_ID   Seed: $SEED"
_log "  Steps:        $N_STEPS"
_log "  Intent:       [$INTENT]"
_log "  Device:       $DEVICE"
_log "  Session log:  $SESSION_LOG"
_log "  Log file:     $LOG_FILE"

# shellcheck disable=SC2086
"$VENV_PYTHON" -m v2_digital_self_replication.cli.run_twin \
    --checkpoint   "$TWIN_CKPT"   \
    --subject-id   "$SUBJECT_ID"  \
    --seed         "$SEED"        \
    --n-steps      "$N_STEPS"     \
    --intent       $INTENT        \
    --log-interval "$LOG_INTERVAL" \
    --device       "$DEVICE"      \
    --session-log  "$SESSION_LOG" \
    --log-level    INFO           \
    2>&1 | tee -a "$LOG_FILE"

_ok "Inference loop complete.  Session summary → $SESSION_LOG"
