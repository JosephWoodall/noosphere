#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Run the digital twin inference loop.
#
# Streams synthetic EEG through the trained twin, drives a simulated arm,
# and triggers online adaptation every 100 steps.
# Shows a rich live dashboard (--no-dashboard for plain log lines).
# Writes a JSON session summary to LOG_DIR/session_latest.json on exit.
#
# Overridable env vars:
#   TWIN_CKPT         full path to twin checkpoint        (default: calibrated.pt or supervised_best.pt)
#   SUBJECT_ID        synthetic subject profile seed      (default: 1)
#   SEED              random seed                         (default: 42)
#   N_STEPS           steps to run (0 = run until Ctrl-C) (default: 2560)
#   INTENT            6-DOF intent vector (space-separated floats) (default: "0.5 0.3 0.0 0.2 0.0 0.4")
#   LOG_INTERVAL      steps between status prints         (default: 256)
#   DEVICE            cpu or cuda                         (default: from env.sh)
#   ZMQ               set to "true" to enable ZMQ bridge + Arduino bridge (default: false)
#   ARDUINO_PORT      serial port for Arduino              (default: /dev/ttyACM0)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

# Prefer calibrated checkpoint if it exists, fall back to supervised_best
_DEFAULT_CKPT="$CHECKPOINT_DIR/supervised_best.pt"
[[ -f "$CHECKPOINT_DIR/calibrated.pt" ]] && _DEFAULT_CKPT="$CHECKPOINT_DIR/calibrated.pt"
TWIN_CKPT="${TWIN_CKPT:-$_DEFAULT_CKPT}"

SUBJECT_ID="${SUBJECT_ID:-1}"
SEED="${SEED:-42}"
N_STEPS="${N_STEPS:-2560}"
INTENT="${INTENT:-0.5 0.3 0.0 0.2 0.0 0.4}"
LOG_INTERVAL="${LOG_INTERVAL:-256}"
ZMQ="${ZMQ:-false}"
ARDUINO_PORT="${ARDUINO_PORT:-/dev/ttyACM0}"
SESSION_LOG="$LOG_DIR/session_latest.json"
LOG_FILE="$LOG_DIR/05_run_twin.log"

ZMQ_FLAG=""
[[ "$ZMQ" == "true" ]] && ZMQ_FLAG="--zmq"

_log "=== Step 5: Run digital twin ==="
_log "  Checkpoint:   $TWIN_CKPT"
_log "  Subject ID:   $SUBJECT_ID   Seed: $SEED"
_log "  Steps:        $N_STEPS"
_log "  Intent:       [$INTENT]"
_log "  Device:       $DEVICE"
_log "  ZMQ:          $ZMQ"
[[ "$ZMQ" == "true" ]] && _log "  Arduino port: $ARDUINO_PORT"
_log "  Session log:  $SESSION_LOG"
_log "  Log file:     $LOG_FILE"

# ── Optionally launch the Arduino bridge as a background process ──────────────
ARDUINO_PID=""
if [[ "$ZMQ" == "true" ]]; then
    _log "  Starting Arduino bridge on $ARDUINO_PORT..."
    "$VENV_PYTHON" -m v2_digital_self_replication.comms.arduino_bridge \
        --port "$ARDUINO_PORT" \
        --log-level INFO \
        >> "$LOG_DIR/arduino_bridge.log" 2>&1 &
    ARDUINO_PID=$!
    _log "  Arduino bridge PID: $ARDUINO_PID (log: $LOG_DIR/arduino_bridge.log)"
    sleep 3   # wait for Arduino reset + serial handshake
fi

# Ensure Arduino bridge is killed when this script exits (Ctrl-C or normal exit)
trap '[[ -n "$ARDUINO_PID" ]] && kill "$ARDUINO_PID" 2>/dev/null && _log "Arduino bridge stopped"' EXIT

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
    $ZMQ_FLAG                     \
    2>&1 | tee -a "$LOG_FILE"

_ok "Inference loop complete.  Session summary → $SESSION_LOG"
