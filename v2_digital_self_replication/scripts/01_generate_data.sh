#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Generate synthetic EEG + physiology training data.
#
# Output: DATA_DIR/*.npy  +  DATA_DIR/metadata.json
#
# Overridable env vars:
#   DATA_DIR        where to write files     (default: v2/.../data/generated)
#   N_SUBJECTS      number of subjects       (default: 10)
#   N_TRIALS        trials per subject       (default: 50)
#   DURATION        trial duration seconds   (default: 4.0)
#   FS              sampling rate Hz         (default: 256)
#   SEED            random seed              (default: 42)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

N_SUBJECTS="${N_SUBJECTS:-10}"
N_TRIALS="${N_TRIALS:-50}"
DURATION="${DURATION:-4.0}"
FS="${FS:-256}"
SEED="${SEED:-42}"
LOG_FILE="$LOG_DIR/01_generate_data.log"

_log "=== Step 1: Generate synthetic data ==="
_log "  Subjects: $N_SUBJECTS   Trials: $N_TRIALS   Duration: ${DURATION}s   FS: ${FS}Hz   Seed: $SEED"
_log "  Output dir: $DATA_DIR"
_log "  Log file:   $LOG_FILE"

"$VENV_PYTHON" -m v2_digital_self_replication.cli.generate_data \
    --n-subjects  "$N_SUBJECTS"  \
    --n-trials    "$N_TRIALS"    \
    --duration    "$DURATION"    \
    --fs          "$FS"          \
    --seed        "$SEED"        \
    --output-dir  "$DATA_DIR"    \
    --log-level   INFO           \
    --log-file    "$LOG_FILE"    \
    2>&1 | tee -a "$LOG_FILE"

_ok "Data generation complete → $DATA_DIR"
