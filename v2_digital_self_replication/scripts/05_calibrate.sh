#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Subject calibration loop.
#
# Runs the countdown-based capture loop, fine-tunes the twin on
# captured data, and saves a calibrated checkpoint.
#
# Overridable env vars:
#   CHECKPOINT_DIR   directory with supervised_best.pt  (default: v2/.../checkpoints)
#   SUBJECT_ID       subject profile seed               (default: 1)
#   SEED             random seed for data gen           (default: 0)
#   N_REPS           repetitions per movement target    (default: 3)
#   CAPTURE_S        capture duration per rep (seconds) (default: 2.0)
#   FT_EPOCHS        fine-tune epochs on captured data  (default: 5)
#   DEVICE           cpu or cuda                        (default: from env.sh)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

SUBJECT_ID="${SUBJECT_ID:-1}"
SEED="${SEED:-0}"
N_REPS="${N_REPS:-3}"
CAPTURE_S="${CAPTURE_S:-2.0}"
FT_EPOCHS="${FT_EPOCHS:-5}"
INPUT_CKPT="${TWIN_CKPT:-$CHECKPOINT_DIR/supervised_best.pt}"
OUTPUT_CKPT="$CHECKPOINT_DIR/calibrated.pt"
LOG_FILE="$LOG_DIR/05_calibrate.log"

_log "=== Step 5: Subject calibration ==="
_log "  Input checkpoint:  $INPUT_CKPT"
_log "  Output checkpoint: $OUTPUT_CKPT"
_log "  Subject ID: $SUBJECT_ID  Seed: $SEED"
_log "  Reps per target: $N_REPS  Capture: ${CAPTURE_S}s  FT epochs: $FT_EPOCHS"
_log "  Device: $DEVICE"

"$VENV_PYTHON" -m v2_digital_self_replication.cli.calibrate \
    --checkpoint   "$INPUT_CKPT"   \
    --output       "$OUTPUT_CKPT"  \
    --subject-id   "$SUBJECT_ID"   \
    --seed         "$SEED"         \
    --n-reps       "$N_REPS"       \
    --capture-s    "$CAPTURE_S"    \
    --ft-epochs    "$FT_EPOCHS"    \
    --device       "$DEVICE"       \
    --log-level    INFO            \
    2>&1 | tee -a "$LOG_FILE"

_ok "Calibration complete → $OUTPUT_CKPT"
