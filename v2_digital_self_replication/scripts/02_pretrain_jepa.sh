#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — JEPA self-supervised pretraining of the stream encoder.
#
# No class labels needed.  Reads data from DATA_DIR, writes encoder checkpoint
# to CHECKPOINT_DIR/jepa_encoder_final.pt.
#
# Overridable env vars:
#   DATA_DIR        input data directory           (default: v2/.../data/generated)
#   CHECKPOINT_DIR  where to save checkpoints      (default: v2/.../checkpoints)
#   JEPA_EPOCHS     number of training epochs      (default: 50)
#   BATCH_SIZE      batch size                     (default: 64)
#   LR              learning rate                  (default: 3e-4)
#   WINDOW_LEN      EEG samples per JEPA window    (default: 256)
#   STRIDE          window stride                  (default: 64)
#   DEVICE          cpu or cuda                    (default: cpu)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

JEPA_EPOCHS="${JEPA_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-3e-4}"
WINDOW_LEN="${WINDOW_LEN:-256}"
STRIDE="${STRIDE:-64}"
LOG_FILE="$LOG_DIR/02_pretrain_jepa.log"

_log "=== Step 2: JEPA pretraining ==="
_log "  Data dir:       $DATA_DIR"
_log "  Checkpoint dir: $CHECKPOINT_DIR"
_log "  Epochs: $JEPA_EPOCHS   Batch: $BATCH_SIZE   LR: $LR   Device: $DEVICE"
_log "  Log file: $LOG_FILE"

"$VENV_PYTHON" -m v2_digital_self_replication.cli.pretrain_jepa \
    --data-dir       "$DATA_DIR"       \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --n-epochs       "$JEPA_EPOCHS"    \
    --batch-size     "$BATCH_SIZE"     \
    --lr             "$LR"             \
    --window-len     "$WINDOW_LEN"     \
    --stride         "$STRIDE"         \
    --device         "$DEVICE"         \
    --log-level      INFO              \
    --log-file       "$LOG_FILE"       \
    2>&1 | tee -a "$LOG_FILE"

_ok "JEPA pretraining complete → $CHECKPOINT_DIR/jepa_encoder_final.pt"
