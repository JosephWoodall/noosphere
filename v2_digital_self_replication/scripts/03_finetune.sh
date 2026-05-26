#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Supervised fine-tuning of the digital twin.
#
# Loads the JEPA encoder from step 2, freezes the backbone,
# and fine-tunes the decoder + last encoder block on labeled data.
# Saves to CHECKPOINT_DIR/supervised_best.pt.
#
# Overridable env vars:
#   DATA_DIR          input data directory                (default: v2/.../data/generated)
#   CHECKPOINT_DIR    checkpoints directory               (default: v2/.../checkpoints)
#   ENCODER_CKPT      path to JEPA encoder checkpoint     (default: CHECKPOINT_DIR/jepa_encoder_final.pt)
#   FT_EPOCHS         fine-tuning epochs                  (default: 5)
#   BATCH_SIZE        batch size                          (default: 64)
#   WINDOW_LEN        EEG window length (samples)         (default: 128 = 500ms @ 256Hz)
#   FREEZE_ENCODER    "true" to freeze backbone           (default: true)
#   DEVICE            cpu or cuda                         (default: cpu)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/env.sh"

ENCODER_CKPT="${ENCODER_CKPT:-$CHECKPOINT_DIR/jepa_encoder_final.pt}"
FT_EPOCHS="${FT_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WINDOW_LEN="${WINDOW_LEN:-128}"
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
LOG_FILE="$LOG_DIR/03_finetune.log"

# Build freeze flag
FREEZE_FLAG="--freeze-encoder"
if [[ "$FREEZE_ENCODER" == "false" ]]; then
    FREEZE_FLAG="--no-freeze-encoder"
fi

_log "=== Step 3: Supervised fine-tuning ==="
_log "  Data dir:       $DATA_DIR"
_log "  Encoder ckpt:   $ENCODER_CKPT"
_log "  Checkpoint dir: $CHECKPOINT_DIR"
_log "  Epochs: $FT_EPOCHS   Batch: $BATCH_SIZE   Freeze encoder: $FREEZE_ENCODER   Device: $DEVICE"
_log "  Log file: $LOG_FILE"

"$VENV_PYTHON" -m v2_digital_self_replication.cli.finetune \
    --data-dir       "$DATA_DIR"       \
    --encoder-ckpt   "$ENCODER_CKPT"   \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --n-epochs       "$FT_EPOCHS"      \
    --batch-size     "$BATCH_SIZE"     \
    --window-len     "$WINDOW_LEN"     \
    $FREEZE_FLAG                       \
    --device         "$DEVICE"         \
    --log-level      INFO              \
    --log-file       "$LOG_FILE"       \
    2>&1 | tee -a "$LOG_FILE"

_ok "Fine-tuning complete → $CHECKPOINT_DIR/supervised_best.pt"
