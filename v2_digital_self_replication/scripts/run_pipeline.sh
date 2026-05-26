#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════════════
# run_pipeline.sh — Full v2 Digital Twin pipeline (cold start to inference)
#
# Runs all five steps in sequence:
#   01 Generate synthetic EEG + physiology training data
#   02 JEPA self-supervised pretraining (no labels)
#   03 Supervised fine-tuning (labeled EEG → motor command)
#   04 Subject calibration (personalize decoder to this subject)
#   05 Inference loop (stream → arm command, online adaptation)
#
# Usage:
#   ./run_pipeline.sh                    # full run with defaults
#   ./run_pipeline.sh --skip-data        # reuse existing data
#   ./run_pipeline.sh --skip-pretrain    # skip JEPA (reuse checkpoint)
#   ./run_pipeline.sh --skip-finetune    # skip supervised ft
#   ./run_pipeline.sh --skip-calibrate   # skip subject calibration
#   ./run_pipeline.sh --quick            # small data + few epochs (smoke test)
#
# Any env var accepted by the individual step scripts can be exported before
# calling this script to override defaults (see each 0N_*.sh header).
# ═════════════════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPTS_DIR/env.sh"

# ── Flags ─────────────────────────────────────────────────────────────────────
SKIP_DATA=false
SKIP_PRETRAIN=false
SKIP_FINETUNE=false
SKIP_CALIBRATE=false
QUICK=false

for arg in "$@"; do
    case "$arg" in
        --skip-data)       SKIP_DATA=true       ;;
        --skip-pretrain)   SKIP_PRETRAIN=true   ;;
        --skip-finetune)   SKIP_FINETUNE=true   ;;
        --skip-calibrate)  SKIP_CALIBRATE=true  ;;
        --quick)           QUICK=true           ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            _err "Unknown argument: $arg  (use --help)"
            exit 1
            ;;
    esac
done

# ── Quick mode: tiny data + minimal epochs for a fast smoke test ───────────────
if [[ "$QUICK" == true ]]; then
    _log "QUICK MODE — reduced settings for smoke test"
    export N_SUBJECTS="${N_SUBJECTS:-2}"
    export N_TRIALS="${N_TRIALS:-5}"
    export DURATION="${DURATION:-2.0}"
    export JEPA_EPOCHS="${JEPA_EPOCHS:-3}"
    export WINDOW_LEN="${WINDOW_LEN:-64}"
    export STRIDE="${STRIDE:-32}"
    export FT_EPOCHS="${FT_EPOCHS:-2}"
    export N_STEPS="${N_STEPS:-512}"
    export N_REPS="${N_REPS:-1}"
    export CAPTURE_S="${CAPTURE_S:-1.0}"
    export CAL_FT_EPOCHS="${CAL_FT_EPOCHS:-1}"
fi

# ── Pipeline ──────────────────────────────────────────────────────────────────
PIPELINE_LOG="$LOG_DIR/pipeline_$(date +'%Y%m%d_%H%M%S').log"
_log "═══════════════════════════════════════════════════════"
_log "  v2 Digital Twin — Full Pipeline"
_log "  Started:        $(date +'%Y-%m-%d %H:%M:%S')"
_log "  Repo root:      $REPO_ROOT"
_log "  Data dir:       $DATA_DIR"
_log "  Checkpoint dir: $CHECKPOINT_DIR"
_log "  Log dir:        $LOG_DIR"
_log "  Pipeline log:   $PIPELINE_LOG"
_log "  Device:         $DEVICE"
_log "═══════════════════════════════════════════════════════"

_start_step() {
    _log ""
    _log "───────────────────────────────────────────────────────"
    _log "  $1"
    _log "───────────────────────────────────────────────────────"
}

T_PIPELINE_START=$(date +%s)
STEPS_RUN=()
STEPS_SKIPPED=()

# ── Step 1: Generate data ─────────────────────────────────────────────────────
if [[ "$SKIP_DATA" == true ]]; then
    _log "SKIP: Step 1 (--skip-data)"
    if [[ ! -f "$DATA_DIR/metadata.json" ]]; then
        _err "Data dir $DATA_DIR has no metadata.json. Cannot skip data generation."
        exit 1
    fi
    STEPS_SKIPPED+=("01_generate_data")
else
    _start_step "Step 1/5 — Generate synthetic data"
    T0=$(date +%s)
    bash "$SCRIPTS_DIR/01_generate_data.sh" 2>&1 | tee -a "$PIPELINE_LOG"
    T1=$(date +%s)
    _ok "Step 1 done in $((T1 - T0))s"
    STEPS_RUN+=("01_generate_data ($((T1 - T0))s)")
fi

# ── Step 2: JEPA pretraining ──────────────────────────────────────────────────
JEPA_CKPT="$CHECKPOINT_DIR/jepa_encoder_final.pt"
if [[ "$SKIP_PRETRAIN" == true ]]; then
    _log "SKIP: Step 2 (--skip-pretrain)"
    if [[ ! -f "$JEPA_CKPT" ]]; then
        _err "JEPA checkpoint not found at $JEPA_CKPT. Cannot skip pretraining."
        exit 1
    fi
    STEPS_SKIPPED+=("02_pretrain_jepa")
else
    _start_step "Step 2/5 — JEPA pretraining (self-supervised, no labels)"
    T0=$(date +%s)
    bash "$SCRIPTS_DIR/02_pretrain_jepa.sh" 2>&1 | tee -a "$PIPELINE_LOG"
    T1=$(date +%s)
    _ok "Step 2 done in $((T1 - T0))s"
    STEPS_RUN+=("02_pretrain_jepa ($((T1 - T0))s)")
fi

# ── Step 3: Supervised fine-tuning ────────────────────────────────────────────
TWIN_CKPT="$CHECKPOINT_DIR/supervised_best.pt"
if [[ "$SKIP_FINETUNE" == true ]]; then
    _log "SKIP: Step 3 (--skip-finetune)"
    if [[ ! -f "$TWIN_CKPT" ]]; then
        _err "Twin checkpoint not found at $TWIN_CKPT. Cannot skip fine-tuning."
        exit 1
    fi
    STEPS_SKIPPED+=("03_finetune")
else
    _start_step "Step 3/5 — Supervised fine-tuning"
    T0=$(date +%s)
    export ENCODER_CKPT="$JEPA_CKPT"
    bash "$SCRIPTS_DIR/03_finetune.sh" 2>&1 | tee -a "$PIPELINE_LOG"
    T1=$(date +%s)
    _ok "Step 3 done in $((T1 - T0))s"
    STEPS_RUN+=("03_finetune ($((T1 - T0))s)")
fi

# ── Step 4: Subject calibration ───────────────────────────────────────────────
CAL_CKPT="$CHECKPOINT_DIR/calibrated.pt"
if [[ "$SKIP_CALIBRATE" == true ]]; then
    _log "SKIP: Step 4 (--skip-calibrate)"
    STEPS_SKIPPED+=("04_calibrate")
    # Fall back to supervised checkpoint if no calibrated one exists
    [[ -f "$CAL_CKPT" ]] || CAL_CKPT="$TWIN_CKPT"
else
    _start_step "Step 4/5 — Subject calibration (personalize to subject $SUBJECT_ID)"
    T0=$(date +%s)
    export TWIN_CKPT="$TWIN_CKPT"
    export FT_EPOCHS="${CAL_FT_EPOCHS:-${FT_EPOCHS:-5}}"
    bash "$SCRIPTS_DIR/04_calibrate.sh" 2>&1 | tee -a "$PIPELINE_LOG"
    T1=$(date +%s)
    _ok "Step 4 done in $((T1 - T0))s"
    STEPS_RUN+=("04_calibrate ($((T1 - T0))s)")
fi

# ── Step 5: Run inference loop ────────────────────────────────────────────────
_start_step "Step 5/5 — Inference loop (online adaptation)"
T0=$(date +%s)
export TWIN_CKPT="$CAL_CKPT"
bash "$SCRIPTS_DIR/05_run_twin.sh" 2>&1 | tee -a "$PIPELINE_LOG"
T1=$(date +%s)
_ok "Step 5 done in $((T1 - T0))s"
STEPS_RUN+=("05_run_twin ($((T1 - T0))s)")

# ── Summary ───────────────────────────────────────────────────────────────────
T_TOTAL=$(($(date +%s) - T_PIPELINE_START))
_log ""
_log "═══════════════════════════════════════════════════════"
_log "  Pipeline complete in ${T_TOTAL}s"
_log "  Steps run:     ${STEPS_RUN[*]:-none}"
_log "  Steps skipped: ${STEPS_SKIPPED[*]:-none}"
_log ""
_log "  Outputs:"
[[ -d "$DATA_DIR" ]]       && _log "    Data:         $DATA_DIR"
[[ -f "$JEPA_CKPT" ]]      && _log "    JEPA encoder: $JEPA_CKPT"
[[ -f "$TWIN_CKPT" ]]      && _log "    Twin (base):  $TWIN_CKPT"
[[ -f "$CAL_CKPT" ]]       && _log "    Twin (calib): $CAL_CKPT"
SESSION_LOG="$LOG_DIR/session_latest.json"
[[ -f "$SESSION_LOG" ]]    && _log "    Session log:  $SESSION_LOG"
_log "    Pipeline log: $PIPELINE_LOG"
_log "═══════════════════════════════════════════════════════"
