#!/usr/bin/env bash
# Shared environment setup — sourced by every v2 script.
# Sets REPO_ROOT, VENV_PYTHON, and PYTHONPATH.  Does not run anything.

# Locate the repo root regardless of where the script is called from.
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export V2_DIR="$(cd "$SCRIPTS_DIR/.." && pwd)"
export REPO_ROOT="$(cd "$V2_DIR/.." && pwd)"

# ── Virtual environment ───────────────────────────────────────────────────────
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "[env.sh] ERROR: venv not found at $VENV_PYTHON"
    echo "         Create it with:  python3 -m venv $REPO_ROOT/.venv"
    echo "         Then install:    $REPO_ROOT/.venv/bin/pip install -r $V2_DIR/requirements.txt"
    exit 1
fi
export VENV_PYTHON

# ── Python path ───────────────────────────────────────────────────────────────
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# ── Default output paths (override via environment before calling script) ─────
export DATA_DIR="${DATA_DIR:-$V2_DIR/data/generated}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-$V2_DIR/checkpoints}"
export LOG_DIR="${LOG_DIR:-$V2_DIR/logs}"
export DEVICE="${DEVICE:-cpu}"

# ── Helpers ───────────────────────────────────────────────────────────────────
_log() { echo "[$(date +'%H:%M:%S')] $*"; }
_ok()  { echo "[$(date +'%H:%M:%S')] ✓ $*"; }
_err() { echo "[$(date +'%H:%M:%S')] ✗ $*" >&2; }

mkdir -p "$LOG_DIR" "$CHECKPOINT_DIR"
