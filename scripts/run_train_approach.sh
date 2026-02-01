#!/bin/bash
# ==============================================================================
# TRAINING CONFIGURATION - Modify these variables as needed
# ==============================================================================
# TOTAL_FRAMES      - Total training frames (default: 500000)
# FRAMES_PER_BATCH  - Frames collected per batch (default: 4096)
# LEARNING_RATE     - Learning rate for optimizer (default: 3e-4)
# DEVICE            - Training device: cuda or cpu (default: cuda)
# SEED              - Random seed for reproducibility (default: 42)
# LOG_DIR           - Directory for logs (default: ./logs)
# SAVE_DIR          - Directory for checkpoints (default: ./checkpoints)
# EXPERIMENT_NAME   - Name for this experiment (default: approach_skill)
# TENSORBOARD_PORT  - Port for TensorBoard web UI (default: 6006)
# ==============================================================================

TOTAL_FRAMES=${TOTAL_FRAMES:-500000}
FRAMES_PER_BATCH=${FRAMES_PER_BATCH:-4096}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-42}
LOG_DIR=${LOG_DIR:-./logs}
SAVE_DIR=${SAVE_DIR:-./checkpoints}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-approach_skill}
TENSORBOARD_PORT=${TENSORBOARD_PORT:-6006}

# ==============================================================================
# Color output helpers
# ==============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==============================================================================
# Cleanup function for graceful exit
# ==============================================================================
TENSORBOARD_PID=""

cleanup() {
    if [ -n "$TENSORBOARD_PID" ] && kill -0 "$TENSORBOARD_PID" 2>/dev/null; then
        info "Stopping TensorBoard (PID: $TENSORBOARD_PID)..."
        kill "$TENSORBOARD_PID" 2>/dev/null
        wait "$TENSORBOARD_PID" 2>/dev/null
        success "TensorBoard stopped"
    fi
}

trap cleanup EXIT INT TERM

# ==============================================================================
# Main script
# ==============================================================================
echo "=============================================================="
echo "Training Approach Skill"
echo "=============================================================="
info "Total frames:      $TOTAL_FRAMES"
info "Frames per batch:  $FRAMES_PER_BATCH"
info "Learning rate:     $LEARNING_RATE"
info "Device:            $DEVICE"
info "Seed:              $SEED"
info "Log directory:     $LOG_DIR"
info "Save directory:    $SAVE_DIR"
info "Experiment name:   $EXPERIMENT_NAME"
echo "=============================================================="

# Create directories
mkdir -p "$LOG_DIR" "$SAVE_DIR"

# Start TensorBoard in background
info "Starting TensorBoard on port $TENSORBOARD_PORT..."
tensorboard --logdir="$LOG_DIR" --port="$TENSORBOARD_PORT" --bind_all 2>/dev/null &
TENSORBOARD_PID=$!

if kill -0 "$TENSORBOARD_PID" 2>/dev/null; then
    success "TensorBoard started at http://localhost:$TENSORBOARD_PORT"
else
    warn "TensorBoard failed to start (may already be running)"
    TENSORBOARD_PID=""
fi

# Run training
info "Starting training..."
python scripts/train_approach.py \
    --total-frames "$TOTAL_FRAMES" \
    --frames-per-batch "$FRAMES_PER_BATCH" \
    --learning-rate "$LEARNING_RATE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --log-dir "$LOG_DIR" \
    --save-dir "$SAVE_DIR" \
    --experiment-name "$EXPERIMENT_NAME"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    success "Training completed successfully!"
else
    error "Training failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
