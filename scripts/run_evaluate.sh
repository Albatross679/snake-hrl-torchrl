#!/bin/bash
# ==============================================================================
# EVALUATION CONFIGURATION - Modify these variables as needed
# ==============================================================================
# CHECKPOINT        - Path to checkpoint file (required)
# POLICY_TYPE       - Policy type: approach, coil, hrl (default: hrl)
# NUM_EPISODES      - Number of evaluation episodes (default: 20)
# DETERMINISTIC     - Use deterministic actions: true/false (default: true)
# SAVE_VIDEOS       - Save video recordings: true/false (default: false)
# VIDEO_DIR         - Directory for video output (default: ./videos)
# OUTPUT_FILE       - Path for results JSON (default: none, prints to stdout)
# DEVICE            - Evaluation device: cuda or cpu (default: cuda)
# ==============================================================================

CHECKPOINT=${CHECKPOINT:-}
POLICY_TYPE=${POLICY_TYPE:-hrl}
NUM_EPISODES=${NUM_EPISODES:-20}
DETERMINISTIC=${DETERMINISTIC:-true}
SAVE_VIDEOS=${SAVE_VIDEOS:-false}
VIDEO_DIR=${VIDEO_DIR:-./videos}
OUTPUT_FILE=${OUTPUT_FILE:-}
DEVICE=${DEVICE:-cuda}

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
# Validation
# ==============================================================================
if [ -z "$CHECKPOINT" ]; then
    error "CHECKPOINT is required. Set it with: CHECKPOINT=/path/to/checkpoint.pt $0"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    error "Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# ==============================================================================
# Main script
# ==============================================================================
echo "=============================================================="
echo "Evaluating Policy"
echo "=============================================================="
info "Checkpoint:        $CHECKPOINT"
info "Policy type:       $POLICY_TYPE"
info "Num episodes:      $NUM_EPISODES"
info "Deterministic:     $DETERMINISTIC"
info "Save videos:       $SAVE_VIDEOS"
if [ "$SAVE_VIDEOS" = "true" ]; then
    info "Video directory:   $VIDEO_DIR"
fi
if [ -n "$OUTPUT_FILE" ]; then
    info "Output file:       $OUTPUT_FILE"
fi
info "Device:            $DEVICE"
echo "=============================================================="

# Create directories if needed
if [ "$SAVE_VIDEOS" = "true" ]; then
    mkdir -p "$VIDEO_DIR"
fi

# Build command
CMD="python scripts/evaluate.py \
    --checkpoint $CHECKPOINT \
    --policy-type $POLICY_TYPE \
    --num-episodes $NUM_EPISODES \
    --device $DEVICE"

if [ "$DETERMINISTIC" = "true" ]; then
    CMD="$CMD --deterministic"
fi

if [ "$SAVE_VIDEOS" = "true" ]; then
    CMD="$CMD --save-videos --video-dir $VIDEO_DIR"
fi

if [ -n "$OUTPUT_FILE" ]; then
    CMD="$CMD --output-file $OUTPUT_FILE"
fi

# Run evaluation
info "Starting evaluation..."
eval $CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    success "Evaluation completed successfully!"
else
    error "Evaluation failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
