#!/usr/bin/env bash
# Reward Design Ablation Study — launch all experiments sequentially.
# Usage: bash script/run_reward_ablation.sh [--parallel N]
#
# Each run: 32 envs, 30 min wall time, seed 42.
# Total sequential time: ~5.5 hours (11 training runs × 30 min).
#
# With --parallel 2, runs 2 experiments at a time (~2.75 hours on multi-GPU).
# GPU assignment: experiments round-robin across CUDA devices.

set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONPATH="$(pwd):$(pwd)/papers"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

COMMON="--task follow_target --num-envs 32 --max-wall-time 30m --seed 42"
RESULTS_DIR="output/ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=== Reward Design Ablation Study ==="
echo "Results dir: $RESULTS_DIR"
echo "Started: $(date)"
echo ""

# Parse args
PARALLEL=1
while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_experiment() {
    local ID="$1"
    local DESC="$2"
    shift 2
    local ARGS="$@"

    echo "[$(date +%H:%M:%S)] Starting $ID: $DESC"
    echo "  Args: $ARGS"

    local LOGFILE="$RESULTS_DIR/${ID}.log"
    python3 -m choi2025.train_ppo $COMMON $ARGS > "$LOGFILE" 2>&1
    local EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] Completed $ID (exit 0)"
    else
        echo "[$(date +%H:%M:%S)] FAILED $ID (exit $EXIT_CODE)"
    fi

    # Copy metrics to results dir for easy comparison
    local RUN_DIR
    RUN_DIR=$(ls -dt output/fixed_follow_target_ppo_lr1e4_32envs_* 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/metrics.jsonl" ]; then
        cp "$RUN_DIR/metrics.jsonl" "$RESULTS_DIR/${ID}_metrics.jsonl"
        cp "$RUN_DIR/config.json" "$RESULTS_DIR/${ID}_config.json"
    fi
}

# ──────────────────────────────────────────────────────────────────
# Group A: Baselines
# ──────────────────────────────────────────────────────────────────

echo "=== Group A: Baselines ==="

# A0: Random baseline — handled by analysis script (no training needed)

run_experiment "A1" "Vanilla dense (paper default)" \
    --dist-weight 1.0 --pbrs-gamma 0 --smooth-weight 0 --heading-weight 0

# ──────────────────────────────────────────────────────────────────
# Group B: Reward Component Ablation (no curriculum)
# ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Group B: Reward Component Ablation ==="

run_experiment "B1" "PBRS only" \
    --dist-weight 0 --pbrs-gamma 0.99 --smooth-weight 0 --heading-weight 0

run_experiment "B2" "Dense(1.0) + PBRS" \
    --dist-weight 1.0 --pbrs-gamma 0.99 --smooth-weight 0 --heading-weight 0

run_experiment "B3" "Dense(0.3) + PBRS" \
    --dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0 --heading-weight 0

run_experiment "B4" "Dense(0.3) + PBRS + smooth" \
    --dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0.02 --heading-weight 0

run_experiment "B5" "Dense(0.3) + PBRS + heading" \
    --dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0 --heading-weight 0.3

run_experiment "B6" "Full stack, no curriculum" \
    --dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0.02 --heading-weight 0.3

# ──────────────────────────────────────────────────────────────────
# Group C: Curriculum Ablation
# ──────────────────────────────────────────────────────────────────

echo ""
echo "=== Group C: Curriculum ==="

run_experiment "C1" "Vanilla + curriculum" \
    --dist-weight 1.0 --pbrs-gamma 0 --smooth-weight 0 --heading-weight 0 \
    --curriculum --warmup-episodes 200

run_experiment "C2" "Dense(1.0) + PBRS + curriculum" \
    --dist-weight 1.0 --pbrs-gamma 0.99 --smooth-weight 0 --heading-weight 0 \
    --curriculum --warmup-episodes 200

run_experiment "C3" "Dense(0.3) + PBRS + smooth + curriculum" \
    --dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0.02 --heading-weight 0 \
    --curriculum --warmup-episodes 200

run_experiment "C4" "Full stack + curriculum" \
    --dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0.02 --heading-weight 0.3 \
    --curriculum --warmup-episodes 200

run_experiment "C5" "Dense(0.3) + PBRS + smooth + curriculum (long warmup)" \
    --dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0.02 --heading-weight 0 \
    --curriculum --warmup-episodes 500

echo ""
echo "=== All experiments complete ==="
echo "Results in: $RESULTS_DIR"
echo "Run analysis: python3 script/analyze_reward_ablation.py $RESULTS_DIR"
echo "Finished: $(date)"
