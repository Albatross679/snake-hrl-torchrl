#!/usr/bin/env bash
# Steepness & PPO Tuning Ablation — 14-hour experiment
# Phase 1: Steepness sweep (S1-S4), ~6h
# Phase 2: PPO tuning (T1-T3), ~4.5h  [uses best k from Phase 1]
# Phase 3: Scale-up (L1), ~3.5h        [uses best overall config]
#
# Usage: bash script/run_steepness_ablation.sh [phase]
#   phase=1  Run Phase 1 only (steepness sweep)
#   phase=2  Run Phase 2 only (requires BEST_K env var)
#   phase=3  Run Phase 3 only (requires BEST_K, BEST_GAE, BEST_ENT env vars)
#   phase=all (default) Run all phases sequentially with auto-analysis

set -euo pipefail
cd "$(dirname "$0")/.."

PHASE="${1:-all}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="output/steepness_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Common args for all runs
COMMON="--task follow_target --num-envs 100 --seed 42"
# Best reward stack from previous ablation (C5)
REWARD_STACK="--dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0.02 --curriculum --warmup-episodes 500"

run_experiment() {
    local ID=$1
    local WALL=$2
    shift 2
    local EXTRA_ARGS="$@"

    echo ""
    echo "=============================================="
    echo "  Running $ID  (wall time: $WALL)"
    echo "  Args: $COMMON $REWARD_STACK $EXTRA_ARGS"
    echo "=============================================="
    echo ""

    python3 -m choi2025.train_ppo $COMMON $REWARD_STACK \
        --max-wall-time "$WALL" \
        $EXTRA_ARGS \
        2>&1 | tee "$RESULTS_DIR/${ID}.log"

    # Copy metrics and config from latest output dir
    LATEST=$(ls -td output/fixed_follow_target_ppo_* 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        cp "$LATEST/metrics.jsonl" "$RESULTS_DIR/${ID}_metrics.jsonl" 2>/dev/null || true
        cp "$LATEST/config.json" "$RESULTS_DIR/${ID}_config.json" 2>/dev/null || true
        echo "  → Copied results to $RESULTS_DIR/${ID}_metrics.jsonl"
    else
        echo "  WARNING: Could not find output directory for $ID"
    fi
}

analyze_phase1() {
    echo ""
    echo "=============================================="
    echo "  Phase 1 Analysis"
    echo "=============================================="
    python3 -c "
import json, sys, os
results_dir = '$RESULTS_DIR'
ids = ['S1', 'S2', 'S3', 'S4']
steepness = {'S1': 1, 'S2': 2, 'S3': 3, 'S4': 5}

print(f'{'ID':>4} {'k':>3} {'Dist(Q4)':>10} {'ΔDist':>8} {'SNR':>6} {'EV':>6} {'Reward':>8}')
print('-' * 55)

best_id, best_dist = None, 999
for rid in ids:
    path = os.path.join(results_dir, f'{rid}_metrics.jsonl')
    if not os.path.exists(path):
        print(f'{rid:>4} {steepness[rid]:>3}   -- file not found --')
        continue
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    n = len(lines)
    q4 = lines[3*n//4:]

    dists = [l.get('dist_to_goal', l.get('mean_dist_to_goal', 0)) for l in q4 if 'dist_to_goal' in l or 'mean_dist_to_goal' in l]
    rewards = [l.get('mean_episode_reward', l.get('episode_reward', 0)) for l in q4]
    evs = [l.get('explained_variance', 0) for l in q4]

    q1 = lines[:n//4]
    dists_q1 = [l.get('dist_to_goal', l.get('mean_dist_to_goal', 0)) for l in q1 if 'dist_to_goal' in l or 'mean_dist_to_goal' in l]

    import numpy as np
    d_q4 = np.mean(dists) if dists else 0
    d_q1 = np.mean(dists_q1) if dists_q1 else 0
    delta = d_q4 - d_q1
    r_mean = np.mean(rewards) if rewards else 0
    ev_mean = np.mean(evs) if evs else 0

    # SNR
    all_rewards_step = [l.get('reward_mean', l.get('mean_reward', 0)) for l in lines]
    all_rewards_std = [l.get('reward_std', l.get('std_reward', 1)) for l in lines]
    snr_vals = [abs(m)/max(s, 1e-8) for m, s in zip(all_rewards_step, all_rewards_std) if s > 0]
    snr = np.mean(snr_vals) if snr_vals else 0

    print(f'{rid:>4} {steepness[rid]:>3} {d_q4:>10.3f} {delta:>+8.3f} {snr:>6.2f} {ev_mean:>6.2f} {r_mean:>8.1f}')

    if d_q4 < best_dist and d_q4 > 0:
        best_dist = d_q4
        best_id = rid

if best_id:
    print(f'\nBest: {best_id} (k={steepness[best_id]}, dist={best_dist:.3f})')
    # Write best k to file for Phase 2
    with open(os.path.join(results_dir, 'BEST_K'), 'w') as f:
        f.write(str(steepness[best_id]))
    print(f'Wrote BEST_K={steepness[best_id]} to {results_dir}/BEST_K')
"
}

# =============================================
# PHASE 1: Steepness Sweep
# =============================================
if [ "$PHASE" = "1" ] || [ "$PHASE" = "all" ]; then
    echo "===== PHASE 1: Steepness Sweep (4 runs × 90min) ====="
    run_experiment S1 90m --reward-steepness 1.0
    run_experiment S2 90m --reward-steepness 2.0
    run_experiment S3 90m --reward-steepness 3.0
    run_experiment S4 90m --reward-steepness 5.0
    analyze_phase1
fi

# =============================================
# PHASE 2: PPO Tuning (uses best k from Phase 1)
# =============================================
if [ "$PHASE" = "2" ] || [ "$PHASE" = "all" ]; then
    # Read best k from Phase 1 analysis or env var
    if [ -n "${BEST_K:-}" ]; then
        K="$BEST_K"
    elif [ -f "$RESULTS_DIR/BEST_K" ]; then
        K=$(cat "$RESULTS_DIR/BEST_K")
    else
        echo "ERROR: No BEST_K found. Run Phase 1 first or set BEST_K env var."
        exit 1
    fi
    echo ""
    echo "===== PHASE 2: PPO Tuning (k=$K, 3 runs × 90min) ====="
    run_experiment T1 90m --reward-steepness "$K" --gae-lambda 0.99
    run_experiment T2 90m --reward-steepness "$K" --entropy-coef 0.02
    run_experiment T3 90m --reward-steepness "$K" --gae-lambda 0.99 --entropy-coef 0.02
fi

# =============================================
# PHASE 3: Scale-up (uses best overall config)
# =============================================
if [ "$PHASE" = "3" ] || [ "$PHASE" = "all" ]; then
    # Read best config — default to best k + combined tuning if no override
    K="${BEST_K:-$(cat "$RESULTS_DIR/BEST_K" 2>/dev/null || echo 5)}"
    GAE="${BEST_GAE:-0.99}"
    ENT="${BEST_ENT:-0.02}"
    echo ""
    echo "===== PHASE 3: Scale-Up (k=$K, λ=$GAE, ent=$ENT, 1 run × 210min) ====="
    run_experiment L1 210m --reward-steepness "$K" --gae-lambda "$GAE" --entropy-coef "$ENT"
fi

echo ""
echo "=============================================="
echo "  All experiments complete!"
echo "  Results: $RESULTS_DIR"
echo "=============================================="
