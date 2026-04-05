#!/usr/bin/env bash
# Phase 2: PPO Tuning (32 envs, k=5, best reward stack)
# Phase 3: Scale-up (best from Phase 2, long run)
set -euo pipefail
cd "$(dirname "$0")/.."

RESULTS_DIR="output/steepness_20260403_234056"
COMMON="--task follow_target --num-envs 32 --seed 42"
REWARD_STACK="--dist-weight 0.3 --pbrs-gamma 0.99 --smooth-weight 0.02 --curriculum --warmup-episodes 500"

run_experiment() {
    local ID=$1
    local WALL=$2
    shift 2
    local EXTRA_ARGS="$@"

    echo ""
    echo "=============================================="
    echo "  Running $ID  (wall time: $WALL, 32 envs)"
    echo "  Extra: $EXTRA_ARGS"
    echo "=============================================="

    python3 -m choi2025.train_ppo $COMMON $REWARD_STACK \
        --max-wall-time "$WALL" \
        $EXTRA_ARGS \
        2>&1 | tee "$RESULTS_DIR/${ID}.log"

    # Copy metrics from latest output dir
    LATEST=$(ls -td output/fixed_follow_target_ppo_*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        cp "$LATEST/metrics.jsonl" "$RESULTS_DIR/${ID}_metrics.jsonl" 2>/dev/null || true
        cp "$LATEST/config.json" "$RESULTS_DIR/${ID}_config.json" 2>/dev/null || true
        echo "  -> Copied results to $RESULTS_DIR/${ID}_metrics.jsonl"
    fi
}

echo "===== PHASE 2: PPO Tuning (3 runs x 90min, 32 envs) ====="
run_experiment T1 90m --gae-lambda 0.99
run_experiment T2 90m --entropy-coef 0.02
run_experiment T3 90m --gae-lambda 0.99 --entropy-coef 0.02

echo ""
echo "===== Phase 2 Analysis ====="
python3 << 'PYEOF'
import json, os, numpy as np

results_dir = "output/steepness_20260403_234056"
ids = ["T1", "T2", "T3"]
labels = {"T1": "GAE=0.99", "T2": "ent=0.02", "T3": "both"}

best_id, best_dist = None, 999
print(f"{'ID':>4} {'Config':>12} {'Dist(Q4)':>10} {'DDist':>8} {'EV':>6} {'GradN':>7} {'StdMin':>7}")
print("-" * 65)

for rid in ids:
    path = os.path.join(results_dir, f"{rid}_metrics.jsonl")
    if not os.path.exists(path):
        print(f"{rid:>4} {labels[rid]:>12}   -- not found --")
        continue
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    n = len(lines)
    q4 = lines[3*n//4:]
    q1 = lines[:n//4]

    d_q4 = np.mean([l["mean_dist_to_goal"] for l in q4])
    d_q1 = np.mean([l["mean_dist_to_goal"] for l in q1])
    ev = np.mean([l["explained_variance"] for l in q4])
    gn = np.mean([l["grad_norm"] for l in q4])
    sm = np.mean([l.get("diagnostics/action_std_min", 0) for l in q4])

    print(f"{rid:>4} {labels[rid]:>12} {d_q4:>10.3f} {d_q4-d_q1:>+8.3f} {ev:>6.2f} {gn:>7.1f} {sm:>7.3f}")
    if d_q4 < best_dist:
        best_dist = d_q4
        best_id = rid

# Compare to C5 baseline
print(f"\nC5 reference (32 envs, 3.6M frames): d=0.543")
for rid in ids:
    path = os.path.join(results_dir, f"{rid}_metrics.jsonl")
    if not os.path.exists(path):
        continue
    with open(path) as f:
        lines = [json.loads(l) for l in f]
    q4 = lines[3*len(lines)//4:]
    d = np.mean([l["mean_dist_to_goal"] for l in q4])
    imp = (0.543 - d) / 0.543 * 100
    print(f"  {rid}: d={d:.3f} ({imp:+.1f}% vs C5)")

if best_id:
    print(f"\nBest Phase 2: {best_id} ({labels[best_id]}), d={best_dist:.3f}")
    # Determine Phase 3 config
    best_cfg = labels[best_id]
    with open(os.path.join(results_dir, "BEST_PHASE2"), "w") as f:
        f.write(best_id)
    print(f"Wrote BEST_PHASE2={best_id} to {results_dir}/BEST_PHASE2")
PYEOF

echo ""
echo "===== PHASE 3: Scale-up (1 run x 270min, 32 envs) ====="

# Determine best Phase 2 config
BEST=$(cat "$RESULTS_DIR/BEST_PHASE2" 2>/dev/null || echo "T3")
echo "Using config from $BEST"

case "$BEST" in
    T1) EXTRA="--gae-lambda 0.99" ;;
    T2) EXTRA="--entropy-coef 0.02" ;;
    T3) EXTRA="--gae-lambda 0.99 --entropy-coef 0.02" ;;
    *)  EXTRA="" ;;
esac

run_experiment L1 270m $EXTRA

echo ""
echo "=============================================="
echo "  All phases complete!"
echo "  Results: $RESULTS_DIR"
echo "=============================================="
