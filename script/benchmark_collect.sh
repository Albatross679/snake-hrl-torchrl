#!/bin/bash
# Benchmark data collection throughput with different worker counts.
# Each worker runs 1 env. Collects 5000 transitions per config.

set -e

TRANSITIONS=5000
BASE_DIR="data/benchmark_collect"
rm -rf "$BASE_DIR"

echo "=== Data Collection Benchmark ==="
echo "Target: $TRANSITIONS transitions per config"
echo "CPUs: $(nproc)"
echo ""

worker_counts=(1 2 4 8 12 16 24 32 48)

printf "%-10s | %s\n" "Workers" "FPS"
echo "-----------|--------"

for w in "${worker_counts[@]}"; do
    save_dir="${BASE_DIR}/w${w}"

    output=$(python3 -m aprx_model_elastica.collect_data \
        --num-transitions $TRANSITIONS \
        --num-workers $w \
        --no-collect-forces \
        --no-sobol \
        --perturbation-fraction 0 \
        --save-dir "$save_dir" \
        --seed 42 \
        2>&1)

    fps=$(echo "$output" | grep -oP '\d+ FPS' | tail -1 | grep -oP '\d+')
    printf "%-10s | %s\n" "$w" "${fps:-ERROR}"

    rm -rf "$save_dir"
done

echo ""
echo "Benchmark complete!"
rm -rf "$BASE_DIR"
