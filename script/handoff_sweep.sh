#!/bin/bash
# Wait for M1 to finish, kill the old sweep, launch W1+W2 only
set -e
cd /home/coder/snake-hrl-torchrl

SWEEP_PID=327823  # current sweep process

echo "Waiting for M1 training to finish (watching process $SWEEP_PID)..."
echo "Current M1 subprocess:"
ps aux | grep train_surrogate | grep -v grep | grep "M1" || true

# Wait for M1's train_surrogate subprocess to finish
while ps aux | grep -v grep | grep "train_surrogate.*--run-name M1" > /dev/null 2>&1; do
    sleep 30
done

echo "M1 finished at $(date)"

# Kill the old sweep process (it would try to start M2 next)
if kill -0 $SWEEP_PID 2>/dev/null; then
    echo "Killing old sweep process $SWEEP_PID..."
    kill $SWEEP_PID 2>/dev/null || true
    sleep 2
fi

echo "Launching new sweep with W1 and W2 only..."
python3 -m aprx_model_elastica.sweep \
  --data-dir data/surrogate_rl_step \
  --device cuda \
  --epochs 200 \
  --output-base output/surrogate

echo "Sweep complete (W1 + W2). Results: output/surrogate/sweep_summary.json"
