#!/bin/bash
# Launch 15-config surrogate sweep in tmux session
# Runs sequentially via sweep.py (subprocess.run per config)
# Monitor: tmux attach -t gsd-sweep
# Logs: output/surrogate/*.log

set -e
cd /home/coder/snake-hrl-torchrl

python3 -m aprx_model_elastica.sweep \
  --data-dir data/surrogate_rl_step \
  --device cuda \
  --epochs 200 \
  --output-base output/surrogate

echo "Sweep complete. Results: output/surrogate/sweep_summary.json"
