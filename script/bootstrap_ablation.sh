#!/bin/bash
# Bootstrap script for reward ablation on a fresh vast.ai instance.
# Usage: curl -sL https://raw.githubusercontent.com/Albatross679/snake-hrl-torchrl/reward-ablation/script/bootstrap_ablation.sh | bash
set -e

echo "=== Bootstrapping ablation study ==="
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

cd /workspace

# Clone or update repo
if [ -d "snake-hrl-torchrl" ]; then
    echo "Repo exists, updating..."
    cd snake-hrl-torchrl
    git fetch origin
    git checkout reward-ablation
    git reset --hard origin/reward-ablation
else
    echo "Cloning repo..."
    git clone --branch reward-ablation https://github.com/Albatross679/snake-hrl-torchrl.git
    cd snake-hrl-torchrl
fi

# Install dependencies
echo "Installing dependencies..."
pip install torch torchrl tensordict wandb numpy scipy 2>&1 | tail -3

# Set wandb key if not set
if [ -z "$WANDB_API_KEY" ]; then
    echo "WARNING: WANDB_API_KEY not set. Runs won't sync to W&B."
    echo "Set it with: export WANDB_API_KEY=your_key"
fi

# Run ablation
echo "=== Starting ablation study ==="
echo "Monitor with: tail -f /workspace/snake-hrl-torchrl/ablation.log"
nohup bash script/run_reward_ablation.sh > ablation.log 2>&1 &
echo "Ablation launched (PID: $!)"
echo "Expected runtime: ~5.5 hours"
