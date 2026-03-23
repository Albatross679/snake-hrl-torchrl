#!/bin/bash
# Wait for SAC training to finish, then launch PPO
echo "$(date): Waiting for SAC training (PID check) to finish..."

while ps aux | grep -E "python.*train.py.*follow_target" | grep -v grep > /dev/null 2>&1; do
    sleep 60
done

echo "$(date): SAC training finished. Launching PPO (50M frames, 100 envs)..."
cd /home/user/snake-hrl-torchrl
nohup python3 papers/choi2025/train_ppo.py --task follow_target --num-envs 100 --seed 42 --total-frames 50000000 > training_ppo_follow.log 2>&1 &
echo "$(date): PPO launched with PID $!"
