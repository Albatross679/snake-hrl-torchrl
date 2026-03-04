"""Training script for DQN gait learning (Shi et al., 2020).

Trains a Q-network to learn locomotion gaits for wheeled or swimming
3-link snake robots using Deep Q-Learning with experience replay.

Usage:
    python -m shi2020.train_shi2020 --robot wheeled --task forward
    python -m shi2020.train_shi2020 --robot swimming --task rotate_left --episodes 5000
"""

import argparse
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from shi2020.configs_shi2020 import (
    ActionConfig,
    GaitTask,
    KinematicRobotConfig,
    RobotType,
    Shi2020Config,
    Shi2020EnvConfig,
)
from shi2020.env_shi2020 import WheeledSnakeEnv, SwimmingSnakeEnv
from src.configs.base import resolve_device
from src.configs.run_dir import setup_run_dir
from src.configs.console import ConsoleLogger


class QNetwork(nn.Module):
    """Q-network: state+action → Q-value.

    Architecture from Section 4.3: Input(5) → 50 ReLU → 10 ReLU → 1 linear.
    """

    def __init__(self, input_dim: int = 5, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [50, 10]

        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_action: torch.Tensor) -> torch.Tensor:
        return self.net(state_action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN gait learning")
    parser.add_argument(
        "--robot",
        type=str,
        default="wheeled",
        choices=[r.value for r in RobotType],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="forward",
        choices=[t.value for t in GaitTask],
    )
    parser.add_argument("--episodes", type=int, default=5000, help="Training episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Config (construct env first so __post_init__ sees robot/task)
    robot_config = KinematicRobotConfig(robot_type=RobotType(args.robot))
    if args.robot == "swimming":
        robot_config.avoid_singularity = False

    action_config = ActionConfig(t_interval=4.0 if args.robot == "swimming" else 2.0)
    env_config = Shi2020EnvConfig(
        robot=robot_config, action=action_config, task=GaitTask(args.task),
    )
    config = Shi2020Config(seed=args.seed, env=env_config)
    config.dqn.num_episodes = args.episodes

    dqn = config.dqn

    # Setup consolidated run directory
    run_dir = setup_run_dir(config)
    save_dir = run_dir / "checkpoints"

    # Environment
    if args.robot == "wheeled":
        env = WheeledSnakeEnv(config.env, device=device)
    else:
        env = SwimmingSnakeEnv(config.env, device=device)

    # Action table
    from shi2020.env_shi2020 import _build_action_table
    action_table = _build_action_table(
        config.env.action.a_max, config.env.action.a_interval
    )
    num_actions = len(action_table)

    with ConsoleLogger(run_dir, None):
        # Networks
        Q = QNetwork(config.network.input_dim, config.network.hidden_dims).to(device)
        Q_target = QNetwork(config.network.input_dim, config.network.hidden_dims).to(device)
        Q_target.load_state_dict(Q.state_dict())

        optimizer = optim.RMSprop(Q.parameters(), lr=dqn.learning_rate)
        memory = deque(maxlen=dqn.memory_size)

        epsilon = dqn.epsilon_init
        total_steps = 0
        best_reward = float("-inf")

        print(f"DQN Training: {args.robot} robot, {args.task} task")
        print(f"  Run directory: {run_dir}")
        print(f"  Actions: {num_actions}, Episodes: {dqn.num_episodes}")

        for episode in range(dqn.num_episodes):
            td = env.reset()
            state = td["observation"].cpu().numpy()
            episode_reward = 0.0

            for t in range(dqn.iterations_per_episode):
                # ε-greedy action selection
                if random.random() < epsilon:
                    action_idx = random.randint(0, num_actions - 1)
                else:
                    # Evaluate Q for all actions
                    state_t = torch.tensor(state, dtype=torch.float32, device=device)
                    actions_t = torch.tensor(
                        action_table, dtype=torch.float32, device=device
                    )
                    sa = torch.cat(
                        [state_t.unsqueeze(0).expand(num_actions, -1), actions_t], dim=1
                    )
                    with torch.no_grad():
                        q_vals = Q(sa).squeeze(-1)
                    action_idx = q_vals.argmax().item()

                # Step environment
                td_action = torch.tensor(action_idx, dtype=torch.int64, device=device)
                from tensordict import TensorDict
                td_in = TensorDict(
                    {"action": td_action}, batch_size=env.batch_size, device=device
                )
                td_next = env.step(td_in)

                next_state = td_next["next", "observation"].cpu().numpy()
                reward = td_next["next", "reward"].item()
                done = td_next["next", "done"].item()

                # Store transition
                memory.append((state, action_idx, reward, next_state, done))
                episode_reward += reward

                # Learn from minibatch
                if len(memory) >= dqn.replay_start:
                    batch = random.sample(list(memory), dqn.minibatch_size)
                    states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*batch)

                    states_t = torch.tensor(
                        np.array(states_b), dtype=torch.float32, device=device
                    )
                    actions_t = torch.tensor(
                        action_table[list(actions_b)], dtype=torch.float32, device=device
                    )
                    rewards_t = torch.tensor(
                        rewards_b, dtype=torch.float32, device=device
                    )
                    next_states_t = torch.tensor(
                        np.array(next_states_b), dtype=torch.float32, device=device
                    )
                    dones_t = torch.tensor(
                        dones_b, dtype=torch.float32, device=device
                    )

                    # Current Q
                    sa_batch = torch.cat([states_t, actions_t], dim=1)
                    q_current = Q(sa_batch).squeeze(-1)

                    # Target Q: max_a' Q_target(s', a')
                    with torch.no_grad():
                        all_actions_t = torch.tensor(
                            action_table, dtype=torch.float32, device=device
                        )
                        max_q_next = torch.zeros(dqn.minibatch_size, device=device)
                        for i in range(dqn.minibatch_size):
                            ns = next_states_t[i].unsqueeze(0).expand(num_actions, -1)
                            sa_next = torch.cat([ns, all_actions_t], dim=1)
                            q_next = Q_target(sa_next).squeeze(-1)
                            max_q_next[i] = q_next.max()

                        target = rewards_t + dqn.gamma * max_q_next * (1 - dones_t)

                    loss = nn.functional.mse_loss(q_current, target)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Update target network
                total_steps += 1
                if total_steps % dqn.target_update_freq == 0:
                    Q_target.load_state_dict(Q.state_dict())

                # Decay epsilon
                epsilon = max(dqn.epsilon_min, epsilon * dqn.epsilon_decay)

                state = next_state
                if done:
                    break

            # Save best checkpoint
            if episode_reward > best_reward:
                best_reward = episode_reward
                torch.save(Q.state_dict(), save_dir / "best.pt")

            if (episode + 1) % 100 == 0:
                print(
                    f"  Episode {episode + 1}/{dqn.num_episodes}: "
                    f"reward={episode_reward:.2f}, ε={epsilon:.4f}"
                )

        # Save final checkpoint
        torch.save(Q.state_dict(), save_dir / "final.pt")

        env.close()
        print("Training complete.")


if __name__ == "__main__":
    main()
