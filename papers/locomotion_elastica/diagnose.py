"""Diagnostic script: run one episode with fixed actions to understand snake behavior."""
import numpy as np
import torch
from tensordict import TensorDict
from locomotion_elastica.config import LocomotionElasticaEnvConfig, GaitType
from locomotion_elastica.env import LocomotionElasticaEnv


def run_episode(action, label, steps=100):
    """Run a short episode with a fixed action and report metrics."""
    config = LocomotionElasticaEnvConfig(gait=GaitType.FORWARD, device="cpu")
    config.randomize_initial_heading = False
    env = LocomotionElasticaEnv(config, device="cpu")

    td = env._reset()

    rewards = []
    v_gs = []
    dists = []

    initial_com = env._get_com(env._get_positions())
    initial_dist = env._get_dist_to_goal(initial_com)

    for step in range(steps):
        td_action = TensorDict({"action": torch.tensor(action, dtype=torch.float32)}, batch_size=[])
        td_next = env._step(td_action)

        r = td_next["reward"].item()
        vg = td_next["v_g"].item()
        dist = td_next["dist_to_goal"].item()
        rewards.append(r)
        v_gs.append(vg)
        dists.append(dist)

    final_com = env._get_com(env._get_positions())
    displacement = np.linalg.norm(final_com - initial_com)
    direction = final_com - initial_com

    # Denormalize action for display
    params = env._serpenoid.denormalize_action(np.array(action))

    print(f"\n=== {label} ===")
    print(f"  Action:       {action}")
    print(f"  Params:       amp={params['amplitude']:.3f}, freq={params['frequency']:.2f}, "
          f"wn={params['wave_number']:.2f}, turn={params['turn_bias']:.2f}")
    print(f"  Initial CoM:  ({initial_com[0]:.4f}, {initial_com[1]:.4f})")
    print(f"  Final CoM:    ({final_com[0]:.4f}, {final_com[1]:.4f})")
    print(f"  Displacement: {displacement:.4f} m")
    print(f"  Direction:    ({direction[0]:.4f}, {direction[1]:.4f})")
    print(f"  Dist to goal: {initial_dist:.4f} -> {dists[-1]:.4f} m")
    print(f"  Mean v_g:     {np.mean(v_gs):.6f} m/s")
    print(f"  Mean reward:  {np.mean(rewards):.4f}")
    print(f"  Total reward: {np.sum(rewards):.4f}")
    print(f"  Speed:        {displacement / (steps * 0.5):.6f} m/s")

    return rewards, v_gs, dists


if __name__ == "__main__":
    print("=" * 60)
    print("AMPLITUDE SWEEP (freq=mid, wave_num=mid, no turn)")
    print("=" * 60)

    # Amplitude sweep: test different curvature amplitudes
    # With new range (0, 5.0): action[0]=-1→0, 0→2.5, 1→5.0
    for amp_action in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        run_episode([amp_action, 0.0, 0.0, 0.0, 0.0], f"Amp action={amp_action}")

    print("\n" + "=" * 60)
    print("WAVE NUMBER SWEEP (amp=mid, freq=mid, no turn)")
    print("=" * 60)

    # Wave number sweep: test different spatial frequencies
    for wn_action in [-0.67, -0.33, 0.0, 0.33, 0.67]:
        run_episode([0.0, 0.0, wn_action, 0.0, 0.0], f"WaveNum action={wn_action}")

    print("\n" + "=" * 60)
    print("FREQUENCY SWEEP (amp=mid, wave_num=mid, no turn)")
    print("=" * 60)

    for freq_action in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        run_episode([0.0, freq_action, 0.0, 0.0, 0.0], f"Freq action={freq_action}")
