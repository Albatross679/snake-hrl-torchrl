"""Motion planning feasibility study for snake locomotion.

Tests sampling-based (RRT) and optimization-based planning approaches
using the DisMech snake simulator as a forward model.

Key questions:
1. Can we characterize discrete motion primitives (forward, turn_left, turn_right)?
2. Can RRT plan paths in reduced (x, y, theta) space using these primitives?
3. What are the limitations for trajectory optimization?

Usage:
    python -m locomotion.test_motion_planning
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from locomotion.config import LocomotionEnvConfig, LocomotionPhysicsConfig
from locomotion.env import LocomotionEnv


# ============================================================================
# Part 1: Motion Primitive Characterization
# ============================================================================

@dataclass
class MotionPrimitive:
    """Result of executing a fixed action for N steps."""
    name: str
    action: np.ndarray
    delta_x: float  # displacement in body-frame forward direction
    delta_y: float  # displacement in body-frame lateral direction
    delta_theta: float  # heading change (radians)
    duration: float  # wall-clock seconds for simulation
    num_steps: int
    trajectory: List[Tuple[float, float, float]] = field(default_factory=list)


def characterize_primitives(
    num_steps: int = 20,
    max_episode_steps: int = 200,
) -> Dict[str, MotionPrimitive]:
    """Run fixed actions and measure resulting motions.

    Tests a library of constant serpenoid actions to build a motion primitive set.
    Each primitive is characterized by (dx, dy, dtheta) in body frame.
    """
    # Define candidate actions: [amplitude, frequency, wave_number, phase, turn_bias]
    # All in normalized [-1, 1] range
    primitives_to_test = {
        # Forward: high amplitude, mid frequency, zero turn bias
        "forward": np.array([0.5, 0.0, 0.0, 0.0, 0.0]),
        # Forward fast: higher amplitude
        "forward_fast": np.array([1.0, 0.5, 0.0, 0.0, 0.0]),
        # Turn left: forward + positive turn bias
        "turn_left": np.array([0.5, 0.0, 0.0, 0.0, 0.5]),
        # Turn left sharp
        "turn_left_sharp": np.array([0.5, 0.0, 0.0, 0.0, 1.0]),
        # Turn right: forward + negative turn bias
        "turn_right": np.array([0.5, 0.0, 0.0, 0.0, -0.5]),
        # Turn right sharp
        "turn_right_sharp": np.array([0.5, 0.0, 0.0, 0.0, -1.0]),
        # Slow crawl
        "slow": np.array([0.2, -0.5, 0.0, 0.0, 0.0]),
        # Stop (zero amplitude)
        "stop": np.array([-1.0, 0.0, 0.0, 0.0, 0.0]),
    }

    results = {}
    config = LocomotionEnvConfig(
        max_episode_steps=max_episode_steps,
        randomize_initial_heading=False,
    )
    env = LocomotionEnv(config=config, device="cpu")

    for name, action in primitives_to_test.items():
        print(f"  Testing primitive: {name} (action={action})")
        t0 = time.time()

        td = env.reset()
        positions = env._get_positions()
        start_com = env._get_com(positions)
        start_heading = env._get_heading_angle(positions)

        trajectory = [(start_com[0], start_com[1], start_heading)]

        import torch
        action_tensor = torch.tensor(action, dtype=torch.float32)

        for step in range(num_steps):
            td_action = td.clone()
            td_action["action"] = action_tensor
            td = env._step(td_action)

            positions = env._get_positions()
            com = env._get_com(positions)
            heading_angle = env._get_heading_angle(positions)
            trajectory.append((com[0], com[1], heading_angle))

        # Compute displacement in initial body frame
        end_com = np.array([trajectory[-1][0], trajectory[-1][1]])
        end_heading = trajectory[-1][2]

        displacement = end_com - start_com
        # Rotate displacement into initial body frame
        cos_h, sin_h = math.cos(start_heading), math.sin(start_heading)
        dx_body = cos_h * displacement[0] + sin_h * displacement[1]
        dy_body = -sin_h * displacement[0] + cos_h * displacement[1]
        dtheta = end_heading - start_heading
        dtheta = (dtheta + math.pi) % (2 * math.pi) - math.pi

        duration = time.time() - t0

        results[name] = MotionPrimitive(
            name=name,
            action=action,
            delta_x=dx_body,
            delta_y=dy_body,
            delta_theta=dtheta,
            duration=duration,
            num_steps=num_steps,
            trajectory=trajectory,
        )

    env.close()
    return results


# ============================================================================
# Part 2: RRT Planner in Reduced (x, y, theta) Space
# ============================================================================

@dataclass
class RRTNode:
    """Node in the RRT tree."""
    x: float
    y: float
    theta: float
    parent_idx: Optional[int] = None
    action_name: Optional[str] = None  # which primitive got us here


@dataclass
class RRTConfig:
    """Configuration for the RRT planner."""
    max_iterations: int = 500
    goal_threshold: float = 0.15  # meters
    goal_bias: float = 0.2  # probability of sampling goal
    x_range: Tuple[float, float] = (-1.0, 3.0)
    y_range: Tuple[float, float] = (-2.0, 2.0)


class SnakeRRT:
    """RRT planner using motion primitives as edges.

    Plans in reduced (x, y, theta) configuration space. Each edge
    corresponds to executing one motion primitive for a fixed duration.
    The simulator is used as the forward model.
    """

    def __init__(
        self,
        primitives: Dict[str, MotionPrimitive],
        config: RRTConfig = None,
    ):
        self.primitives = primitives
        self.config = config or RRTConfig()
        self.nodes: List[RRTNode] = []
        self.rng = np.random.default_rng(42)

    def _distance(self, n1: RRTNode, n2: RRTNode) -> float:
        """Weighted distance metric combining position and heading."""
        dx = n1.x - n2.x
        dy = n1.y - n2.y
        dtheta = abs((n1.theta - n2.theta + math.pi) % (2 * math.pi) - math.pi)
        # Weight heading less than position (snake can turn while moving)
        return math.sqrt(dx**2 + dy**2 + 0.1 * dtheta**2)

    def _sample_random(self, goal: RRTNode) -> RRTNode:
        """Sample a random configuration, with goal bias."""
        if self.rng.random() < self.config.goal_bias:
            return goal
        x = self.rng.uniform(*self.config.x_range)
        y = self.rng.uniform(*self.config.y_range)
        theta = self.rng.uniform(-math.pi, math.pi)
        return RRTNode(x=x, y=y, theta=theta)

    def _nearest(self, target: RRTNode) -> int:
        """Find nearest node in tree to target."""
        best_idx = 0
        best_dist = float("inf")
        for i, node in enumerate(self.nodes):
            d = self._distance(node, target)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx

    def _steer(self, from_node: RRTNode, primitive: MotionPrimitive) -> RRTNode:
        """Apply a motion primitive from a given state.

        Uses the primitive's (dx, dy, dtheta) transformed to world frame.
        """
        cos_t = math.cos(from_node.theta)
        sin_t = math.sin(from_node.theta)

        # Transform body-frame displacement to world frame
        world_dx = cos_t * primitive.delta_x - sin_t * primitive.delta_y
        world_dy = sin_t * primitive.delta_x + cos_t * primitive.delta_y

        new_x = from_node.x + world_dx
        new_y = from_node.y + world_dy
        new_theta = from_node.theta + primitive.delta_theta
        new_theta = (new_theta + math.pi) % (2 * math.pi) - math.pi

        return RRTNode(
            x=new_x,
            y=new_y,
            theta=new_theta,
            parent_idx=None,
            action_name=primitive.name,
        )

    def plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
    ) -> Optional[List[RRTNode]]:
        """Run RRT from start to goal.

        Args:
            start: (x, y, theta) start configuration
            goal: (x, y, theta) goal configuration

        Returns:
            Path as list of RRTNodes, or None if planning fails.
        """
        self.nodes = [RRTNode(x=start[0], y=start[1], theta=start[2])]
        goal_node = RRTNode(x=goal[0], y=goal[1], theta=goal[2])

        # Select useful primitives (skip "stop")
        active_primitives = {
            k: v for k, v in self.primitives.items()
            if k != "stop" and abs(v.delta_x) + abs(v.delta_y) > 1e-4
        }

        for iteration in range(self.config.max_iterations):
            # Sample random point
            q_rand = self._sample_random(goal_node)

            # Find nearest node
            near_idx = self._nearest(q_rand)
            near_node = self.nodes[near_idx]

            # Try each primitive from nearest node, pick best
            best_new = None
            best_dist = float("inf")

            for prim_name, prim in active_primitives.items():
                candidate = self._steer(near_node, prim)
                d = self._distance(candidate, q_rand)
                if d < best_dist:
                    best_dist = d
                    best_new = candidate
                    best_new.action_name = prim_name

            if best_new is None:
                continue

            # Add to tree
            best_new.parent_idx = near_idx
            self.nodes.append(best_new)

            # Check goal
            goal_dist = math.sqrt(
                (best_new.x - goal_node.x) ** 2
                + (best_new.y - goal_node.y) ** 2
            )
            if goal_dist < self.config.goal_threshold:
                return self._extract_path(len(self.nodes) - 1)

        return None  # Failed

    def _extract_path(self, goal_idx: int) -> List[RRTNode]:
        """Extract path from root to goal node."""
        path = []
        idx = goal_idx
        while idx is not None:
            path.append(self.nodes[idx])
            idx = self.nodes[idx].parent_idx
        path.reverse()
        return path


# ============================================================================
# Part 3: Simulate-and-Verify (run RRT path in actual simulator)
# ============================================================================

def verify_rrt_path(
    path: List[RRTNode],
    primitives: Dict[str, MotionPrimitive],
    steps_per_primitive: int = 20,
) -> Tuple[List[Tuple[float, float, float]], float]:
    """Execute an RRT path in the actual simulator and measure error.

    Returns:
        (actual_trajectory, final_position_error)
    """
    config = LocomotionEnvConfig(
        max_episode_steps=len(path) * steps_per_primitive + 10,
        randomize_initial_heading=False,
    )
    env = LocomotionEnv(config=config, device="cpu")
    td = env.reset()

    import torch
    actual_trajectory = []
    positions = env._get_positions()
    com = env._get_com(positions)
    heading = env._get_heading_angle(positions)
    actual_trajectory.append((com[0], com[1], heading))

    for node in path[1:]:  # skip start node
        action = primitives[node.action_name].action
        action_tensor = torch.tensor(action, dtype=torch.float32)

        for _ in range(steps_per_primitive):
            td_action = td.clone()
            td_action["action"] = action_tensor
            td = env._step(td_action)

        positions = env._get_positions()
        com = env._get_com(positions)
        heading = env._get_heading_angle(positions)
        actual_trajectory.append((com[0], com[1], heading))

    env.close()

    # Compute error vs planned goal
    planned_goal = (path[-1].x, path[-1].y)
    actual_end = (actual_trajectory[-1][0], actual_trajectory[-1][1])
    error = math.sqrt(
        (planned_goal[0] - actual_end[0]) ** 2
        + (planned_goal[1] - actual_end[1]) ** 2
    )
    return actual_trajectory, error


# ============================================================================
# Part 4: Feasibility Analysis
# ============================================================================

def analyze_feasibility(primitives: Dict[str, MotionPrimitive]) -> str:
    """Generate a feasibility report for motion planning approaches."""
    lines = []
    lines.append("=" * 70)
    lines.append("MOTION PLANNING FEASIBILITY REPORT")
    lines.append("=" * 70)

    # 1. Configuration Space Analysis
    lines.append("\n1. CONFIGURATION SPACE ANALYSIS")
    lines.append("-" * 40)
    lines.append("  Full state: 21 nodes x 2 (XY) = 42 DOF + velocities")
    lines.append("  Control space: 5-dim serpenoid (amp, freq, wave_num, phase, turn_bias)")
    lines.append("  Task space: 3-dim (x, y, heading)")
    lines.append("  Constraint: Non-holonomic (cannot move laterally)")
    lines.append("  Physics: DisMech rod + RFT anisotropic friction")

    # 2. Motion Primitive Summary
    lines.append("\n2. MOTION PRIMITIVES")
    lines.append("-" * 40)
    lines.append(f"  {'Name':<20} {'dx(m)':<10} {'dy(m)':<10} {'dtheta(deg)':<12} {'sim_time(s)'}")
    for name, p in primitives.items():
        lines.append(
            f"  {name:<20} {p.delta_x:<10.4f} {p.delta_y:<10.4f} "
            f"{math.degrees(p.delta_theta):<12.2f} {p.duration:.2f}"
        )

    # 3. Feasibility per approach
    lines.append("\n3. APPROACH FEASIBILITY")
    lines.append("-" * 40)

    # Check if we have meaningful forward motion
    fwd = primitives.get("forward")
    has_forward = fwd and abs(fwd.delta_x) > 0.001

    # Check if turns produce heading change
    tl = primitives.get("turn_left")
    tr = primitives.get("turn_right")
    has_steering = (
        tl and abs(tl.delta_theta) > 0.01
        and tr and abs(tr.delta_theta) > 0.01
    )

    # Sim speed
    avg_sim_time = np.mean([p.duration for p in primitives.values()])

    lines.append(f"\n  Forward motion detected: {'YES' if has_forward else 'NO'}")
    if has_forward:
        lines.append(f"    Forward displacement per primitive: {fwd.delta_x:.4f} m")
    lines.append(f"  Steering control detected: {'YES' if has_steering else 'NO'}")
    if has_steering:
        lines.append(f"    Left turn rate: {math.degrees(tl.delta_theta):.2f} deg/prim")
        lines.append(f"    Right turn rate: {math.degrees(tr.delta_theta):.2f} deg/prim")
    lines.append(f"  Avg simulation time per primitive: {avg_sim_time:.3f} s")

    lines.append("\n  a) SAMPLING-BASED (RRT, RRT*, PRM)")
    if has_forward and has_steering:
        lines.append("     FEASIBLE with caveats:")
        lines.append("     + Motion primitives provide controllable (dx, dy, dtheta)")
        lines.append("     + RRT can plan in reduced (x, y, theta) space")
        lines.append("     + PRM can build a roadmap for repeated queries")
        lines.append(f"     - Simulation cost: ~{avg_sim_time:.2f}s per edge evaluation")
        lines.append(f"       (500 RRT iterations ~ {500 * avg_sim_time:.0f}s if sim-verified)")
        lines.append("     - Open-loop execution accumulates drift")
        lines.append("     - Non-holonomic constraints limit reachability")
        lines.append("     RECOMMENDATION: Use motion-primitive RRT with offline primitive")
        lines.append("       characterization (fast) + online sim verification (selective)")
    else:
        lines.append("     LIMITED: Insufficient motion control for effective planning")

    lines.append("\n  b) OPTIMIZATION-BASED (CHOMP, TrajOpt)")
    lines.append("     CHALLENGING:")
    lines.append("     - DisMech simulator is NOT differentiable")
    lines.append("       (implicit Euler solver, contact dynamics)")
    lines.append("     - Cannot compute analytical gradients for trajectory optimization")
    lines.append("     - Finite-difference gradients possible but expensive:")
    lines.append(f"       5-dim action x {avg_sim_time:.2f}s/eval = {5 * avg_sim_time:.1f}s per gradient")
    lines.append("     - CHOMP requires smooth cost landscape (violated by contact)")
    lines.append("     ALTERNATIVE: CMA-ES or other gradient-free optimization")
    lines.append("       could optimize short trajectory segments")

    lines.append("\n  c) LATTICE-BASED PLANNING")
    if has_forward and has_steering:
        lines.append("     MOST PRACTICAL APPROACH:")
        lines.append("     + Pre-compute motion primitive lattice offline")
        lines.append("     + A* or Dijkstra search over discretized (x, y, theta)")
        lines.append("     + Deterministic, repeatable, no simulation at planning time")
        lines.append("     - Resolution limited by primitive granularity")
        lines.append("     - Requires re-planning for dynamic obstacles")
    else:
        lines.append("     LIMITED: Need reliable primitives first")

    lines.append("\n  d) HYBRID: RL + PLANNING")
    lines.append("     RECOMMENDED FOR THIS PROJECT:")
    lines.append("     + Use RL (PPO) for low-level locomotion (already implemented)")
    lines.append("     + Use planning for high-level waypoint sequencing")
    lines.append("     + The HRL architecture already supports this hierarchy:")
    lines.append("       Meta-controller (planner) -> Sub-policies (RL)")
    lines.append("     + Planning provides interpretable high-level behavior")
    lines.append("     + RL handles the complex physics-dependent locomotion")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("SNAKE MOTION PLANNING FEASIBILITY STUDY")
    print("=" * 70)

    # Step 1: Characterize motion primitives
    print("\nStep 1: Characterizing motion primitives...")
    primitives = characterize_primitives(num_steps=20)

    print("\n  Primitive results:")
    for name, p in primitives.items():
        print(
            f"    {name:<20} dx={p.delta_x:+.4f}m  dy={p.delta_y:+.4f}m  "
            f"dtheta={math.degrees(p.delta_theta):+.2f}deg  ({p.duration:.2f}s)"
        )

    # Step 2: Run RRT
    print("\nStep 2: Running RRT planner...")
    rrt = SnakeRRT(
        primitives=primitives,
        config=RRTConfig(
            max_iterations=500,
            goal_threshold=0.15,
            goal_bias=0.2,
        ),
    )

    # Plan from origin heading +x to goal at (1.5, 0.5)
    start = (0.0, 0.0, 0.0)
    goal = (1.5, 0.5, 0.0)
    print(f"  Planning from {start} to {goal}...")

    t0 = time.time()
    path = rrt.plan(start, goal)
    plan_time = time.time() - t0

    if path is not None:
        print(f"  Path found! {len(path)} nodes, planning time: {plan_time:.3f}s")
        print("  Path actions:")
        for i, node in enumerate(path):
            action = node.action_name or "start"
            print(f"    [{i}] ({node.x:.3f}, {node.y:.3f}, {math.degrees(node.theta):.1f}deg) via {action}")

        # Step 3: Verify in simulator
        print("\nStep 3: Verifying path in simulator...")
        actual_traj, error = verify_rrt_path(path, primitives, steps_per_primitive=20)
        print(f"  Final position error (planned vs actual): {error:.4f} m")
        print(f"  Actual endpoint: ({actual_traj[-1][0]:.3f}, {actual_traj[-1][1]:.3f})")
        print(f"  Planned endpoint: ({path[-1].x:.3f}, {path[-1].y:.3f})")
    else:
        print(f"  RRT failed to find path in {rrt.config.max_iterations} iterations ({plan_time:.3f}s)")

    # Step 4: Feasibility report
    print("\n")
    report = analyze_feasibility(primitives)
    print(report)

    return primitives, path


if __name__ == "__main__":
    primitives, path = main()
