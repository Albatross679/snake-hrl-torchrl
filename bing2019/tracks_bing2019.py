"""Target trajectory generators for locomotion tasks.

Faithfully ported from the original TracksGenerator (Bing et al., IJCAI 2019).
"""

import math
import numpy as np


class TracksGenerator:
    """Generates target trajectories for the snake to follow."""

    def __init__(
        self,
        target_v: float = 0.3,
        head_target_dist: float = 4.0,
        target_dist_min: float = 2.0,
        target_dist_max: float = 6.0,
    ):
        self.target_v = target_v
        self.head_target_dist = head_target_dist
        self.target_dist_min = target_dist_min
        self.target_dist_max = target_dist_max

        # State for random track generator
        self._current_segment_idx = 0
        self._current_segment_start_x = 0.0
        self._current_segment_start_y = 0.0

    @staticmethod
    def calculate_distance(head_x, head_y, target_x, target_y):
        return math.sqrt((target_x - head_x) ** 2 + (target_y - head_y) ** 2)

    def reset(self):
        """Reset internal state for random track generator."""
        self._current_segment_idx = 0
        self._current_segment_start_x = 0.0
        self._current_segment_start_y = 0.0

    def gen_line_step(self, head_x, head_y, target_x, target_y, dt):
        """Generate next target position along a straight line."""
        x = target_x
        y = 0.0

        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)

        if current_dist < self.target_dist_min:
            x = head_x + self.target_dist_min
        elif current_dist > self.target_dist_max:
            pass
        else:
            x += self.target_v * dt

        return x, y

    def gen_wave_step(self, head_x, head_y, target_x, target_y, dt):
        """Generate next target position along a sinusoidal wave."""
        start_sin_at = 5
        period = 0.2
        amplitude = 5

        x = target_x
        y = target_y
        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)

        if x >= start_sin_at:
            y = amplitude * np.sin(period * (x - start_sin_at))
        else:
            y = 0.0

        if current_dist < self.target_dist_min:
            x = head_x + self.target_dist_min
        elif current_dist > self.target_dist_max:
            pass
        else:
            y_diff = np.abs(y - target_y)
            y_way = np.sqrt(y_diff ** 2 + y_diff ** 2)
            x = x + (self.target_v * dt) - 5 * y_way * dt

        return x, y

    def gen_zigzag_step(self, head_x, head_y, target_x, target_y, dt):
        """Generate next target position along a zigzag path."""
        c = 10
        d = 0.056
        e = 4
        start_sin_at = 5

        x = target_x
        y = target_y
        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)

        def a():
            return c * (-1 + 2 * math.fmod(math.floor(d * (x + e)), 2))

        def b():
            return -c * math.fmod(math.floor(d * (x + e)), 2)

        if current_dist < self.target_dist_min:
            x = head_x + self.target_dist_min
        elif current_dist > self.target_dist_max:
            pass
        else:
            if x >= start_sin_at:
                x += (self.target_v * dt) * 0.872  # 60 degree correction
                y = (d * (x + e) - math.floor(d * (x + e))) * a() + b() + c / 2
            else:
                x += self.target_v * dt
                y = 0.0

        return x, y

    def gen_circle_step(self, head_x, head_y, target_x, target_y, dt):
        """Generate next target position along a circular path."""
        start_sin_at = 10
        radius = start_sin_at

        x = target_x
        y = target_y
        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)

        if current_dist < self.target_dist_min:
            x = head_x + self.target_dist_min
        elif current_dist > self.target_dist_max:
            pass
        else:
            x += self.target_v * dt

            if x >= start_sin_at:
                alpha_head = math.degrees(math.atan2(head_y, head_x))
                a = current_dist + self.target_v * dt
                b_val = radius
                c_val = radius
                cos_alpha = (b_val ** 2 + c_val ** 2 - a ** 2) / (2 * b_val * c_val)
                cos_alpha = max(-1.0, min(1.0, cos_alpha))  # clamp for numerical safety
                beta = math.degrees(math.acos(cos_alpha))
                alpha_target = alpha_head + beta
                x = math.cos(math.radians(alpha_target)) * radius
                y = math.sin(math.radians(alpha_target)) * radius

        return x, y

    def _gen_track_angles(self, angles, target_x, target_y, dist):
        """Helper for random track: advance along piecewise linear segments."""
        segment_angle = angles[self._current_segment_idx]
        segment_length = 5

        segment_dist = math.sqrt(
            (self._current_segment_start_x - target_x) ** 2
            + (self._current_segment_start_y - target_y) ** 2
        )
        new_segment_dist = dist + segment_dist

        if new_segment_dist > segment_length:
            self._current_segment_start_x += (
                math.cos(math.radians(segment_angle)) * segment_length
            )
            self._current_segment_start_y += (
                math.sin(math.radians(segment_angle)) * segment_length
            )

            if self._current_segment_idx < len(angles) - 1:
                self._current_segment_idx += 1

            segment_angle = angles[self._current_segment_idx]
            new_segment_dist = new_segment_dist - segment_length

        x = self._current_segment_start_x + math.cos(math.radians(segment_angle)) * new_segment_dist
        y = self._current_segment_start_y + math.sin(math.radians(segment_angle)) * new_segment_dist

        return x, y

    def gen_random_step(self, head_x, head_y, target_x, target_y, dt, seed=6):
        """Generate next target position along a random piecewise path."""
        max_degree_target_change = 60
        segments = 15

        rng = np.random.RandomState(seed)
        angles = [0]
        for _ in range(segments):
            a = rng.randint(
                angles[-1] - max_degree_target_change,
                angles[-1] + max_degree_target_change,
            )
            angles.append(a)

        current_dist = self.calculate_distance(head_x, head_y, target_x, target_y)

        if current_dist < self.target_dist_min:
            dist = self.target_dist_min - current_dist + self.target_v * dt
        elif current_dist > self.target_dist_max:
            dist = 0.0
        else:
            dist = self.target_v * dt

        x, y = self._gen_track_angles(angles, target_x, target_y, dist)
        return x, y

    def step(self, track_type, head_x, head_y, target_x, target_y, dt, **kwargs):
        """Dispatch to the appropriate track generator.

        Args:
            track_type: One of "line", "wave", "zigzag", "circle", "random".
            head_x, head_y: Current head position.
            target_x, target_y: Current target position.
            dt: Simulation timestep.

        Returns:
            Tuple of (new_target_x, new_target_y).
        """
        generators = {
            "line": self.gen_line_step,
            "wave": self.gen_wave_step,
            "zigzag": self.gen_zigzag_step,
            "circle": self.gen_circle_step,
            "random": self.gen_random_step,
        }
        if track_type not in generators:
            raise ValueError(f"Unknown track type: {track_type}. Choose from {list(generators)}")

        gen = generators[track_type]
        if track_type == "random":
            return gen(head_x, head_y, target_x, target_y, dt, **kwargs)
        return gen(head_x, head_y, target_x, target_y, dt)
