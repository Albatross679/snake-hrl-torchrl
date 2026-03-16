"""A* global path planner and occupancy grid (Jiang et al., 2024).

Provides the first layer of the four-layer control hierarchy:
A* planner generates a waypoint sequence through the maze,
which the RL agent follows one waypoint at a time.
"""

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np


class OccupancyGrid:
    """2D binary occupancy grid for path planning.

    Converts between world coordinates and grid indices.

    Args:
        width: Grid width (number of cells).
        height: Grid height (number of cells).
        resolution: Cell size in world units (meters).
        origin: World coordinate of grid cell (0, 0).
    """

    def __init__(
        self,
        width: int,
        height: int,
        resolution: float = 0.5,
        origin: Tuple[float, float] = (0.0, 0.0),
    ):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = origin
        self.grid = np.zeros((height, width), dtype=np.int8)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        gx = int((wx - self.origin[0]) / self.resolution)
        gy = int((wy - self.origin[1]) / self.resolution)
        gx = max(0, min(gx, self.width - 1))
        gy = max(0, min(gy, self.height - 1))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid indices to world coordinates (cell center)."""
        wx = gx * self.resolution + self.origin[0] + self.resolution / 2
        wy = gy * self.resolution + self.origin[1] + self.resolution / 2
        return wx, wy

    def set_obstacle(self, gx: int, gy: int) -> None:
        """Mark a cell as occupied."""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            self.grid[gy, gx] = 1

    def is_free(self, gx: int, gy: int) -> bool:
        """Check if a cell is free (not occupied and in bounds)."""
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.grid[gy, gx] == 0
        return False

    def set_walls_from_segments(
        self,
        walls: List[Tuple[float, float, float, float]],
        inflation: float = 0.0,
    ) -> None:
        """Mark grid cells occupied by wall segments.

        Args:
            walls: List of (x1, y1, x2, y2) wall segments in world coords.
            inflation: Extra radius around walls (meters) for safety margin.
        """
        for x1, y1, x2, y2 in walls:
            # Rasterize line segment onto grid
            gx1, gy1 = self.world_to_grid(x1, y1)
            gx2, gy2 = self.world_to_grid(x2, y2)

            # Bresenham-like rasterization
            steps = max(abs(gx2 - gx1), abs(gy2 - gy1), 1)
            for t in range(steps + 1):
                frac = t / steps
                gx = int(round(gx1 + frac * (gx2 - gx1)))
                gy = int(round(gy1 + frac * (gy2 - gy1)))

                # Inflate around the point
                inflate_cells = int(math.ceil(inflation / self.resolution))
                for dx in range(-inflate_cells, inflate_cells + 1):
                    for dy in range(-inflate_cells, inflate_cells + 1):
                        self.set_obstacle(gx + dx, gy + dy)


class AStarPlanner:
    """A* path planner on an occupancy grid.

    Uses 8-connected grid with Euclidean heuristic.

    Args:
        grid: OccupancyGrid instance.
    """

    def __init__(self, grid: OccupancyGrid):
        self.grid = grid
        # 8-connected neighbors: (dx, dy, cost)
        self._neighbors = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2)),
        ]

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """Find shortest path from start to goal in world coordinates.

        Args:
            start: Start position (world coords).
            goal: Goal position (world coords).

        Returns:
            List of waypoints in world coords, or None if no path found.
        """
        sx, sy = self.grid.world_to_grid(*start)
        gx, gy = self.grid.world_to_grid(*goal)

        if not self.grid.is_free(sx, sy) or not self.grid.is_free(gx, gy):
            return None

        # Priority queue: (f_cost, g_cost, x, y)
        open_set = [(0.0, 0.0, sx, sy)]
        came_from = {}
        g_score = {(sx, sy): 0.0}

        while open_set:
            f, g, cx, cy = heapq.heappop(open_set)

            if (cx, cy) == (gx, gy):
                # Reconstruct path
                path = []
                node = (gx, gy)
                while node in came_from:
                    path.append(self.grid.grid_to_world(*node))
                    node = came_from[node]
                path.append(self.grid.grid_to_world(sx, sy))
                path.reverse()
                return self._simplify_path(path)

            if g > g_score.get((cx, cy), float("inf")):
                continue

            for dx, dy, cost in self._neighbors:
                nx, ny = cx + dx, cy + dy
                if not self.grid.is_free(nx, ny):
                    continue

                new_g = g + cost
                if new_g < g_score.get((nx, ny), float("inf")):
                    g_score[(nx, ny)] = new_g
                    h = math.sqrt((nx - gx) ** 2 + (ny - gy) ** 2)
                    heapq.heappush(open_set, (new_g + h, new_g, nx, ny))
                    came_from[(nx, ny)] = (cx, cy)

        return None  # No path found

    def _simplify_path(
        self,
        path: List[Tuple[float, float]],
        tolerance: float = 0.1,
    ) -> List[Tuple[float, float]]:
        """Simplify path using Ramer-Douglas-Peucker algorithm.

        Args:
            path: List of waypoints.
            tolerance: Maximum deviation allowed (world units).

        Returns:
            Simplified path.
        """
        if len(path) <= 2:
            return path

        # Find the point with maximum distance from the line segment
        start = np.array(path[0])
        end = np.array(path[-1])
        line_vec = end - start
        line_len = np.linalg.norm(line_vec)

        max_dist = 0.0
        max_idx = 0

        for i in range(1, len(path) - 1):
            point = np.array(path[i])
            if line_len < 1e-10:
                dist = np.linalg.norm(point - start)
            else:
                # Distance from point to line
                cross = abs(np.cross(line_vec, start - point))
                dist = cross / line_len
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > tolerance:
            left = self._simplify_path(path[:max_idx + 1], tolerance)
            right = self._simplify_path(path[max_idx:], tolerance)
            return left[:-1] + right
        else:
            return [path[0], path[-1]]
