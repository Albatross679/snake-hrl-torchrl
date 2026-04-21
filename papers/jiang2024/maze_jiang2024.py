"""Kruskal's maze generator and MJCF wall geometry (Jiang et al., 2024).

Generates randomized maze layouts using Kruskal's algorithm with union-find.
Converts wall structure to MuJoCo XML for simulation.
"""

from typing import List, Optional, Tuple

import numpy as np


class _UnionFind:
    """Union-Find (Disjoint Set Union) data structure."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


class KruskalMazeGenerator:
    """Randomized Kruskal's algorithm for maze generation.

    Generates a perfect maze (every cell reachable from every other cell)
    on a rows x cols grid, then converts to wall segments in world coordinates.

    Args:
        rows: Number of cell rows.
        cols: Number of cell columns.
        cell_size: Size of each cell in world units (meters).
        origin: World coordinate of the maze's bottom-left corner.
        wall_height: Height of maze walls (meters).
        wall_thickness: Thickness of maze walls (meters).
    """

    def __init__(
        self,
        rows: int = 5,
        cols: int = 5,
        cell_size: float = 2.0,
        origin: Tuple[float, float] = (0.0, 0.0),
        wall_height: float = 0.3,
        wall_thickness: float = 0.1,
    ):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.origin = origin
        self.wall_height = wall_height
        self.wall_thickness = wall_thickness
        self._walls: Optional[List[Tuple[float, float, float, float]]] = None

    def generate(self, seed: Optional[int] = None) -> List[Tuple[float, float, float, float]]:
        """Generate maze walls using randomized Kruskal's algorithm.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            List of wall segments as (x1, y1, x2, y2) in world coordinates.
        """
        rng = np.random.RandomState(seed)

        rows, cols = self.rows, self.cols
        n_cells = rows * cols

        # Build list of all internal edges (walls between adjacent cells)
        edges = []
        for r in range(rows):
            for c in range(cols):
                cell_id = r * cols + c
                if c < cols - 1:
                    edges.append((cell_id, cell_id + 1, "v", r, c))  # vertical wall
                if r < rows - 1:
                    edges.append((cell_id, cell_id + cols, "h", r, c))  # horizontal wall

        # Shuffle edges
        rng.shuffle(edges)

        # Run Kruskal's
        uf = _UnionFind(n_cells)
        removed_walls = set()

        for cell_a, cell_b, orientation, r, c in edges:
            if uf.union(cell_a, cell_b):
                removed_walls.add((orientation, r, c))

        # Convert remaining internal walls + boundary walls to segments
        walls = []
        ox, oy = self.origin
        cs = self.cell_size

        # Boundary walls
        total_w = cols * cs
        total_h = rows * cs
        walls.append((ox, oy, ox + total_w, oy))                    # bottom
        walls.append((ox, oy + total_h, ox + total_w, oy + total_h))  # top
        walls.append((ox, oy, ox, oy + total_h))                    # left
        walls.append((ox + total_w, oy, ox + total_w, oy + total_h))  # right

        # Internal walls (those not removed by Kruskal's)
        walls_internal = []
        for cell_a, cell_b, orient, row, col in edges:
            if (orient, row, col) in removed_walls:
                continue
            if orient == "v":
                # Vertical wall: right side of cell (row, col)
                x = ox + (col + 1) * cs
                y1 = oy + row * cs
                y2 = oy + (row + 1) * cs
                walls_internal.append((x, y1, x, y2))
            else:
                # Horizontal wall: top side of cell (row, col)
                y = oy + (row + 1) * cs
                x1 = ox + col * cs
                x2 = ox + (col + 1) * cs
                walls_internal.append((x1, y, x2, y))

        walls.extend(walls_internal)
        self._walls = walls
        return walls

    def to_occupancy_grid(
        self,
        resolution: float = 0.25,
        inflation: float = 0.3,
    ) -> "OccupancyGrid":
        """Convert maze walls to an occupancy grid.

        Args:
            resolution: Grid cell size (meters).
            inflation: Safety margin around walls (meters).

        Returns:
            OccupancyGrid instance.
        """
        from jiang2024.planner_jiang2024 import OccupancyGrid

        if self._walls is None:
            raise RuntimeError("Call generate() before to_occupancy_grid()")

        total_w = self.cols * self.cell_size
        total_h = self.rows * self.cell_size
        width = int(np.ceil(total_w / resolution)) + 2
        height = int(np.ceil(total_h / resolution)) + 2

        grid = OccupancyGrid(
            width=width,
            height=height,
            resolution=resolution,
            origin=(self.origin[0] - resolution, self.origin[1] - resolution),
        )
        grid.set_walls_from_segments(self._walls, inflation=inflation)
        return grid

    def to_mjcf_bodies(self) -> str:
        """Convert maze walls to MuJoCo XML body elements.

        Returns:
            XML string containing <body> elements for each wall segment.
        """
        if self._walls is None:
            raise RuntimeError("Call generate() before to_mjcf_bodies()")

        bodies = []
        wh = self.wall_height
        wt = self.wall_thickness

        for i, (x1, y1, x2, y2) in enumerate(self._walls):
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            cz = wh / 2

            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx * dx + dy * dy)

            if length < 1e-6:
                continue

            # Wall as a box: half-sizes
            hx = length / 2
            hy = wt / 2
            hz = wh / 2

            # Rotation angle around z-axis
            angle = np.degrees(np.arctan2(dy, dx))

            bodies.append(
                f'        <body name="wall_{i}" pos="{cx:.3f} {cy:.3f} {cz:.3f}" '
                f'euler="0 0 {angle:.1f}">\n'
                f'            <geom type="box" size="{hx:.3f} {hy:.3f} {hz:.3f}" '
                f'rgba="0.6 0.6 0.6 1" conaffinity="1" condim="3"/>\n'
                f'        </body>'
            )

        return "\n".join(bodies)

    def get_cell_center(self, row: int, col: int) -> Tuple[float, float]:
        """Get world coordinates of a cell's center.

        Args:
            row: Cell row index.
            col: Cell column index.

        Returns:
            (x, y) world coordinates.
        """
        ox, oy = self.origin
        cs = self.cell_size
        return (ox + (col + 0.5) * cs, oy + (row + 0.5) * cs)
