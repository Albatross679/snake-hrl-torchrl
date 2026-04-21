"""Pipe tunnel geometry for dismech-rods rod-to-rod contact.

Generates fixed rod limbs that approximate a cylindrical pipe's inner surface.
Each cross-sectional ring is a separate limb (open polygon) with all nodes
pinned via boundary conditions. The existing multi-limb contact (FCL + Lumelsky)
handles snake-to-wall forces automatically.

Usage with dismech-rods::

    pipe = PipeGeometry.straight(
        start=np.array([0, 0, 0]),
        direction=np.array([1, 0, 0]),
        length=1.0,
        radius=0.05,
    )
    pipe.add_to_simulation(soft_robots, forces, snake_col_group=0x0001)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PipeRing:
    """A single cross-sectional ring of the pipe wall.

    Attributes:
        nodes: (n_sides, 3) array of node positions on the ring.
        center: (3,) center of the ring.
        normal: (3,) unit normal of the ring plane (pipe axis direction).
    """
    nodes: np.ndarray
    center: np.ndarray
    normal: np.ndarray


@dataclass
class PipeGeometry:
    """Collection of rings forming a pipe tunnel.

    Attributes:
        rings: List of PipeRing objects.
        radius: Inner radius of the pipe.
        n_sides: Number of polygon sides per ring.
        ring_spacing: Axial distance between consecutive rings.
        wall_radius: Capsule radius of wall rod edges.
        stagger: Whether adjacent rings are rotated by half the angular spacing.
    """
    rings: List[PipeRing] = field(default_factory=list)
    radius: float = 0.05
    n_sides: int = 12
    ring_spacing: float = 0.04
    wall_radius: float = 0.002
    stagger: bool = True

    # --- Factory methods ---

    @classmethod
    def straight(
        cls,
        start: np.ndarray,
        direction: np.ndarray,
        length: float,
        radius: float = 0.05,
        n_sides: int = 12,
        ring_spacing: float = 0.04,
        wall_radius: float = 0.002,
        stagger: bool = True,
    ) -> PipeGeometry:
        """Create a straight pipe.

        Args:
            start: (3,) starting center of the pipe.
            direction: (3,) unnormalized pipe axis direction.
            length: Total length of the pipe.
            radius: Inner radius.
            n_sides: Polygon sides per ring.
            ring_spacing: Axial distance between rings.
            wall_radius: Capsule radius of wall edges.
            stagger: Rotate alternate rings by half angular step.

        Returns:
            PipeGeometry with rings arranged along a straight axis.
        """
        direction = np.asarray(direction, dtype=float)
        axis = direction / np.linalg.norm(direction)
        start = np.asarray(start, dtype=float)

        n_rings = max(2, int(np.ceil(length / ring_spacing)) + 1)
        actual_spacing = length / (n_rings - 1)

        pipe = cls(
            radius=radius,
            n_sides=n_sides,
            ring_spacing=actual_spacing,
            wall_radius=wall_radius,
            stagger=stagger,
        )

        for i in range(n_rings):
            center = start + i * actual_spacing * axis
            angle_offset = (np.pi / n_sides) * (i % 2) if stagger else 0.0
            ring = _make_ring(center, axis, radius, n_sides, angle_offset)
            pipe.rings.append(ring)

        return pipe

    @classmethod
    def curved(
        cls,
        center_of_curvature: np.ndarray,
        start_angle: float,
        sweep_angle: float,
        bend_radius: float,
        pipe_radius: float = 0.05,
        bend_plane_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
        n_sides: int = 12,
        ring_spacing: float = 0.04,
        wall_radius: float = 0.002,
        stagger: bool = True,
    ) -> PipeGeometry:
        """Create a curved pipe section (arc in a plane).

        Args:
            center_of_curvature: (3,) center point of the bend arc.
            start_angle: Starting angle (radians) in the bend plane.
            sweep_angle: Total angle swept (radians, positive = CCW).
            bend_radius: Distance from center of curvature to pipe centerline.
            pipe_radius: Inner radius of the pipe cross-section.
            bend_plane_normal: (3,) normal to the plane of the bend.
            n_sides: Polygon sides per ring.
            ring_spacing: Arc-length distance between rings.
            wall_radius: Capsule radius of wall edges.
            stagger: Rotate alternate rings.

        Returns:
            PipeGeometry with rings arranged along a circular arc.
        """
        center_of_curvature = np.asarray(center_of_curvature, dtype=float)
        bend_plane_normal = np.asarray(bend_plane_normal, dtype=float)
        bend_plane_normal = bend_plane_normal / np.linalg.norm(bend_plane_normal)

        arc_length = abs(sweep_angle) * bend_radius
        n_rings = max(2, int(np.ceil(arc_length / ring_spacing)) + 1)

        # Build a local coordinate frame for the bend plane
        # e1, e2 span the plane; bend_plane_normal is the out-of-plane direction
        e1 = _perpendicular_vector(bend_plane_normal)
        e2 = np.cross(bend_plane_normal, e1)

        pipe = cls(
            radius=pipe_radius,
            n_sides=n_sides,
            ring_spacing=ring_spacing,
            wall_radius=wall_radius,
            stagger=stagger,
        )

        angles = np.linspace(start_angle, start_angle + sweep_angle, n_rings)
        for i, theta in enumerate(angles):
            # Centerline position on the arc
            radial = np.cos(theta) * e1 + np.sin(theta) * e2
            center = center_of_curvature + bend_radius * radial

            # Pipe axis = tangent to the arc (perpendicular to radial in the plane)
            tangent = -np.sin(theta) * e1 + np.cos(theta) * e2
            if sweep_angle < 0:
                tangent = -tangent

            angle_offset = (np.pi / n_sides) * (i % 2) if stagger else 0.0
            ring = _make_ring(center, tangent, pipe_radius, n_sides, angle_offset)
            pipe.rings.append(ring)

        return pipe

    @classmethod
    def from_centerline(
        cls,
        points: np.ndarray,
        radius: float = 0.05,
        n_sides: int = 12,
        wall_radius: float = 0.002,
        stagger: bool = True,
    ) -> PipeGeometry:
        """Create a pipe from an arbitrary centerline path.

        Places one ring at each provided centerline point. The pipe axis at
        each ring is estimated from the local tangent direction.

        Args:
            points: (n_points, 3) centerline positions.
            radius: Inner radius.
            n_sides: Polygon sides per ring.
            wall_radius: Capsule radius.
            stagger: Rotate alternate rings.

        Returns:
            PipeGeometry following the centerline.
        """
        points = np.asarray(points, dtype=float)
        assert points.ndim == 2 and points.shape[1] == 3
        n = len(points)
        assert n >= 2

        # Estimate tangent at each point via central differences
        tangents = np.zeros_like(points)
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        for i in range(1, n - 1):
            tangents[i] = points[i + 1] - points[i - 1]
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        tangents = tangents / norms

        spacings = np.linalg.norm(np.diff(points, axis=0), axis=1)
        avg_spacing = float(np.mean(spacings)) if len(spacings) > 0 else 0.04

        pipe = cls(
            radius=radius,
            n_sides=n_sides,
            ring_spacing=avg_spacing,
            wall_radius=wall_radius,
            stagger=stagger,
        )

        for i in range(n):
            angle_offset = (np.pi / n_sides) * (i % 2) if stagger else 0.0
            ring = _make_ring(points[i], tangents[i], radius, n_sides, angle_offset)
            pipe.rings.append(ring)

        return pipe

    # --- Concatenation ---

    def extend(self, other: PipeGeometry) -> PipeGeometry:
        """Append another pipe's rings to this one (in-place). Returns self."""
        self.rings.extend(other.rings)
        return self

    # --- Integration with dismech-rods ---

    def add_to_simulation(
        self,
        soft_robots,
        forces=None,
        snake_col_group: int = 0x0001,
        wall_col_group: Optional[int] = None,
        wall_density: float = 1000.0,
        wall_youngs_modulus: float = 1e8,
        wall_poisson_ratio: float = 0.5,
        contact_col_limit: float = 5e-3,
        contact_delta: float = 2e-3,
        contact_k_scaler: float = 1e4,
        contact_friction: bool = True,
        contact_nu: float = 1e-3,
    ) -> dict:
        """Add pipe wall limbs to a dismech-rods simulation.

        This must be called AFTER adding the snake limb but BEFORE
        calling sim_manager.initialize([]).

        Args:
            soft_robots: py_dismech.SoftRobots handle.
            forces: py_dismech.ForceContainer handle. If provided, a ContactForce
                is automatically added.
            snake_col_group: Collision group bitmask of the snake limb.
            wall_col_group: Collision group for wall limbs. Defaults to
                snake_col_group (so walls collide with snake).
            wall_density: Density for wall rods (irrelevant, nodes are locked).
            wall_youngs_modulus: Young's modulus (high = effectively rigid).
            wall_poisson_ratio: Poisson ratio for wall rods.
            contact_col_limit: FCL broad-phase distance buffer.
            contact_delta: Penalty activation distance.
            contact_k_scaler: Contact stiffness scaling.
            contact_friction: Enable friction.
            contact_nu: Friction smoothing parameter.

        Returns:
            Dict with metadata: limb_indices, n_rings, n_wall_edges.
        """
        import py_dismech

        if wall_col_group is None:
            wall_col_group = snake_col_group

        first_wall_limb = len(soft_robots.limbs)
        n_wall_edges = 0

        for ring in self.rings:
            nodes = [ring.nodes[j].tolist() for j in range(len(ring.nodes))]
            nodes_vec3 = [np.array(n) for n in nodes]

            soft_robots.addLimb(
                nodes_vec3,
                wall_density,
                self.wall_radius,
                wall_youngs_modulus,
                wall_poisson_ratio,
                0.0,  # mu (friction coeff for rod bending, irrelevant here)
                wall_col_group,
            )

            limb_idx = len(soft_robots.limbs) - 1
            n_edges = len(ring.nodes) - 1
            for edge_idx in range(n_edges):
                soft_robots.lockEdge(limb_idx, edge_idx)
            n_wall_edges += n_edges

        last_wall_limb = len(soft_robots.limbs) - 1

        # Add contact force if forces container is provided
        if forces is not None:
            contact = py_dismech.ContactForce(
                soft_robots,
                contact_col_limit,
                contact_delta,
                contact_k_scaler,
                contact_friction,
                contact_nu,
                False,  # self_contact=False (only inter-limb)
            )
            forces.addForce(contact)

        return {
            "first_wall_limb": first_wall_limb,
            "last_wall_limb": last_wall_limb,
            "n_rings": len(self.rings),
            "n_wall_edges": n_wall_edges,
        }

    # --- Queries ---

    @property
    def n_rings(self) -> int:
        return len(self.rings)

    @property
    def total_wall_edges(self) -> int:
        return sum(len(r.nodes) - 1 for r in self.rings)

    @property
    def centerline(self) -> np.ndarray:
        """(n_rings, 3) array of ring center positions."""
        return np.array([r.center for r in self.rings])

    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the pipe (approximate).

        Projects the point onto the nearest ring and checks radial distance.
        """
        point = np.asarray(point, dtype=float)
        centers = self.centerline
        dists = np.linalg.norm(centers - point, axis=1)
        nearest = np.argmin(dists)

        ring = self.rings[nearest]
        to_point = point - ring.center
        # Project out the axial component
        axial = np.dot(to_point, ring.normal)
        radial_vec = to_point - axial * ring.normal
        radial_dist = np.linalg.norm(radial_vec)

        return radial_dist < self.radius

    def get_wall_nodes_flat(self) -> np.ndarray:
        """Get all wall node positions as a flat (N, 3) array."""
        all_nodes = [r.nodes for r in self.rings]
        return np.vstack(all_nodes)


# --- Internal helpers ---

def _make_ring(
    center: np.ndarray,
    axis: np.ndarray,
    radius: float,
    n_sides: int,
    angle_offset: float = 0.0,
) -> PipeRing:
    """Create a polygonal ring of nodes around a center point.

    Args:
        center: (3,) center of the ring.
        axis: (3,) unit normal to the ring plane (pipe axis direction).
        radius: Radial distance from center to nodes.
        n_sides: Number of nodes in the polygon.
        angle_offset: Rotation offset (radians) for staggering.

    Returns:
        PipeRing with n_sides nodes.
    """
    axis = axis / np.linalg.norm(axis)

    # Build orthonormal frame: axis, e1, e2
    e1 = _perpendicular_vector(axis)
    e2 = np.cross(axis, e1)

    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False) + angle_offset
    nodes = np.zeros((n_sides, 3))
    for i, theta in enumerate(angles):
        nodes[i] = center + radius * (np.cos(theta) * e1 + np.sin(theta) * e2)

    return PipeRing(nodes=nodes, center=center.copy(), normal=axis.copy())


def _perpendicular_vector(v: np.ndarray) -> np.ndarray:
    """Return a unit vector perpendicular to v."""
    v = v / np.linalg.norm(v)
    # Pick the axis least aligned with v
    candidates = np.eye(3)
    dots = np.abs(candidates @ v)
    least_aligned = candidates[np.argmin(dots)]
    perp = np.cross(v, least_aligned)
    return perp / np.linalg.norm(perp)
