"""Tests for pipe wall geometry generation."""

import numpy as np
import pytest

from src.physics.pipe_geometry import PipeGeometry, PipeRing, _make_ring, _perpendicular_vector


class TestPerpendicularVector:
    def test_perpendicular_to_x(self):
        v = np.array([1.0, 0.0, 0.0])
        perp = _perpendicular_vector(v)
        assert abs(np.dot(v, perp)) < 1e-10
        assert abs(np.linalg.norm(perp) - 1.0) < 1e-10

    def test_perpendicular_to_arbitrary(self):
        v = np.array([1.0, 2.0, 3.0])
        perp = _perpendicular_vector(v)
        assert abs(np.dot(v / np.linalg.norm(v), perp)) < 1e-10
        assert abs(np.linalg.norm(perp) - 1.0) < 1e-10

    def test_perpendicular_to_z(self):
        v = np.array([0.0, 0.0, 1.0])
        perp = _perpendicular_vector(v)
        assert abs(np.dot(v, perp)) < 1e-10


class TestMakeRing:
    def test_node_count(self):
        ring = _make_ring(np.zeros(3), np.array([1, 0, 0]), radius=0.05, n_sides=12)
        assert ring.nodes.shape == (12, 3)

    def test_nodes_on_circle(self):
        center = np.array([1.0, 2.0, 3.0])
        axis = np.array([0.0, 0.0, 1.0])
        radius = 0.05
        ring = _make_ring(center, axis, radius, n_sides=16)

        for node in ring.nodes:
            radial = node - center
            # Axial component should be zero
            assert abs(np.dot(radial, axis)) < 1e-10
            # Radial distance should equal radius
            assert abs(np.linalg.norm(radial) - radius) < 1e-10

    def test_center_and_normal_stored(self):
        center = np.array([5.0, 0.0, 0.0])
        axis = np.array([1.0, 0.0, 0.0])
        ring = _make_ring(center, axis, 0.1, 8)
        np.testing.assert_allclose(ring.center, center)
        np.testing.assert_allclose(ring.normal, axis / np.linalg.norm(axis))

    def test_stagger_offset(self):
        ring0 = _make_ring(np.zeros(3), np.array([1, 0, 0]), 0.05, 12, angle_offset=0.0)
        ring1 = _make_ring(np.zeros(3), np.array([1, 0, 0]), 0.05, 12, angle_offset=np.pi / 12)
        # Nodes should differ due to offset
        assert not np.allclose(ring0.nodes, ring1.nodes)

    def test_arbitrary_axis(self):
        center = np.zeros(3)
        axis = np.array([1.0, 1.0, 1.0])
        radius = 0.1
        ring = _make_ring(center, axis, radius, n_sides=8)

        axis_norm = axis / np.linalg.norm(axis)
        for node in ring.nodes:
            # Projection onto axis should be zero
            assert abs(np.dot(node - center, axis_norm)) < 1e-10
            # Radial distance should equal radius
            assert abs(np.linalg.norm(node - center) - radius) < 1e-10


class TestStraightPipe:
    def test_basic_creation(self):
        pipe = PipeGeometry.straight(
            start=np.zeros(3),
            direction=np.array([1, 0, 0]),
            length=1.0,
            radius=0.05,
            n_sides=12,
            ring_spacing=0.1,
        )
        assert pipe.n_rings >= 2
        assert pipe.radius == 0.05
        assert pipe.n_sides == 12

    def test_ring_centers_along_axis(self):
        pipe = PipeGeometry.straight(
            start=np.array([0, 0, 0]),
            direction=np.array([1, 0, 0]),
            length=1.0,
            ring_spacing=0.25,
        )
        centers = pipe.centerline
        # All centers should have y=0, z=0
        np.testing.assert_allclose(centers[:, 1], 0, atol=1e-10)
        np.testing.assert_allclose(centers[:, 2], 0, atol=1e-10)
        # X should be monotonically increasing from 0 to 1
        assert centers[0, 0] == pytest.approx(0.0)
        assert centers[-1, 0] == pytest.approx(1.0)

    def test_ring_spacing(self):
        pipe = PipeGeometry.straight(
            start=np.zeros(3),
            direction=np.array([0, 1, 0]),
            length=0.5,
            ring_spacing=0.05,
        )
        centers = pipe.centerline
        spacings = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        # All spacings should be approximately equal
        np.testing.assert_allclose(spacings, spacings[0], rtol=0.01)

    def test_stagger_alternates_rings(self):
        pipe = PipeGeometry.straight(
            start=np.zeros(3),
            direction=np.array([1, 0, 0]),
            length=0.2,
            ring_spacing=0.1,
            stagger=True,
        )
        if pipe.n_rings >= 2:
            # Adjacent rings should have different angular positions
            r0_angles = _ring_angles(pipe.rings[0])
            r1_angles = _ring_angles(pipe.rings[1])
            assert not np.allclose(r0_angles, r1_angles, atol=1e-6)

    def test_no_stagger(self):
        pipe = PipeGeometry.straight(
            start=np.zeros(3),
            direction=np.array([1, 0, 0]),
            length=0.2,
            ring_spacing=0.1,
            stagger=False,
        )
        if pipe.n_rings >= 2:
            r0_angles = _ring_angles(pipe.rings[0])
            r1_angles = _ring_angles(pipe.rings[1])
            np.testing.assert_allclose(r0_angles, r1_angles, atol=1e-6)

    def test_total_wall_edges(self):
        pipe = PipeGeometry.straight(
            start=np.zeros(3),
            direction=np.array([1, 0, 0]),
            length=1.0,
            n_sides=12,
            ring_spacing=0.1,
        )
        # Each ring has n_sides nodes and n_sides - 1 edges
        expected_edges_per_ring = pipe.n_sides - 1
        assert pipe.total_wall_edges == pipe.n_rings * expected_edges_per_ring

    def test_contains_point_center(self):
        pipe = PipeGeometry.straight(
            start=np.zeros(3),
            direction=np.array([1, 0, 0]),
            length=1.0,
            radius=0.05,
        )
        # Center of pipe should be inside
        assert pipe.contains_point(np.array([0.5, 0, 0]))
        # Point well outside should not be
        assert not pipe.contains_point(np.array([0.5, 0.1, 0]))

    def test_diagonal_direction(self):
        pipe = PipeGeometry.straight(
            start=np.zeros(3),
            direction=np.array([1, 1, 0]),
            length=1.0,
            radius=0.05,
        )
        # Centerline should follow the diagonal
        centers = pipe.centerline
        direction = centers[-1] - centers[0]
        direction /= np.linalg.norm(direction)
        expected = np.array([1, 1, 0]) / np.sqrt(2)
        np.testing.assert_allclose(direction, expected, atol=1e-10)


class TestCurvedPipe:
    def test_basic_creation(self):
        pipe = PipeGeometry.curved(
            center_of_curvature=np.zeros(3),
            start_angle=0.0,
            sweep_angle=np.pi / 2,
            bend_radius=0.5,
            pipe_radius=0.05,
        )
        assert pipe.n_rings >= 2

    def test_90_degree_bend(self):
        pipe = PipeGeometry.curved(
            center_of_curvature=np.zeros(3),
            start_angle=0.0,
            sweep_angle=np.pi / 2,
            bend_radius=0.5,
            pipe_radius=0.05,
            ring_spacing=0.05,
        )
        centers = pipe.centerline

        # All centers should be at bend_radius=0.5 from origin
        dists = np.linalg.norm(centers[:, :2], axis=1)
        np.testing.assert_allclose(dists, 0.5, atol=1e-10)
        # Arc should sweep 90 degrees: first and last centers should be perpendicular
        v0 = centers[0, :2] / np.linalg.norm(centers[0, :2])
        v1 = centers[-1, :2] / np.linalg.norm(centers[-1, :2])
        assert abs(np.dot(v0, v1)) < 1e-6  # perpendicular

    def test_ring_centers_on_arc(self):
        bend_radius = 0.3
        pipe = PipeGeometry.curved(
            center_of_curvature=np.zeros(3),
            start_angle=0.0,
            sweep_angle=np.pi,
            bend_radius=bend_radius,
            pipe_radius=0.05,
        )
        centers = pipe.centerline
        # All centers should be at bend_radius from origin (in xy plane)
        dists = np.linalg.norm(centers[:, :2], axis=1)
        np.testing.assert_allclose(dists, bend_radius, atol=1e-10)


class TestFromCenterline:
    def test_follows_path(self):
        # Zigzag path
        points = np.array([
            [0, 0, 0],
            [0.1, 0.05, 0],
            [0.2, 0, 0],
            [0.3, 0.05, 0],
            [0.4, 0, 0],
        ], dtype=float)
        pipe = PipeGeometry.from_centerline(points, radius=0.03)
        centers = pipe.centerline
        np.testing.assert_allclose(centers, points)
        assert pipe.n_rings == 5

    def test_nodes_at_correct_radius(self):
        points = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        radius = 0.05
        pipe = PipeGeometry.from_centerline(points, radius=radius, n_sides=8)

        for ring in pipe.rings:
            for node in ring.nodes:
                radial = node - ring.center
                axial_component = np.dot(radial, ring.normal) * ring.normal
                radial_component = radial - axial_component
                assert abs(np.linalg.norm(radial_component) - radius) < 1e-10


class TestExtend:
    def test_concatenation(self):
        p1 = PipeGeometry.straight(np.zeros(3), np.array([1, 0, 0]), 0.5, radius=0.05)
        p2 = PipeGeometry.straight(np.array([0.5, 0, 0]), np.array([1, 0, 0]), 0.5, radius=0.05)
        n1, n2 = p1.n_rings, p2.n_rings
        p1.extend(p2)
        assert p1.n_rings == n1 + n2


class TestGetWallNodesFlat:
    def test_shape(self):
        pipe = PipeGeometry.straight(np.zeros(3), np.array([1, 0, 0]), 0.5,
                                     n_sides=8, ring_spacing=0.1)
        nodes = pipe.get_wall_nodes_flat()
        assert nodes.shape == (pipe.n_rings * pipe.n_sides, 3)


class TestAddToSimulation:
    """Integration tests requiring py_dismech. Skipped if not installed."""

    @pytest.fixture
    def sim_setup(self):
        py_dismech = pytest.importorskip("py_dismech")
        sim = py_dismech.SimulationManager()
        sim.sim_params.dt = 0.01
        max_iter = py_dismech.MaxIterations()
        max_iter.num_iters = 100
        sim.sim_params.max_iter = max_iter
        sim.render_params.renderer = py_dismech.HEADLESS
        return sim, py_dismech

    def test_adds_limbs(self, sim_setup):
        sim, py_dismech = sim_setup
        soft_robots = sim.soft_robots

        # Add snake limb first
        soft_robots.addLimb(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.3, 0.0, 0.0]),
            10, 1000.0, 0.005, 1e6, 0.5, 0.0, 0x0001,
        )
        assert len(soft_robots.limbs) == 1

        pipe = PipeGeometry.straight(
            start=np.array([-0.05, 0, 0]),
            direction=np.array([1, 0, 0]),
            length=0.4,
            radius=0.05,
            n_sides=8,
            ring_spacing=0.1,
        )

        info = pipe.add_to_simulation(soft_robots, sim.forces, snake_col_group=0x0001)

        assert info["first_wall_limb"] == 1
        assert info["n_rings"] == pipe.n_rings
        assert info["n_wall_edges"] == pipe.n_rings * (pipe.n_sides - 1)
        assert len(soft_robots.limbs) == 1 + pipe.n_rings

    def test_simulation_initializes(self, sim_setup):
        """Verify the simulation can initialize with pipe walls present."""
        sim, py_dismech = sim_setup
        soft_robots = sim.soft_robots

        soft_robots.addLimb(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.2, 0.0, 0.0]),
            6, 1000.0, 0.005, 1e6, 0.5, 0.0, 0x0001,
        )

        pipe = PipeGeometry.straight(
            start=np.array([-0.02, 0, 0]),
            direction=np.array([1, 0, 0]),
            length=0.24,
            radius=0.04,
            n_sides=6,
            ring_spacing=0.08,
        )
        pipe.add_to_simulation(soft_robots, sim.forces)

        # This should not raise
        sim.initialize([])


# --- Test helpers ---

def _ring_angles(ring: PipeRing) -> np.ndarray:
    """Get angular positions of ring nodes relative to center."""
    radials = ring.nodes - ring.center
    # Project out the normal
    projected = radials - np.outer(radials @ ring.normal, ring.normal)
    # Get angles using first node's direction as reference
    e1 = projected[0] / np.linalg.norm(projected[0])
    e2 = np.cross(ring.normal, e1)
    angles = np.arctan2(projected @ e2, projected @ e1)
    return angles
