"""Pinned headings must PREFER the exact direction, not just anything within the
45-degree cone (the rotatable-gizmo feature exposes arbitrary angles)."""
import numpy as np

from piperouter_solver.models import RouteRequest, WireType
from piperouter_solver.solver import Solver


def _wire():
    return WireType(id="w", label="w", kind="wire", outer_diameter_mm=4.0,
                    min_bend_radius_mm=20.0, cost_per_m=1.0, mass_per_m_kg=0.1,
                    max_temp_c=200.0, em_sensitivity=0.0, color=(1.0, 1.0, 0.0))


def _first_step(res, start_cell):
    # cells[0] is the first cell AFTER the source, so the departure direction is
    # cells[0] - start_cell (the cone/alignment-gated first move).
    d = np.asarray(res.cells[0], float) - np.asarray(start_cell, float)
    return d / (np.linalg.norm(d) + 1e-12)


def test_departure_follows_exact_heading_ray(empty_stack):
    # heading at azimuth ~30 deg (not a lattice direction): the departure stub
    # rasterizes the EXACT ray, so the direction over the stub must match the arrow
    # far better than any single lattice offset (+X is 30 deg off, diagonal 15 deg off).
    s = empty_stack
    h = np.array([np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0])
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((0, 2, 1))),
        end=tuple(s.frame.grid_to_world((9, 2, 1))), connectivity=26,
        start_heading=tuple(h), weights={"smoothing": 0.0, "bend": 0.0}))
    assert res.status == "routed"
    d = np.asarray(res.cells[1], float) - np.asarray((0, 2, 1), float)  # over the stub
    d /= np.linalg.norm(d)
    assert float(d @ h) > 0.99, f"stub direction {d} vs heading {h}"


def test_departure_follows_plain_axis_when_aligned(empty_stack):
    # a heading exactly along +X must depart along +X (no reason to pay the diagonal)
    s = empty_stack
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((0, 5, 1))),
        end=tuple(s.frame.grid_to_world((9, 5, 1))), connectivity=26,
        start_heading=(1.0, 0.0, 0.0), weights={"smoothing": 0.0, "bend": 0.0}))
    assert res.status == "routed"
    step = _first_step(res, (0, 5, 1))
    assert float(step @ np.array([1.0, 0.0, 0.0])) > 0.99


def _steps(cells):
    return [tuple(int(b - a) for a, b in zip(p, q)) for p, q in zip(cells[:-1], cells[1:])]


def test_start_heading_forces_straight_exit(empty_stack):
    # heading +Y with bend weight 0: without the stub the route could turn 90 degrees
    # after ONE voxel. The stub forces >= 2 straight cells along the arrow first.
    s = empty_stack
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((2, 2, 1))),
        end=tuple(s.frame.grid_to_world((9, 2, 1))), connectivity=26,
        start_heading=(0.0, 1.0, 0.0), weights={"smoothing": 0.0, "bend": 0.0}))
    assert res.status == "routed"
    first = [np.subtract(res.cells[0], (2, 2, 1))] + _steps(res.cells[:3])
    assert all(tuple(d) == (0, 1, 0) for d in first[:2]), \
        f"exit not straight along +Y: {first}"


def test_end_heading_forces_straight_arrival(empty_stack):
    # arrival along +X: the LAST >= 2 steps into the goal must be +X even with bend 0.
    s = empty_stack
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((0, 8, 1))),
        end=tuple(s.frame.grid_to_world((9, 2, 1))), connectivity=26,
        end_heading=(1.0, 0.0, 0.0), weights={"smoothing": 0.0, "bend": 0.0}))
    assert res.status == "routed"
    last = _steps(res.cells[-3:])
    assert all(d == (1, 0, 0) for d in last), f"arrival not straight along +X: {last}"


def test_blocked_runway_shortens_stub_instead_of_failing(empty_stack):
    # a wall right after the start along the heading: the stub shortens and the wire
    # still routes (graceful degradation, no no_path).
    s = empty_stack
    s.occupancy[2, 4, :] = 1                 # blocks the +Y runway 2 cells out
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((2, 2, 1))),
        end=tuple(s.frame.grid_to_world((9, 2, 1))), connectivity=26,
        start_heading=(0.0, 1.0, 0.0), weights={"smoothing": 0.0, "bend": 0.0}))
    assert res.status == "routed"
