import numpy as np

from piperouter_solver import planners
from piperouter_solver.models import WireType


def _wire(**kw):
    base = dict(id="w", label="w", kind="wire", outer_diameter_mm=10.0,
                min_bend_radius_mm=50.0, cost_per_m=1.0, mass_per_m_kg=0.1,
                max_temp_c=200.0, em_sensitivity=1.0, color=(1.0, 0.0, 0.0))
    base.update(kw)
    return WireType(**base)


def test_every_global_planner_routes_around_a_wall_collision_free(empty_stack):
    s = empty_stack                 # 10x10x3
    s.occupancy[5, :8, :] = 1       # wall across x=5, gap only at y=8,9
    wire = _wire()
    weights = {"surface": 1.0, "bend": 1.0}
    a, b = (0, 5, 1), (9, 5, 1)
    blocked = planners.blocked_mask(s, wire, 0.0, None)

    for name, cls in planners.GLOBAL_PLANNERS.items():
        cells = cls().plan(s, wire, weights, 26, a, b, None, 0.0, None, None)
        assert cells, f"{name}: no path found"
        for c in cells:
            assert not blocked[tuple(c)], f"{name}: routed through blocked cell {c}"
        assert np.linalg.norm(np.subtract(cells[0], a)) <= 3   # starts near start
        assert np.linalg.norm(np.subtract(cells[-1], b)) <= 3  # ends near goal


def _path_len(cells, cell_size):
    p = [np.asarray(c, dtype=float) for c in cells]
    return cell_size * sum(float(np.linalg.norm(b - a)) for a, b in zip(p[:-1], p[1:]))


def test_octree_lattice_matches_lattice(empty_stack):
    # the corridor (octree) + banded lattice must produce a route of essentially the same
    # length as the full lattice (it runs the same lattice, just restricted to a band).
    s = empty_stack
    s.occupancy[5, :8, :] = 1
    wire = _wire()
    weights = {"surface": 1.0, "bend": 1.0}
    a, b = (0, 5, 1), (9, 5, 1)
    lat = planners.LatticeGlobal().plan(s, wire, weights, 26, a, b, None, 0.0, None, None)
    olt = planners.OctreeLatticeGlobal().plan(s, wire, weights, 26, a, b, None, 0.0, None, None)
    assert lat and olt
    cs = s.frame.cell_size
    assert abs(_path_len(olt, cs) - _path_len(lat, cs)) < 3 * cs   # within a few cells


def test_rrt_is_deterministic(empty_stack):
    s = empty_stack
    s.occupancy[5, :8, :] = 1
    wire = _wire()
    p1 = planners.RRTGlobal().plan(s, wire, {}, 26, (0, 5, 1), (9, 5, 1), None, 0.0, None, None)
    p2 = planners.RRTGlobal().plan(s, wire, {}, 26, (0, 5, 1), (9, 5, 1), None, 0.0, None, None)
    assert p1 == p2 and p1 is not None        # fixed seed -> identical route
