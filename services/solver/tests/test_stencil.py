"""The implicit lattice solver must agree with the materialized graph.

stencil.solve relaxes the (cell x heading) lattice in place instead of building it, so
these tests pin the property that makes that safe: identical edge semantics, and
therefore identical path cost, against ExpandedLatticeBuilder + shortest_path.
"""
import numpy as np
import pytest

from piperouter_solver import backend, fields, planners, stencil
from piperouter_solver.grids import GridStack
from piperouter_solver.lattice import ExpandedLatticeBuilder
from piperouter_solver.models import GridFrame, WireType


def _wire():
    return WireType(id="w", label="w", kind="wire", outer_diameter_mm=1.0,
                    min_bend_radius_mm=30.0, cost_per_m=1.0, mass_per_m_kg=0.1,
                    max_temp_c=1e9, em_sensitivity=0.5, color=(1, 1, 0))


def _stack(shape, p, seed):
    rng = np.random.default_rng(seed)
    occ = (rng.random(shape) < p).astype(np.uint8)
    occ[0, 0, 0] = 0
    occ[-1, -1, -1] = 0
    return GridStack(
        frame=GridFrame(bounds_min=np.zeros(3), cell_size=0.05, res_xyz=shape),
        occupancy=occ, surface_dist=(rng.random(shape) * 3).astype(np.float32),
        thermal=np.full(shape, 20.0, dtype=np.float32),
        em=rng.random(shape).astype(np.float32))


def _both(shape, p, seed, bend=1.0, conn=26):
    st, w = _stack(shape, p, seed), _wire()
    weights = {"surface": 2.0, "em": 1.0, "bend": bend}
    a, b = (0, 0, 0), (shape[0] - 1, shape[1] - 1, shape[2] - 1)
    g = ExpandedLatticeBuilder().build(st, w, weights, conn, a, b, None, clearance_m=0.0)
    _path, ref_cost = backend.shortest_path(g.src, g.dst, g.weight, g.n_nodes,
                                            g.source_id, g.sink_id)
    blocked = st.dilate_occupancy(w.radius_m).astype(bool) | fields.melt_mask(st, w)
    soft = fields.soft_cost_field(st, w, weights)
    offs = stencil.offsets_for(conn)
    lut = stencil.build_turn_lut(offs, w.min_bend_radius_mm,
                                 st.frame.cell_size * 1000.0, bend)
    cells, cost = stencil.solve(~blocked, soft, st.frame.cell_size, offs, lut, a, b,
                                xp=np)
    return ref_cost, cost, cells, a, b


@pytest.mark.parametrize("shape,p,seed,bend", [
    ((10, 10, 4), 0.10, 1, 1.0), ((12, 9, 6), 0.18, 2, 1.0),
    ((14, 14, 3), 0.22, 3, 3.0), ((9, 9, 9), 0.05, 4, 0.0),
    ((16, 7, 5), 0.30, 5, 1.0),
])
def test_cost_matches_the_materialized_graph(shape, p, seed, bend):
    ref, got, _cells, _a, _b = _both(shape, p, seed, bend)
    assert got == pytest.approx(ref, abs=1e-4)


def test_path_is_contiguous_and_spans_the_endpoints():
    _ref, _got, cells, a, b = _both((12, 9, 6), 0.18, 2)
    assert cells[0] == a and cells[-1] == b
    offs = set(stencil.offsets_for(26))
    for p_, q in zip(cells[:-1], cells[1:]):
        assert tuple(int(y - x) for x, y in zip(p_, q)) in offs


def test_path_never_enters_a_blocked_cell():
    st, w = _stack((14, 11, 5), 0.25, 9), _wire()
    weights = {"surface": 1.0, "bend": 1.0}
    blocked = st.dilate_occupancy(w.radius_m).astype(bool) | fields.melt_mask(st, w)
    soft = fields.soft_cost_field(st, w, weights)
    offs = stencil.offsets_for(26)
    lut = stencil.build_turn_lut(offs, w.min_bend_radius_mm,
                                 st.frame.cell_size * 1000.0, 1.0)
    cells, cost = stencil.solve(~blocked, soft, st.frame.cell_size, offs, lut,
                                (0, 0, 0), (13, 10, 4), xp=np)
    if cells is not None:
        assert np.isfinite(cost)
        for c in cells:
            assert not blocked[c]


def test_unreachable_goal_reports_no_path():
    shape = (9, 9, 3)
    occ = np.zeros(shape, dtype=np.uint8)
    occ[4, :, :] = 1                       # solid wall, no opening
    st = GridStack(frame=GridFrame(bounds_min=np.zeros(3), cell_size=0.05,
                                   res_xyz=shape),
                   occupancy=occ, surface_dist=np.ones(shape, dtype=np.float32),
                   thermal=np.full(shape, 20.0, dtype=np.float32),
                   em=np.zeros(shape, dtype=np.float32))
    w = _wire()
    blocked = st.dilate_occupancy(w.radius_m).astype(bool)
    soft = fields.soft_cost_field(st, w, {"surface": 1.0})
    offs = stencil.offsets_for(26)
    lut = stencil.build_turn_lut(offs, w.min_bend_radius_mm, 50.0, 1.0)
    cells, cost = stencil.solve(~blocked, soft, 0.05, offs, lut,
                                (0, 4, 1), (8, 4, 1), xp=np)
    assert cells is None and cost == float("inf")


def test_dense_planner_is_registered_and_routes():
    st, w = _stack((11, 9, 4), 0.12, 6), _wire()
    g = planners.make_global("dense")
    assert isinstance(g, planners.DenseGlobal)
    cells = g.plan(st, w, {"surface": 1.0, "bend": 1.0}, 26, (0, 0, 0), (10, 8, 3),
                   None, 0.0)
    assert cells and cells[-1] == (10, 8, 3)
