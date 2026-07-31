import numpy as np

from piperouter_solver.lattice import ExpandedLatticeBuilder, LatticeGraph
from piperouter_solver.models import WireType


def _wire():
    return WireType(
        id="t", label="t", kind="wire", outer_diameter_mm=10.0,
        min_bend_radius_mm=50.0, cost_per_m=1.0, mass_per_m_kg=0.1,
        max_temp_c=200.0, em_sensitivity=0.0, color=(1.0, 0.0, 0.0),
    )


def _cells(g):
    return {tuple(int(v) for v in row) for row in g.ordinal_cells}


def test_build_returns_graph_with_source_and_sink(empty_stack):
    builder = ExpandedLatticeBuilder()
    g = builder.build(
        empty_stack, _wire(), weights={}, connectivity=6,
        start_cell=(0, 0, 0), goal_cell=(9, 0, 0), extra_obstacles=None,
    )
    assert isinstance(g, LatticeGraph)
    assert g.source_id == g.n_nodes - 2
    assert g.sink_id == g.n_nodes - 1
    assert g.src.shape == g.dst.shape == g.weight.shape
    assert (g.weight >= 0).all()


def test_blocked_cells_have_no_nodes(empty_stack):
    s = empty_stack
    s.occupancy[5, 0, 0] = 1
    builder = ExpandedLatticeBuilder()
    g = builder.build(
        s, _wire(), weights={}, connectivity=6,
        start_cell=(0, 0, 0), goal_cell=(9, 0, 0), extra_obstacles=None,
    )
    # no free-cell ordinal maps back to the blocked cell
    assert (5, 0, 0) not in _cells(g)


def test_bend_weight_scales_turn_penalty(empty_stack):
    wire = WireType(
        id="t", label="t", kind="wire", outer_diameter_mm=10.0,
        min_bend_radius_mm=400.0, cost_per_m=1.0, mass_per_m_kg=0.1,
        max_temp_c=200.0, em_sensitivity=0.0, color=(1.0, 0.0, 0.0))
    b = ExpandedLatticeBuilder()
    g0 = b.build(empty_stack, wire, weights={"bend": 0.0}, connectivity=26,
                 start_cell=(0, 5, 1), goal_cell=(9, 5, 1), extra_obstacles=None)
    g4 = b.build(empty_stack, wire, weights={"bend": 4.0}, connectivity=26,
                 start_cell=(0, 5, 1), goal_cell=(9, 5, 1), extra_obstacles=None)
    # bend=0 removes all turn cost, so any bend weight lifts the costliest turn edges
    assert g4.weight.max() > g0.weight.max()


def test_extra_obstacles_remove_cells(empty_stack):
    builder = ExpandedLatticeBuilder()
    obstacles = np.zeros(empty_stack.frame.res_xyz, dtype=bool)
    obstacles[3, 0, 0] = True
    g = builder.build(
        empty_stack, _wire(), weights={}, connectivity=6,
        start_cell=(0, 0, 0), goal_cell=(9, 0, 0), extra_obstacles=obstacles,
    )
    assert (3, 0, 0) not in _cells(g)
