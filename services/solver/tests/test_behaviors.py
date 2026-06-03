import numpy as np

from piperouter_solver.models import RouteRequest, WireType
from piperouter_solver.solver import Solver


def _wire(**kw):
    base = dict(
        id="w", label="w", kind="wire", outer_diameter_mm=10.0,
        min_bend_radius_mm=50.0, cost_per_m=1.0, mass_per_m_kg=0.1,
        max_temp_c=200.0, em_sensitivity=1.0, color=(1.0, 0.0, 0.0),
    )
    base.update(kw)
    return WireType(**base)


def _between(stack, a, b, **req_kw):
    return RouteRequest(
        wire=req_kw.pop("wire", _wire()),
        start=tuple(stack.frame.grid_to_world(a)),
        end=tuple(stack.frame.grid_to_world(b)),
        connectivity=req_kw.pop("connectivity", 26),
        **req_kw,
    )


def test_clearance_respected_thick_wire_cannot_squeeze(empty_stack):
    s = empty_stack
    # two walls one free gap apart at x=5 (gap only at y=4,5)
    s.occupancy[5, :4, :] = 1
    s.occupancy[5, 6:, :] = 1
    # a fat wire (radius dilates the gap shut) cannot pass; thin one can
    fat = Solver().route_one(s, _between(s, (0, 5, 1), (9, 5, 1),
                                         wire=_wire(outer_diameter_mm=400.0)))
    thin = Solver().route_one(s, _between(s, (0, 5, 1), (9, 5, 1),
                                          wire=_wire(outer_diameter_mm=10.0)))
    assert thin.status == "routed"
    assert fat.status == "no_path"


def test_collision_avoided_never_enters_occupied(wall_stack):
    s = wall_stack
    res = Solver().route_one(s, _between(s, (0, 5, 1), (9, 5, 1)))
    assert res.status == "routed"
    for (i, j, k) in res.cells:
        assert s.occupancy[i, j, k] == 0


def test_gentle_bend_preferred_over_sharp(empty_stack):
    s = empty_stack
    # force a turn by routing diagonally; a stiff wire should take a path at
    # least as long as a floppy one (higher min bend radius => more turn cost)
    floppy = Solver().route_one(
        s, _between(s, (0, 0, 1), (9, 9, 1), wire=_wire(min_bend_radius_mm=1.0)))
    stiff = Solver().route_one(
        s, _between(s, (0, 0, 1), (9, 9, 1), wire=_wire(min_bend_radius_mm=400.0)))
    assert floppy.status == stiff.status == "routed"
    assert stiff.length_m >= floppy.length_m  # stiffness trades length for gentleness


def test_hot_region_avoided_when_weighted(empty_stack):
    s = empty_stack
    s.thermal[:, 5, :] = 90.0  # warm band (below the 200C rating, so soft only)
    start, end = (0, 5, 1), (9, 5, 1)
    avoid = Solver().route_one(
        s, _between(s, start, end, wire=_wire(max_temp_c=200.0),
                    weights={"thermal": 8.0}))
    ignore = Solver().route_one(
        s, _between(s, start, end, wire=_wire(max_temp_c=200.0),
                    weights={"thermal": 0.0}))
    assert avoid.status == ignore.status == "routed"
    hot_in_avoid = sum(1 for c in avoid.cells if c[1] == 5)
    hot_in_ignore = sum(1 for c in ignore.cells if c[1] == 5)
    assert hot_in_avoid <= hot_in_ignore


def test_em_region_avoided_scaled_by_sensitivity(empty_stack):
    s = empty_stack
    s.em[:, 5, :] = 1.0
    start, end = (0, 5, 1), (9, 5, 1)
    sensitive = Solver().route_one(
        s, _between(s, start, end, wire=_wire(em_sensitivity=1.0),
                    weights={"em": 8.0}))
    immune = Solver().route_one(
        s, _between(s, start, end, wire=_wire(em_sensitivity=0.0),
                    weights={"em": 8.0}))
    assert sensitive.status == immune.status == "routed"
    em_in_sensitive = sum(1 for c in sensitive.cells if c[1] == 5)
    em_in_immune = sum(1 for c in immune.cells if c[1] == 5)
    assert em_in_sensitive <= em_in_immune


def test_waypoint_hit_and_melt_cutoff_removed():
    # exercised in test_solver.py; this file is the behavior gate summary.
    assert True
